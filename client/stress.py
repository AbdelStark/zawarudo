"""Concurrent WMCP stress tester for the Docker Compose demo.

Where :mod:`client.traffic` keeps a calm, serial background stream so dashboards have data,
this module drives a *high-load* client: many concurrent workers submitting a configurable,
randomized mix of operations (``score`` / ``plan`` / ``rollout`` / ``encode``) with varied tensor
shapes, an optional slice of intentionally-invalid requests, and a latency-percentile summary at
the end. It is dependency-free (stdlib threads + urllib) and reuses the same payload builders as
the demo client, so it runs in the existing ``wmcp-pusht-client`` image with no extra deps.

Everything is configurable via ``WMCP_STRESS_*`` environment variables (see ``main``); the numeric
knobs accept the literal ``auto`` to pick a backend-aware default (lighter shapes/concurrency for the
CPU ``lewm`` backend, heavier for ``mock``). CLI flags override env, which overrides the auto default.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from . import payloads
from .demo import wait_until_ready
from .wmcp_client import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, WMCPClient, WMCPError

OPERATIONS = ("score", "plan", "rollout", "encode")


# --- configuration -------------------------------------------------------------------------------


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _auto(value: Optional[str]) -> bool:
    """Whether a raw value means 'pick a backend-aware default'."""
    return value is None or value.strip() == "" or value.strip().lower() == "auto"


def _auto_int(value: Optional[str], *, mock: int, lewm: int, backend: str) -> int:
    if _auto(value):
        return lewm if backend == "lewm" else mock
    return int(value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class StressConfig:
    concurrency: int
    duration: float           # seconds; 0 = until ``total`` (or forever if total is 0 too)
    total: int                # total request cap; 0 = unbounded
    target_rps: float         # 0 = unthrottled (as fast as the workers can go)
    ops: tuple[tuple[str, int], ...]   # weighted (operation, weight) mix
    invalid_ratio: float
    min_candidates: int
    max_candidates: int
    min_horizon: int
    max_horizon: int
    min_iterations: int
    max_iterations: int
    ramp: float               # seconds to stagger workers up to full concurrency
    report_interval: float
    seed: int
    real_pixels: bool

    def pixel_encoding(self, backend: str) -> str:
        return "base64" if backend == "lewm" and self.real_pixels else "uri"


def parse_ops(spec: str) -> tuple[tuple[str, int], ...]:
    """Parse an operation mix like ``"score:4,plan:2,rollout:2,encode:1"`` or ``"score,plan"``."""
    mix: list[tuple[str, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        name, _, weight = token.partition(":")
        name = name.strip().lower()
        if name not in OPERATIONS:
            raise ValueError(f"unknown operation '{name}' (expected one of {', '.join(OPERATIONS)})")
        mix.append((name, int(weight) if weight.strip() else 1))
    if not mix:
        raise ValueError("operation mix is empty")
    return tuple(mix)


def config_from_env(backend: str) -> StressConfig:
    return StressConfig(
        concurrency=_auto_int(os.getenv("WMCP_STRESS_CONCURRENCY"), mock=24, lewm=8, backend=backend),
        duration=float(os.getenv("WMCP_STRESS_DURATION", "120")),
        total=int(os.getenv("WMCP_STRESS_TOTAL", "0")),
        target_rps=float(os.getenv("WMCP_STRESS_TARGET_RPS", "0")),
        ops=parse_ops(os.getenv("WMCP_STRESS_OPS", "score:4,plan:2,rollout:2,encode:1")),
        invalid_ratio=float(os.getenv("WMCP_STRESS_INVALID_RATIO", "0.02")),
        min_candidates=_auto_int(os.getenv("WMCP_STRESS_MIN_CANDIDATES"), mock=16, lewm=4, backend=backend),
        max_candidates=_auto_int(os.getenv("WMCP_STRESS_MAX_CANDIDATES"), mock=192, lewm=16, backend=backend),
        min_horizon=_auto_int(os.getenv("WMCP_STRESS_MIN_HORIZON"), mock=4, lewm=2, backend=backend),
        max_horizon=_auto_int(os.getenv("WMCP_STRESS_MAX_HORIZON"), mock=16, lewm=6, backend=backend),
        min_iterations=_auto_int(os.getenv("WMCP_STRESS_MIN_ITERATIONS"), mock=3, lewm=1, backend=backend),
        max_iterations=_auto_int(os.getenv("WMCP_STRESS_MAX_ITERATIONS"), mock=8, lewm=3, backend=backend),
        ramp=float(os.getenv("WMCP_STRESS_RAMP", "5")),
        report_interval=float(os.getenv("WMCP_STRESS_REPORT_INTERVAL", "5")),
        seed=int(os.getenv("WMCP_STRESS_SEED", "11")),
        real_pixels=_truthy(os.getenv("WMCP_STRESS_REAL_PIXELS", "true")),
    )


# --- live statistics -----------------------------------------------------------------------------


@dataclass
class Stats:
    """Thread-safe rolling counters and latency samples."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    submitted: int = 0
    ok: int = 0
    failed: int = 0
    expected_errors: int = 0
    transport_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    per_op: dict[str, int] = field(default_factory=dict)
    started_at: float = 0.0

    def record(self, operation: str, *, ok: bool, expected: bool, transport: bool, latency_ms: float) -> None:
        with self.lock:
            self.submitted += 1
            self.per_op[operation] = self.per_op.get(operation, 0) + 1
            self.latencies_ms.append(latency_ms)
            if ok:
                self.ok += 1
            elif expected:
                self.expected_errors += 1
            elif transport:
                self.transport_errors += 1
            else:
                self.failed += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "submitted": self.submitted,
                "ok": self.ok,
                "failed": self.failed,
                "expected_errors": self.expected_errors,
                "transport_errors": self.transport_errors,
                "per_op": dict(self.per_op),
                "latencies_ms": list(self.latencies_ms),
            }


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[int(rank)]
    return sorted_values[low] * (high - rank) + sorted_values[high] * (rank - low)


# --- payload building ----------------------------------------------------------------------------


def build_request(operation: str, cfg: StressConfig, backend: str, rng: random.Random, request_id: str) -> dict[str, Any]:
    """A randomized request for ``operation`` within the configured shape ranges."""
    encoding = cfg.pixel_encoding(backend)
    seed = rng.randrange(1_000_000)
    candidates = rng.randint(cfg.min_candidates, max(cfg.min_candidates, cfg.max_candidates))
    horizon = rng.randint(cfg.min_horizon, max(cfg.min_horizon, cfg.max_horizon))

    if operation == "score":
        return payloads.score_request(
            request_id, s=candidates, t=horizon, inline_actions=True, pixel_encoding=encoding, seed=seed
        )
    if operation == "rollout":
        return payloads.rollout_request(
            request_id, s=candidates, t=horizon, inline_actions=True, pixel_encoding=encoding, seed=seed
        )
    if operation == "encode":
        return payloads.encode_request(request_id, pixel_encoding=encoding)
    if operation == "plan":
        iterations = rng.randint(cfg.min_iterations, max(cfg.min_iterations, cfg.max_iterations))
        return payloads.plan_request(
            request_id, horizon=horizon, iterations=iterations, candidates=candidates, pixel_encoding=encoding, seed=seed
        )
    raise ValueError(f"unsupported operation: {operation}")  # pragma: no cover


# --- worker loop ---------------------------------------------------------------------------------


def _weighted_choice(rng: random.Random, ops: tuple[tuple[str, int], ...]) -> str:
    names = [name for name, _ in ops]
    weights = [weight for _, weight in ops]
    return rng.choices(names, weights=weights, k=1)[0]


def _worker(
    worker_id: int,
    client: WMCPClient,
    cfg: StressConfig,
    backend: str,
    stats: Stats,
    stop: threading.Event,
    deadline: Optional[float],
) -> None:
    rng = random.Random(cfg.seed + worker_id * 7919)
    # Ramp: stagger worker start so concurrency climbs smoothly instead of a thundering herd.
    if cfg.ramp > 0 and cfg.concurrency > 1:
        stop.wait(cfg.ramp * worker_id / cfg.concurrency)
    per_worker_min_interval = cfg.concurrency / cfg.target_rps if cfg.target_rps > 0 else 0.0
    counter = 0

    while not stop.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            break
        if cfg.total and stats.submitted >= cfg.total:
            break

        counter += 1
        invalid = cfg.invalid_ratio > 0 and rng.random() < cfg.invalid_ratio
        operation = _weighted_choice(rng, cfg.ops)
        request_id = f"stress-w{worker_id}-{operation}-{counter}"
        if invalid:
            request: dict[str, Any] = {"wmcp_version": "0.1", "inputs": {}}
        else:
            request = build_request(operation, cfg, backend, rng, request_id)

        started = time.perf_counter()
        try:
            getattr(client, operation)(request)
            latency_ms = (time.perf_counter() - started) * 1000.0
            stats.record(operation, ok=True, expected=False, transport=False, latency_ms=latency_ms)
        except WMCPError:
            latency_ms = (time.perf_counter() - started) * 1000.0
            stats.record(operation, ok=False, expected=invalid, transport=False, latency_ms=latency_ms)
        except OSError:
            latency_ms = (time.perf_counter() - started) * 1000.0
            stats.record(operation, ok=False, expected=False, transport=True, latency_ms=latency_ms)

        if per_worker_min_interval:
            elapsed = (time.perf_counter() - started)
            if per_worker_min_interval > elapsed:
                stop.wait(per_worker_min_interval - elapsed)


def _reporter(cfg: StressConfig, stats: Stats, stop: threading.Event) -> None:
    last_submitted = 0
    last_time = time.monotonic()
    while not stop.wait(cfg.report_interval):
        snap = stats.snapshot()
        now = time.monotonic()
        window = max(now - last_time, 1e-9)
        rps = (snap["submitted"] - last_submitted) / window
        elapsed = now - stats.started_at
        lat = sorted(snap["latencies_ms"])
        p95 = _percentile(lat, 0.95)
        print(
            f"[{elapsed:6.1f}s] sent={snap['submitted']:>7} rps={rps:7.1f} "
            f"ok={snap['ok']} expected_err={snap['expected_errors']} "
            f"failed={snap['failed']} transport_err={snap['transport_errors']} p95={p95:7.1f}ms",
            flush=True,
        )
        last_submitted = snap["submitted"]
        last_time = now


def _print_summary(cfg: StressConfig, backend: str, stats: Stats, wall_seconds: float) -> None:
    snap = stats.snapshot()
    lat = sorted(snap["latencies_ms"])
    submitted = snap["submitted"]
    throughput = submitted / wall_seconds if wall_seconds > 0 else 0.0
    error_total = snap["failed"] + snap["transport_errors"]
    error_rate = (error_total / submitted * 100.0) if submitted else 0.0

    print("\n" + "=" * 68, flush=True)
    print("WMCP stress test summary", flush=True)
    print("=" * 68, flush=True)
    print(f"  backend             {backend}", flush=True)
    print(f"  concurrency         {cfg.concurrency}", flush=True)
    print(f"  wall time           {wall_seconds:.1f}s", flush=True)
    print(f"  requests sent       {submitted}", flush=True)
    print(f"  throughput          {throughput:.1f} req/s", flush=True)
    print(f"  ok                  {snap['ok']}", flush=True)
    print(f"  expected errors     {snap['expected_errors']} (intentional invalid)", flush=True)
    print(f"  unexpected failures {snap['failed']}", flush=True)
    print(f"  transport errors    {snap['transport_errors']}", flush=True)
    print(f"  error rate          {error_rate:.2f}% (unexpected only)", flush=True)
    if lat:
        print("  latency ms          "
              f"p50={_percentile(lat, 0.50):.1f} "
              f"p90={_percentile(lat, 0.90):.1f} "
              f"p95={_percentile(lat, 0.95):.1f} "
              f"p99={_percentile(lat, 0.99):.1f} "
              f"max={lat[-1]:.1f}", flush=True)
    if snap["per_op"]:
        mix = " ".join(f"{op}={count}" for op, count in sorted(snap["per_op"].items()))
        print(f"  per-operation       {mix}", flush=True)
    print("=" * 68, flush=True)


# --- orchestration -------------------------------------------------------------------------------


def run_stress(client: WMCPClient, cfg: Optional[StressConfig] = None) -> int:
    if not wait_until_ready(client, attempts=120, delay=1.0):
        print(f"service at {client.base_url} did not become ready", flush=True)
        return 1

    backend = str(client.metadata().get("runtime", {}).get("backend") or "mock")
    cfg = cfg or config_from_env(backend)

    # Don't let the ramp-up outlast a short run, or late workers never start.
    if cfg.duration > 0 and cfg.ramp > cfg.duration * 0.5:
        cfg = replace(cfg, ramp=cfg.duration * 0.5)

    horizon = "infinite" if (cfg.duration <= 0 and cfg.total <= 0) else (
        f"{cfg.total} requests" if cfg.duration <= 0 else f"{cfg.duration:.0f}s"
    )
    ops_desc = ",".join(f"{name}:{weight}" for name, weight in cfg.ops)
    print(
        f"stress backend={backend} concurrency={cfg.concurrency} run={horizon} "
        f"ops={ops_desc} target_rps={'unthrottled' if cfg.target_rps <= 0 else cfg.target_rps} "
        f"invalid_ratio={cfg.invalid_ratio} "
        f"candidates={cfg.min_candidates}-{cfg.max_candidates} horizon={cfg.min_horizon}-{cfg.max_horizon}",
        flush=True,
    )

    stats = Stats()
    stop = threading.Event()
    stats.started_at = time.monotonic()
    deadline = stats.started_at + cfg.duration if cfg.duration > 0 else None

    reporter = threading.Thread(target=_reporter, args=(cfg, stats, stop), daemon=True)
    reporter.start()

    workers = [
        threading.Thread(target=_worker, args=(i, client, cfg, backend, stats, stop, deadline), daemon=True)
        for i in range(cfg.concurrency)
    ]
    for worker in workers:
        worker.start()

    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        print("\ninterrupted — stopping workers", flush=True)
    finally:
        stop.set()

    wall_seconds = time.monotonic() - stats.started_at
    _print_summary(cfg, backend, stats, wall_seconds)
    # Non-zero exit only on unexpected failures, so the container surfaces real problems.
    return 1 if stats.failed > 0 else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Concurrent WMCP stress tester for the Push-T demo")
    parser.add_argument("--base-url", default=os.getenv("WMCP_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model-id", default=os.getenv("WMCP_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--concurrency", type=int, default=None, help="overrides WMCP_STRESS_CONCURRENCY")
    parser.add_argument("--duration", type=float, default=None, help="seconds to run (0 = until --total / forever)")
    parser.add_argument("--total", type=int, default=None, help="total request cap (0 = unbounded)")
    parser.add_argument("--target-rps", type=float, default=None, help="throttle to this aggregate rps (0 = unthrottled)")
    parser.add_argument("--ops", default=None, help='operation mix, e.g. "score:4,plan:2,rollout:2,encode:1"')
    parser.add_argument("--invalid-ratio", type=float, default=None, help="fraction of intentionally invalid requests")
    args = parser.parse_args(argv)

    # CLI flags win over env: fold them into the environment that run_stress reads (after it
    # detects the backend), so there is a single source of truth and no double readiness probe.
    cli_env = {
        "WMCP_STRESS_CONCURRENCY": args.concurrency,
        "WMCP_STRESS_DURATION": args.duration,
        "WMCP_STRESS_TOTAL": args.total,
        "WMCP_STRESS_TARGET_RPS": args.target_rps,
        "WMCP_STRESS_OPS": args.ops,
        "WMCP_STRESS_INVALID_RATIO": args.invalid_ratio,
    }
    for key, value in cli_env.items():
        if value is not None:
            os.environ[key] = str(value)

    timeout = float(os.getenv("WMCP_STRESS_TIMEOUT", os.getenv("WMCP_TIMEOUT", "120")))
    client = WMCPClient(args.base_url, model_id=args.model_id, timeout=timeout)
    return run_stress(client)


if __name__ == "__main__":
    raise SystemExit(main())
