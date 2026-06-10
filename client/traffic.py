"""Continuous WMCP traffic generator for the Docker Compose demo.

The generator is intentionally dependency-free and reuses the same payload builders as the demo
client. It keeps the real ``lewm`` backend on a small request profile while using larger mock shapes
to make Prometheus and Grafana useful immediately after ``docker compose up``.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import payloads
from .demo import wait_until_ready
from .wmcp_client import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, WMCPClient, WMCPError


@dataclass(frozen=True)
class TrafficCall:
    operation: str
    request: dict[str, Any]
    expect_error: bool = False


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _backend_name(metadata: dict[str, Any]) -> str:
    return str(metadata.get("runtime", {}).get("backend") or "mock")


def _pixel_encoding(backend: str, *, real_pixels: bool) -> str:
    return "base64" if backend == "lewm" and real_pixels else "uri"


def _score_shape(backend: str, cycle: int) -> tuple[int, int]:
    if backend == "lewm":
        return (6 + (cycle % 3) * 2, 4)
    candidates = (16, 32, 64, 128)[cycle % 4]
    horizon = (4, 8, 12, 16)[cycle % 4]
    return candidates, horizon


def build_cycle(
    backend: str,
    cycle: int,
    *,
    seed: int = 0,
    profile: str = "steady",
    include_invalid: bool = True,
    real_pixels: bool = True,
) -> list[TrafficCall]:
    """Return a deterministic set of requests for one traffic cycle.

    ``real_pixels`` is true in Docker so the LeWM backend receives materialized pixels. Tests can set
    it false to validate envelopes without allocating large base64 image payloads.
    """
    encoding = _pixel_encoding(backend, real_pixels=real_pixels)
    score_candidates, score_horizon = _score_shape(backend, cycle)
    plan_horizon = 4 if backend == "lewm" else 8 + (cycle % 3) * 4
    iterations = 2 if backend == "lewm" else 4 + (cycle % 4)
    plan_candidates = 12 if backend == "lewm" else 48 + (cycle % 4) * 16
    rollout_candidates = 4 if backend == "lewm" else 12 + (cycle % 3) * 8
    rollout_horizon = 3 if backend == "lewm" else 4 + (cycle % 3) * 2
    request_seed = seed + cycle * 17

    calls = [
        TrafficCall("score", payloads.score_request(
            f"traffic-score-{cycle}",
            s=score_candidates,
            t=score_horizon,
            inline_actions=True,
            pixel_encoding=encoding,
            seed=request_seed,
        )),
        TrafficCall("plan", payloads.plan_request(
            f"traffic-plan-{cycle}",
            horizon=plan_horizon,
            iterations=iterations,
            candidates=plan_candidates,
            pixel_encoding=encoding,
            seed=request_seed + 1,
        )),
    ]

    if cycle % 2 == 0:
        calls.append(TrafficCall("encode", payloads.encode_request(
            f"traffic-encode-{cycle}",
            pixel_encoding=encoding,
        )))
    if cycle % 3 == 0:
        calls.append(TrafficCall("rollout", payloads.rollout_request(
            f"traffic-rollout-{cycle}",
            s=rollout_candidates,
            t=rollout_horizon,
            inline_actions=True,
            pixel_encoding=encoding,
            seed=request_seed + 2,
        )))
    if profile == "burst" and backend != "lewm":
        calls.extend([
            TrafficCall("score", payloads.score_request(
                f"traffic-burst-score-{cycle}",
                s=256,
                t=16,
                inline_actions=True,
                seed=request_seed + 3,
            )),
            TrafficCall("rollout", payloads.rollout_request(
                f"traffic-burst-rollout-{cycle}",
                s=64,
                t=12,
                inline_actions=True,
                seed=request_seed + 4,
            )),
        ])
    if include_invalid and cycle % 5 == 0:
        calls.append(TrafficCall("score", {"wmcp_version": "0.1", "inputs": {}}, expect_error=True))
    return calls


def _invoke(client: WMCPClient, call: TrafficCall) -> dict[str, Any]:
    method = getattr(client, call.operation)
    return method(call.request)


def run_traffic(
    client: WMCPClient,
    *,
    loops: int = 0,
    interval: float = 3.0,
    profile: str = "steady",
    include_invalid: bool = True,
    real_pixels: bool = True,
    seed: int = 0,
) -> int:
    if not wait_until_ready(client, attempts=90, delay=1.0):
        print(f"service at {client.base_url} did not become ready", flush=True)
        return 1

    metadata = client.metadata()
    backend = _backend_name(metadata)
    print(
        f"traffic profile={profile} backend={backend} interval={interval}s loops={'infinite' if loops == 0 else loops}",
        flush=True,
    )

    cycle = 0
    while loops == 0 or cycle < loops:
        cycle += 1
        started = time.perf_counter()
        for call in build_cycle(
            backend,
            cycle,
            seed=seed,
            profile=profile,
            include_invalid=include_invalid,
            real_pixels=real_pixels,
        ):
            try:
                response = _invoke(client, call)
                summary = response.get("diagnostics") or response.get("outputs", {})
                print(f"cycle={cycle} op={call.operation} ok summary={summary}", flush=True)
            except WMCPError as exc:
                if call.expect_error:
                    print(f"cycle={cycle} op={call.operation} expected_error={exc.code}", flush=True)
                else:
                    print(f"cycle={cycle} op={call.operation} error={exc}", flush=True)
            except OSError as exc:
                print(f"cycle={cycle} op={call.operation} transport_error={exc}", flush=True)

        elapsed = time.perf_counter() - started
        if interval > elapsed:
            time.sleep(interval - elapsed)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate background WMCP traffic for demo dashboards")
    parser.add_argument("--base-url", default=os.getenv("WMCP_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model-id", default=os.getenv("WMCP_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--loops", type=int, default=int(os.getenv("WMCP_TRAFFIC_LOOPS", "0")))
    parser.add_argument("--interval", type=float, default=float(os.getenv("WMCP_TRAFFIC_INTERVAL", "3")))
    parser.add_argument("--profile", default=os.getenv("WMCP_TRAFFIC_PROFILE", "steady"), choices=("steady", "burst"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("WMCP_TRAFFIC_SEED", "7")))
    parser.add_argument(
        "--include-invalid",
        action=argparse.BooleanOptionalAction,
        default=_truthy(os.getenv("WMCP_TRAFFIC_INCLUDE_INVALID", "true")),
    )
    parser.add_argument(
        "--real-pixels",
        action=argparse.BooleanOptionalAction,
        default=_truthy(os.getenv("WMCP_TRAFFIC_REAL_PIXELS", "true")),
    )
    args = parser.parse_args(argv)

    timeout = float(os.getenv("WMCP_TIMEOUT", "240"))
    client = WMCPClient(args.base_url, model_id=args.model_id, timeout=timeout)
    return run_traffic(
        client,
        loops=args.loops,
        interval=args.interval,
        profile=args.profile,
        include_invalid=args.include_invalid,
        real_pixels=args.real_pixels,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
