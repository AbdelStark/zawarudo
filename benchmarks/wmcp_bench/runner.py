"""Async load generator: fire N requests at a profile's shape with bounded concurrency.

Concurrency is real (threads via ``asyncio.to_thread``) so a blocking ``urllib`` poster parallelises;
the ``poster`` seam lets tests drive a FastAPI ``TestClient`` instead of a socket.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .payloads import build_request
from .profiles import Profile

MODEL_ID = "lewm-pusht"

# (method, path, json_body|None) -> (status_code, parsed_json|None)
Poster = Callable[[str, str, Optional[dict]], "tuple[int, Optional[dict]]"]


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (q / 100.0)
    floor = int(k)
    ceil = min(floor + 1, len(ordered) - 1)
    if floor == ceil:
        return ordered[floor]
    return ordered[floor] + (ordered[ceil] - ordered[floor]) * (k - floor)


@dataclass
class ProfileResult:
    name: str
    operation: str
    count: int
    concurrency: int
    elapsed_s: float
    latencies_ms: list[float]
    errors_by_code: dict[str, int]
    candidate_counts: list[int]

    @property
    def errors(self) -> int:
        return sum(self.errors_by_code.values())

    @property
    def ok(self) -> int:
        return self.count - self.errors

    @property
    def error_rate(self) -> float:
        return self.errors / self.count if self.count else 0.0

    @property
    def rps(self) -> float:
        return self.count / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def p50(self) -> float:
        return percentile(self.latencies_ms, 50)

    @property
    def p90(self) -> float:
        return percentile(self.latencies_ms, 90)

    @property
    def p95(self) -> float:
        return percentile(self.latencies_ms, 95)

    @property
    def p99(self) -> float:
        return percentile(self.latencies_ms, 99)

    def candidate_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {}
        for c in self.candidate_counts:
            dist[c] = dist.get(c, 0) + 1
        return dist


def _request_spec(profile: Profile, index: int) -> tuple[str, str, Optional[dict]]:
    rid = f"bench-{profile.name}-{index}"
    op = profile.operation
    if op == "health":
        return "GET", "/healthz", None
    if op == "metadata":
        return "GET", f"/wmcp/v1/models/{MODEL_ID}", None
    return "POST", f"/wmcp/v1/models/{MODEL_ID}:{op}", build_request(profile, rid)


async def _one(poster: Poster, method: str, path: str, body: Optional[dict]) -> tuple[float, Optional[str], Optional[int]]:
    start = time.perf_counter()
    error: Optional[str] = None
    candidates: Optional[int] = None
    try:
        code, parsed = await asyncio.to_thread(poster, method, path, body)
        if code >= 400:
            detail = parsed.get("detail", {}) if isinstance(parsed, dict) else {}
            error = str(detail.get("code", "HTTP_ERROR")) if isinstance(detail, dict) else "HTTP_ERROR"
        elif isinstance(parsed, dict):
            diag = parsed.get("diagnostics", {})
            if isinstance(diag, dict) and "candidate_count" in diag:
                candidates = int(diag["candidate_count"])
    except Exception as exc:  # noqa: BLE001 - record transport failures as an error code
        error = type(exc).__name__
    return (time.perf_counter() - start) * 1000.0, error, candidates


async def run_profile(
    poster: Poster,
    profile: Profile,
    *,
    total_requests: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> ProfileResult:
    n = total_requests if total_requests is not None else profile.default_requests
    conc = concurrency if concurrency is not None else max(1, profile.concurrency)
    sem = asyncio.Semaphore(conc)

    async def worker(index: int) -> tuple[float, Optional[str], Optional[int]]:
        method, path, body = _request_spec(profile, index)
        async with sem:
            return await _one(poster, method, path, body)

    start = time.perf_counter()
    results = await asyncio.gather(*(worker(i) for i in range(n)))
    elapsed = time.perf_counter() - start

    errors: dict[str, int] = {}
    candidate_counts: list[int] = []
    for _, err, cand in results:
        if err:
            errors[err] = errors.get(err, 0) + 1
        if cand is not None:
            candidate_counts.append(cand)

    return ProfileResult(
        name=profile.name,
        operation=profile.operation,
        count=n,
        concurrency=conc,
        elapsed_s=elapsed,
        latencies_ms=[r[0] for r in results],
        errors_by_code=errors,
        candidate_counts=candidate_counts,
    )
