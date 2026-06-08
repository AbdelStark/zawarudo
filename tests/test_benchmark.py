"""Tests for the benchmark harness (issue #8)."""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from wmcp_jepa_service.schemas import RequestEnvelope
from wmcp_jepa_service.server import app

from benchmarks.wmcp_bench import PROFILES, render_report, run_profile
from benchmarks.wmcp_bench.metadata import backend_from_readyz, collect_metadata
from benchmarks.wmcp_bench.payloads import build_request
from benchmarks.wmcp_bench.runner import percentile


def make_poster():
    tc = TestClient(app)

    def poster(method: str, path: str, body: Optional[dict]) -> "tuple[int, Optional[dict]]":
        resp = tc.request(method, path, json=body) if body is not None else tc.request(method, path)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, None

    return poster


def test_profiles_match_plan() -> None:
    medium = PROFILES["score-medium"]
    assert (medium.operation, medium.s, medium.t, medium.concurrency) == ("score", 256, 16, 8)
    assert PROFILES["score-large"].s == 1024
    assert set(PROFILES) >= {"smoke-score", "score-small", "score-medium", "rollout-medium", "plan-medium"}


def test_build_request_shapes_validate() -> None:
    req = build_request(PROFILES["score-medium"], "r1")
    assert req["inputs"]["action_candidates"]["tensor"]["shape"] == [1, 256, 16, 10]
    RequestEnvelope.model_validate(req)
    plan = build_request(PROFILES["plan-medium"], "r2")
    assert plan["parameters"]["candidates"] == 256
    RequestEnvelope.model_validate(plan)


def test_percentile() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


def test_run_score_profile_against_app() -> None:
    result = asyncio.run(run_profile(make_poster(), PROFILES["smoke-score"], total_requests=12, concurrency=4))
    assert result.count == 12
    assert result.errors == 0
    assert len(result.latencies_ms) == 12
    assert result.p95 >= 0.0
    assert result.candidate_counts and all(c == 4 for c in result.candidate_counts)  # smoke S=4


def test_run_health_profile_against_app() -> None:
    result = asyncio.run(run_profile(make_poster(), PROFILES["health"], total_requests=10))
    assert result.count == 10
    assert result.errors == 0


def test_render_report_has_required_context() -> None:
    result = asyncio.run(run_profile(make_poster(), PROFILES["smoke-score"], total_requests=5))
    meta = collect_metadata(base_url="http://testserver", backend="mock")
    report = render_report(meta, [result], title="t")
    for needle in ("## Results", "smoke-score", "Backend: mock", "p95", "Dynamic batching config", "Next actions"):
        assert needle in report


def test_metadata_has_required_fields() -> None:
    meta = collect_metadata(base_url="http://x", backend="mock")
    for key in ("timestamp", "service_commit", "backend", "hardware", "python", "dtype", "dynamic_batching"):
        assert key in meta
    assert backend_from_readyz({"backend": "lewm"}) == "lewm"
    assert backend_from_readyz(None) == "unknown"
