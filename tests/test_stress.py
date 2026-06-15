"""Tests for the concurrent WMCP stress tester (client.stress).

Pure helpers (ops parsing, percentiles, backend-aware config, payload variety) are tested directly;
``run_stress`` is driven end-to-end through the demo client's ``requester`` seam against the real
FastAPI app in-process, so the threaded load path, request mix, and percentile summary all execute
without a socket.
"""

from __future__ import annotations

import random
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from wmcp_jepa_service.schemas import RequestEnvelope
from wmcp_jepa_service.server import app

from client import WMCPClient, WMCPError
from client.stress import (
    StressConfig,
    build_request,
    config_from_env,
    parse_ops,
    run_stress,
    _percentile,
)


def make_client(model_id: str = "lewm-pusht") -> WMCPClient:
    tc = TestClient(app)

    def requester(method: str, path: str, body: Optional[dict]) -> dict:
        resp = tc.request(method, path, json=body) if body is not None else tc.request(method, path)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", {}) or {}
            except ValueError:
                detail = {}
            raise WMCPError(resp.status_code, str(detail.get("code", "ERR")), str(detail.get("message", "")))
        return resp.json()

    return WMCPClient("http://testserver", model_id=model_id, requester=requester)


def base_config(**overrides) -> StressConfig:
    cfg = config_from_env("mock")
    return StressConfig(**{**cfg.__dict__, **overrides})


def test_parse_ops_weighted_and_plain() -> None:
    assert parse_ops("score:4,plan:2,rollout:2,encode:1") == (
        ("score", 4), ("plan", 2), ("rollout", 2), ("encode", 1),
    )
    assert parse_ops("score, plan") == (("score", 1), ("plan", 1))


def test_parse_ops_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError):
        parse_ops("score,teleport")


def test_percentiles_interpolate() -> None:
    values = [float(i) for i in range(1, 101)]
    assert _percentile(values, 0.50) == pytest.approx(50.5)
    assert _percentile(values, 0.95) == pytest.approx(95.05)
    assert _percentile([], 0.5) != _percentile([], 0.5)  # nan


def test_config_is_backend_aware() -> None:
    lewm = config_from_env("lewm")
    mock = config_from_env("mock")
    # The CPU backend gets a lighter default load than the mock.
    assert lewm.concurrency < mock.concurrency
    assert lewm.max_candidates < mock.max_candidates


def test_build_request_envelopes_validate_and_vary() -> None:
    cfg = base_config(min_candidates=8, max_candidates=8, min_horizon=4, max_horizon=4)
    rng = random.Random(0)
    for op in ("score", "plan", "rollout", "encode"):
        req = build_request(op, cfg, "mock", rng, f"id-{op}")
        RequestEnvelope.model_validate(req)
        assert req["operation"] == op
    plan = build_request("plan", cfg, "mock", rng, "id-plan")
    assert plan["parameters"]["candidates"] == 8
    assert plan["parameters"]["horizon"] == 4


def test_run_stress_end_to_end_bounded() -> None:
    cfg = base_config(
        concurrency=4,
        duration=0,        # bounded by total, not time
        total=40,
        ramp=0,
        report_interval=999,   # no periodic report inside the short run
        invalid_ratio=0.25,    # exercise the expected-error path
        max_candidates=16,
    )
    rc = run_stress(make_client(), cfg)
    assert rc == 0  # no unexpected failures


def test_run_stress_counts_invalid_as_expected_not_failure() -> None:
    # Every request is intentionally invalid -> all land as expected errors, run still succeeds.
    cfg = base_config(concurrency=2, duration=0, total=10, ramp=0, report_interval=999, invalid_ratio=1.0)
    assert run_stress(make_client(), cfg) == 0
