"""Tests for the Push-T demo client (issue #6).

The client's ``requester`` seam is wired to a FastAPI ``TestClient`` so the client talks to the real
app in-process — exercising payload building, transport, response parsing, and rendering end to end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from wmcp_jepa_service.schemas import RequestEnvelope
from wmcp_jepa_service.server import app

from client import WMCPClient, WMCPError, payloads
from client.demo import run_demo, summarize
from client.render import render_html, summarize_html
from client.traffic import build_cycle


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


def test_payload_envelopes_validate() -> None:
    for req in (
        payloads.score_request(s=16, t=8),
        payloads.rollout_request(s=16, t=8),
        payloads.plan_request(horizon=8),
        payloads.encode_request(),
    ):
        RequestEnvelope.model_validate(req)  # raises on a malformed envelope
    score = payloads.score_request(b=1, s=16, t=8)
    assert score["inputs"]["action_candidates"]["tensor"]["shape"] == [1, 16, 8, 10]


def test_traffic_generator_builds_valid_envelopes() -> None:
    calls = build_cycle("mock", 1, seed=3, include_invalid=False, real_pixels=False)
    assert {call.operation for call in calls} == {"score", "plan"}
    for call in calls:
        RequestEnvelope.model_validate(call.request)


def test_traffic_generator_uses_lewm_safe_shapes_without_real_pixels() -> None:
    calls = build_cycle("lewm", 5, seed=3, include_invalid=True, real_pixels=False)
    by_operation = {call.operation: call for call in calls if not call.expect_error}
    score_shape = by_operation["score"].request["inputs"]["action_candidates"]["tensor"]["shape"]
    plan_params = by_operation["plan"].request["parameters"]
    assert score_shape[1] <= 10
    assert score_shape[2] == 4
    assert plan_params["candidates"] <= 12
    assert plan_params["iterations"] <= 2
    assert any(call.expect_error for call in calls)


def test_metadata_and_readyz() -> None:
    client = make_client()
    assert client.readyz()["status"] == "ready"
    assert client.metadata()["model_id"] == "lewm-pusht"


def test_run_demo_end_to_end() -> None:
    client = make_client()
    result = run_demo(client, s=8, t=4, horizon=4, seed=0)

    score_out = result["score"]["outputs"]
    assert score_out["costs"]["shape"] == [1, 8]
    assert len(score_out["best_index"]) == 1

    plan_out = result["plan"]["outputs"]
    assert plan_out["best_action_sequence"]["shape"] == [1, 4, 10]
    assert plan_out["first_action"]["shape"] == [1, 10]
    assert len(plan_out["best_cost"]) == 1


def test_summarize_is_human_readable() -> None:
    result = run_demo(make_client(), s=8, t=4, horizon=4)
    text = summarize(result)
    assert "best_index" in text
    assert "best_cost" in text
    assert "lewm-pusht" in text


def test_render_html_writes_page(tmp_path: Path) -> None:
    result = run_demo(make_client(), s=8, t=4, horizon=4)
    out = render_html(result, tmp_path / "demo.html")
    content = out.read_text()
    assert "Push-T" in content
    assert "candidate costs" in content
    assert "lewm-pusht" in content
    # static, self-contained page: no external scripts/stylesheets
    assert "<script" not in content
    assert "cdn" not in content.lower()
    assert summarize_html(result) == content


def test_model_not_found_raises() -> None:
    client = make_client(model_id="does-not-exist")
    with pytest.raises(WMCPError) as excinfo:
        client.score(payloads.score_request())
    assert excinfo.value.status == 404
    assert excinfo.value.code == "MODEL_NOT_FOUND"
