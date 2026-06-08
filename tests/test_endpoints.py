"""Endpoint contract tests using the committed example payloads (RFC-0005 acceptance criterion 2).

Run against the default mock backend so they need no weights — they assert each RFC-0005 endpoint
responds with the documented output keys/shapes for the ``examples/*.json`` requests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wmcp_jepa_service.server import app

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MODEL = "lewm-pusht"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text())


def test_metadata_endpoint(client: TestClient) -> None:
    resp = client.get(f"/wmcp/v1/models/{MODEL}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL
    assert {"metadata", "encode", "rollout", "score", "plan"}.issubset(set(body["supported_operations"]))
    assert body["latent_space"]["dimension"] == 192


def test_list_models_endpoint(client: TestClient) -> None:
    resp = client.get("/wmcp/v1/models")
    assert resp.status_code == 200
    assert resp.json()["models"][0]["model_id"] == MODEL


def test_score_example_payload(client: TestClient) -> None:
    resp = client.post(f"/wmcp/v1/models/{MODEL}:score", json=_load("score_request.json"))
    assert resp.status_code == 200, resp.text
    out = resp.json()["outputs"]
    assert out["costs"]["shape"] == [1, 256] and out["costs"]["layout"] == "B,S"
    assert len(out["best_index"]) == 1
    assert "min" in out["cost_statistics"] and "max" in out["cost_statistics"]


def test_rollout_example_payload(client: TestClient) -> None:
    resp = client.post(f"/wmcp/v1/models/{MODEL}:rollout", json=_load("rollout_request.json"))
    assert resp.status_code == 200, resp.text
    out = resp.json()["outputs"]
    assert out["predicted_latents"]["shape"][-1] == 192
    assert out["predicted_latents"]["layout"] == "B,S,T,D"


def test_plan_example_payload(client: TestClient) -> None:
    resp = client.post(f"/wmcp/v1/models/{MODEL}:plan", json=_load("plan_request.json"))
    assert resp.status_code == 200, resp.text
    out = resp.json()["outputs"]
    assert out["best_action_sequence"]["shape"][-1] == 10
    assert out["first_action"]["shape"][-1] == 10
    assert len(out["best_cost"]) == 1
    assert "iterations" in out["planner_diagnostics"]


def test_encode_endpoint(client: TestClient) -> None:
    body = {
        "wmcp_version": "0.1", "request_id": "enc-1",
        "inputs": {"observation_history": {"modality": "rgb", "tensor": {
            "kind": "tensor", "encoding": "uri", "dtype": "uint8", "shape": [1, 3, 3, 224, 224],
            "layout": "B,H,C,224,224", "uri": "memory://obs.npy"}}},
    }
    resp = client.post(f"/wmcp/v1/models/{MODEL}:encode", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["outputs"]["latents"]["shape"][-1] == 192


def test_unknown_model_returns_404(client: TestClient) -> None:
    resp = client.post("/wmcp/v1/models/not-a-model:score", json=_load("score_request.json"))
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "MODEL_NOT_FOUND"
