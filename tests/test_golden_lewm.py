"""Golden numerical validation for the real LeWMRuntime (issue #5).

The vendored model loads the published ``quentinll/lewm-pusht`` checkpoint with ``strict=True``
(0 missing/unexpected keys — see test_model_package / #3), so it is numerically identical to upstream.
This test pins the verified runtime's costs on fixed inputs and re-checks them through the full
service path (decode → preprocess → get_cost → format), within tolerance, plus no-grad / device /
action-scaler invariants.

Requires the real model package at ``.artifacts/model-package/lewm-pusht`` (built by
``scripts/build_model_package.py real``; 72 MB, git-ignored) and the ``lewm`` extra — skipped otherwise.

RFC-0005 open issues resolved by the runtime contract:
- #3 (goal-frame handling): the goal cost uses the LAST goal-frame embedding (``criterion`` compares
  the final predicted latent to ``goal_emb[..., -1:, :]``); 1+ goal frames are accepted, last is used.
- #4 (history-action optionality): action candidates ``[B,S,T,A]`` include the history actions; the
  model splits ``[H, T-H]`` internally, so the first H actions align with the H observation frames.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

PKG = Path(".artifacts/model-package/lewm-pusht")
FIXTURE = Path(__file__).parent / "fixtures" / "golden_lewm.json"


def _require_real_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    if not (PKG / "weights.safetensors").exists():
        pytest.skip(f"real model package not present at {PKG} (build with scripts/build_model_package.py real)")
    from wmcp_jepa_service.runtime_lewm import LeWMRuntime

    return LeWMRuntime(str(PKG), device="cpu")


def _golden_inputs(spec: dict):
    import numpy as np

    b, s, t, h = spec["B"], spec["S"], spec["T"], spec["H"]
    rs = np.random.RandomState(spec["seed"])
    obs = (rs.rand(b, h, 224, 224, 3) * 255).astype(np.uint8)
    goal = (rs.rand(b, 1, 224, 224, 3) * 255).astype(np.uint8)
    acts = rs.randn(b, s, t, 10).astype(np.float32)
    return obs, goal, acts


def _b64(arr) -> str:
    import numpy as np

    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def _score_request(spec: dict):
    from wmcp_jepa_service.schemas import RequestEnvelope

    obs, goal, acts = _golden_inputs(spec)
    b, s, t, h = spec["B"], spec["S"], spec["T"], spec["H"]
    return RequestEnvelope(
        request_id="golden", operation="score", model="lewm-pusht",
        inputs={
            "observation_history": {"modality": "rgb", "tensor": {"kind": "tensor", "encoding": "base64",
                "dtype": "uint8", "shape": [b, h, 224, 224, 3], "layout": "B,H,224,224,C", "data_b64": _b64(obs)}},
            "goal": {"modality": "rgb", "tensor": {"kind": "tensor", "encoding": "base64",
                "dtype": "uint8", "shape": [b, 1, 224, 224, 3], "layout": "B,G,224,224,C", "data_b64": _b64(goal)}},
            "action_candidates": {"space": "continuous", "tensor": {"kind": "tensor", "encoding": "base64",
                "dtype": "float32", "shape": [b, s, t, 10], "layout": "B,S,T,A", "data_b64": _b64(acts)}},
        },
    )


def test_golden_costs_within_tolerance() -> None:
    import numpy as np

    rt = _require_real_runtime()
    spec = json.loads(FIXTURE.read_text())
    resp = asyncio.run(rt.score(_score_request(spec)))

    costs = np.array(resp.outputs["costs"]["data"])
    expected = np.array(spec["expected_costs"])
    assert resp.outputs["costs"]["shape"] == [spec["B"], spec["S"]]
    assert np.allclose(costs, expected, atol=spec["atol"], rtol=spec["rtol"]), (
        f"max abs diff {np.abs(costs - expected).max():.6f} exceeds atol {spec['atol']}"
    )
    assert resp.outputs["best_index"] == spec["expected_best_index"]
    assert set(resp.outputs["cost_statistics"]) == {"min", "mean", "max"}


def test_runtime_is_frozen_on_cpu() -> None:
    rt = _require_real_runtime()
    assert rt.device == "cpu"
    assert all(not p.requires_grad for p in rt.model.parameters())
    assert all(str(p.device) == "cpu" for p in rt.model.parameters())
    assert not rt.model.training


def test_action_scaler_roundtrip_and_bounds() -> None:
    rt = _require_real_runtime()
    import numpy as np

    x = np.random.RandomState(0).randn(1, 4, 6, 10).astype("float32")
    restored = rt.action_scaler.inverse_transform(rt.action_scaler.transform(x))
    assert np.allclose(np.asarray(restored), x, atol=1e-5)
    assert rt.action_scaler.bounds is not None
    assert len(rt.action_scaler.bounds["low"]) == 10
