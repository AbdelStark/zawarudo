"""Tests for the real LeWMRuntime backend (issue #3).

torch-gated: skipped where the `lewm` extra (torch + transformers + safetensors) isn't installed.
Builds a package from a randomly-initialized model with the *real* architecture/config, so it exercises
the full decode → preprocess → model → format path (and CEM planner) without the 72 MB checkpoint.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

# Real published Push-T config (sans Hydra `_target_`); build_lewm_from_config hardcodes the classes.
CONFIG: dict[str, Any] = {
    "encoder": {"size": "tiny", "patch_size": 14, "image_size": 224},
    "predictor": {"num_frames": 3, "input_dim": 192, "hidden_dim": 192, "output_dim": 192,
                  "depth": 6, "heads": 16, "mlp_dim": 2048, "dim_head": 64, "dropout": 0.0, "emb_dropout": 0.0},
    "action_encoder": {"input_dim": 10, "emb_dim": 192},
    "projector": {"input_dim": 192, "output_dim": 192, "hidden_dim": 2048},
    "pred_proj": {"input_dim": 192, "output_dim": 192, "hidden_dim": 2048},
}


def _build_package(tmp: Path) -> Path:
    import torch
    from safetensors.torch import save_file

    from wmcp_jepa_service import lewm_model, model_package as mp, packaging

    torch.manual_seed(0)
    model = lewm_model.build_lewm_from_config(CONFIG)
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    out = tmp / "pkg"
    out.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out / "weights.safetensors"))
    descriptor = {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")} for k, v in state.items()}
    manifest = packaging.pusht_manifest(
        weights_file="weights.safetensors", weights_format="safetensors", tensors=descriptor, config=CONFIG
    )
    packaging._write_json(out / mp.MANIFEST_FILE, manifest)
    packaging._write_aux_files(out, config=CONFIG)
    mp.write_checksums(out)
    return out


def _b64(arr: Any) -> str:
    import numpy as np

    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def _img(b: int, f: int, *, inline: bool = False) -> dict[str, Any]:
    import numpy as np

    arr = (np.random.RandomState(0).rand(b, f, 224, 224, 3) * 255).astype(np.uint8)
    tensor: dict[str, Any] = {"kind": "tensor", "dtype": "uint8", "shape": [b, f, 224, 224, 3], "layout": "B,H,224,224,C"}
    if inline:
        tensor["encoding"] = "inline"
        tensor["data"] = arr.tolist()
    else:
        tensor["encoding"] = "base64"
        tensor["data_b64"] = _b64(arr)
    return {"modality": "rgb", "tensor": tensor}


def _act(b: int, s: int, t: int) -> dict[str, Any]:
    import numpy as np

    arr = np.random.RandomState(1).randn(b, s, t, 10).astype("float32")
    return {"space": "continuous",
            "tensor": {"kind": "tensor", "encoding": "base64", "dtype": "float32",
                       "shape": [b, s, t, 10], "layout": "B,S,T,A", "data_b64": _b64(arr)}}


def _runtime(tmp: Path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from wmcp_jepa_service.runtime_lewm import LeWMRuntime

    return LeWMRuntime(str(_build_package(tmp)), device="cpu")


def test_score_returns_costs_and_stats(tmp_path: Path) -> None:
    from wmcp_jepa_service.schemas import RequestEnvelope

    rt = _runtime(tmp_path)
    req = RequestEnvelope(
        request_id="s", operation="score", model="lewm-pusht",
        inputs={"observation_history": _img(1, 1), "goal": _img(1, 1), "action_candidates": _act(1, 4, 5)},
    )
    resp = asyncio.run(rt.score(req))
    assert resp.outputs["costs"]["shape"] == [1, 4]
    assert resp.outputs["costs"]["layout"] == "B,S"
    assert len(resp.outputs["best_index"]) == 1
    assert set(resp.outputs["cost_statistics"]) == {"min", "mean", "max"}


def test_plan_returns_sequence_and_first_action(tmp_path: Path) -> None:
    from wmcp_jepa_service.schemas import RequestEnvelope

    rt = _runtime(tmp_path)
    req = RequestEnvelope(
        request_id="p", operation="plan", model="lewm-pusht",
        inputs={"observation_history": _img(1, 1), "goal": _img(1, 1)},
        parameters={"horizon": 4, "iterations": 3, "candidates": 16, "seed": 0},
    )
    resp = asyncio.run(rt.plan(req))
    assert resp.outputs["best_action_sequence"]["shape"] == [1, 4, 10]
    assert resp.outputs["first_action"]["shape"] == [1, 10]
    assert len(resp.outputs["best_cost"]) == 1
    diag = resp.outputs["planner_diagnostics"]
    assert diag["iterations"] == 3 and len(diag["best_cost_by_iteration"]) == 3


def test_encode_returns_latents(tmp_path: Path) -> None:
    from wmcp_jepa_service.schemas import RequestEnvelope

    rt = _runtime(tmp_path)
    resp = asyncio.run(rt.encode(RequestEnvelope(
        request_id="e", operation="encode", model="lewm-pusht", inputs={"observation_history": _img(1, 3)})))
    assert resp.outputs["latents"]["shape"] == [1, 3, 192]


def test_inline_and_base64_decode_agree(tmp_path: Path) -> None:
    import numpy as np

    from wmcp_jepa_service.schemas import RequestEnvelope

    rt = _runtime(tmp_path)

    def run(inline: bool) -> list:
        req = RequestEnvelope(
            request_id="d", operation="score", model="lewm-pusht",
            inputs={"observation_history": _img(1, 1, inline=inline), "goal": _img(1, 1, inline=inline),
                    "action_candidates": _act(1, 3, 4)},
        )
        return asyncio.run(rt.score(req)).outputs["costs"]["data"]

    assert np.allclose(np.array(run(True)), np.array(run(False)), atol=1e-4)
