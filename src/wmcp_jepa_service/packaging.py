"""Author trusted model packages (see ``model_package`` for the consume/verify side).

This module *builds* packages: it assembles a schema-conforming manifest from the RFC-0005 Push-T
profile plus the upstream config, writes the auxiliary descriptors (preprocessing, action space,
action scaler), serialises weights, and writes ``checksums.txt``.

Two weight paths:
- ``build_package`` writes a portable, stdlib-only ``weights.json`` (used by tests, ``--synthetic``,
  and demos — never real weights).
- ``build_real_package`` converts an upstream ``weights.pt`` to ``weights.safetensors`` (lazy torch +
  safetensors), the preferred runtime artifact (RFC-0005 open issue #2).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_package import (
    ACTION_SCALER_FILE,
    ACTION_SPACE_FILE,
    CONFIG_FILE,
    JSON_WEIGHTS,
    MANIFEST_FILE,
    PREPROCESSING_FILE,
    SAFETENSORS,
    SOURCES_LOCK_FILE,
    write_checksums,
)

# RFC-0005 Push-T model profile (rfc/0005-pusht-demo-profile.md "Model profile").
PUSHT_LATENT_DIM = 192
PUSHT_ACTION_DIM = 10
PUSHT_IMAGE_SIZE = 224
PUSHT_HISTORY_SIZE = 3

DEFAULT_LIMITS = {
    "max_batch": 8,
    "max_candidates": 1024,
    "max_horizon": 64,
    "max_planner_iterations": 10,
}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shape_of(nested: Any) -> tuple[int, ...]:
    """Infer the shape of a (possibly ragged-checked) nested list."""
    shape: list[int] = []
    cur = nested
    while isinstance(cur, (list, tuple)):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return tuple(shape)


def _to_nested(array: Any) -> Any:
    """Coerce a torch/numpy array or nested list into plain python nested lists."""
    if isinstance(array, (list, tuple)):
        return list(array)
    if hasattr(array, "tolist"):
        return array.tolist()
    raise TypeError(f"cannot serialise weight of type {type(array)!r}")


def tensor_descriptor(shape: Sequence[int], dtype: str = "float32") -> dict[str, Any]:
    return {"shape": [int(x) for x in shape], "dtype": dtype}


def pusht_manifest(
    *,
    weights_file: str,
    weights_format: str,
    tensors: Mapping[str, Mapping[str, Any]],
    model_id: str = "lewm-pusht",
    source_repository: str = "https://github.com/lucas-maes/le-wm",
    source_revision: str = "unpinned",
    artifact_uri: str = "https://huggingface.co/quentinll/lewm-pusht",
    artifact_revision: str = "unpinned",
    framework: str = "pytorch",
    runtime_class: str = "wmcp_jepa_service.runtime.LeWMRuntime",
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest conforming to ``schemas/model-manifest.schema.json`` (+ a ``weights`` block)."""
    latent_dim = int((config or {}).get("latent_dim", PUSHT_LATENT_DIM))
    action_dim = int((config or {}).get("action_dim", PUSHT_ACTION_DIM))
    return {
        "schema_version": "0.1",
        "model_id": model_id,
        "model_family": "jepa",
        "model_type": "action_conditioned_world_model",
        "task": "pusht",
        "source_repository": source_repository,
        "source_revision": source_revision,
        "artifact_uri": artifact_uri,
        "artifact_revision": artifact_revision,
        "framework": framework,
        "runtime": {
            "backend": "lewm",
            "class": runtime_class,
            "python": ">=3.10",
            "device": "cuda",
        },
        "supported_operations": ["metadata", "encode", "predict", "rollout", "score", "plan"],
        "inputs": {
            "observation_history": {"layout": "B,H,C,224,224 or B,H,224,224,C", "history_size": PUSHT_HISTORY_SIZE},
            "action_candidates": {"layout": "B,S,T,A", "action_dim": action_dim},
            "goal": {"layout": "B,G,C,224,224 or B,G,224,224,C"},
        },
        "outputs": {
            "encode": {"latents": "B,H,D"},
            "rollout": {"predicted_latents": "B,S,T,D"},
            "score": {"costs": "B,S", "best_index": "B", "cost_statistics": ["min", "mean", "max"]},
            "plan": {"best_action_sequence": "B,T,A", "first_action": "B,A", "best_cost": "B"},
        },
        "preprocessing": {"ref": PREPROCESSING_FILE},
        "action_space": {"ref": ACTION_SPACE_FILE, "action_dim": action_dim},
        "latent_space": {"dimension": latent_dim, "dtype": "float32"},
        "limits": dict(DEFAULT_LIMITS),
        "weights": {
            "file": weights_file,
            "format": weights_format,
            "tensors": dict(tensors),
        },
    }


def default_preprocessing() -> dict[str, Any]:
    return {
        "image_size": PUSHT_IMAGE_SIZE,
        "history_size": PUSHT_HISTORY_SIZE,
        "channels": 3,
        "resize": [PUSHT_IMAGE_SIZE, PUSHT_IMAGE_SIZE],
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "layout": "B,H,C,224,224",
        "dtype": "float32",
        "note": "Placeholder ImageNet normalization; confirm against upstream LeWorldModel preprocessing.",
    }


def default_action_space() -> dict[str, Any]:
    return {
        "space": "continuous",
        "action_dim": PUSHT_ACTION_DIM,
        "layout": "B,S,T,A",
        "bounds": {"low": [-1.0] * PUSHT_ACTION_DIM, "high": [1.0] * PUSHT_ACTION_DIM},
        "note": "Bounds are placeholders pending upstream confirmation (RFC-0005 open issue #1).",
    }


def default_action_scaler() -> dict[str, Any]:
    return {
        "kind": "identity",
        "action_dim": PUSHT_ACTION_DIM,
        "bounds": {"low": [-1.0] * PUSHT_ACTION_DIM, "high": [1.0] * PUSHT_ACTION_DIM},
        "note": "Identity scaling assumed; confirm the upstream action scaler (RFC-0005 open issue #1).",
    }


def default_sources_lock() -> dict[str, Any]:
    return {
        "note": "Pin exact commits/revisions before building a production package (LEWM_INTEGRATION_GUIDE §1).",
        "sources": {
            "le-wm": {"url": "https://github.com/lucas-maes/le-wm", "commit": "UNPINNED"},
            "stable-worldmodel": {"url": "https://github.com/galilai-group/stable-worldmodel", "commit": "UNPINNED"},
            "stable-pretraining": {"url": "https://github.com/rbalestr-lab/stable-pretraining", "commit": "UNPINNED"},
        },
        "artifact": {"repo": "quentinll/lewm-pusht", "revision": "UNPINNED"},
    }


def _write_aux_files(out_dir: Path, *, config: Mapping[str, Any] | None) -> None:
    _write_json(out_dir / CONFIG_FILE, dict(config or {}))
    _write_json(out_dir / PREPROCESSING_FILE, default_preprocessing())
    _write_json(out_dir / ACTION_SPACE_FILE, default_action_space())
    _write_json(out_dir / ACTION_SCALER_FILE, default_action_scaler())
    if not (out_dir / SOURCES_LOCK_FILE).exists():
        _write_json(out_dir / SOURCES_LOCK_FILE, default_sources_lock())


def build_package(
    out_dir: str | Path,
    *,
    named_weights: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    model_id: str = "lewm-pusht",
    dtype: str = "float32",
    weights_file: str = "weights.json",
) -> Path:
    """Write a portable JSON-weights package and ``checksums.txt``. Returns the package dir.

    ``named_weights`` maps tensor name -> nested list (or any array with ``.tolist()``). This is the
    stdlib-only path used by tests, ``--synthetic``, and demos. It must never be used for real weights.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nested = {name: _to_nested(arr) for name, arr in named_weights.items()}
    tensors_blob = {
        name: {"shape": list(_shape_of(value)), "dtype": dtype, "data": value} for name, value in nested.items()
    }
    _write_json(out_dir / weights_file, {"__format__": "wmcp-json-weights-v1", "tensors": tensors_blob})

    descriptor = {name: tensor_descriptor(_shape_of(value), dtype) for name, value in nested.items()}
    manifest = pusht_manifest(
        weights_file=weights_file, weights_format=JSON_WEIGHTS, tensors=descriptor, model_id=model_id, config=config
    )
    _write_json(out_dir / MANIFEST_FILE, manifest)
    _write_aux_files(out_dir, config=config)
    write_checksums(out_dir)
    return out_dir


def build_synthetic_package(out_dir: str | Path, *, model_id: str = "lewm-pusht") -> Path:
    """Build a tiny, schema-valid package with deterministic placeholder weights (no real checkpoint)."""
    named_weights = {
        "encoder.weight": [[float((i + j) % 5) for j in range(8)] for i in range(4)],
        "encoder.bias": [0.0, 0.0, 0.0, 0.0],
        "predictor.weight": [[0.1 * ((i * 4 + j) % 7) for j in range(4)] for i in range(4)],
    }
    config = {
        "latent_dim": PUSHT_LATENT_DIM,
        "action_dim": PUSHT_ACTION_DIM,
        "image_size": PUSHT_IMAGE_SIZE,
        "history_size": PUSHT_HISTORY_SIZE,
        "synthetic": True,
    }
    return build_package(out_dir, named_weights=named_weights, config=config, model_id=model_id)


def build_real_package(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    model_id: str = "lewm-pusht",
    source_revision: str = "unpinned",
    artifact_revision: str = "unpinned",
) -> Path:
    """Convert an upstream checkpoint dir (``config.json`` + ``weights.pt``) into a safetensors package.

    Requires the ``torch`` and ``safetensors`` libraries (lazy import). Raises if the source is missing.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = source_dir / "weights.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"expected upstream weights at {pt_path}")
    config: dict[str, Any] = {}
    if (source_dir / CONFIG_FILE).exists():
        config = json.loads((source_dir / CONFIG_FILE).read_text(encoding="utf-8"))

    torch = importlib.import_module("torch")  # type: Any
    safetensors_torch = importlib.import_module("safetensors.torch")  # type: Any

    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    state = {name: tensor.contiguous() for name, tensor in state.items()}
    weights_file = "weights.safetensors"
    safetensors_torch.save_file(state, str(out_dir / weights_file))

    descriptor = {
        name: tensor_descriptor(tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""))
        for name, tensor in state.items()
    }
    manifest = pusht_manifest(
        weights_file=weights_file,
        weights_format=SAFETENSORS,
        tensors=descriptor,
        model_id=model_id,
        source_revision=source_revision,
        artifact_revision=artifact_revision,
        config=config,
    )
    _write_json(out_dir / MANIFEST_FILE, manifest)
    _write_aux_files(out_dir, config=config)
    write_checksums(out_dir)
    return out_dir
