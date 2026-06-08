"""Build WMCP request envelopes at profile shapes.

Two encodings (stdlib only):
- ``uri`` (default): URI-backed tensors (shape-only on the wire) — tiny payloads even at S=1024; the
  mock backend reads only the declared shape.
- ``base64``: real bytes inline (uint8 images via ``os.urandom``, float32 actions via ``array``) — used
  to drive the real ``lewm`` backend, which materialises and runs on the actual tensor values.
"""

from __future__ import annotations

import array
import base64
import os
import random
from typing import Any

from .profiles import Profile

IMAGE = 224


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _b64_uint8(n: int) -> str:
    return base64.b64encode(os.urandom(n)).decode()


def _b64_float32(n: int, *, seed: int = 0) -> str:
    rng = random.Random(seed)
    values = array.array("f", (rng.uniform(-1.0, 1.0) for _ in range(n)))
    return base64.b64encode(values.tobytes()).decode()


def _tensor(
    shape: list[int], layout: str, *, dtype: str = "float32", encoding: str = "uri", uri: str = "memory://bench.npy"
) -> dict[str, Any]:
    ref: dict[str, Any] = {"kind": "tensor", "encoding": encoding, "dtype": dtype, "shape": shape, "layout": layout}
    if encoding == "uri":
        ref["uri"] = uri
    elif encoding == "base64":
        ref["data_b64"] = _b64_uint8(_numel(shape)) if dtype == "uint8" else _b64_float32(_numel(shape))
    else:
        raise ValueError(f"unsupported encoding: {encoding}")
    return ref


def _observation(b: int, h: int, encoding: str) -> dict[str, Any]:
    return {"modality": "rgb",
            "tensor": _tensor([b, h, 3, IMAGE, IMAGE], "B,H,C,224,224", dtype="uint8", encoding=encoding, uri="memory://obs.npy")}


def _goal(b: int, encoding: str) -> dict[str, Any]:
    return {"modality": "rgb",
            "tensor": _tensor([b, 1, 3, IMAGE, IMAGE], "B,G,C,224,224", dtype="uint8", encoding=encoding, uri="memory://goal.npy")}


def _actions(p: Profile, encoding: str) -> dict[str, Any]:
    return {"space": "continuous",
            "tensor": _tensor([p.b, p.s, p.t, p.a], "B,S,T,A", encoding=encoding, uri="memory://actions.npy")}


def build_request(profile: Profile, request_id: str, *, encoding: str = "uri") -> dict[str, Any]:
    """Build the WMCP request envelope for a profile (health/metadata have no body)."""
    op = profile.operation
    if op in ("health", "metadata"):
        raise ValueError(f"{op} has no request body")

    base: dict[str, Any] = {"wmcp_version": "0.1", "request_id": request_id, "operation": op, "model": "lewm-pusht"}
    if op == "score":
        base["inputs"] = {
            "observation_history": _observation(profile.b, profile.h, encoding),
            "goal": _goal(profile.b, encoding),
            "action_candidates": _actions(profile, encoding),
        }
        base["parameters"] = {"history_size": profile.h, "horizon": profile.t}
    elif op == "rollout":
        base["inputs"] = {
            "observation_history": _observation(profile.b, profile.h, encoding),
            "action_candidates": _actions(profile, encoding),
        }
        base["parameters"] = {"history_size": profile.h, "horizon": profile.t}
    elif op == "plan":
        base["inputs"] = {
            "observation_history": _observation(profile.b, profile.h, encoding),
            "goal": _goal(profile.b, encoding),
        }
        base["parameters"] = {
            "planner": "cem", "horizon": profile.t, "iterations": 5,
            "candidates": profile.s, "elite_fraction": 0.1,
        }
    else:
        raise ValueError(f"unsupported operation: {op}")
    return base
