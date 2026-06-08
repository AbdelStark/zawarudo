"""Build URI-backed WMCP request envelopes at profile shapes.

The harness sends URI-backed tensors (shape-only on the wire) so payloads stay tiny even at S=1024;
the service resolves the URI for real backends, while the mock reads only the declared shape.
"""

from __future__ import annotations

from typing import Any

from .profiles import Profile

IMAGE = 224


def _tensor(shape: list[int], layout: str, *, dtype: str = "float32", uri: str = "memory://bench.npy") -> dict[str, Any]:
    return {"kind": "tensor", "encoding": "uri", "dtype": dtype, "shape": shape, "layout": layout, "uri": uri}


def _observation(b: int, h: int) -> dict[str, Any]:
    return {"modality": "rgb", "tensor": _tensor([b, h, 3, IMAGE, IMAGE], "B,H,C,224,224", dtype="uint8", uri="memory://obs.npy")}


def _goal(b: int) -> dict[str, Any]:
    return {"modality": "rgb", "tensor": _tensor([b, 1, 3, IMAGE, IMAGE], "B,G,C,224,224", dtype="uint8", uri="memory://goal.npy")}


def _actions(p: Profile) -> dict[str, Any]:
    return {"space": "continuous", "tensor": _tensor([p.b, p.s, p.t, p.a], "B,S,T,A", uri="memory://actions.npy")}


def build_request(profile: Profile, request_id: str) -> dict[str, Any]:
    """Build the WMCP request envelope for a profile (None for control-plane health/metadata)."""
    op = profile.operation
    if op in ("health", "metadata"):
        raise ValueError(f"{op} has no request body")

    base: dict[str, Any] = {"wmcp_version": "0.1", "request_id": request_id, "operation": op, "model": "lewm-pusht"}
    if op == "score":
        base["inputs"] = {
            "observation_history": _observation(profile.b, profile.h),
            "goal": _goal(profile.b),
            "action_candidates": _actions(profile),
        }
        base["parameters"] = {"history_size": profile.h, "horizon": profile.t}
    elif op == "rollout":
        base["inputs"] = {
            "observation_history": _observation(profile.b, profile.h),
            "action_candidates": _actions(profile),
        }
        base["parameters"] = {"history_size": profile.h, "horizon": profile.t}
    elif op == "plan":
        base["inputs"] = {"observation_history": _observation(profile.b, profile.h), "goal": _goal(profile.b)}
        base["parameters"] = {
            "planner": "cem",
            "horizon": profile.t,
            "iterations": 5,
            "candidates": profile.s,
            "elite_fraction": 0.1,
        }
    else:
        raise ValueError(f"unsupported operation: {op}")
    return base
