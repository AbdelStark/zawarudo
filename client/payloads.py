"""Build WMCP request envelopes for the Push-T demo (RFC-0005 input profile).

Tensors are always sent as a ``TensorRef`` (never raw arrays). Observations/goals default to
URI-backed references (real images live behind a URI the service resolves); action candidates can be
URI-backed or tiny inline arrays so the demo runs end-to-end against the mock backend.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

ACTION_DIM = 10
IMAGE_SIZE = 224
HISTORY_SIZE = 3


def tensor_ref(
    shape: Sequence[int],
    layout: str,
    *,
    encoding: str = "uri",
    dtype: str = "float32",
    uri: Optional[str] = None,
    data: Any = None,
) -> dict[str, Any]:
    ref: dict[str, Any] = {"kind": "tensor", "encoding": encoding, "dtype": dtype, "shape": list(shape), "layout": layout}
    if encoding == "uri":
        ref["uri"] = uri or "memory://tensor.npy"
    elif encoding == "inline":
        ref["data"] = data
    else:
        raise ValueError(f"unsupported encoding for demo payloads: {encoding}")
    return ref


def observation(uri: str, *, b: int = 1, history: int = HISTORY_SIZE, dtype: str = "uint8") -> dict[str, Any]:
    shape = [b, history, 3, IMAGE_SIZE, IMAGE_SIZE]
    return {"modality": "rgb", "tensor": tensor_ref(shape, "B,H,C,224,224", encoding="uri", dtype=dtype, uri=uri)}


def goal(uri: str, *, b: int = 1, g: int = 1, dtype: str = "uint8") -> dict[str, Any]:
    shape = [b, g, 3, IMAGE_SIZE, IMAGE_SIZE]
    return {"modality": "rgb", "tensor": tensor_ref(shape, "B,G,C,224,224", encoding="uri", dtype=dtype, uri=uri)}


def action_candidates(
    b: int, s: int, t: int, *, inline: bool = False, seed: int = 0, uri: str = "memory://actions.npy"
) -> dict[str, Any]:
    shape = [b, s, t, ACTION_DIM]
    if inline:
        rng = random.Random(seed)
        data = [[[[rng.uniform(-1.0, 1.0) for _ in range(ACTION_DIM)] for _ in range(t)] for _ in range(s)] for _ in range(b)]
        tensor = tensor_ref(shape, "B,S,T,A", encoding="inline", dtype="float32", data=data)
    else:
        tensor = tensor_ref(shape, "B,S,T,A", encoding="uri", dtype="float32", uri=uri)
    return {
        "space": "continuous",
        "tensor": tensor,
        "bounds": {"low": [-1.0] * ACTION_DIM, "high": [1.0] * ACTION_DIM},
    }


def _envelope(request_id: str, operation: str, inputs: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "wmcp_version": "0.1",
        "request_id": request_id,
        "operation": operation,
        "model": "lewm-pusht",
        "inputs": inputs,
        "parameters": parameters,
        "return_options": {"include_candidate_costs": True, "include_best_index": True, "include_diagnostics": True},
    }


def score_request(
    request_id: str = "demo-score",
    *,
    b: int = 1,
    s: int = 16,
    t: int = 8,
    obs_uri: str = "memory://demo/history.npy",
    goal_uri: str = "memory://demo/goal.npy",
    inline_actions: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    inputs = {
        "observation_history": observation(obs_uri, b=b),
        "goal": goal(goal_uri, b=b),
        "action_candidates": action_candidates(b, s, t, inline=inline_actions, seed=seed),
    }
    return _envelope(request_id, "score", inputs, {"history_size": HISTORY_SIZE, "horizon": t, "seed": seed})


def rollout_request(
    request_id: str = "demo-rollout",
    *,
    b: int = 1,
    s: int = 16,
    t: int = 8,
    obs_uri: str = "memory://demo/history.npy",
    inline_actions: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    inputs = {
        "observation_history": observation(obs_uri, b=b),
        "action_candidates": action_candidates(b, s, t, inline=inline_actions, seed=seed),
    }
    return _envelope(request_id, "rollout", inputs, {"history_size": HISTORY_SIZE, "horizon": t, "seed": seed})


def encode_request(
    request_id: str = "demo-encode", *, b: int = 1, obs_uri: str = "memory://demo/history.npy"
) -> dict[str, Any]:
    return _envelope(request_id, "encode", {"observation_history": observation(obs_uri, b=b)}, {"history_size": HISTORY_SIZE})


def plan_request(
    request_id: str = "demo-plan",
    *,
    horizon: int = 8,
    iterations: int = 5,
    candidates: int = 64,
    obs_uri: str = "memory://demo/history.npy",
    goal_uri: str = "memory://demo/goal.npy",
    seed: int = 0,
) -> dict[str, Any]:
    inputs = {"observation_history": observation(obs_uri), "goal": goal(goal_uri)}
    params = {
        "planner": "cem",
        "horizon": horizon,
        "iterations": iterations,
        "candidates": candidates,
        "elite_fraction": 0.1,
        "seed": seed,
        "action_bounds": {"low": [-1.0] * ACTION_DIM, "high": [1.0] * ACTION_DIM},
    }
    return _envelope(request_id, "plan", inputs, params)
