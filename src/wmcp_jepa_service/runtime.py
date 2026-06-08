from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, Protocol

from .schemas import ModelMetadata, RequestEnvelope, ResponseEnvelope
from .observability import MODEL_COMPUTE, MODEL_LOADED


class WorldModelBackend(Protocol):
    def metadata(self) -> ModelMetadata: ...
    async def encode(self, request: RequestEnvelope) -> ResponseEnvelope: ...
    async def predict(self, request: RequestEnvelope) -> ResponseEnvelope: ...
    async def rollout(self, request: RequestEnvelope) -> ResponseEnvelope: ...
    async def score(self, request: RequestEnvelope) -> ResponseEnvelope: ...
    async def plan(self, request: RequestEnvelope) -> ResponseEnvelope: ...


class MockWorldModelBackend:
    """Deterministic-enough mock backend for API and observability development.

    Replace this with a LeWMRuntime that loads the pinned Push-T checkpoint and calls
    upstream encode/rollout/get_cost functions.
    """

    def __init__(self, model_id: str = "lewm-pusht", revision: str = "mock", backend: str = "mock") -> None:
        self.model_id = model_id
        self.revision = revision
        self.backend = backend
        MODEL_LOADED.labels(model_id, revision, backend).set(1)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_revision=self.revision,
            input_shapes={
                "observation_history": "B,H,C,224,224 or B,H,224,224,C",
                "action_candidates": "B,S,T,10",
                "goal": "B,G,C,224,224 or B,G,224,224,C",
            },
            limits={"max_batch": 8, "max_candidates": 1024, "max_horizon": 64, "max_planner_iterations": 10},
            runtime={"backend": self.backend, "device": "cpu", "dynamic_batching": False},
        )

    def _shape_from_action_candidates(self, request: RequestEnvelope) -> tuple[int, int, int, int]:
        action = request.inputs.get("action_candidates", {})
        tensor = action.get("tensor", {}) if isinstance(action, dict) else {}
        shape = tensor.get("shape") or [1, 4, 4, 10]
        if len(shape) != 4:
            return (1, 4, 4, 10)
        return tuple(int(x) for x in shape)  # type: ignore[return-value]

    async def encode(self, request: RequestEnvelope) -> ResponseEnvelope:
        start = time.perf_counter()
        await asyncio.sleep(0.002)
        MODEL_COMPUTE.labels(self.model_id, "encode", self.backend).observe(time.perf_counter() - start)
        return ResponseEnvelope(
            request_id=request.request_id,
            operation="encode",
            model=self.model_id,
            model_revision=self.revision,
            outputs={
                "latents": {
                    "kind": "tensor",
                    "encoding": "inline",
                    "dtype": "float32",
                    "shape": [1, 3, 192],
                    "layout": "B,H,D",
                    "data": [[[0.0] * 192 for _ in range(3)]],
                }
            },
            diagnostics={"mock": True},
        )

    async def predict(self, request: RequestEnvelope) -> ResponseEnvelope:
        return await self.encode(request)

    async def rollout(self, request: RequestEnvelope) -> ResponseEnvelope:
        b, s, t, _a = self._shape_from_action_candidates(request)
        start = time.perf_counter()
        await asyncio.sleep(min(0.05, 0.001 + s * t * 0.000001))
        MODEL_COMPUTE.labels(self.model_id, "rollout", self.backend).observe(time.perf_counter() - start)
        return ResponseEnvelope(
            request_id=request.request_id,
            operation="rollout",
            model=self.model_id,
            model_revision=self.revision,
            outputs={
                "predicted_latents": {
                    "kind": "tensor",
                    "encoding": "uri",
                    "dtype": "float32",
                    "shape": [b, s, t, 192],
                    "layout": "B,S,T,D",
                    "uri": f"memory://{request.request_id}/predicted_latents.npy",
                }
            },
            diagnostics={"mock": True, "candidate_count": s, "horizon": t},
        )

    async def score(self, request: RequestEnvelope) -> ResponseEnvelope:
        b, s, t, _a = self._shape_from_action_candidates(request)
        start = time.perf_counter()
        await asyncio.sleep(min(0.05, 0.001 + s * t * 0.000001))
        MODEL_COMPUTE.labels(self.model_id, "score", self.backend).observe(time.perf_counter() - start)
        rng = random.Random(request.parameters.get("seed", 0))
        costs = [[rng.random() for _ in range(s)] for _ in range(b)]
        best = [min(range(s), key=lambda i: costs[row][i]) for row in range(b)]
        return ResponseEnvelope(
            request_id=request.request_id,
            operation="score",
            model=self.model_id,
            model_revision=self.revision,
            outputs={
                "costs": {"kind": "tensor", "encoding": "inline", "dtype": "float32", "shape": [b, s], "layout": "B,S", "data": costs},
                "best_index": best,
                "cost_statistics": {"min": min(min(row) for row in costs), "max": max(max(row) for row in costs)},
            },
            diagnostics={"mock": True, "candidate_count": s, "horizon": t},
        )

    async def plan(self, request: RequestEnvelope) -> ResponseEnvelope:
        params: Dict[str, Any] = request.parameters
        b = 1
        horizon = int(params.get("horizon", 16))
        action_dim = 10
        iterations = int(params.get("iterations", 5))
        rng = random.Random(params.get("seed", 0))
        best_cost_by_iteration = [1.0 / (i + 1) for i in range(iterations)]
        sequence = [[[rng.uniform(-1, 1) for _ in range(action_dim)] for _ in range(horizon)] for _ in range(b)]
        return ResponseEnvelope(
            request_id=request.request_id,
            operation="plan",
            model=self.model_id,
            model_revision=self.revision,
            outputs={
                "best_action_sequence": {"kind": "tensor", "encoding": "inline", "dtype": "float32", "shape": [b, horizon, action_dim], "layout": "B,T,A", "data": sequence},
                "first_action": {"kind": "tensor", "encoding": "inline", "dtype": "float32", "shape": [b, action_dim], "layout": "B,A", "data": [sequence[0][0]]},
                "best_cost": [best_cost_by_iteration[-1]],
                "planner_diagnostics": {"iterations": iterations, "best_cost_by_iteration": best_cost_by_iteration},
            },
            diagnostics={"mock": True},
        )
