from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import RequestEnvelope


@dataclass(frozen=True)
class WorkloadShape:
    batch: int | None = None
    candidates: int | None = None
    horizon: int | None = None
    action_dim: int | None = None
    planner_iterations: int | None = None

    def span_attributes(self) -> dict[str, int]:
        attributes: dict[str, int] = {}
        if self.batch is not None:
            attributes["tensor.batch"] = self.batch
        if self.candidates is not None:
            attributes["tensor.candidates"] = self.candidates
        if self.horizon is not None:
            attributes["tensor.horizon"] = self.horizon
        if self.action_dim is not None:
            attributes["tensor.action_dim"] = self.action_dim
        if self.planner_iterations is not None:
            attributes["planner.iterations"] = self.planner_iterations
        return attributes


def workload_shape_from_request(request: RequestEnvelope) -> WorkloadShape:
    """Extract low-cardinality workload dimensions from a validated WMCP request."""
    parameters = request.parameters if isinstance(request.parameters, dict) else {}
    batch, candidates, horizon, action_dim = _action_candidate_shape(request.inputs)

    if batch is None:
        batch = _batch_from_inputs(request.inputs)

    if request.operation == "plan":
        batch = batch or 1
        candidates = _positive_int(parameters.get("candidates")) or candidates
        horizon = _positive_int(parameters.get("horizon")) or horizon
        action_dim = action_dim or _action_dim_from_bounds(parameters.get("action_bounds"))

    return WorkloadShape(
        batch=batch,
        candidates=candidates,
        horizon=horizon,
        action_dim=action_dim,
        planner_iterations=_positive_int(parameters.get("iterations")),
    )


def _action_candidate_shape(inputs: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    action = inputs.get("action_candidates") if isinstance(inputs, dict) else None
    shape = _tensor_shape(action)
    if len(shape) != 4:
        return None, None, None, None
    return tuple(_positive_int(value) for value in shape)  # type: ignore[return-value]


def _batch_from_inputs(inputs: dict[str, Any]) -> int | None:
    if not isinstance(inputs, dict):
        return None
    for key in ("observation_history", "goal", "latents"):
        shape = _tensor_shape(inputs.get(key))
        if shape:
            return _positive_int(shape[0])
    return None


def _tensor_shape(node: Any) -> list[Any]:
    if not isinstance(node, dict):
        return []
    ref = node.get("tensor") if isinstance(node.get("tensor"), dict) else node
    shape = ref.get("shape") if isinstance(ref, dict) else None
    return shape if isinstance(shape, list) else []


def _action_dim_from_bounds(bounds: Any) -> int | None:
    if not isinstance(bounds, dict):
        return None
    low = bounds.get("low")
    return len(low) if isinstance(low, list) and low else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
