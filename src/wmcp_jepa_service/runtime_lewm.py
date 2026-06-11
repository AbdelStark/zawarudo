"""Real Push-T LeWorldModel runtime — the `lewm` backend (replaces the mock for real serving).

Loads a trusted, checksum-verified model package (issue #2), builds the vendored LeWM model
(``lewm_model``), and serves encode/rollout/score/plan behind the same ``WorldModelBackend`` Protocol
as the mock (ADR-0001 — the API stays engine-agnostic). CPU-capable; AMP/compile stay OFF until golden
validation (#5) passes (LEWM_INTEGRATION_GUIDE §5).

Input contract (WMCP ``TensorRef``):
- observation_history / goal: RGB frames. ``uint8`` inputs are preprocessed (scale→ImageNet
  normalize); ``float*`` inputs are assumed already preprocessed CHW. HWC (last dim 3) is permuted.
- action_candidates: ``[B,S,T,A]`` float, assumed already in the model's (z-scored) action space —
  the package action scaler is applied (identity by default; RFC-0005 open issue #1).
"""

from __future__ import annotations

import base64
import os
import time
import urllib.request
from io import BytesIO
from typing import Any

import numpy as np
import torch

from . import telemetry
from .lewm_model import build_lewm_from_config
from .model_package import ActionScaler, load_package
from .observability import MODEL_COMPUTE, MODEL_LOADED, PLANNER_ITERATIONS
from .runtime_logging import backend_operation, log_backend_loading, log_backend_ready
from .schemas import ModelMetadata, RequestEnvelope, ResponseEnvelope

_DTYPE_TO_NP = {
    "uint8": np.uint8, "float16": np.float16, "float32": np.float32,
    "float64": np.float64, "int32": np.int32, "int64": np.int64,
}
# Above this many elements, latent/rollout outputs are written to the latent store and returned by uri.
_INLINE_LIMIT = 100_000


class LeWMRuntime:
    """Serves the real Push-T checkpoint. Satisfies the ``WorldModelBackend`` Protocol."""

    def __init__(self, package_path: str, *, device: str = "cpu", latent_store: str | None = None) -> None:
        self.device = device
        load_start = time.perf_counter()
        log_backend_loading(backend="lewm", package_path=package_path, device=device)
        loaded = load_package(package_path, device=device, build_model=build_lewm_from_config)
        self.model = loaded.model
        self.model_id = loaded.model_id
        self.revision = loaded.revision
        self.backend = "lewm"
        self.preprocessing: dict[str, Any] = loaded.preprocessing or {}
        self.action_scaler: ActionScaler = loaded.action_scaler
        norm = self.preprocessing.get("normalize", {})
        self._mean = torch.tensor(norm.get("mean", [0.485, 0.456, 0.406]), device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(norm.get("std", [0.229, 0.224, 0.225]), device=device).view(1, 3, 1, 1)
        self._image = int(self.preprocessing.get("image_size", 224))
        self.latent_store = latent_store if latent_store else os.getenv("WMCP_LATENT_STORE", ".artifacts/latents")
        MODEL_LOADED.labels(self.model_id, self.revision, self.backend).set(1)
        log_backend_ready(
            backend=self.backend,
            model_id=self.model_id,
            revision=self.revision,
            device=self.device,
            image_size=self._image,
            latent_store=self.latent_store,
            load_ms=round((time.perf_counter() - load_start) * 1000, 3),
            synthetic_outputs=False,
        )

    # --- metadata ----------------------------------------------------------------------------

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
            runtime={"backend": self.backend, "device": self.device, "dynamic_batching": False},
        )

    # --- tensor decoding -----------------------------------------------------------------------

    def _decode_tensor(self, ref: dict[str, Any]) -> torch.Tensor:
        """Decode a WMCP ``TensorRef`` (inline | base64 | uri) into a torch tensor of declared shape."""
        shape = [int(x) for x in ref.get("shape", [])]
        np_dtype = _DTYPE_TO_NP.get(ref.get("dtype", "float32"), np.float32)
        encoding = ref.get("encoding")
        if encoding == "inline":
            if ref.get("data") is None:
                raise ValueError("inline tensor missing data")
            arr = np.asarray(ref["data"], dtype=np_dtype).reshape(shape)
        elif encoding == "base64":
            if not ref.get("data_b64"):
                raise ValueError("base64 tensor missing data_b64")
            arr = np.frombuffer(base64.b64decode(ref["data_b64"]), dtype=np_dtype).reshape(shape)
        elif encoding == "uri":
            arr = self._fetch_uri(str(ref.get("uri", "")), np_dtype).reshape(shape)
        else:
            raise ValueError(f"unsupported tensor encoding: {encoding}")
        out = np.ascontiguousarray(arr)
        if not out.flags.writeable:  # e.g. base64 frombuffer -> read-only
            out = out.copy()
        return torch.from_numpy(out).to(self.device)

    def _fetch_uri(self, uri: str, np_dtype: Any) -> np.ndarray:
        """Resolve a tensor URI. Supports file:// / local paths and http(s):// pointing at .npy."""
        if uri.startswith(("http://", "https://")):
            with urllib.request.urlopen(uri, timeout=30) as resp:  # noqa: S310 - operator-provided
                return np.load(BytesIO(resp.read()), allow_pickle=False)
        path = uri[len("file://") :] if uri.startswith("file://") else uri
        if "://" in path:
            raise ValueError(f"cannot resolve tensor uri scheme: {uri!r} (supported: file://, http(s)://)")
        if not os.path.exists(path):
            raise ValueError(f"tensor uri not found: {uri!r} (send inline/base64 pixels, or a reachable .npy uri)")
        return np.load(path, allow_pickle=False).astype(np_dtype)

    def _decode_pixels(self, node: dict[str, Any]) -> torch.Tensor:
        """Decode an observation/goal node to a preprocessed ``(B, F, 3, image, image)`` float tensor."""
        ref = node["tensor"] if isinstance(node, dict) and "tensor" in node else node
        raw = self._decode_tensor(ref)
        if raw.ndim != 5:
            raise ValueError(f"expected a 5-D image tensor (B,F,...), got shape {tuple(raw.shape)}")
        # HWC -> CHW when the last dim is the channel.
        if raw.shape[-1] == 3 and raw.shape[2] != 3:
            raw = raw.permute(0, 1, 4, 2, 3).contiguous()
        b, f = raw.shape[0], raw.shape[1]
        x = raw.reshape(b * f, *raw.shape[2:]).float()
        if str(ref.get("dtype", "float32")) == "uint8":
            x = x / 255.0
            x = (x - self._mean) / self._std
        if x.shape[-1] != self._image or x.shape[-2] != self._image:
            x = torch.nn.functional.interpolate(x, size=(self._image, self._image), mode="bilinear", align_corners=False)
        return x.reshape(b, f, 3, self._image, self._image)

    def _decode_actions(self, node: dict[str, Any]) -> torch.Tensor:
        ref = node["tensor"] if isinstance(node, dict) and "tensor" in node else node
        acts = self._decode_tensor(ref).float()
        if acts.ndim != 4:
            raise ValueError(f"action_candidates must be [B,S,T,A], got {tuple(acts.shape)}")
        return self.action_scaler.transform(acts)

    # --- output helpers ------------------------------------------------------------------------

    def _tensor_output(self, request_id: str, name: str, tensor: torch.Tensor, layout: str) -> dict[str, Any]:
        arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
        shape = [int(x) for x in arr.shape]
        if arr.size <= _INLINE_LIMIT:
            return {"kind": "tensor", "encoding": "inline", "dtype": "float32", "shape": shape, "layout": layout,
                    "data": arr.tolist()}
        os.makedirs(self.latent_store, exist_ok=True)
        path = os.path.join(self.latent_store, f"{request_id}_{name}.npy")
        np.save(path, arr)
        return {"kind": "tensor", "encoding": "uri", "dtype": "float32", "shape": shape, "layout": layout,
                "uri": f"file://{os.path.abspath(path)}"}

    # --- operations ----------------------------------------------------------------------------

    async def encode(self, request: RequestEnvelope) -> ResponseEnvelope:
        with backend_operation(request, backend=self.backend, model_id=self.model_id, revision=self.revision) as log_fields:
            with telemetry.span("wmcp.preprocess", **{"wmcp.operation": "encode"}):
                pixels = self._decode_pixels(request.inputs["observation_history"])  # (B,F,3,img,img)
                log_fields["input.pixels_shape"] = list(pixels.shape)
            start = time.perf_counter()
            with telemetry.span("wmcp.model.encode"), torch.inference_mode():
                info = self.model.encode({"pixels": pixels})
                emb = info["emb"]  # (B,F,192)
            MODEL_COMPUTE.labels(self.model_id, "encode", self.backend).observe(time.perf_counter() - start)
            log_fields["output.latents_shape"] = list(emb.shape)
            return ResponseEnvelope(
                request_id=request.request_id,
                operation="encode",
                model=self.model_id,
                model_revision=self.revision,
                outputs={"latents": self._tensor_output(request.request_id, "latents", emb, "B,H,D")},
                diagnostics={"backend": self.backend, "history": int(emb.shape[1])},
            )

    async def predict(self, request: RequestEnvelope) -> ResponseEnvelope:
        return await self.encode(request)

    def _obs_goal_actions(self, request: RequestEnvelope) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        obs = self._decode_pixels(request.inputs["observation_history"])  # (B,F,3,img,img)
        actions = self._decode_actions(request.inputs["action_candidates"])  # (B,S,T,A)
        goal = self._decode_pixels(request.inputs["goal"]) if "goal" in request.inputs else None
        return obs, goal, actions

    def _build_info(self, obs: torch.Tensor, goal: torch.Tensor | None) -> dict[str, torch.Tensor]:
        # model expects (B, 1, F, 3, img, img): a leading sample-placeholder dim, then F history frames.
        info = {"pixels": obs.unsqueeze(1)}
        if goal is not None:
            info["goal"] = goal.unsqueeze(1)
        return info

    async def rollout(self, request: RequestEnvelope) -> ResponseEnvelope:
        with backend_operation(request, backend=self.backend, model_id=self.model_id, revision=self.revision) as log_fields:
            with telemetry.span("wmcp.preprocess", **{"wmcp.operation": "rollout"}):
                obs, _goal, actions = self._obs_goal_actions(request)
                log_fields["input.observation_shape"] = list(obs.shape)
                log_fields["input.actions_shape"] = list(actions.shape)
            b, s, t = actions.shape[:3]
            start = time.perf_counter()
            with (
                telemetry.span("wmcp.model.rollout", **{"wmcp.candidate_count": s, "wmcp.horizon": t}),
                torch.inference_mode(),
            ):
                info = self.model.rollout(self._build_info(obs, None), actions)
                predicted = info["predicted_emb"]  # (B,S,F+T?,192)
            MODEL_COMPUTE.labels(self.model_id, "rollout", self.backend).observe(time.perf_counter() - start)
            log_fields["output.predicted_latents_shape"] = list(predicted.shape)
            return ResponseEnvelope(
                request_id=request.request_id,
                operation="rollout",
                model=self.model_id,
                model_revision=self.revision,
                outputs={
                    "predicted_latents": self._tensor_output(
                        request.request_id,
                        "predicted_latents",
                        predicted,
                        "B,S,T,D",
                    )
                },
                diagnostics={"backend": self.backend, "candidate_count": s, "horizon": t},
            )

    def _score(self, obs: torch.Tensor, goal: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model.get_cost(self._build_info(obs, goal), actions)  # (B,S)

    async def score(self, request: RequestEnvelope) -> ResponseEnvelope:
        with backend_operation(request, backend=self.backend, model_id=self.model_id, revision=self.revision) as log_fields:
            with telemetry.span("wmcp.preprocess", **{"wmcp.operation": "score"}):
                obs, goal, actions = self._obs_goal_actions(request)
                log_fields["input.observation_shape"] = list(obs.shape)
                log_fields["input.goal_shape"] = list(goal.shape) if goal is not None else None
                log_fields["input.actions_shape"] = list(actions.shape)
            if goal is None:
                raise ValueError("score requires a 'goal' input")
            b, s, t = actions.shape[:3]
            start = time.perf_counter()
            with telemetry.span("wmcp.model.score", **{"wmcp.candidate_count": s, "wmcp.horizon": t}):
                costs = self._score(obs, goal, actions)  # (B,S)
            MODEL_COMPUTE.labels(self.model_id, "score", self.backend).observe(time.perf_counter() - start)
            best_index = torch.argmin(costs, dim=1).tolist()
            log_fields["output.costs_shape"] = list(costs.shape)
            log_fields["output.best_index"] = [int(i) for i in best_index]
            return ResponseEnvelope(
                request_id=request.request_id,
                operation="score",
                model=self.model_id,
                model_revision=self.revision,
                outputs={
                    "costs": self._tensor_output(request.request_id, "costs", costs, "B,S"),
                    "best_index": [int(i) for i in best_index],
                    "cost_statistics": {
                        "min": float(costs.min()),
                        "mean": float(costs.mean()),
                        "max": float(costs.max()),
                    },
                },
                diagnostics={"backend": self.backend, "candidate_count": s, "horizon": t},
            )

    async def plan(self, request: RequestEnvelope) -> ResponseEnvelope:
        with backend_operation(request, backend=self.backend, model_id=self.model_id, revision=self.revision) as log_fields:
            params = request.parameters
            horizon = int(params.get("horizon", 16))
            iterations = int(params.get("iterations", 5))
            population = int(params.get("candidates", 256))
            elite_frac = float(params.get("elite_fraction", 0.1))
            action_dim = 10
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(params.get("seed", 0)))
            log_fields["planner.population"] = population
            log_fields["planner.elite_fraction"] = elite_frac

            with telemetry.span("wmcp.preprocess", **{"wmcp.operation": "plan"}):
                obs = self._decode_pixels(request.inputs["observation_history"])
                goal = self._decode_pixels(request.inputs["goal"]) if "goal" in request.inputs else None
                log_fields["input.observation_shape"] = list(obs.shape)
                log_fields["input.goal_shape"] = list(goal.shape) if goal is not None else None
            if goal is None:
                raise ValueError("plan requires a 'goal' input")
            b = obs.shape[0]
            if b != 1:
                raise ValueError("plan supports batch size 1")

            n_elite = min(population, max(2, int(population * elite_frac)))
            mean = torch.zeros(horizon, action_dim, device=self.device)
            std = torch.ones(horizon, action_dim, device=self.device)
            best_seq = mean.clone()
            best_cost = float("inf")
            best_cost_by_iteration: list[float] = []

            start = time.perf_counter()
            with telemetry.span("wmcp.model.plan", **{"wmcp.planner_iterations": iterations, "wmcp.horizon": horizon}):
                for _ in range(iterations):
                    noise = torch.randn(population, horizon, action_dim, generator=generator, device=self.device)
                    candidates = (mean + std * noise).unsqueeze(0)  # (1, P, T, A)
                    costs = self._score(obs, goal, candidates)[0]  # (P,)
                    elite_idx = torch.topk(costs, n_elite, largest=False).indices
                    elites = candidates[0, elite_idx]  # (n_elite, T, A)
                    mean = elites.mean(dim=0)
                    std = elites.std(dim=0, unbiased=False).clamp_min(1e-3)
                    iter_best = float(costs[elite_idx[0]])
                    if iter_best < best_cost:
                        best_cost = iter_best
                        best_seq = candidates[0, elite_idx[0]].clone()
                    best_cost_by_iteration.append(best_cost)
            MODEL_COMPUTE.labels(self.model_id, "plan", self.backend).observe(time.perf_counter() - start)
            PLANNER_ITERATIONS.labels(self.model_id, "plan").observe(iterations)

            seq = best_seq.unsqueeze(0)  # (1,T,A)
            log_fields["output.action_sequence_shape"] = list(seq.shape)
            log_fields["planner.best_cost"] = best_cost
            return ResponseEnvelope(
                request_id=request.request_id,
                operation="plan",
                model=self.model_id,
                model_revision=self.revision,
                outputs={
                    "best_action_sequence": self._tensor_output(request.request_id, "plan", seq, "B,T,A"),
                    "first_action": {
                        "kind": "tensor",
                        "encoding": "inline",
                        "dtype": "float32",
                        "shape": [1, action_dim],
                        "layout": "B,A",
                        "data": [seq[0, 0].tolist()],
                    },
                    "best_cost": [best_cost],
                    "planner_diagnostics": {
                        "iterations": iterations,
                        "candidates": population,
                        "best_cost_by_iteration": best_cost_by_iteration,
                    },
                },
                diagnostics={"backend": self.backend, "horizon": horizon},
            )
