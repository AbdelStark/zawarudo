---
name: runtime-backend
description: How a WorldModelBackend is implemented. The real `LeWMRuntime` (the `lewm` backend) already lives in runtime_lewm.py; this covers how it works and how to update the checkpoint or add another backend. Activate for any work on runtime.py / runtime_lewm.py, model loading, encode/predict/rollout/score/plan compute, torch/transformers integration, device placement, or wiring WMCP_BACKEND.
prerequisites: the `lewm` extra (torch + transformers + einops + safetensors); a trusted model package (see model-packaging skill)
---

# Runtime Backend

<purpose>
The runtime executes model operations behind the `WorldModelBackend` Protocol so the WMCP API stays
engine-agnostic (ADR-0001). Two backends ship: the default `MockWorldModelBackend` (`runtime.py`,
synthetic outputs for contract tests) and the real `LeWMRuntime` (`runtime_lewm.py`, the `lewm`
backend on the Push-T checkpoint). Both satisfy the Protocol; adding a backend must not change the API.
</purpose>

<context>
- Protocol (`runtime.py`): `metadata()`, and `async` `encode/predict/rollout/score/plan`, each taking a
  `RequestEnvelope` and returning a `ResponseEnvelope`.
- Mock behavior to mirror in shape (not values): `encode`→latents `[B,H,192]`; `rollout`→
  `predicted_latents [B,S,T,192]` via uri; `score`→`costs [B,S]` + `best_index` + `cost_statistics`;
  `plan`→`best_action_sequence [B,T,A]`, `first_action [B,A]`, `best_cost`, `planner_diagnostics`.
- Latent dim is 192 (`ModelMetadata.latent_space`). Action dim is 10 for Push-T. Action tensor layout
  `[B,S,T,A]`.
- Metrics: wrap real compute in `MODEL_COMPUTE.labels(model, op, backend).observe(...)`; set
  `MODEL_LOADED` on construction. `_handle()` already records request-level latency/counts.
- Upstream: `LEWM_INTEGRATION_GUIDE.md` is the authoritative integration recipe; `SOURCES.md` lists
  upstream repos (le-wm, stable-worldmodel). LeWorldModel exposes `encode`, `predict`, `rollout`, and a
  goal-conditioned `get_cost`-style function.
</context>

<procedure>
To update the Push-T checkpoint or add a new world-model backend (the real `LeWMRuntime` already
follows these steps in `runtime_lewm.py`):
1. Pin upstream commits (le-wm, stable-worldmodel, stable-pretraining) — do not float `main`.
2. Acquire/convert a trusted model package (see `model-packaging` skill). Never load weights from a request.
3. Implement the backend (e.g. `LeWMRuntime`) in its own module (`runtime_lewm.py`):
   - `__init__`: load pinned model from package path; `.to(device).eval()`; freeze params; load
     preprocessor + action scaler; set `MODEL_LOADED`.
   - each op: decode `TensorRef` inputs → preprocess/normalize → `torch.inference_mode()` compute →
     format as `ResponseEnvelope` with `TensorRef` outputs (large tensors by `uri`).
4. Wire selection in `server.py`: choose backend from `WMCP_BACKEND` (`mock` | `lewm`), keep Mock as default.
5. Golden-validate (see `testing` skill) before exposing: assert cost shape + numerical tolerance vs upstream.
6. Only then optimize: flatten candidate dim into batch, AMP/float16 after numeric check, `torch.compile`,
   reuse goal encodings within a plan request. Correctness first.
</procedure>

<patterns>
<do>
— Keep the API/envelope handling identical to the mock; only the compute changes.
— `torch.inference_mode()` for all forward passes; freeze the model; assert no grads.
— Decode tensors by `TensorRef.encoding`; honor `dtype`, `shape`, `layout`.
</do>
<dont>
— Don't put model code in `server.py`. — Don't accept weights/code from request bodies.
— Don't enable AMP/`torch.compile` before numerical validation passes.
— Don't return huge latents inline — use `uri`.
</dont>
</patterns>

<examples>
Skeleton below; the **real, working implementation** is `runtime_lewm.py` (read it first):
```python
class LeWMRuntime:  # satisfies WorldModelBackend
    def __init__(self, package_path: str, device: str = "cuda"):
        self.model = load_pinned_lewm_model(package_path).to(device).eval()
        freeze(self.model)
        self.preprocessor = load_preprocessor(package_path)
        self.action_scaler = load_action_scaler(package_path)
        MODEL_LOADED.labels(self.model_id, self.revision, "lewm").set(1)

    async def score(self, request):
        obs  = self.preprocessor.decode_observation(request.inputs["observation_history"])
        goal = self.preprocessor.decode_observation(request.inputs["goal"])
        acts = self.action_scaler.transform(load_tensor(request.inputs["action_candidates"]))
        with torch.inference_mode():
            costs = self.model.get_cost({"pixels": obs, "goal": goal}, acts)
        return format_score_response(request, costs)  # -> ResponseEnvelope, costs as TensorRef [B,S]
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| Output shape ≠ mock | wrong layout/order | match `[B,S]` costs, `[B,S,T,192]` rollout latents |
| Grad memory blowup | model not frozen / no inference_mode | freeze + `torch.inference_mode()` |
| Numeric mismatch vs upstream | normalization/AMP | verify action scaler & bounds; disable AMP until validated |
</troubleshooting>

<references>
— src/wmcp_jepa_service/runtime_lewm.py: the real `LeWMRuntime` (`lewm` backend) — decode → inference → CEM plan
— src/wmcp_jepa_service/lewm_model.py: vendored LeWorldModel (ViT encoder + predictor)
— src/wmcp_jepa_service/runtime.py: Protocol + MockWorldModelBackend (the shape contract to mirror)
— LEWM_INTEGRATION_GUIDE.md: full integration recipe + perf knobs
— SOURCES.md, adr/ADR-0001-inference-engine.md: upstream + engine rationale
</references>
