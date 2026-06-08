---
name: wmcp-api
description: The WMCP-aligned HTTP contract for this service — request/response envelopes, the six operations (metadata/encode/predict/rollout/score/plan), TensorRef payloads, error codes, and the KServe V2 adapter. Activate whenever adding/changing an endpoint, an operation, a request field, the error model, or when a client integration question arises. This contract is GATED — schema changes need approval and must stay in sync across schemas.py, openapi.yaml, and schemas/*.schema.json.
prerequisites: pydantic v2; FastAPI
---

# WMCP API

<purpose>
Defines the stable external contract the service exposes. All routes live in `server.py`; all shapes
live in `schemas.py`. Read this before touching either.
</purpose>

<context>
- Envelopes (`schemas.py`): `RequestEnvelope`, `ResponseEnvelope`, `ErrorEnvelope`, `ModelMetadata`.
- `RequestEnvelope`: `wmcp_version, request_id, operation, model, model_revision?, trace, inputs,
  parameters, return_options`. `operation` ∈ metadata|encode|predict|rollout|score|plan.
- `ResponseEnvelope`: `…, outputs: dict, diagnostics: dict`.
- Tensors are always a `TensorRef`: `{kind:"tensor", encoding: inline|base64|uri, dtype, shape, layout,
  data?|data_b64?|uri?, sha256?}`. A `model_validator` enforces encoding↔payload (inline→`data`,
  base64→`data_b64`, uri→`uri`).
- `Observation{modality, tensor, preprocessing}`; `ActionTensor{space, tensor, normalization?, bounds?}`.
- Routes use the `:operation` suffix style, e.g. `POST /wmcp/v1/models/{id}:score`.
- The model id gate: `server.py` rejects any `model != WMCP_MODEL_ID` with 404 `MODEL_NOT_FOUND`.
- Errors: `_handle()` maps Pydantic `ValidationError`→422 `INVALID_ARGUMENT`, re-raises `HTTPException`
  (404/400), and wraps everything else →500 `INTERNAL`. Body shape: `{detail:{code,message}}`.
- KServe V2 adapter `POST /v2/models/{name}/infer` reads `parameters.operation` (default `score`) and
  re-dispatches through `_handle()`.
</context>

<procedure>
1. Determine if you are changing the CONTRACT (envelope/operation/field) or just behavior.
2. Contract change → GATED: get approval, then edit `schemas.py` AND mirror in `api/openapi.yaml`
   AND `schemas/wmcp-message.schema.json`. They are one artifact in three files.
3. New operation: add to `RequestEnvelope.operation` Literal → `WorldModelBackend` Protocol →
   backend impl → route delegating to `_handle("<op>", model_id, body, backend.<op>)`.
4. Behavior-only change: edit the backend method; no schema edit.
5. Always finish: add/extend a contract test, then `ruff check . && mypy src && pytest -q`.
</procedure>

<patterns>
<do>
— Route every request through `_handle()` so metrics + error mapping stay uniform.
— Return large tensors (`predicted_latents`, latents) via `encoding:"uri"`.
— Keep `request_id` echoed back in the response envelope.
</do>
<dont>
— Don't raise bare exceptions in routes for client errors → raise `HTTPException` with a `{code,message}`.
— Don't add raw arrays to `outputs` — wrap in a `TensorRef`.
— Don't diverge `schemas.py` from `api/openapi.yaml` / `schemas/*.schema.json`.
</dont>
</patterns>

<examples>
Minimal score request (also `examples/score_request.json`, `tests/test_contracts.py`):
```json
{ "wmcp_version":"0.1", "request_id":"test-1", "operation":"score", "model":"lewm-pusht",
  "inputs": { "action_candidates": { "space":"continuous",
    "tensor": { "kind":"tensor","encoding":"uri","dtype":"float32",
                "shape":[1,4,4,10],"layout":"B,S,T,A","uri":"memory://actions.npy" } } } }
```
`shape [B,S,T,A]` = batch, candidate-sequences, horizon, action-dim.
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| 422 on a request that "looks right" | TensorRef encoding↔payload mismatch | match `encoding` to `data`/`data_b64`/`uri` |
| 404 MODEL_NOT_FOUND | `model` ≠ `WMCP_MODEL_ID` | use configured id |
| New op returns 400 in KServe path | op not in adapter dispatch map | add to dispatch in `server.py` |
</troubleshooting>

<references>
— src/wmcp_jepa_service/schemas.py: envelope + TensorRef definitions
— src/wmcp_jepa_service/server.py: routes, `_handle`, KServe adapter
— api/openapi.yaml, schemas/wmcp-message.schema.json: external contract mirrors
— examples/*.json: canonical request/response fixtures
</references>
</content>
