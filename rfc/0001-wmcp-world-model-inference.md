# RFC-0001: WMCP extension for world-model inference

| Field | Value |
|---|---|
| Status | Draft |
| Authors | WMCP-JEPA Serve working draft |
| Created | 2026-06-08 |
| Target | WMCP core protocol |

## Abstract

This RFC defines a WMCP extension for serving action-conditioned world models. It introduces standard operations for encoding observations, predicting latent states, rolling out future trajectories, scoring candidate action sequences, and planning with model-predictive control.

The motivating implementation is a Push-T JEPA/LeWorldModel service, but the protocol is intended to support future V-JEPA-like and other latent world-model checkpoints.

## Motivation

World-model serving differs from ordinary classification, embedding, and text-generation serving. Inputs may include image histories, continuous actions, goals, candidate action tensors, and planner configuration. Outputs may include latent embeddings, predicted trajectories, costs, selected actions, and diagnostics. Without a standard protocol, every model demo exposes incompatible ad hoc APIs.

WMCP should make world-model inference:

- typed and self-describing;
- observable and benchmarkable;
- compatible with deployment systems;
- reusable across tasks and model families;
- safe from ambiguous action/observation semantics.

## Terminology

| Term | Definition |
|---|---|
| Observation | Sensor-derived input such as RGB frames, proprioception, depth, or state vectors. |
| Action | Control input applied to the environment or used to condition prediction. |
| History | Ordered context window of previous observations/actions. |
| Latent | Learned representation used by the model for prediction or scoring. |
| Goal | Target observation, state, latent, or task condition. |
| Candidate action sequence | A proposed sequence of actions over a rollout horizon. |
| Rollout | Model-predicted future latent/state trajectory conditioned on actions. |
| Cost | Scalar or vector value measuring candidate disagreement with the goal. |
| Planner | Algorithm that searches over actions using model predictions/costs. |

## Protocol versioning

Every request MUST include `wmcp_version`. Servers MUST reject unsupported major versions. Servers SHOULD include a `supported_versions` list in model metadata.

```json
{
  "wmcp_version": "0.1",
  "operation": "score",
  "model": "lewm-pusht"
}
```

## Common request envelope

A WMCP world-model request MUST include:

- `wmcp_version`
- `request_id`
- `operation`
- `model`
- `inputs`

It SHOULD include:

- `model_revision`
- `trace.traceparent`
- `parameters`
- `return_options`
- `client_metadata`

```json
{
  "wmcp_version": "0.1",
  "request_id": "uuid",
  "operation": "rollout",
  "model": "lewm-pusht",
  "model_revision": "hf:quentinll/lewm-pusht:<revision>",
  "trace": {"traceparent": "..."},
  "inputs": {},
  "parameters": {},
  "return_options": {}
}
```

## Common response envelope

A successful response MUST include:

- `wmcp_version`
- `request_id`
- `model`
- `model_revision`
- `operation`
- `outputs`
- `diagnostics` or `telemetry`

```json
{
  "wmcp_version": "0.1",
  "request_id": "uuid",
  "model": "lewm-pusht",
  "model_revision": "...",
  "operation": "score",
  "outputs": {},
  "diagnostics": {
    "latency_ms": 42.1
  }
}
```

## Standard operations

### `metadata`

Returns model and runtime capabilities.

### `encode`

Converts observations/goals into model latents.

### `predict`

Predicts next latent(s) from current latents and actions. This MAY be internal-only for some deployments.

### `rollout`

Predicts a future latent/state trajectory for one or more action sequences.

### `score`

Scores candidate action sequences against a goal.

### `plan`

Searches for an action sequence using model-based planning.

## Tensor representation

Tensors MUST declare:

- `kind`
- `encoding`
- `dtype`
- `shape`
- `layout`

Tensors MAY include:

- `data` for inline JSON;
- `data_b64` for base64 payloads;
- `uri` for external object references;
- `sha256` for checksum verification;
- `normalization` metadata.

Example:

```json
{
  "kind": "tensor",
  "encoding": "uri",
  "dtype": "uint8",
  "shape": [1, 3, 224, 224, 3],
  "layout": "B,T,H,W,C",
  "uri": "s3://wmcp-demo/obs.npy",
  "sha256": "..."
}
```

## Observation representation

Observations MUST describe modality and layout.

```json
{
  "modality": "rgb",
  "tensor": {
    "kind": "tensor",
    "encoding": "uri",
    "dtype": "uint8",
    "shape": [1, 3, 224, 224, 3],
    "layout": "B,T,H,W,C",
    "uri": "s3://..."
  },
  "preprocessing": {
    "resize": [224, 224],
    "normalization": "model_default"
  }
}
```

## Action representation

Actions MUST describe action dimension, coordinate frame or semantics where known, bounds, and normalization.

```json
{
  "space": "continuous",
  "dtype": "float32",
  "shape": [1, 256, 16, 10],
  "layout": "B,S,T,A",
  "normalization": "model_default",
  "tensor": {
    "kind": "tensor",
    "encoding": "uri",
    "dtype": "float32",
    "shape": [1, 256, 16, 10],
    "layout": "B,S,T,A",
    "uri": "s3://..."
  }
}
```

## Error response

Errors MUST be structured and machine-readable.

```json
{
  "wmcp_version": "0.1",
  "request_id": "uuid",
  "error": {
    "code": "INVALID_TENSOR_SHAPE",
    "message": "Expected action dimension 10.",
    "retryable": false,
    "details": {"field": "inputs.action_candidates"}
  }
}
```

## Compatibility with KServe V2

A WMCP world-model server MAY expose a KServe V2-compatible endpoint. KServe V2 compatibility SHOULD be an adapter, because WMCP carries semantics that are not present in generic infer requests: action-space meaning, rollout horizon, planner settings, latent return options, and goal-conditioned costs.

## Security considerations

1. Servers MUST validate tensor shapes and declared payload sizes before decoding large payloads.
2. Servers MUST enforce max horizon, candidate count, batch size, and planner iterations.
3. Servers MUST NOT load arbitrary code, model paths, or untrusted checkpoint artifacts from inference requests.
4. URI-based payloads MUST use allowlisted schemes and credentials.
5. Raw request IDs, user IDs, and URI paths MUST NOT become high-cardinality metric labels.

## Open questions

1. Should WMCP define binary gRPC tensors as mandatory for production conformance?
2. Should latents be standardized as opaque handles rather than arrays?
3. Should `plan` be a core operation or a higher-level extension built on `score`?
4. Should cost values be dimensionless model-specific scores or standardized task metrics?
5. How should world-model services declare action semantics across robot embodiments?
