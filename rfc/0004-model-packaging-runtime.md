# RFC-0004: Model packaging and runtime manifest for WMCP world models

| Field | Value |
|---|---|
| Status | Draft |
| Created | 2026-06-08 |
| Depends on | RFC-0001 |

## Abstract

This RFC defines a model package and runtime manifest format for WMCP world-model services. World models require more than weights: they require architecture config, preprocessing, action spaces, normalizers, supported operations, runtime adapter metadata, limits, and observability labels.

## Motivation

A checkpoint alone is insufficient to serve a world model correctly. For action-conditioned planning, the runtime must know image preprocessing, action scaling, action bounds, history size, horizon limits, latent shapes, cost functions, and code/runtime compatibility. Without a manifest, services risk silently producing invalid action scores or plans.

## Package layout

```text
model-package/
├── manifest.json
├── config.json
├── weights.pt | model.safetensors | triton-model-repository/
├── preprocessing.json
├── action_space.json
├── normalizers/
│   ├── action_scaler.json
│   └── observation_stats.json
├── checksums.txt
└── README.md
```

## Manifest schema

A manifest MUST include:

```json
{
  "schema_version": "0.1",
  "model_id": "lewm-pusht",
  "model_family": "jepa",
  "model_type": "action_conditioned_world_model",
  "task": "pusht",
  "source_repository": "https://github.com/lucas-maes/le-wm",
  "source_revision": "<git-sha>",
  "artifact_uri": "hf://quentinll/lewm-pusht",
  "artifact_revision": "<hf-revision>",
  "artifact_sha256": "<sha256>",
  "framework": "pytorch",
  "runtime": {
    "backend": "ray-pytorch",
    "class": "wmcp_jepa_service.runtime.LeWMRuntime",
    "python": "3.10"
  },
  "supported_operations": ["metadata", "encode", "rollout", "score", "plan"],
  "inputs": {},
  "outputs": {},
  "preprocessing": {},
  "action_space": {},
  "latent_space": {},
  "limits": {}
}
```

See `schemas/model-manifest.schema.json` for a concrete draft.

## Artifact safety

Servers MUST NOT load arbitrary model artifacts from request payloads. Artifact URIs are deployment-time configuration, not inference-time user inputs.

Recommended safety policy:

1. Pin artifact revisions.
2. Verify checksums before loading.
3. Load pickle-based PyTorch files only from trusted artifacts.
4. Prefer `safetensors` or graph-export formats when feasible.
5. Isolate conversion in a build/init job rather than request path.
6. Produce a signed model package or attestation for production.

## Runtime declaration

The manifest MUST declare supported runtime backends. Example:

```json
{
  "runtime_backends": [
    {
      "name": "ray-pytorch",
      "status": "primary",
      "device": "cuda",
      "supports_dynamic_batching": true
    },
    {
      "name": "triton-python",
      "status": "experimental",
      "device": "cuda",
      "supports_dynamic_batching": true
    }
  ]
}
```

## Preprocessing declaration

Preprocessing MUST be explicit:

```json
{
  "image": {
    "input_layouts": ["B,T,H,W,C", "B,T,C,H,W"],
    "target_layout": "B,T,C,H,W",
    "resize": [224, 224],
    "dtype": "float32",
    "scale": "0_1",
    "normalization": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  }
}
```

If upstream model-specific normalization differs, the manifest MUST use model-specific values.

## Action-space declaration

Action-space declaration MUST include dimension, dtype, shape conventions, normalization, and bounds.

```json
{
  "type": "continuous",
  "dimension": 10,
  "layout": "B,S,T,A",
  "dtype": "float32",
  "normalization": {
    "name": "model_default",
    "artifact": "normalizers/action_scaler.json"
  },
  "bounds": {
    "low": [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    "high": [1,1,1,1,1,1,1,1,1,1]
  }
}
```

The `bounds` example above is a placeholder unless verified from the source evaluation artifacts.

## Operation declaration

Each operation SHOULD declare input and output shapes.

```json
{
  "operation": "score",
  "inputs": {
    "observation_history": "B,H,C,224,224",
    "goal": "B,G,C,224,224",
    "action_candidates": "B,S,T,10"
  },
  "outputs": {
    "costs": "B,S",
    "best_index": "B"
  }
}
```

## Compatibility levels

| Level | Requirements |
|---|---|
| L0 | Manifest and metadata load. |
| L1 | Encode and rollout. |
| L2 | Score. |
| L3 | Plan. |
| L4 | Production observability and benchmark conformance. |

The Push-T demo should target L4.

## Open questions

1. Should WMCP require OCI model packages?
2. Should model packages support multiple runtime artifacts under one manifest?
3. Should action normalizers be represented using a standard scaler schema?
4. Should artifact signatures be mandatory for public deployment?
5. Should WMCP define a canonical model-card extension?
