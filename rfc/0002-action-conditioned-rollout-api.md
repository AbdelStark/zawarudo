# RFC-0002: Action-conditioned rollout, scoring, and planning APIs

| Field | Value |
|---|---|
| Status | Draft |
| Created | 2026-06-08 |
| Depends on | RFC-0001 |

## Abstract

This RFC specifies the semantics of `rollout`, `score`, and `plan` operations for action-conditioned world models. These operations are central to JEPA-based world-model inference services because they expose the predictive and planning capabilities that distinguish world models from ordinary embedding services.

## Shape conventions

Recommended symbolic dimensions:

| Symbol | Meaning |
|---|---|
| `B` | Client batch size. |
| `H` | Observation/action history length. |
| `S` | Number of candidate action sequences. |
| `T` | Future rollout horizon. |
| `A` | Action dimension. |
| `D` | Latent dimension. |
| `C` | Image channels. |

Recommended layouts:

- Observation pixels: `B,H,C,Height,Width` or `B,H,Height,Width,C` with explicit layout.
- Action history: `B,H,A`.
- Candidate actions: `B,S,T,A`.
- Predicted embeddings: `B,S,T,D`.
- Costs: `B,S`.

Servers MUST not infer ambiguous layouts. Clients MUST declare layouts.

## Rollout operation

### Request

A rollout request MUST include:

- observation history or current latent;
- candidate action sequences;
- rollout horizon;
- return options.

```json
{
  "wmcp_version": "0.1",
  "request_id": "uuid",
  "operation": "rollout",
  "model": "lewm-pusht",
  "inputs": {
    "observation_history": {"modality": "rgb", "tensor": {"kind": "tensor"}},
    "action_candidates": {"space": "continuous", "tensor": {"kind": "tensor"}}
  },
  "parameters": {
    "history_size": 3,
    "horizon": 16
  },
  "return_options": {
    "include_predicted_latents": true,
    "latent_encoding": "uri"
  }
}
```

### Response

A rollout response MUST include one of:

- `predicted_latents`; or
- `predicted_latents_ref`; or
- an error explaining why latents were not returned.

It SHOULD include timing diagnostics and shape metadata.

```json
{
  "outputs": {
    "predicted_latents": {
      "kind": "tensor",
      "encoding": "uri",
      "dtype": "float32",
      "shape": [1, 256, 16, 192],
      "layout": "B,S,T,D",
      "uri": "s3://wmcp-demo/out/latents.npy"
    }
  },
  "diagnostics": {
    "timing_ms": {
      "preprocess": 4.2,
      "queue_wait": 1.1,
      "model_compute": 38.7,
      "serialize": 0.8
    }
  }
}
```

## Score operation

### Request

A score request MUST include:

- observation history or current latent;
- goal observation or goal latent;
- candidate action sequences.

It MAY include a cost function name if the model package supports multiple cost functions.

```json
{
  "operation": "score",
  "model": "lewm-pusht",
  "inputs": {
    "observation_history": {"modality": "rgb", "tensor": {"kind": "tensor"}},
    "goal": {"modality": "rgb", "tensor": {"kind": "tensor"}},
    "action_candidates": {"space": "continuous", "tensor": {"kind": "tensor"}}
  },
  "parameters": {
    "history_size": 3,
    "horizon": 16,
    "cost": "goal_latent_mse"
  },
  "return_options": {
    "include_candidate_costs": true,
    "include_best_index": true,
    "include_predicted_latents": false
  }
}
```

### Response

A score response MUST include `costs` and SHOULD include `best_index`, `cost_statistics`, and timing diagnostics.

```json
{
  "outputs": {
    "costs": {
      "kind": "tensor",
      "encoding": "inline",
      "dtype": "float32",
      "shape": [1, 4],
      "layout": "B,S",
      "data": [[0.91, 0.42, 0.57, 0.12]]
    },
    "best_index": [3],
    "cost_statistics": {
      "min": 0.12,
      "mean": 0.505,
      "max": 0.91
    }
  }
}
```

## Plan operation

### Request

A plan request MUST include observation history, goal, planner name, horizon, and action constraints. It SHOULD include a seed for reproducibility.

```json
{
  "operation": "plan",
  "model": "lewm-pusht",
  "inputs": {
    "observation_history": {"modality": "rgb", "tensor": {"kind": "tensor"}},
    "goal": {"modality": "rgb", "tensor": {"kind": "tensor"}}
  },
  "parameters": {
    "planner": "cem",
    "horizon": 16,
    "iterations": 5,
    "candidates": 256,
    "elite_fraction": 0.1,
    "seed": 1234,
    "action_bounds": {
      "low": [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
      "high": [1,1,1,1,1,1,1,1,1,1]
    }
  }
}
```

### Response

A plan response MUST include:

- `best_action_sequence`;
- `first_action`;
- `best_cost`;
- planner diagnostics.

It SHOULD include per-iteration best/mean costs.

```json
{
  "outputs": {
    "best_action_sequence": {
      "kind": "tensor",
      "encoding": "inline",
      "dtype": "float32",
      "shape": [1, 16, 10],
      "layout": "B,T,A",
      "data": []
    },
    "first_action": {
      "kind": "tensor",
      "encoding": "inline",
      "dtype": "float32",
      "shape": [1, 10],
      "layout": "B,A",
      "data": []
    },
    "best_cost": [0.12],
    "planner_diagnostics": {
      "iterations": 5,
      "candidates_per_iteration": 256,
      "best_cost_by_iteration": [0.55, 0.31, 0.22, 0.15, 0.12]
    }
  }
}
```

## Determinism and reproducibility

Servers SHOULD support deterministic planning when the request includes a seed and deterministic backend settings are enabled. Responses SHOULD report whether execution was deterministic.

```json
{
  "diagnostics": {
    "deterministic": true,
    "seed": 1234,
    "backend_determinism": "torch_deterministic_algorithms"
  }
}
```

## Planner cancellation and timeout

Servers MUST enforce planner timeouts. If a plan request times out, the server MAY return the best candidate found so far when `return_partial_on_timeout=true`. The response MUST mark the result as partial.

```json
{
  "error": {
    "code": "TIMEOUT",
    "message": "Planner exceeded 10s timeout.",
    "retryable": true
  },
  "partial_outputs": {
    "best_action_sequence": {},
    "best_cost": [0.22]
  }
}
```

## Conformance requirements

A compliant implementation MUST:

1. Validate declared tensor layouts and shapes.
2. Return cost shape `[B,S]` for `score`.
3. Return action shape `[B,T,A]` for `plan`.
4. Expose model metadata describing action dimension and supported horizon limits.
5. Include diagnostics with timing breakdowns.
6. Emit standard telemetry described in RFC-0003.

## Open questions

1. Should `score` accept arbitrary user-defined cost functions, or only model-declared costs?
2. Should WMCP standardize candidate generation algorithms such as CEM, MPPI, and random shooting?
3. Should a plan endpoint be allowed to maintain session state between receding-horizon calls?
4. How should partial planner outputs be represented on cancellation?
