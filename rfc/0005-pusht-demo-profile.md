# RFC-0005: Push-T demo conformance profile

| Field | Value |
|---|---|
| Status | Draft |
| Created | 2026-06-08 |
| Depends on | RFC-0001, RFC-0002, RFC-0003, RFC-0004 |

## Abstract

This RFC defines a concrete conformance profile for the first end-to-end WMCP world-model demo: serving a Push-T LeWorldModel checkpoint through a production-grade API.

The profile is intended to validate WMCP fields, deployment practices, runtime choices, telemetry, and benchmark methodology.

## Model profile

| Field | Value |
|---|---|
| `model_id` | `lewm-pusht` |
| `model_family` | `jepa` |
| `model_type` | `action_conditioned_world_model` |
| `task` | `pusht` |
| `source` | LeWorldModel repository |
| `artifact` | Hugging Face `quentinll/lewm-pusht` |
| `observation_modality` | RGB pixels |
| `image_size` | 224x224 target input |
| `history_size` | 3 default |
| `action_dim` | 10 from published config |
| `latent_dim` | 192 from published config |
| `operations` | `metadata`, `encode`, `rollout`, `score`, `plan` |

## Required endpoints

A conforming Push-T demo service MUST expose:

1. `GET /healthz`
2. `GET /readyz`
3. `GET /metrics`
4. `GET /wmcp/v1/models`
5. `GET /wmcp/v1/models/lewm-pusht`
6. `POST /wmcp/v1/models/lewm-pusht:encode`
7. `POST /wmcp/v1/models/lewm-pusht:rollout`
8. `POST /wmcp/v1/models/lewm-pusht:score`
9. `POST /wmcp/v1/models/lewm-pusht:plan`

## Required demo artifacts

1. Model manifest.
2. Example rollout request.
3. Example score request.
4. Example plan request.
5. Docker Compose file.
6. Kubernetes deployment manifest.
7. Prometheus config.
8. OpenTelemetry Collector config.
9. Grafana dashboard.
10. Load-test report.

## Input profile

Observation history:

```text
shape: B,H,C,224,224 or B,H,224,224,C
B: 1 for baseline demo
H: 3
C: 3
encoding: inline for tiny examples; URI/base64 for real images
```

Candidate actions:

```text
shape: B,S,T,10
S baseline values: 16, 64, 256
T baseline values: 8, 16, 32
A: 10
```

Goal:

```text
shape: B,G,C,224,224 or B,G,224,224,C
G: implementation-defined; at least one goal frame
```

## Output profile

Score response:

```text
costs: B,S float32
best_index: B int64
cost_statistics: min, mean, max
```

Plan response:

```text
best_action_sequence: B,T,10 float32
first_action: B,10 float32
best_cost: B float32
planner_diagnostics: iterations, candidates, best_cost_by_iteration
```

## Observability profile

The demo MUST expose:

- `wmcp_requests_total`
- `wmcp_request_latency_seconds`
- `wmcp_model_compute_seconds`
- `wmcp_queue_wait_seconds`
- `wmcp_candidate_count`
- `wmcp_rollout_horizon`
- `wmcp_planner_iterations`
- `wmcp_input_validation_errors_total`
- model readiness metric
- GPU metrics if running with GPU

The demo MUST emit traces with `wmcp.request`, `wmcp.validate`, `wmcp.preprocess`, `wmcp.model.rollout`, `wmcp.model.score`, and `wmcp.serialize` spans.

## Benchmark matrix

| Profile | B | S | T | Operation | Purpose |
|---|---:|---:|---:|---|---|
| smoke | 1 | 4 | 4 | score | Correctness and shape. |
| small | 1 | 16 | 8 | score/rollout | Local demo. |
| medium | 1 | 256 | 16 | score/plan | Main demo. |
| large | 1 | 1024 | 32 | score | Stress and OOM boundary. |
| concurrent | 1 | 64 | 16 | score | Dynamic batching behavior. |

Every benchmark report MUST include:

- hardware;
- GPU driver/runtime;
- container image digest;
- model revision/checksum;
- backend;
- dtype;
- request shape;
- p50/p95/p99 latency;
- throughput;
- GPU utilization;
- GPU memory;
- queue wait;
- batch size distribution.

## Acceptance criteria

A Push-T demo is conforming when:

1. All required endpoints respond successfully with real model integration.
2. Contract tests pass using example JSON payloads.
3. The score endpoint returns `[B,S]` costs for all benchmark profiles that fit configured limits.
4. The plan endpoint returns `[B,T,10]` actions and a `[B,10]` first action.
5. Prometheus and OTel telemetry are visible during the demo.
6. The benchmark report includes the full benchmark context.
7. At least five RFC issues are filed based on implementation learnings.

## Known open issues for Push-T

1. Confirm the exact action scaler and action bounds used by upstream LeWorldModel evaluation.
2. Confirm whether the HF `weights.pt` should be converted to a safer runtime artifact before deployment.
3. Confirm canonical goal-frame handling for Push-T score/planning.
4. Confirm whether history actions are required or optional for each endpoint.
5. Confirm latency targets after benchmarking on selected GPU hardware.
