# Product Requirements Document: WMCP-JEPA Serve

## 1. Product summary

`wmcp-jepa-serve` is a backend inference service for JEPA-based, action-conditioned world models. The first demo serves a Push-T LeWorldModel checkpoint and exposes model primitives through a WMCP-aligned API for encoding observations, rolling out latent futures, scoring candidate actions, and planning with model-predictive control.

The product goal is to demonstrate that world models can be served with the same operational rigor expected from modern LLM and embedding services: typed API contracts, model versioning, batching, health/readiness, latency breakdowns, GPU telemetry, OpenTelemetry traces, Prometheus metrics, deployment manifests, and conformance tests.

## 2. Problem statement

World models are increasingly useful for robotics, planning, simulation, and model-based control, but current tooling is mostly research-code oriented. A user can train or evaluate a model from a notebook or repository script, yet cannot easily deploy it as a production service with stable API semantics, observability, autoscaling, model registry metadata, or standardized action/latent/cost outputs.

vLLM has become a strong reference point for serving LLMs, but JEPA world models do not naturally fit token-by-token autoregressive text generation. The product must therefore identify which parts of the vLLM stack can be reused conceptually, which parts can be extended experimentally, and which serving engines are better suited for the first production-quality demo.

## 3. Goals

### Product goals

1. Provide the first production-grade API service for an action-conditioned JEPA world model.
2. Demonstrate end-to-end Push-T inference using a pretrained LeWorldModel checkpoint.
3. Validate and refine the WMCP RFC by implementing concrete request/response schemas and telemetry requirements.
4. Make world-model inference observable, benchmarkable, and deployable.
5. Establish a reusable architecture for future V-JEPA, V-JEPA 2 action-conditioned, and related latent world-model checkpoints.

### Engineering goals

1. Wrap LeWorldModel primitives (`encode`, `predict`, `rollout`, `score`, `plan`) behind a stable service contract.
2. Support GPU execution, dynamic batching, queueing/backpressure, and optional multi-replica deployment.
3. Emit OpenTelemetry traces, structured logs, and Prometheus metrics with per-operation latency and model-compute breakdowns.
4. Package model artifacts with manifest metadata, preprocessing configuration, shape constraints, and checksums.
5. Provide a Docker Compose demo and Kubernetes manifests suitable for local and cluster deployment.

## 4. Non-goals

1. Training or fine-tuning JEPA models.
2. Guaranteeing physical-robot safety or closed-loop real-world control safety.
3. Building a generic simulator service.
4. Replacing stable-worldmodel for environment management, data collection, or policy evaluation.
5. Committing to vLLM as the production runtime before a real plugin feasibility benchmark.
6. Supporting arbitrary user-uploaded model code in production.

## 5. Target users

| User | Needs |
|---|---|
| Robotics/world-model researcher | Deploy a checkpoint behind an API and inspect latent rollout behavior. |
| Infrastructure engineer | Operate the service with metrics, logs, traces, autoscaling, health checks, and reproducible builds. |
| WMCP RFC author | Validate protocol fields against a real world-model workload. |
| Demo/evaluation user | Run a Push-T planning request and obtain an action sequence, costs, and diagnostics. |
| Future model integrator | Add V-JEPA-like or other action-conditioned world models without redesigning the protocol. |

## 6. Primary use cases

### UC-1: Model metadata discovery

A client queries the service for model metadata and receives supported operations, tensor shapes, version/revision, preprocessing requirements, action semantics, and runtime capabilities.

### UC-2: Encode observation or goal

A client submits observation pixels and receives a latent embedding or embedding reference. This validates image preprocessing, batching, and latent output management.

### UC-3: Roll out candidate futures

A client submits observation history and candidate action sequences. The service returns predicted latent trajectories, optionally compressed, truncated, or stored as artifact references.

### UC-4: Score candidate action sequences

A client submits observation history, a goal observation, and candidate action sequences. The service returns a cost per candidate plus diagnostics.

### UC-5: Plan with MPC/CEM

A client submits observation history, a goal, horizon, action bounds, and planner parameters. The service performs iterative candidate generation/scoring and returns the best action sequence, first action, cost curve, and telemetry.

### UC-6: Observe and debug production behavior

An operator views request rate, p50/p95/p99 latency, queue wait, batch size, GPU utilization, GPU memory, model compute time, validation errors, and planner iteration timings.

## 7. MVP scope

The MVP is a single-model Push-T deployment that supports:

1. `GET /healthz` and `GET /readyz`.
2. `GET /wmcp/v1/models` and `GET /wmcp/v1/models/{model_id}`.
3. `POST /wmcp/v1/models/{model_id}:encode`.
4. `POST /wmcp/v1/models/{model_id}:rollout`.
5. `POST /wmcp/v1/models/{model_id}:score`.
6. `POST /wmcp/v1/models/{model_id}:plan`.
7. `/metrics` in Prometheus format.
8. OpenTelemetry trace export to an OTLP collector.
9. Docker Compose deployment with service, Prometheus, OpenTelemetry Collector, and Grafana.
10. Contract tests for schemas and example payloads.

## 8. Future scope

1. Triton backend for optimized `encode`/`rollout`/`score` once model graph boundaries are stable.
2. KServe custom runtime and/or ServingRuntime integration.
3. vLLM plugin feasibility spike for world-model pooling/action-rollout workloads.
4. Model registry integration and multiple checkpoint support.
5. Streaming rollout diagnostics for long-horizon or high-candidate planning.
6. On-device/edge inference profile.
7. V-JEPA 2 action-conditioned model support.
8. Zero-copy image/tensor transport over gRPC or shared memory.

## 9. Requirements

### Functional requirements

| ID | Requirement | Priority |
|---|---|---:|
| FR-001 | Load a pinned Push-T LeWorldModel checkpoint and expose model metadata. | P0 |
| FR-002 | Validate all request inputs against WMCP schemas before inference. | P0 |
| FR-003 | Support image observation history with explicit dtype, shape, color layout, and normalization metadata. | P0 |
| FR-004 | Support action candidate tensors with explicit horizon, candidate count, dtype, and action dimension. | P0 |
| FR-005 | Return structured outputs with costs, selected actions, optional latent embeddings, and diagnostics. | P0 |
| FR-006 | Expose health and readiness endpoints. | P0 |
| FR-007 | Emit Prometheus metrics and OTel traces. | P0 |
| FR-008 | Support request IDs and trace propagation. | P0 |
| FR-009 | Support dynamic batching for compatible `encode`, `rollout`, and `score` requests. | P1 |
| FR-010 | Provide local Docker Compose demo and Kubernetes manifests. | P1 |
| FR-011 | Support KServe V2-compatible inference endpoint as an adapter. | P2 |
| FR-012 | Support Triton backend adapter. | P2 |
| FR-013 | Support vLLM plugin experiment. | P2 |

### Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-001 | Availability | Single-replica demo must recover from worker restart; production target should support at least two replicas. |
| NFR-002 | Latency visibility | p50/p95/p99 latency must be tracked per endpoint and per model operation. |
| NFR-003 | Throughput visibility | Request rate, batch size, queue wait, planner iteration count, and candidate count must be tracked. |
| NFR-004 | GPU visibility | GPU utilization, memory used, memory reserved, OOM count, and model load time must be tracked. |
| NFR-005 | Reproducibility | Model manifest must pin source, revision, architecture config, preprocessing, and artifact checksums. |
| NFR-006 | Compatibility | API must remain model-family-neutral: Push-T is the first profile, not the protocol itself. |
| NFR-007 | Security | Service must not load arbitrary untrusted pickle/model code from user requests. |
| NFR-008 | Operability | Logs must be structured JSON and include request ID, trace ID, model ID, operation, status, and latency. |
| NFR-009 | Payload governance | Max request size, max candidate count, max horizon, max batch size, and timeout must be configurable. |

## 10. Initial SLOs and benchmark targets

These targets are intentionally provisional until measured on the actual hardware and checkpoint. They should be used as acceptance thresholds for the first internal demo, not as public claims.

| Operation | Initial target | Notes |
|---|---|---|
| Metadata/health | p95 < 50 ms | CPU-only, no GPU dependency except readiness. |
| Encode | p95 < 200 ms for small batches | Includes image decode/preprocess; should separate preprocess from GPU time. |
| Rollout | p95 < 500 ms for moderate candidate batches | Depends on horizon/candidate count. |
| Score | p95 < 700 ms for moderate candidate batches | Includes goal encoding and candidate costs. |
| Plan | p95 < 3 s for first demo profile | Depends heavily on CEM iterations/candidates. |

Every SLO must be reported with hardware, batch dimensions, candidate count, horizon, dtype, and backend.

## 11. Acceptance criteria

### Demo acceptance

1. A clean clone of the implementation repository can run the local demo with Docker Compose.
2. The service exposes health, readiness, metadata, score, rollout, and plan endpoints.
3. A Push-T request returns a valid action sequence and candidate costs.
4. Prometheus shows request counters and latency histograms.
5. OpenTelemetry traces show validation, preprocessing, queueing, model compute, and serialization spans.
6. Grafana dashboard displays request rate, latency, GPU memory, GPU utilization, queue wait, and planner diagnostics.
7. The service passes schema-contract tests using the example payloads in this archive.

### RFC validation acceptance

1. Every required WMCP field is represented in request/response examples.
2. The Push-T demo identifies at least five protocol open questions or refinements.
3. Model manifest includes enough information for another runtime to reproduce preprocessing and invocation semantics.
4. Observability RFC defines names, labels, and cardinality rules for standard metrics.

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Upstream LeWorldModel/stable-worldmodel APIs change | Integration churn | Pin commits; wrap upstream APIs behind a narrow runtime adapter. |
| Checkpoint uses unsafe PyTorch pickle loading | Supply-chain risk | Load only pinned trusted artifacts in build/init phase; convert to safer format where feasible; store checksums. |
| Action preprocessing/scaling is incomplete | Incorrect planning outputs | Treat action scaler/normalizer as required model package artifact; add golden tests against upstream eval. |
| vLLM adapter is expensive and not performant | Schedule risk | Keep vLLM as research spike, not MVP dependency. |
| Large inline image payloads hurt latency | Poor UX/SLO misses | Support tensor references/URIs and request-size limits; add binary/gRPC later. |
| Planning latency too high for demo | Demo risk | Start with score/rollout first; tune candidate count/horizon; batch across candidates. |

## 13. Product milestones

| Milestone | Deliverables |
|---|---|
| M0: Protocol and package | WMCP draft schemas, model manifest, example requests, mock service. |
| M1: Real checkpoint integration | Load Push-T checkpoint; implement encode/rollout/score; golden tests. |
| M2: Planner integration | Add CEM/MPC plan endpoint; diagnostics; reproducibility seeds. |
| M3: Observability | Prometheus, OTel traces, Grafana dashboard, structured logs. |
| M4: Deployable demo | Docker Compose and Kubernetes deployment; load-test report. |
| M5: Runtime optimization | Evaluate Ray Serve vs Triton; optional vLLM plugin spike. |
| M6: RFC refinement | Publish findings and protocol changes from implementation. |
