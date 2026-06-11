# RFC-0003: Observability and telemetry for WMCP world-model services

| Field | Value |
|---|---|
| Status | Draft |
| Created | 2026-06-08 |
| Depends on | RFC-0001, RFC-0002 |

## Abstract

This RFC defines standard metrics, traces, and structured logs for WMCP world-model inference services. The goal is to make world-model inference as observable as production LLM serving while accounting for world-model-specific dimensions such as horizon, candidate count, rollout steps, and planner iterations.

## Requirements

A production-grade WMCP world-model server MUST expose:

1. Request counters by model, operation, and status.
2. Latency histograms by model, operation, and status.
3. Validation error counters by operation and error code.
4. Queue wait, preprocessing, model compute, and serialization timing.
5. Batching metrics where batching is used.
6. Planner diagnostics where planning is exposed.
7. Model load/readiness metrics.
8. Trace propagation using W3C `traceparent`.
9. Structured logs with request and trace correlation.

## Metrics

### Required counters

| Metric | Labels | Description |
|---|---|---|
| `wmcp_requests_total` | `model`, `operation`, `status` | Total requests. |
| `wmcp_input_validation_errors_total` | `operation`, `code` | Validation failures. |
| `wmcp_backend_errors_total` | `model`, `operation`, `code` | Backend/model errors. |
| `wmcp_gpu_oom_total` | `model`, `backend` | GPU out-of-memory events. |

### Required histograms

| Metric | Labels | Description |
|---|---|---|
| `wmcp_request_latency_seconds` | `model`, `operation`, `status` | End-to-end latency. |
| `wmcp_validation_latency_seconds` | `operation` | Schema and policy validation. |
| `wmcp_payload_decode_seconds` | `operation`, `encoding` | Payload decode time. |
| `wmcp_preprocess_latency_seconds` | `model`, `operation` | Image/tensor preprocessing. |
| `wmcp_queue_wait_seconds` | `model`, `operation` | Time queued for replica/batch. |
| `wmcp_model_compute_seconds` | `model`, `operation`, `backend` | Model execution time. |
| `wmcp_serialize_latency_seconds` | `operation` | Response serialization time. |
| `wmcp_batch_size` | `model`, `operation` | Number of requests in a batch. |
| `wmcp_candidate_count` | `model`, `operation` | Number of candidate actions. |
| `wmcp_rollout_horizon` | `model`, `operation` | Future horizon length. |
| `wmcp_planner_iterations` | `model`, `planner` | Planner iterations. |

### Required gauges

| Metric | Labels | Description |
|---|---|---|
| `wmcp_model_loaded` | `model`, `revision`, `backend` | 1 when loaded and usable. |
| `wmcp_model_load_seconds` | `model`, `revision`, `backend` | Last load duration. |
| `wmcp_replica_inflight_requests` | `model`, `replica` | Active requests. |

GPU metrics SHOULD be collected with platform-native exporters such as DCGM on NVIDIA deployments, or runtime-specific metrics when available.

## Metric label cardinality rules

Metric labels MUST NOT include:

- `request_id`
- `trace_id`
- raw user IDs
- raw URI paths
- exception messages
- arbitrary model revision strings if revisions are unbounded

Labels SHOULD be limited to controlled vocabularies. High-cardinality values belong in traces/logs, not Prometheus labels.

## Tracing

### Required root span

Every request MUST create or continue a `wmcp.request` span.

Required span attributes:

- `wmcp.version`
- `wmcp.operation`
- `wmcp.request_id`
- `model.id`
- `model.revision`
- `runtime.backend`
- `http.method`
- `http.route`

### Required child spans

| Span | Description |
|---|---|
| `wmcp.auth` | Authentication/authorization. |
| `wmcp.validate` | Request schema and limit validation. |
| `wmcp.payload.decode` | Base64/URI/tensor loading. |
| `wmcp.preprocess` | Image resize/normalization/action scaling. |
| `wmcp.scheduler.enqueue` | Queueing for replica/batch. |
| `wmcp.batch.form` | Batch assembly. |
| `wmcp.model.encode` | Model encode. |
| `wmcp.model.rollout` | Model rollout. |
| `wmcp.model.score` | Cost computation. |
| `wmcp.planner.iteration` | One planner iteration. |
| `wmcp.serialize` | Response encoding. |

### Optional span events

- `model.load.start`
- `model.load.complete`
- `batch.flush`
- `planner.elite_selected`
- `partial_result_returned`
- `gpu_oom`

## Structured logs

Logs MUST be structured JSON in production mode.

Required fields:

```json
{
  "timestamp": "2026-06-08T10:00:00.000Z",
  "level": "INFO",
  "message": "wmcp request complete",
  "request_id": "uuid",
  "trace_id": "trace-id",
  "span_id": "span-id",
  "model_id": "lewm-pusht",
  "model_revision": "hf:quentinll/lewm-pusht:<revision>",
  "operation": "score",
  "status_code": 200,
  "latency_ms": 42.1,
  "candidate_count": 256,
  "horizon": 16,
  "batch_size": 1,
  "error_code": null
}
```

## Service-level objectives

SLOs MUST be tied to benchmark profiles. For example:

```text
Operation: score
Model: lewm-pusht
Hardware: NVIDIA L4
Input: B=1, S=256, T=16, A=10, image=224x224, H=3
SLO: p95 end-to-end latency < 700 ms
```

SLOs without hardware and shape context are invalid.

## Dashboards

A production dashboard SHOULD include:

1. Request rate by operation.
2. Error rate by stable code from `wmcp_request_errors_total`.
3. p50/p95/p99 request latency by operation.
4. Queue wait and batch size.
5. Model compute time by operation.
6. Candidate count and horizon distributions.
7. Planner best-cost convergence.
8. GPU utilization and memory.
9. Model load/readiness status.
10. Top validation failures.

Services SHOULD expose in-flight request and readiness gauges (`wmcp_inflight_requests`,
`wmcp_service_ready`) so local dashboards can distinguish idle, saturated, and unavailable states
without relying on raw logs.

## Alert recommendations

| Alert | Condition |
|---|---|
| Service down | `/readyz` fails for > 2 minutes. |
| High error rate | 5xx rate > 2% for 5 minutes. |
| High validation failure rate | Validation failures spike above baseline. |
| Latency regression | p95 latency > SLO for 10 minutes. |
| GPU OOM | Any OOM in production. |
| Model not loaded | `wmcp_model_loaded == 0` for ready deployment. |
| Queue saturation | p95 queue wait > 50% of total latency. |

## Privacy and governance

Observability data MUST NOT include raw image payloads, action tensors, or full external object URIs. For debugging, servers MAY log redacted shape/dtype metadata and stable artifact hashes.

## Open questions

1. Should WMCP define mandatory histogram buckets?
2. Should WMCP standardize semantic-convention names under OpenTelemetry?
3. Should planner diagnostics be emitted as metrics, traces, logs, or response fields only?
4. How should distributed planning traces be represented across multiple workers?
