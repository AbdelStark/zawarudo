---
name: observability
description: The metrics, traces, and dashboards this service emits — Prometheus metric objects in observability.py, the /metrics endpoint, OTLP tracing config, and the Grafana dashboard. Activate when adding a metric, instrumenting a code path, debugging latency, or wiring telemetry. Metric names and labels are a consumed contract — extend, don't rename.
prerequisites: prometheus-client; opentelemetry-sdk/otlp
---

# Observability

<purpose>
Make every operation measurable: request counts, end-to-end + model-compute latency, and workload shape
(candidate count, rollout horizon). Dashboards and alerts depend on these exact names/labels.
</purpose>

<context>
- Metrics live in `observability.py` (all `prometheus_client` objects):
  `wmcp_requests_total{model,operation,status}`, `wmcp_request_latency_seconds{model,operation,status}`,
  `wmcp_model_compute_seconds{model,operation,backend}`, `wmcp_queue_wait_seconds{model,operation}`,
  `wmcp_candidate_count{model,operation}`, `wmcp_rollout_horizon{model,operation}`,
  `wmcp_model_loaded{model,revision,backend}`, `wmcp_input_validation_errors_total{operation,code}`.
- `_handle()` in `server.py` owns REQUESTS + REQUEST_LATENCY + VALIDATION_ERRORS, and records
  candidate/horizon from `action_candidates` shape `[B,S,T,A]` via `_record_request_shape`.
- Backends own MODEL_COMPUTE (wrap the actual compute) and MODEL_LOADED (set at construction).
- `observe_latency(histogram, *labels)` is a contextmanager helper for ad-hoc timing.
- Exposed at `GET /metrics`. OTLP endpoint from `WMCP_OTEL_EXPORTER_OTLP_ENDPOINT`.
- Ops stack (`deployment/`): prometheus.yaml scrape config, otel-collector.yaml, grafana-dashboard.json.
</context>

<procedure>
1. New metric → add the object to `observability.py` with explicit label names.
2. Increment/observe at the right layer: request-level in `_handle`, compute-level in the backend.
3. Histograms with domain ranges (counts, horizons) need explicit `buckets=` (see CANDIDATE_COUNT).
4. If consumed by dashboards/alerts, update `deployment/grafana-dashboard.json` + any prometheus rules.
5. Verify locally: `curl localhost:8080/metrics | grep wmcp_`.
</procedure>

<patterns>
<do>
— Keep label cardinality bounded (`model`, `operation`, `status`, `backend`) — no unbounded ids.
— Wrap real compute in MODEL_COMPUTE so model time is separable from request overhead.
</do>
<dont>
— Don't rename or relabel existing metrics (breaks dashboards/alerts) — add new ones.
— Don't put high-cardinality values (request_id, uri) in labels.
</dont>
</patterns>

<examples>
```python
from .observability import MODEL_COMPUTE, observe_latency
with observe_latency(MODEL_COMPUTE, model_id, "score", backend):
    costs = compute_costs(...)
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| metric missing on /metrics | object defined but never observed | add an `.inc()/.observe()` call path |
| candidate/horizon always empty | action tensor shape ≠ 4-dim `[B,S,T,A]` | ensure request uses 4-d action_candidates |
</troubleshooting>

<references>
— src/wmcp_jepa_service/observability.py · src/wmcp_jepa_service/server.py (`_handle`, `_record_request_shape`)
— deployment/{prometheus.yaml,otel-collector.yaml,grafana-dashboard.json} · rfc/0003-observability-telemetry.md
</references>
