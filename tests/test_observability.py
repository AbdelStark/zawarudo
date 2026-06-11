"""Tests for telemetry wiring: metrics presence, env gates, spans, structured logging (issue #4)."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from wmcp_jepa_service import telemetry
from wmcp_jepa_service.server import app

SCORE_BODY = {
    "wmcp_version": "0.1",
    "request_id": "t-score",
    "inputs": {
        "action_candidates": {
            "space": "continuous",
            "tensor": {
                "kind": "tensor",
                "encoding": "uri",
                "dtype": "float32",
                "shape": [1, 16, 8, 10],
                "layout": "B,S,T,A",
                "uri": "memory://actions.npy",
            },
        }
    },
    "parameters": {"seed": 0},
}
PLAN_BODY = {
    "wmcp_version": "0.1",
    "request_id": "t-plan",
    "parameters": {"horizon": 8, "iterations": 5, "candidates": 32, "seed": 1},
}
# Missing the required `request_id` -> RequestEnvelope validation fails (422).
INVALID_BODY = {"wmcp_version": "0.1", "parameters": {}}

REQUIRED_METRICS = [
    "wmcp_requests_total",
    "wmcp_inflight_requests",
    "wmcp_request_errors_total",
    "wmcp_request_latency_seconds",
    "wmcp_validation_latency_seconds",
    "wmcp_serialize_latency_seconds",
    "wmcp_model_compute_seconds",
    "wmcp_queue_wait_seconds",
    "wmcp_batch_size",
    "wmcp_candidate_count",
    "wmcp_rollout_horizon",
    "wmcp_planner_iterations",
    "wmcp_input_validation_errors_total",
    "wmcp_model_loaded",
]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _span_capture() -> InMemorySpanExporter:
    provider = telemetry.get_provider()
    assert provider is not None, "tracer provider must be initialised by server import"
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    exporter.clear()
    return exporter


def test_metrics_exposes_all_required_names(client: TestClient) -> None:
    assert client.post("/wmcp/v1/models/lewm-pusht:score", json=SCORE_BODY).status_code == 200
    assert client.post("/wmcp/v1/models/lewm-pusht:plan", json=PLAN_BODY).status_code == 200
    assert client.post("/wmcp/v1/models/lewm-pusht:score", json=INVALID_BODY).status_code == 422  # populate errors

    body = client.get("/metrics").text
    missing = [name for name in REQUIRED_METRICS if name not in body]
    assert not missing, f"/metrics missing: {missing}"


def test_planner_iterations_metric_observed(client: TestClient) -> None:
    client.post("/wmcp/v1/models/lewm-pusht:plan", json=PLAN_BODY)
    body = client.get("/metrics").text
    assert "wmcp_planner_iterations_count" in body
    assert 'wmcp_planner_iterations_bucket{' in body
    assert 'wmcp_candidate_count_count{model="lewm-pusht",operation="plan"}' in body
    assert 'wmcp_batch_size_count{model="lewm-pusht",operation="plan"}' in body


def test_request_error_metric_uses_stable_error_codes(client: TestClient) -> None:
    assert client.post("/wmcp/v1/models/lewm-pusht:score", json=INVALID_BODY).status_code == 422
    body = client.get("/metrics").text
    assert 'wmcp_request_errors_total{code="INVALID_ARGUMENT",model="lewm-pusht",operation="score"}' in body


def test_metrics_gated_by_enable_prometheus(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WMCP_ENABLE_PROMETHEUS", "false")
    assert client.get("/metrics").status_code == 404
    monkeypatch.setenv("WMCP_ENABLE_PROMETHEUS", "true")
    assert client.get("/metrics").status_code == 200


def test_score_emits_rfc_spans(client: TestClient) -> None:
    exporter = _span_capture()
    client.post("/wmcp/v1/models/lewm-pusht:score", json=SCORE_BODY)
    names = {s.name for s in exporter.get_finished_spans()}
    assert {"wmcp.request", "wmcp.validate", "wmcp.preprocess", "wmcp.model.score", "wmcp.serialize"}.issubset(names)


def test_plan_emits_model_span(client: TestClient) -> None:
    exporter = _span_capture()
    client.post("/wmcp/v1/models/lewm-pusht:plan", json=PLAN_BODY)
    names = {s.name for s in exporter.get_finished_spans()}
    assert {"wmcp.request", "wmcp.validate", "wmcp.model.plan", "wmcp.serialize"}.issubset(names)


def test_request_span_records_error_status(client: TestClient) -> None:
    exporter = _span_capture()
    client.post("/wmcp/v1/models/lewm-pusht:score", json=INVALID_BODY)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["wmcp.request"].status.status_code.name == "ERROR"
    assert spans["wmcp.request"].attributes["error.code"] == "INVALID_ARGUMENT"


def test_request_span_joins_traceparent_and_records_workload_shape(client: TestClient) -> None:
    exporter = _span_capture()
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    parent_id = "00f067aa0ba902b7"
    body = {**SCORE_BODY, "trace": {"traceparent": f"00-{trace_id}-{parent_id}-01"}}
    client.post("/wmcp/v1/models/lewm-pusht:score", json=body)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    request_span = spans["wmcp.request"]
    assert f"{request_span.context.trace_id:032x}" == trace_id
    assert request_span.parent is not None
    assert f"{request_span.parent.span_id:016x}" == parent_id
    assert request_span.attributes["tensor.batch"] == 1
    assert request_span.attributes["tensor.candidates"] == 16
    assert request_span.attributes["tensor.horizon"] == 8
    assert request_span.attributes["tensor.action_dim"] == 10


def test_json_log_formatter_emits_valid_json() -> None:
    formatter = telemetry.JsonLogFormatter()
    record = logging.LogRecord("wmcp", logging.INFO, __file__, 1, "hello", None, None)
    record.extra_fields = {"request_id": "r-1", "operation": "score"}
    parsed = json.loads(formatter.format(record))
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "hello"
    assert parsed["logger"] == "wmcp"
    assert parsed["request_id"] == "r-1"
    assert parsed["operation"] == "score"


def test_configure_logging_honors_level() -> None:
    telemetry.configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING
    telemetry.configure_logging("INFO")  # restore for other tests
    assert logging.getLogger().level == logging.INFO


def test_otel_endpoint_enables_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyExporter(SpanExporter):
        def export(self, spans):  # type: ignore[no-untyped-def]
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    import opentelemetry.exporter.otlp.proto.grpc.trace_exporter as otlp_mod

    monkeypatch.setattr(otlp_mod, "OTLPSpanExporter", lambda **kwargs: _DummyExporter())
    tracer = telemetry.init_tracing("http://localhost:4317")
    assert tracer is not None
    with tracer.start_as_current_span("probe"):
        pass
