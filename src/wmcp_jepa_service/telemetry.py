"""OpenTelemetry tracing + structured JSON logging, configured from the telemetry env vars.

The WMCP API stays engine-agnostic; this module owns *how* telemetry is wired so ``server.py`` only
opens spans. Tracing always records into an SDK ``TracerProvider`` (so spans are testable) and, when
``WMCP_OTEL_EXPORTER_OTLP_ENDPOINT`` is set, additionally exports to an OTLP collector over gRPC.

RFC-0005 span vocabulary: ``wmcp.request``, ``wmcp.validate``, ``wmcp.preprocess``,
``wmcp.model.rollout``, ``wmcp.model.score``, ``wmcp.serialize``.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span

SERVICE_NAME = "wmcp-jepa-service"
_TRACER_NAME = "wmcp_jepa_service"

log = logging.getLogger("wmcp")

_PROVIDER: TracerProvider | None = None


# --- structured logging ------------------------------------------------------------------------


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per log record, enriched with the active trace/span ids when present."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            payload["trace_id"] = format(ctx.trace_id, "032x")
            payload["span_id"] = format(ctx.span_id, "016x")
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger and honor ``WMCP_LOG_LEVEL``."""
    resolved = getattr(logging, str(level).upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(resolved)
    log.setLevel(resolved)


# --- tracing -----------------------------------------------------------------------------------


def init_tracing(otlp_endpoint: str | None = None, *, service_name: str = SERVICE_NAME) -> trace.Tracer:
    """Initialise the SDK tracer provider (once) and, if an endpoint is given, an OTLP gRPC exporter."""
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = TracerProvider(resource=Resource.create({"service.name": service_name}))
        trace.set_tracer_provider(_PROVIDER)
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=otlp_endpoint.startswith("http://"))
            _PROVIDER.add_span_processor(BatchSpanProcessor(exporter))
            log.info("OTLP trace export enabled", extra={"extra_fields": {"endpoint": otlp_endpoint}})
        except Exception as exc:  # noqa: BLE001 - never let telemetry setup crash the service
            log.warning("failed to enable OTLP exporter: %s", exc)
    return trace.get_tracer(_TRACER_NAME)


def get_provider() -> TracerProvider | None:
    """Return the active SDK tracer provider (used by tests to attach in-memory exporters)."""
    return _PROVIDER


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


def extract_trace_context(
    trace_fields: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> Context | None:
    """Build an OTel context from W3C trace fields without putting request IDs in labels."""
    carrier: dict[str, str] = {}
    for source in (headers, trace_fields):
        if not source:
            continue
        for key in ("traceparent", "tracestate"):
            value = source.get(key)
            if isinstance(value, str) and value:
                carrier[key] = value
    return extract(carrier) if carrier else None


@contextmanager
def span(name: str, context: Context | None = None, /, **attributes: Any) -> Iterator[Span]:
    """Start ``name`` as the current span, setting non-None attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, context=context) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current
