from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from opentelemetry.trace import Status, StatusCode
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from . import telemetry
from .observability import (
    BATCH_SIZE,
    CANDIDATE_COUNT,
    QUEUE_WAIT,
    REQUEST_ERRORS,
    REQUEST_IN_FLIGHT,
    REQUEST_LATENCY,
    REQUESTS,
    ROLLOUT_HORIZON,
    SERIALIZE_LATENCY,
    SERVICE_READY,
    VALIDATION_LATENCY,
    VALIDATION_ERRORS,
    observe_latency,
    update_gpu_metrics,
)
from .request_shape import WorkloadShape, workload_shape_from_request
from .runtime import MockWorldModelBackend, WorldModelBackend
from .schemas import RequestEnvelope

MODEL_ID = os.getenv("WMCP_MODEL_ID", "lewm-pusht")
BACKEND_NAME = os.getenv("WMCP_BACKEND", "mock")
LOG_LEVEL = os.getenv("WMCP_LOG_LEVEL", "INFO")
OTEL_ENDPOINT = os.getenv("WMCP_OTEL_EXPORTER_OTLP_ENDPOINT")

telemetry.configure_logging(LOG_LEVEL)
telemetry.init_tracing(OTEL_ENDPOINT)
log = logging.getLogger("wmcp")


def _make_backend() -> WorldModelBackend:
    """Select the backend from WMCP_BACKEND. `mock` (default) keeps the contract-test path; `lewm`
    loads the real Push-T runtime from a trusted package (torch imported lazily only when selected)."""
    if BACKEND_NAME == "lewm":
        from .runtime_lewm import LeWMRuntime  # lazy: torch/transformers only for the real backend

        package = os.getenv("WMCP_MODEL_PACKAGE", "/models/lewm-pusht")
        device = os.getenv("WMCP_HF_DEVICE", "cpu")
        log.info("loading lewm backend", extra={"extra_fields": {"package": package, "device": device}})
        return LeWMRuntime(package, device=device)
    return MockWorldModelBackend(model_id=MODEL_ID, revision="mock", backend=BACKEND_NAME)


backend: WorldModelBackend = _make_backend()
app = FastAPI(title="WMCP-JEPA Serve", version="0.1.0")
SERVICE_READY.labels(MODEL_ID, BACKEND_NAME).set(1)


def _prometheus_enabled() -> bool:
    return os.getenv("WMCP_ENABLE_PROMETHEUS", "true").strip().lower() in ("1", "true", "yes", "on")


def _record_workload_shape(req: RequestEnvelope) -> WorkloadShape:
    shape = workload_shape_from_request(req)
    if shape.batch is not None:
        BATCH_SIZE.labels(req.model, req.operation).observe(float(shape.batch))
    if shape.candidates is not None:
        CANDIDATE_COUNT.labels(req.model, req.operation).observe(float(shape.candidates))
    if shape.horizon is not None:
        ROLLOUT_HORIZON.labels(req.model, req.operation).observe(float(shape.horizon))
    return shape


def _body_trace(body: dict) -> dict:
    trace_fields = body.get("trace", {}) if isinstance(body, dict) else {}
    return trace_fields if isinstance(trace_fields, dict) else {}


def _http_error_code(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
        return detail["code"]
    return f"HTTP_{exc.status_code}"


def _internal_error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "out of memory" in message and ("cuda" in message or "gpu" in message):
        return "GPU_OOM"
    if isinstance(exc, TimeoutError):
        return "TIMEOUT"
    return "INTERNAL"


async def _handle(operation: str, model_id: str, request: Request, fn: Callable[[RequestEnvelope], Awaitable[object]]) -> object:
    body = await request.json()
    return await _handle_body(operation, model_id, body, request.headers, fn)


async def _handle_body(
    operation: str,
    model_id: str,
    body: dict,
    headers: Mapping[str, str] | None,
    fn: Callable[[RequestEnvelope], Awaitable[object]],
) -> object:
    start = time.perf_counter()
    status = "ok"
    error_code: str | None = None
    request_id = body.get("request_id") if isinstance(body, dict) else None
    trace_context = telemetry.extract_trace_context(_body_trace(body), headers)
    with REQUEST_IN_FLIGHT.labels(model_id, operation).track_inprogress():
        with telemetry.span(
            "wmcp.request",
            trace_context,
            **{
                "wmcp.operation": operation,
                "wmcp.model": model_id,
                "wmcp.version": body.get("wmcp_version") if isinstance(body, dict) else None,
                "wmcp.request_id": request_id,
                "model.id": model_id,
                "runtime.backend": BACKEND_NAME,
            },
        ) as request_span:
            req: RequestEnvelope | None = None
            workload_shape = WorkloadShape()
            try:
                with observe_latency(VALIDATION_LATENCY, operation):
                    with telemetry.span("wmcp.validate"):
                        req = RequestEnvelope.model_validate({**body, "operation": operation, "model": model_id})
                        request_id = req.request_id
                        if req.model != MODEL_ID:
                            raise HTTPException(
                                status_code=404,
                                detail={"code": "MODEL_NOT_FOUND", "message": f"Unknown model {req.model}"},
                            )
                        workload_shape = _record_workload_shape(req)
                for key, value in workload_shape.span_attributes().items():
                    request_span.set_attribute(key, value)
                # Time spent between request receipt and backend dispatch (validation overhead today;
                # dynamic-batching queue wait once Ray Serve lands).
                QUEUE_WAIT.labels(model_id, operation).observe(time.perf_counter() - start)
                result = await fn(req)
                with observe_latency(SERIALIZE_LATENCY, operation):
                    with telemetry.span("wmcp.serialize"):
                        if hasattr(result, "model_dump"):
                            result = result.model_dump()
                return result
            except ValidationError as exc:
                status = "invalid"
                error_code = "INVALID_ARGUMENT"
                VALIDATION_ERRORS.labels(operation, "VALIDATION_ERROR").inc()
                request_span.record_exception(exc)
                request_span.set_status(Status(StatusCode.ERROR, "validation error"))
                raise HTTPException(status_code=422, detail={"code": error_code, "message": str(exc)}) from exc
            except HTTPException as exc:
                status = "error"
                error_code = _http_error_code(exc)
                request_span.set_status(Status(StatusCode.ERROR, str(exc.detail)))
                raise
            except Exception as exc:  # noqa: BLE001
                status = "error"
                error_code = _internal_error_code(exc)
                request_span.record_exception(exc)
                request_span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise HTTPException(status_code=500, detail={"code": error_code, "message": str(exc)}) from exc
            finally:
                elapsed = time.perf_counter() - start
                request_span.set_attribute("wmcp.status", status)
                if error_code is not None:
                    request_span.set_attribute("error.code", error_code)
                    REQUEST_ERRORS.labels(model_id, operation, error_code).inc()
                REQUESTS.labels(model_id, operation, status).inc()
                REQUEST_LATENCY.labels(model_id, operation, status).observe(elapsed)
                log.info(
                    "wmcp request completed",
                    extra={
                        "extra_fields": {
                            "request_id": request_id,
                            "model": model_id,
                            "operation": operation,
                            "status": status,
                            "error_code": error_code,
                            "latency_ms": round(elapsed * 1000, 3),
                            "backend": BACKEND_NAME,
                            **workload_shape.span_attributes(),
                        }
                    },
                )


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    SERVICE_READY.labels(MODEL_ID, BACKEND_NAME).set(1)
    return {"status": "ready", "model": MODEL_ID, "backend": BACKEND_NAME}


@app.get("/metrics")
def metrics() -> Response:
    if not _prometheus_enabled():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "prometheus metrics disabled"})
    update_gpu_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/wmcp/v1/models")
def list_models() -> dict:
    return {"models": [backend.metadata().model_dump()]}


@app.get("/wmcp/v1/models/{model_id}")
def get_model(model_id: str) -> dict:
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND"})
    return backend.metadata().model_dump()


@app.post("/wmcp/v1/models/{model_id}:encode")
async def encode(model_id: str, request: Request) -> object:
    return await _handle("encode", model_id, request, backend.encode)


@app.post("/wmcp/v1/models/{model_id}:predict")
async def predict(model_id: str, request: Request) -> object:
    return await _handle("predict", model_id, request, backend.predict)


@app.post("/wmcp/v1/models/{model_id}:rollout")
async def rollout(model_id: str, request: Request) -> object:
    return await _handle("rollout", model_id, request, backend.rollout)


@app.post("/wmcp/v1/models/{model_id}:score")
async def score(model_id: str, request: Request) -> object:
    return await _handle("score", model_id, request, backend.score)


@app.post("/wmcp/v1/models/{model_id}:plan")
async def plan(model_id: str, request: Request) -> object:
    return await _handle("plan", model_id, request, backend.plan)


@app.post("/v2/models/{model_name}/infer")
async def kserve_v2_adapter(model_name: str, request: Request) -> object:
    body = await request.json()
    # Minimal adapter placeholder. Production implementation should map KServe V2 inputs
    # to a WMCP operation declared in body.parameters.operation.
    operation = body.get("parameters", {}).get("operation", "score")
    wmcp_body = {
        "wmcp_version": "0.1",
        "request_id": body.get("id", "kserve-request"),
        "operation": operation,
        "model": model_name,
        "inputs": body.get("inputs", {}),
        "parameters": body.get("parameters", {}),
    }
    dispatch = {"encode": backend.encode, "predict": backend.predict, "rollout": backend.rollout, "score": backend.score, "plan": backend.plan}
    if operation not in dispatch:
        raise HTTPException(status_code=400, detail={"code": "UNSUPPORTED_OPERATION"})
    return await _handle_body(operation, model_name, wmcp_body, request.headers, dispatch[operation])
