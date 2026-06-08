from __future__ import annotations

import os
import time
from typing import Callable, Awaitable

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from .observability import CANDIDATE_COUNT, REQUEST_LATENCY, REQUESTS, ROLLOUT_HORIZON, VALIDATION_ERRORS
from .runtime import MockWorldModelBackend, WorldModelBackend
from .schemas import RequestEnvelope

MODEL_ID = os.getenv("WMCP_MODEL_ID", "lewm-pusht")
BACKEND_NAME = os.getenv("WMCP_BACKEND", "mock")

backend: WorldModelBackend = MockWorldModelBackend(model_id=MODEL_ID, revision="mock", backend=BACKEND_NAME)
app = FastAPI(title="WMCP-JEPA Serve", version="0.1.0")


def _record_request_shape(req: RequestEnvelope) -> None:
    action = req.inputs.get("action_candidates") if isinstance(req.inputs, dict) else None
    if not isinstance(action, dict):
        return
    tensor = action.get("tensor", {})
    shape = tensor.get("shape", [])
    if len(shape) == 4:
        _b, s, t, _a = shape
        CANDIDATE_COUNT.labels(req.model, req.operation).observe(float(s))
        ROLLOUT_HORIZON.labels(req.model, req.operation).observe(float(t))


async def _handle(operation: str, model_id: str, body: dict, fn: Callable[[RequestEnvelope], Awaitable[object]]) -> object:
    start = time.perf_counter()
    status = "ok"
    try:
        req = RequestEnvelope.model_validate({**body, "operation": operation, "model": model_id})
        if req.model != MODEL_ID:
            raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": f"Unknown model {req.model}"})
        _record_request_shape(req)
        result = await fn(req)
        return result
    except ValidationError as exc:
        status = "invalid"
        VALIDATION_ERRORS.labels(operation, "VALIDATION_ERROR").inc()
        raise HTTPException(status_code=422, detail={"code": "INVALID_ARGUMENT", "message": str(exc)}) from exc
    except HTTPException:
        status = "error"
        raise
    except Exception as exc:  # noqa: BLE001
        status = "error"
        raise HTTPException(status_code=500, detail={"code": "INTERNAL", "message": str(exc)}) from exc
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(model_id, operation, status).inc()
        REQUEST_LATENCY.labels(model_id, operation, status).observe(elapsed)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    return {"status": "ready", "model": MODEL_ID, "backend": BACKEND_NAME}


@app.get("/metrics")
def metrics() -> Response:
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
    return await _handle("encode", model_id, await request.json(), backend.encode)


@app.post("/wmcp/v1/models/{model_id}:predict")
async def predict(model_id: str, request: Request) -> object:
    return await _handle("predict", model_id, await request.json(), backend.predict)


@app.post("/wmcp/v1/models/{model_id}:rollout")
async def rollout(model_id: str, request: Request) -> object:
    return await _handle("rollout", model_id, await request.json(), backend.rollout)


@app.post("/wmcp/v1/models/{model_id}:score")
async def score(model_id: str, request: Request) -> object:
    return await _handle("score", model_id, await request.json(), backend.score)


@app.post("/wmcp/v1/models/{model_id}:plan")
async def plan(model_id: str, request: Request) -> object:
    return await _handle("plan", model_id, await request.json(), backend.plan)


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
    return await _handle(operation, model_name, wmcp_body, dispatch[operation])
