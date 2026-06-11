from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .request_shape import workload_shape_from_request
from .schemas import RequestEnvelope

log = logging.getLogger("wmcp.backend")


def log_backend_ready(*, backend: str, model_id: str, revision: str, **fields: Any) -> None:
    log.info(
        "backend ready",
        extra={"extra_fields": {"backend": backend, "model": model_id, "model_revision": revision, **fields}},
    )


def log_backend_loading(*, backend: str, package_path: str, device: str) -> None:
    log.info(
        "backend loading",
        extra={"extra_fields": {"backend": backend, "package": package_path, "device": device}},
    )


def log_mock_backend_warning(*, backend: str, model_id: str, revision: str) -> None:
    log.warning(
        "mock backend active; requests return synthetic outputs, not real world-model inference",
        extra={"extra_fields": {"backend": backend, "model": model_id, "model_revision": revision}},
    )


@contextmanager
def backend_operation(
    request: RequestEnvelope,
    *,
    backend: str,
    model_id: str,
    revision: str,
    **fields: Any,
) -> Iterator[dict[str, Any]]:
    shape = workload_shape_from_request(request)
    log_fields: dict[str, Any] = {
        "backend": backend,
        "model": model_id,
        "model_revision": revision,
        "operation": request.operation,
        "request_id": request.request_id,
        **shape.span_attributes(),
        **fields,
    }
    start = time.perf_counter()
    log.info("backend operation started", extra={"extra_fields": log_fields})
    try:
        yield log_fields
    except Exception as exc:
        log_fields["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
        log_fields["error_type"] = type(exc).__name__
        log.exception("backend operation failed", extra={"extra_fields": log_fields})
        raise
    else:
        log_fields["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
        log.info("backend operation completed", extra={"extra_fields": log_fields})
