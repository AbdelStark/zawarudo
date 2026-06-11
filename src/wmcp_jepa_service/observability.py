from __future__ import annotations

import importlib
import time
from contextlib import contextmanager
from typing import Any, Iterator

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "wmcp_requests_total",
    "Total WMCP requests",
    ["model", "operation", "status"],
)
REQUEST_LATENCY = Histogram(
    "wmcp_request_latency_seconds",
    "End-to-end WMCP request latency",
    ["model", "operation", "status"],
)
REQUEST_IN_FLIGHT = Gauge(
    "wmcp_inflight_requests",
    "WMCP requests currently being handled",
    ["model", "operation"],
)
REQUEST_ERRORS = Counter(
    "wmcp_request_errors_total",
    "WMCP requests that ended with an error code",
    ["model", "operation", "code"],
)
VALIDATION_LATENCY = Histogram(
    "wmcp_validation_latency_seconds",
    "WMCP envelope validation latency",
    ["operation"],
)
SERIALIZE_LATENCY = Histogram(
    "wmcp_serialize_latency_seconds",
    "WMCP response serialization latency",
    ["operation"],
)
MODEL_COMPUTE = Histogram(
    "wmcp_model_compute_seconds",
    "Model compute latency",
    ["model", "operation", "backend"],
)
QUEUE_WAIT = Histogram(
    "wmcp_queue_wait_seconds",
    "Queue wait latency",
    ["model", "operation"],
)
CANDIDATE_COUNT = Histogram(
    "wmcp_candidate_count",
    "Candidate action count",
    ["model", "operation"],
    buckets=(1, 4, 16, 64, 128, 256, 512, 1024, 2048),
)
ROLLOUT_HORIZON = Histogram(
    "wmcp_rollout_horizon",
    "Rollout horizon",
    ["model", "operation"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)
BATCH_SIZE = Histogram(
    "wmcp_batch_size",
    "Observed request batch size",
    ["model", "operation"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)
PLANNER_ITERATIONS = Histogram(
    "wmcp_planner_iterations",
    "Planner iterations per plan request",
    ["model", "operation"],
    buckets=(1, 2, 3, 5, 8, 10, 15, 20, 30, 50),
)
MODEL_LOADED = Gauge(
    "wmcp_model_loaded",
    "Whether the model is loaded",
    ["model", "revision", "backend"],
)
VALIDATION_ERRORS = Counter(
    "wmcp_input_validation_errors_total",
    "Input validation errors",
    ["operation", "code"],
)
SERVICE_READY = Gauge(
    "wmcp_service_ready",
    "1 when the service has loaded a backend and is ready to serve",
    ["model", "backend"],
)

# Optional GPU metrics — populated best-effort when torch + CUDA are available (RFC-0005).
GPU_AVAILABLE = Gauge(
    "wmcp_gpu_available",
    "1 if a CUDA device is available",
    ["device"],
)
GPU_MEMORY_USED = Gauge(
    "wmcp_gpu_memory_used_bytes",
    "GPU memory currently allocated by torch",
    ["device"],
)


def update_gpu_metrics() -> None:
    """Refresh GPU gauges if torch+CUDA are present; a no-op otherwise (never raises)."""
    try:
        torch: Any = importlib.import_module("torch")  # optional dependency
        if not torch.cuda.is_available():
            return
        for index in range(torch.cuda.device_count()):
            device = f"cuda:{index}"
            GPU_AVAILABLE.labels(device).set(1)
            GPU_MEMORY_USED.labels(device).set(float(torch.cuda.memory_allocated(index)))
    except Exception:  # noqa: BLE001 - metrics must never break the request path
        return


@contextmanager
def observe_latency(histogram: Histogram, *labels: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.labels(*labels).observe(time.perf_counter() - start)
