from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

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


@contextmanager
def observe_latency(histogram: Histogram, *labels: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.labels(*labels).observe(time.perf_counter() - start)
