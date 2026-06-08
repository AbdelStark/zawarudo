"""Self-contained async load-test harness for the WMCP Push-T service (issue #8).

Stdlib only — concurrency via ``asyncio.to_thread`` over a blocking ``urllib`` poster, so it runs in
any Python 3.9+ environment without adding dependencies. Implements the RFC-0005 / load-test-plan
profiles, aggregates p50/p90/p95/p99 latency + throughput + error-rate, captures the required benchmark
metadata, and fills the plan's report template.
"""

from __future__ import annotations

from .profiles import PROFILES, Profile
from .report import render_report
from .runner import ProfileResult, run_profile

__all__ = ["PROFILES", "Profile", "ProfileResult", "run_profile", "render_report"]
