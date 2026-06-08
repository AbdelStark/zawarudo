"""Benchmark profiles from the load-test plan (benchmarks/load-test-plan.md "Test profiles")."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    operation: str  # health | metadata | score | rollout | plan
    b: int
    s: int
    t: int
    h: int
    a: int
    concurrency: int
    default_requests: int
    purpose: str


PROFILE_LIST: tuple[Profile, ...] = (
    Profile("health", "health", 0, 0, 0, 0, 0, 16, 200, "Control-plane latency."),
    Profile("smoke-score", "score", 1, 4, 4, 3, 10, 1, 20, "Correctness."),
    Profile("score-small", "score", 1, 16, 8, 3, 10, 4, 100, "Local demo."),
    Profile("score-medium", "score", 1, 256, 16, 3, 10, 8, 200, "Main target."),
    Profile("score-large", "score", 1, 1024, 32, 3, 10, 4, 60, "Stress."),
    Profile("rollout-medium", "rollout", 1, 256, 16, 3, 10, 8, 120, "Latent output overhead."),
    Profile("plan-medium", "plan", 1, 256, 16, 3, 10, 1, 30, "CEM/MPC end-to-end."),
    Profile("plan-concurrent", "plan", 1, 128, 16, 3, 10, 4, 60, "Planner queueing."),
)

PROFILES: dict[str, Profile] = {p.name: p for p in PROFILE_LIST}
