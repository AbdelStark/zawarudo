# Skill Registry — wmcp-jepa-service

Last updated: 2026-06-15

Canonical location `.codex/skills/`; symlinked at `.claude/skills/` and `.agents/skills/`.
Load a skill by reading its file when its triggers match the current work.

| Skill            | File                  | Triggers                                                        | Priority |
|------------------|-----------------------|----------------------------------------------------------------|----------|
| WMCP API         | wmcp-api.md           | envelope, operation, endpoint, TensorRef, route, error code, kserve | Core |
| Runtime Backend  | runtime-backend.md    | backend, runtime, LeWM, mock, encode/rollout/score/plan, torch, checkpoint | Core |
| Testing          | testing.md            | test, contract, pytest, golden, validation, fixture            | Core     |
| Observability    | observability.md      | metric, prometheus, otel, trace, grafana, latency, histogram   | Extend   |
| Deployment       | deployment.md         | docker, compose, k8s, kubernetes, kserve, deploy, manifest     | Extend   |
| Model Packaging  | model-packaging.md    | manifest, weights, package, safetensors, checksum, safe load   | Extend   |

## Recommended (not yet scaffolded)
- [ ] `planning-cem-mpc.md` — extract the `lewm` backend's in-process CEM planner (`runtime_lewm.py`) if planning grows.
- [ ] `batching-backpressure.md` — Ray Serve dynamic batching + queue/backpressure (phase-2).
- [ ] `benchmarking.md` — load-test matrix execution (see `benchmarks/load-test-plan.md`).
