# WMCP-JEPA Serve: production inference service dossier

**Generated:** 2026-06-08  
**Project codename:** `wmcp-jepa-serve`  
**Primary demo target:** Push-T action-conditioned JEPA world model using `quentinll/lewm-pusht` / LeWorldModel.

This archive contains a complete project dossier for building a production-grade backend service that serves JEPA-style, action-conditioned world models through a WMCP-aligned API. It is written as an implementation-ready package: product requirements, technical specifications, API schemas, RFC drafts, deployment manifests, observability standards, and a small reference service skeleton.

The recommendation is to implement the first production demo with **FastAPI + Ray Serve + PyTorch** behind a WMCP API layer, package it for Docker Compose and Kubernetes, and keep **Triton Inference Server** as the phase-two optimization path. **vLLM should not be the first inference core** for Push-T/LeWM because its strongest architecture is for token-oriented LLM generation and pooling, while the world-model workload is a custom action-conditioned latent rollout and cost/plan computation. vLLM remains useful as a design reference and as a research spike for a plugin/pooling-style adapter.

## One-command demo

Bring up the whole demo — **client + backend + monitoring** — with one command:

```bash
make demo            # docker compose up: client + backend + prometheus + otel + grafana
# or, on an NVIDIA GPU host:
make demo-gpu
make demo-down       # stop + clean volumes
```

What happens: the backend starts, becomes healthy, then the one-shot **client** runs a
`metadata → score → plan` cycle against it and writes a result page to `.artifacts/demo/demo.html`.

| Service | URL | Notes |
|---|---|---|
| WMCP API | http://localhost:8080 | `/readyz`, `/metrics`, `/wmcp/v1/models/lewm-pusht:{score,plan,...}` |
| Prometheus | http://localhost:9090 | scrapes the service `/metrics` |
| Grafana | http://localhost:3000 | `admin` / `admin`; dashboards auto-provisioned |

By default the backend runs the **mock** runtime (no weights needed). To serve the real Push-T
checkpoint on CPU or GPU, build the model package and switch the backend:

```bash
make package                                  # build a model package (see models/lewm-pusht/README.md)
cp .env.example .env                          # then set HF_TOKEN + WMCP_BACKEND=lewm
WMCP_BACKEND=lewm make demo
```

## Contents

```text
.
├── PRD.md
├── TECHNICAL_SPEC.md
├── SOURCES.md
├── research/
│   └── engine-evaluation.md
├── adr/
│   └── ADR-0001-inference-engine.md
├── rfc/
│   ├── 0001-wmcp-world-model-inference.md
│   ├── 0002-action-conditioned-rollout-api.md
│   ├── 0003-observability-telemetry.md
│   ├── 0004-model-packaging-runtime.md
│   └── 0005-pusht-demo-profile.md
├── api/
│   └── openapi.yaml
├── schemas/
│   ├── wmcp-message.schema.json
│   └── model-manifest.schema.json
├── examples/
│   ├── rollout_request.json
│   ├── score_request.json
│   ├── plan_request.json
│   └── metadata_response.json
├── deployment/
│   ├── docker-compose.yaml
│   ├── prometheus.yaml
│   ├── otel-collector.yaml
│   ├── grafana-dashboard.json
│   └── k8s/
│       ├── deployment.yaml
│       └── kserve-inferenceservice.yaml
├── benchmarks/
│   └── load-test-plan.md
├── src/wmcp_jepa_service/
│   ├── __init__.py
│   ├── server.py
│   ├── runtime.py
│   ├── schemas.py
│   └── observability.py
├── tests/
│   └── test_contracts.py
├── Dockerfile
├── pyproject.toml
└── Makefile
```

## Recommended first milestone

Build a deployable Push-T demo with the following operations:

| Operation | Purpose | Required for MVP |
|---|---|---:|
| `metadata` | Model card, manifest, supported tasks, shapes, runtime capabilities | Yes |
| `encode` | Convert observation/goal pixels to latent embeddings | Yes |
| `rollout` | Predict latent trajectories for candidate action sequences | Yes |
| `score` | Return goal-conditioned costs for candidate action sequences | Yes |
| `plan` | Run CEM/MPC over the score function and return a selected action plan | Yes |

The LeWorldModel implementation already exposes `encode`, `predict`, `rollout`, and goal-conditioned `get_cost`-style functionality, so the first service should wrap those primitives rather than attempting to force the model into a text-generation inference engine.

## What is explicitly not included

This archive does **not** include downloaded model weights or a verified running LeWM container. The service skeleton is intentionally lightweight and uses a mock runtime by default. Production implementation should pin model artifact revisions, run a real checkpoint-conversion step, and verify shape/action normalization against the upstream LeWorldModel and stable-worldmodel evaluation code.

## Fast path to implementation

1. Convert or load the HF checkpoint into the LeWorldModel runtime using the upstream loader path.
2. Replace the mock runtime in `src/wmcp_jepa_service/runtime.py` with a `LeWMRuntime` implementation.
3. Run schema-contract tests and golden inference tests.
4. Deploy locally with `deployment/docker-compose.yaml`.
5. Promote to Kubernetes via `deployment/k8s/deployment.yaml`; optionally wrap with KServe once runtime behavior is stable.
6. Benchmark with the matrix in `benchmarks/load-test-plan.md`.

## Source limitations and assumptions

The current WMCP RFC text was not provided, so the WMCP interface in this archive is a proposed working draft. It is designed to be easy to refactor into your canonical WMCP fields while still validating the important standardization questions: typed observations/actions, latent outputs, cost/plan semantics, telemetry, model packaging, and conformance profiles.
