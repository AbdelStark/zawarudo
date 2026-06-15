<identity>
wmcp-jepa-service (`wmcp-jepa-serve`): a production inference backend that serves JEPA-style,
action-conditioned world models through a WMCP-aligned HTTP API. First target: a Push-T
LeWorldModel checkpoint. Operations: encode, predict, rollout, score, plan.
Status: v0.1.0 — API + observability + deployment complete, plus a real `LeWMRuntime` (the `lewm`
backend) serving the Push-T checkpoint on CPU/GPU. A deterministic `MockWorldModelBackend` (the default
`mock` backend) is kept for contract tests and weightless demos; `WMCP_BACKEND` selects between them.
</identity>

<stack>
| Layer        | Technology              | Version    | Notes                                              |
|--------------|-------------------------|------------|----------------------------------------------------|
| Language     | Python                  | >=3.10     | Docker pins 3.10-slim                              |
| Web API      | FastAPI + uvicorn       | 0.111+ / 0.30+ | ASGI; entry `wmcp_jepa_service.server:app`     |
| Validation   | Pydantic                | v2 (2.7+)  | All API contracts live in `schemas.py`            |
| Metrics      | prometheus-client       | 0.20+      | `/metrics` endpoint                                |
| Tracing      | opentelemetry-* (sdk/otlp) | 1.25+   | OTLP exporter                                      |
| Runtime (opt)| ray[serve]              | 2.30+      | extra `ray`; phase-2 serving core (ADR-0001)      |
| Model (real) | torch, transformers, einops, safetensors | 2.3+ | extra `lewm`; the real Push-T backend (vendored model) |
| Package mgr  | uv                      | (uv.lock)  | `uv.lock` is the source of truth                  |
| Build        | setuptools (src-layout) | 69+        | package discovered under `src/`                   |
| Tests        | pytest + httpx          | 8.2+       | extra `dev`                                        |
| Lint/Types   | ruff (line-length 120), mypy | 0.5+ / 1.10+ | extra `dev`                                  |
</stack>

<structure>
```
src/wmcp_jepa_service/        # the only Python package [agent: create/modify with care]
├── server.py                 #   FastAPI app, routes, _handle() dispatch, KServe v2 adapter
├── runtime.py                #   WorldModelBackend Protocol + MockWorldModelBackend (default `mock`)
├── runtime_lewm.py           #   LeWMRuntime — the real `lewm` backend (decode → inference → CEM plan)
├── lewm_model.py             #   vendored LeWorldModel (ViT encoder + predictor); torch/transformers/einops
├── model_package.py          #   trusted-package loader: checksums, manifest, shapes, safe weight load
├── packaging.py              #   author side — build synthetic/real model packages + checksums
├── request_shape.py          #   low-cardinality workload dims (batch/candidates/horizon) for metrics
├── runtime_logging.py        #   structured backend-operation logging context managers
├── telemetry.py              #   OTel tracing setup + JSON log formatter + span() helper
├── schemas.py                #   Pydantic API contract (envelopes, TensorRef) [GATED — see boundaries]
├── observability.py          #   Prometheus metric objects + observe_latency()
└── __init__.py               #   __version__
tests/test_contracts.py       # schema-contract tests [agent: create/modify]
api/openapi.yaml              # WMCP HTTP contract [GATED]
schemas/*.schema.json         # WMCP message + model-manifest JSON Schemas [GATED]
examples/*.json               # request/response fixtures [agent: create/modify]
deployment/                   # docker-compose, prometheus, otel, grafana, k8s/ [GATED]
adr/  rfc/  research/          # governance + decision records [GATED — append-only via PR]
benchmarks/  PRD.md  TECHNICAL_SPEC.md  SOURCES.md  LEWM_INTEGRATION_GUIDE.md  # docs [agent: modify]
```
</structure>

<commands>
| Task          | Command                                            | Notes                                  |
|---------------|----------------------------------------------------|----------------------------------------|
| Install       | `uv sync --extra dev` (or `make install`)          | Makefile uses `pip install -e .[dev]`  |
| Run (dev)     | `make run`                                          | uvicorn :8080, `--reload`              |
| Test          | `make test` / `pytest -q`                           | contract tests, runs in <1s            |
| Lint          | `ruff check .`                                      | line-length 120                        |
| Format        | `ruff format .`                                     |                                        |
| Type check    | `mypy src`                                           | zero errors expected                   |
| Docker build  | `make docker-build`                                 | tag `wmcp-jepa-service:local`          |
| Compose up    | `make compose-up`                                   | service+prometheus+otel+grafana        |
| Compose down  | `make compose-down`                                 |                                        |
| Smoke check   | `curl localhost:8080/readyz` ; `curl localhost:8080/wmcp/v1/models` |                |
</commands>

<api_surface>
Base path `/wmcp/v1`. Model id is set by env `WMCP_MODEL_ID` (default `lewm-pusht`); requests to any
other id return 404 `MODEL_NOT_FOUND`.

| Method/Path                                | Operation | Backend method   |
|--------------------------------------------|-----------|------------------|
| GET  `/healthz`, `/readyz`, `/metrics`     | —         | liveness/ready/prom |
| GET  `/wmcp/v1/models[/{id}]`              | metadata  | `backend.metadata()` |
| POST `/wmcp/v1/models/{id}:encode`        | encode    | `backend.encode` |
| POST `/wmcp/v1/models/{id}:predict`       | predict   | `backend.predict`|
| POST `/wmcp/v1/models/{id}:rollout`       | rollout   | `backend.rollout`|
| POST `/wmcp/v1/models/{id}:score`         | score     | `backend.score`  |
| POST `/wmcp/v1/models/{id}:plan`          | plan      | `backend.plan`   |
| POST `/v2/models/{name}/infer`            | (mapped)  | KServe V2 adapter → WMCP |

Every request is a `RequestEnvelope`; every success is a `ResponseEnvelope`; errors return
`{detail: {code, message}}` with codes `INVALID_ARGUMENT`(422), `MODEL_NOT_FOUND`(404),
`UNSUPPORTED_OPERATION`(400), `INTERNAL`(500). See skill `wmcp-api`.
</api_surface>

<config>
Env vars (read in `server.py`; defaults shown):
`WMCP_MODEL_ID=lewm-pusht` · `WMCP_BACKEND=mock` (`mock` | `lewm`) · `WMCP_OTEL_EXPORTER_OTLP_ENDPOINT` ·
`WMCP_ENABLE_PROMETHEUS=true` · `WMCP_LOG_LEVEL=INFO`. The `lewm` backend also reads
`WMCP_MODEL_PACKAGE=/models/lewm-pusht`, `WMCP_HF_DEVICE=cpu` (`cpu` | `cuda`), and
`WMCP_LATENT_STORE=.artifacts/latents`. `server.py:_make_backend()` selects the backend from
`WMCP_BACKEND` (the `lewm` path lazy-imports torch/transformers).
</config>

<conventions>
<code_style>
`from __future__ import annotations` at top of every module. Type-annotate all signatures.
Async route handlers; backend methods are `async def` returning `ResponseEnvelope`.
Pydantic v2 (`model_validate`, `model_dump`, `model_validator(mode="after")`). ruff, 120 cols.
Tensors are never raw arrays on the wire — always a `TensorRef` (`encoding` ∈ inline|base64|uri).
</code_style>
<patterns>
<do>
— Add new operations by: (1) extend `WorldModelBackend` Protocol, (2) implement in backend,
  (3) add route delegating to `_handle(op, model_id, request, backend.<op>)`, (4) add a contract test.
— Route all request flow through `_handle()` so metrics/latency/error-mapping stay uniform.
— Emit `MODEL_COMPUTE` around the actual compute; let `_handle` own `REQUESTS`/`REQUEST_LATENCY`.
— Keep runtime behind the `WorldModelBackend` Protocol so the API stays runtime-agnostic (ADR-0001).
— Return latents/large tensors by `uri`, not `inline`, to keep payloads small.
</do>
<dont>
— Don't put model-execution logic in `server.py` — it belongs in a backend module (`runtime.py` mock, `runtime_lewm.py` real).
— Don't change envelope shapes in `schemas.py` without updating `api/openapi.yaml`,
  `schemas/*.schema.json`, and the contract tests (they are one contract — GATED).
— Don't accept model weights or model code from inference request bodies (security; PRD non-goal #6).
— Don't `pip install` ad-hoc — declare deps in `pyproject.toml` extras and resync.
</dont>
</patterns>
<commit_conventions>Conventional Commits: `feat(scope): …`, `fix(scope): …` (see `git log`).</commit_conventions>
</conventions>

<workflows>
<real_backend>
The real `LeWMRuntime` (`lewm` backend) lives in `runtime_lewm.py` and serves the Push-T checkpoint.
Follow this same procedure (skill `runtime-backend`) to update the checkpoint or add a new backend:
1. Pin upstream commits (le-wm, stable-worldmodel) per `LEWM_INTEGRATION_GUIDE.md` / `scripts/pin_sources.py`.
2. Build a trusted model package (manifest, config, weights, preprocessing, action scaler, checksums)
   with `scripts/build_model_package.py` (`packaging.py`).
3. Implement the backend behind `WorldModelBackend`; load+freeze model, `torch.inference_mode()`.
4. Golden-validate (`tests/test_golden_lewm.py`): assert cost shape + numerical tolerance vs upstream.
5. Select it via `WMCP_BACKEND` in `server.py:_make_backend()`; keep Mock as the default contract-test backend.
</real_backend>
<add_operation_or_field>
1. Edit `schemas.py` (GATED — get approval). 2. Mirror in `api/openapi.yaml` + `schemas/*.schema.json`.
3. Implement in backend + route. 4. Add/extend `tests/test_contracts.py`. 5. `ruff check . && mypy src && pytest -q`.
</add_operation_or_field>
</workflows>

<boundaries>
<forbidden>
DO NOT modify, read, or commit: `.env`, `.env.*`, secrets/keys. (`.claude/`, `.agents/harness/`,
`.venv/`, `.artifacts/`, `.runs/`, `data/raw/` are gitignored — don't commit them.)
NEVER load model weights or model code supplied in an inference request payload.
</forbidden>
<gated>
Modify ONLY with explicit human approval (these are public contracts / infra / governance):
— `src/wmcp_jepa_service/schemas.py`, `api/openapi.yaml`, `schemas/*.schema.json` (the WMCP contract)
— `deployment/**` (compose, prometheus, otel, grafana, k8s, kserve)
— `pyproject.toml` dependency changes
— `adr/**`, `rfc/**` (append new records via PR; don't rewrite accepted ones)
</gated>
<safety_checks>
Before any destructive op (delete, overwrite, schema change, dependency bump): state what you'll do,
state what could break (which consumers/tests), and wait for confirmation.
</safety_checks>
</boundaries>

<troubleshooting>
| Symptom                                   | Cause                                  | Fix                                          |
|-------------------------------------------|----------------------------------------|----------------------------------------------|
| 404 `MODEL_NOT_FOUND` on a valid request  | request `model` ≠ `WMCP_MODEL_ID`      | use `lewm-pusht` or set `WMCP_MODEL_ID`      |
| 422 `INVALID_ARGUMENT`                     | envelope failed Pydantic validation    | check `TensorRef.encoding` ↔ payload field   |
| `tensor requires data/data_b64/uri`        | encoding/payload mismatch in `TensorRef`| inline→`data`, base64→`data_b64`, uri→`uri`  |
| ImportError torch/transformers             | the `lewm` extra isn't installed       | `uv sync --extra lewm` (real-backend deps)   |
| metrics absent                             | scraping wrong port                    | `/metrics` on :8080                          |
<recovery>
1. Read the full error (most contain the fix). 2. Confirm the file/command exists.
3. `uv sync --extra dev` (dependency drift is the #1 cause). 4. `git status` for stray edits.
5. Still stuck → state the problem and ask.
</recovery>
</troubleshooting>

<environment>
Harness: Claude Code (and Codex/agents via symlinked skills). Shell + git available. The service is a
mock by default — running it needs no GPU or weights. The real `lewm` backend needs the `lewm` extra + a built model package.
</environment>

<skills>
Canonical skills in `.codex/skills/` (symlinked at `.claude/skills/`, `.agents/skills/`). Load by reading
the file when entering its domain. Registry: `.codex/skills/_index.md`.
— `wmcp-api`: WMCP envelopes, operations, error codes, TensorRef, KServe adapter.
— `runtime-backend`: implement/extend a `WorldModelBackend` (the real `LeWMRuntime` is in `runtime_lewm.py`).
— `testing`: contract tests + golden numerical validation against upstream.
— `observability`: Prometheus metrics, OTel traces, Grafana, metric conventions.
— `deployment`: Docker Compose, Kubernetes, KServe InferenceService.
— `model-packaging`: trusted model package format + safe-loading policy.
</skills>

<memory>
<decisions>
2026-06-08 Ray Serve + FastAPI + PyTorch as MVP engine — runs arbitrary action-conditioned PyTorch
  with dynamic batching — rejected vLLM (token-gen oriented), Triton (needs stable graph; phase-2),
  KServe (control plane, not engine), TorchServe (archival). See adr/ADR-0001.
2026-06-08 Runtime hidden behind `WorldModelBackend` Protocol — keeps WMCP API independent of engine.
2026-06-08 WMCP schema is a working draft — upstream WMCP RFC text not yet available; designed to
  refactor into canonical fields (see README "Source limitations").
2026-06-08 Real `LeWMRuntime` shipped as the `lewm` backend — vendored LeWorldModel (no Hydra/env stack),
  checksum-verified model package, `torch.inference_mode()`, in-process CEM planner. Mock stays the
  default contract-test backend; Ray Serve dynamic batching remains phase-2.
</decisions>
<lessons>
— TensorRef on the wire (never raw arrays) keeps payloads typed and lets large latents move by uri.
— `_handle()` centralizes metrics + error mapping; every operation must go through it.
</lessons>
</memory>
