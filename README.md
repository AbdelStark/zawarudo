# Za Warudo

**HTTP serving for action-conditioned JEPA world models.**

Serve a Push-T [LeWorldModel](https://huggingface.co/quentinll/lewm-pusht) checkpoint behind a typed,
WMCP-aligned API (`encode`, `rollout`, `score`, `plan`) with first-class Prometheus metrics and
OpenTelemetry traces. Swap the real PyTorch checkpoint for a zero-dependency mock with one env var.
CPU or GPU. One command to a running demo.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-425CC7?logo=opentelemetry&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)

---

## Quickstart

```bash
make demo-local                 # backend + client, end to end (mock, no weights)
make demo-local BACKEND=lewm    # the real Push-T checkpoint on CPU
make demo                       # full stack: dashboard + traffic + backend + Prometheus + OTel + Grafana
```

`make demo-local` boots the API, runs a `metadata → score → plan` cycle against it, and writes an
HTML view. No Docker required.

`make demo` and `make demo-lewm` expose the interactive dashboard at `http://localhost:8088`. It can
send direct WMCP requests, run batches, start a browser-side traffic mix, and read live Prometheus
queries through the dashboard proxy. The compose stack also starts a low-pressure `traffic-generator`
service so Grafana and the dashboard have useful request, latency, planner, validation-error, and
shape metrics without manual clicks.

## Install

```bash
uv sync --extra dev                 # API + tooling
uv sync --extra dev --extra lewm    # + the real LeWorldModel backend (torch, transformers)
make run                            # uvicorn on :8080
```

## API

Base path `/wmcp/v1`. Every request is a typed envelope; tensors travel as a `TensorRef`
(`inline` / `base64` / `uri`), never raw arrays.

| Method · Path | Operation | Output |
|---|---|---|
| `GET  /wmcp/v1/models/{id}` | metadata | model card, shapes, limits |
| `POST /wmcp/v1/models/{id}:encode` | encode | latents `[B,H,192]` |
| `POST /wmcp/v1/models/{id}:rollout` | rollout | predicted latents `[B,S,T,192]` |
| `POST /wmcp/v1/models/{id}:score` | score | goal-conditioned costs `[B,S]` + `best_index` |
| `POST /wmcp/v1/models/{id}:plan` | plan | CEM/MPC plan `[B,T,10]` + first action `[B,10]` |
| `GET  /healthz` · `/readyz` · `/metrics` | system | liveness · readiness · Prometheus |
| `POST /v2/models/{name}/infer` | KServe V2 | Open Inference Protocol adapter |

```bash
curl -s localhost:8080/wmcp/v1/models/lewm-pusht:score \
  -H 'content-type: application/json' -d @examples/score_request.json \
  | jq '.outputs | {best_index, shape: .costs.shape}'
# { "best_index": [42], "shape": [1, 256] }
```

Errors return `{detail: {code, message}}` with `INVALID_ARGUMENT` (422), `MODEL_NOT_FOUND` (404),
`UNSUPPORTED_OPERATION` (400), `INTERNAL` (500).

## Backends

| `WMCP_BACKEND` | Engine | Needs |
|---|---|---|
| `mock` *(default)* | deterministic stub with exact output shapes | nothing |
| `lewm` | real `quentinll/lewm-pusht` checkpoint, CPU or GPU | `--extra lewm` + a model package |

The runtime lives behind a `WorldModelBackend` protocol, so the API never changes when the engine
does. The model is **vendored** (no Hydra/gym/pygame research stack) and loads only from a trusted,
checksum-verified package, never from a request payload.

```bash
# build a safetensors package from the HF checkpoint, then serve it
python scripts/build_model_package.py real --source <hf-dir> --out .artifacts/model-package/lewm-pusht
WMCP_BACKEND=lewm WMCP_MODEL_PACKAGE=.artifacts/model-package/lewm-pusht make run
```

A package is a self-describing directory (`manifest.json`, `weights.safetensors`, `preprocessing.json`,
action scaler, `checksums.txt`); the loader verifies checksums and tensor shapes, freezes the model,
and runs under `torch.inference_mode()`.

## Observability

Every operation is measured and traced.

- **Metrics** (`/metrics`): `wmcp_requests_total`, `wmcp_request_latency_seconds`,
  `wmcp_model_compute_seconds`, `wmcp_queue_wait_seconds`, `wmcp_candidate_count`,
  `wmcp_rollout_horizon`, `wmcp_planner_iterations`, `wmcp_input_validation_errors_total`, GPU gauges.
- **Traces**: OTel spans `wmcp.request → validate → preprocess → model.{score,rollout,plan} → serialize`,
  exported over OTLP.

`make demo` provisions:

- Interactive frontend: `http://localhost:8088`
- Grafana: `http://localhost:3000` with `WMCP LeWM Operations` and `WMCP Traffic Stimulator`
- Prometheus: `http://localhost:9090`
- Tempo: `http://localhost:3200`

The frontend proxies `/api/*` to the WMCP service and `/prometheus/*` to Prometheus, so direct
requests and live metrics work from the same browser origin.

## Benchmark

```bash
make run &
python benchmarks/run_benchmark.py --profile score-medium    # or --all
```

Async load generator across the profiles (`smoke` → `score-{small,medium,large}` → `rollout`/`plan`),
reporting p50/p90/p95/p99, throughput, and error rate with full run context. Reports land in
[`benchmarks/reports/`](benchmarks/reports/).

## Configuration

| Env var | Default | |
|---|---|---|
| `WMCP_MODEL_ID` | `lewm-pusht` | served model id |
| `WMCP_BACKEND` | `mock` | `mock` \| `lewm` |
| `WMCP_MODEL_PACKAGE` | `/models/lewm-pusht` | package dir (lewm backend) |
| `WMCP_HF_DEVICE` | `cpu` | `cpu` \| `cuda` |
| `WMCP_OTEL_EXPORTER_OTLP_ENDPOINT` | unset | OTLP gRPC collector endpoint |
| `WMCP_ENABLE_PROMETHEUS` | `true` | gate `/metrics` |
| `WMCP_LOG_LEVEL` | `INFO` | structured JSON log level |

## Deploy

```bash
make demo            # docker compose: dashboard + traffic + client + backend + monitoring
make demo-lewm       # same stack with the real LeWM backend on CPU
make demo-gpu        # + NVIDIA device reservation
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/kserve-inferenceservice.yaml   # KServe InferenceService
```

## Develop

```bash
make test                           # pytest
uv run --extra dev ruff check .     # lint (line length 120)
uv run --extra dev mypy src         # types
```

```text
src/wmcp_jepa_service/   server.py · runtime.py · runtime_lewm.py · lewm_model.py
                         model_package.py · packaging.py · schemas.py · observability.py · telemetry.py
client/                  stdlib WMCP client + demo + HTML view
scripts/                 build_model_package.py · pin_sources.py · run_demo_local.sh
deployment/              docker-compose*.yaml · prometheus · otel-collector · tempo · grafana · k8s/
benchmarks/              wmcp_bench/ harness + reports/
```

Stack: FastAPI · Pydantic v2 · PyTorch · Transformers · prometheus-client · OpenTelemetry · uv.

## License

The vendored LeWorldModel code and the `quentinll/lewm-pusht` checkpoint are MIT-licensed
(upstream [`galilai-group/stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel)).
