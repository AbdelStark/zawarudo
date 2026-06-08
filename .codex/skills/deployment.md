---
name: deployment
description: How this service is packaged and deployed — Dockerfile, Docker Compose (service + prometheus + otel + grafana), Kubernetes Deployment, and the optional KServe InferenceService. Activate for container/build, local-stack, or cluster work. All of deployment/** is GATED (infra) — change with approval.
prerequisites: docker; (cluster) kubectl/KServe
---

# Deployment

<purpose>
Ship the service locally (Compose) and to a cluster (k8s / KServe) without changing application code.
</purpose>

<context>
- `Dockerfile`: `python:3.10-slim`, installs the package editable, `EXPOSE 8080`,
  runs `uvicorn wmcp_jepa_service.server:app --host 0.0.0.0 --port 8080`.
- `deployment/docker-compose.yaml`: services `wmcp-jepa-service` (:8080), `prometheus` (:9090),
  `otel-collector` (:4317/4318/8889), `grafana` (:3000, admin/admin). Service env: `WMCP_MODEL_ID`,
  `WMCP_BACKEND`, `WMCP_OTEL_EXPORTER_OTLP_ENDPOINT`, `WMCP_ENABLE_PROMETHEUS`, `WMCP_LOG_LEVEL`.
  GPU block is commented — enable on an NVIDIA host with the Container Toolkit.
- `deployment/k8s/deployment.yaml`: Kubernetes Deployment/Service.
- `deployment/k8s/kserve-inferenceservice.yaml`: optional KServe wrapper; the service already exposes a
  `/v2/models/{name}/infer` adapter for the KServe V2 protocol.
- Make targets: `make docker-build`, `make compose-up`, `make compose-down`.
</context>

<procedure>
1. Local image: `make docker-build`.
2. Full local stack: `make compose-up` → service :8080, Prometheus :9090, Grafana :3000.
3. Smoke: `curl localhost:8080/readyz`, `curl localhost:8080/metrics`.
4. GPU host: uncomment the `deploy.resources.reservations.devices` block before `compose up`.
5. Cluster: apply `deployment/k8s/deployment.yaml`; wrap with KServe only once a real runtime is stable.
</procedure>

<patterns>
<do>
— Keep the base image at python:3.10-slim for the mock/API image; use a dedicated, dependency-pinned
  image for the real LeWM runtime (heavy torch + upstream deps).
— Set `WMCP_BACKEND` via env, not code, once backend selection is wired.
</do>
<dont>
— Don't bake model weights into the generic image — mount/package separately (see model-packaging).
— Don't promote to KServe before the runtime + golden tests are stable (ADR-0001 follow-ups).
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| compose service can't reach otel | wrong endpoint | `WMCP_OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317` |
| GPU not used | reservation block commented | uncomment + install NVIDIA Container Toolkit |
| KServe infer 400 | unsupported operation | set `parameters.operation` to a known op |
</troubleshooting>

<references>
— Dockerfile · deployment/docker-compose.yaml · deployment/k8s/* · rfc/0004-model-packaging-runtime.md
</references>
</content>
