# Load-test and benchmark plan

## Purpose

Measure correctness, latency, throughput, batching behavior, GPU utilization, memory, and planner convergence for the Push-T WMCP-JEPA service.

## Required benchmark metadata

Every benchmark run must record:

- benchmark timestamp;
- git commit of service;
- container image digest;
- model artifact revision and checksum;
- backend (`mock`, `ray-pytorch`, `triton`, etc.);
- hardware model;
- GPU count and memory;
- CUDA/driver versions;
- Python/PyTorch versions;
- dtype and compile settings;
- request encoding mode;
- batch/candidate/horizon/action dimensions;
- dynamic batching configuration;
- planner configuration.

## Test profiles

| Profile | Operation | B | S | T | H | A | Concurrency | Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---|
| health | health/metadata | - | - | - | - | - | 16 | Control-plane latency. |
| smoke-score | score | 1 | 4 | 4 | 3 | 10 | 1 | Correctness. |
| score-small | score | 1 | 16 | 8 | 3 | 10 | 4 | Local demo. |
| score-medium | score | 1 | 256 | 16 | 3 | 10 | 8 | Main target. |
| score-large | score | 1 | 1024 | 32 | 3 | 10 | 4 | Stress. |
| rollout-medium | rollout | 1 | 256 | 16 | 3 | 10 | 8 | Latent output overhead. |
| plan-medium | plan | 1 | 256 | 16 | 3 | 10 | 1 | CEM/MPC end-to-end. |
| plan-concurrent | plan | 1 | 128 | 16 | 3 | 10 | 4 | Planner queueing. |

## Measurements

- Request rate.
- p50/p90/p95/p99 end-to-end latency.
- p50/p95 validation latency.
- p50/p95 preprocess latency.
- p50/p95 queue wait.
- p50/p95 model compute latency.
- p50/p95 serialization latency.
- Batch size distribution.
- Candidate count distribution.
- GPU utilization.
- GPU memory allocated/reserved.
- OOM count.
- Error rate by code.
- Planner best cost by iteration.

## Commands

Example smoke request:

```bash
curl -s http://localhost:8080/wmcp/v1/models/lewm-pusht:score \
  -H 'Content-Type: application/json' \
  --data @examples/score_request.json | jq .
```

Example load-test shape generator should produce URI-backed payloads and then issue concurrent HTTP requests. For the real benchmark implementation, use Locust, k6, or a Python asyncio client.

## Acceptance thresholds for demo

These are placeholders until measured on target hardware.

| Operation/profile | Acceptance |
|---|---|
| health/metadata | p95 < 50 ms |
| smoke-score | 0 errors, valid `[B,S]` costs |
| score-medium | p95 < 700 ms on selected GPU or documented exception |
| rollout-medium | p95 < 900 ms with URI latents or documented exception |
| plan-medium | p95 < 3 s or documented tuning issue |
| observability | Metrics and traces visible for every operation |

## Report template

```markdown
# Benchmark report

- Date:
- Service commit:
- Container image:
- Model revision/checksum:
- Backend:
- Hardware:
- Driver/CUDA/PyTorch:
- Dynamic batching config:

## Results

| Profile | RPS | p50 | p95 | p99 | GPU util | GPU mem | Error rate |
|---|---:|---:|---:|---:|---:|---:|---:|

## Findings

## Bottlenecks

## RFC implications

## Next actions
```
