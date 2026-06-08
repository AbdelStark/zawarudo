# Push-T demo — RFC-0005 acceptance run log

End-to-end conformance evidence for epic #1 against `rfc/0005-pusht-demo-profile.md`. Reproduced
2026-06-08 on macOS (Apple Silicon, CPU) with the real `quentinll/lewm-pusht` checkpoint
(revision `22b330c28c27ead4bfd1888615af1340e3fe9052`).

## Reproduce

```bash
# 1. Build the real model package (needs HF_TOKEN + the `lewm` extra)
python scripts/pin_sources.py --package models/lewm-pusht
python scripts/build_model_package.py real --source <hf-download> --out .artifacts/model-package/lewm-pusht
python scripts/build_model_package.py verify --package .artifacts/model-package/lewm-pusht

# 2. One command: client + real backend + monitoring (CPU)
make demo-lewm            # docker compose -f docker-compose.yaml -f docker-compose.lewm.yaml up --build
# Grafana http://localhost:3000 (admin/admin) · Prometheus :9090 · Tempo :3200
```

## Acceptance criteria (RFC-0005 §"Acceptance criteria")

| # | Criterion | Status | Evidence |
|---|---|:---:|---|
| 1 | All required endpoints respond with **real** model integration | ✅ | `make demo-lewm`: `/readyz` → `backend: lewm`, `metadata` rev `22b330c`; client ran `metadata → score → plan` (exit 0). Live HTTP smoke of encode/rollout/score/plan. |
| 2 | Contract tests pass using example JSON payloads | ✅ | `tests/test_endpoints.py` posts every endpoint with `examples/*.json` → 200 + documented output keys. |
| 3 | `score` returns `[B,S]` costs | ✅ | Real backend: `costs [1,16]` (and `[1,256]` in `score-medium`); golden test asserts shape + values within tol. |
| 4 | `plan` returns `[B,T,10]` actions + `[B,10]` first action | ✅ | Real backend: `best_action_sequence [1,8,10]`, `first_action [1,10]`, CEM `best_cost_by_iteration` converging. |
| 5 | Prometheus **and** OTel telemetry visible during the demo | ✅ | Prometheus scraped `wmcp_model_compute_seconds{backend="lewm"}` for score+plan and `wmcp_planner_iterations`; Tempo held the service's `wmcp.request`/`wmcp.model.plan` traces; Grafana auto-provisioned Prometheus+Tempo datasources + the **WMCP-JEPA Serve** dashboard. |
| 6 | Benchmark report includes the full benchmark context | ✅ | `benchmarks/reports/score-medium-lewm.md` (real lewm CPU; commit, backend, hardware, PyTorch, encoding, candidate distribution, p50/p95/p99). |
| 7 | ≥5 RFC issues filed from implementation learnings | ✅ | See "Follow-up RFC issues" below. |

## Sample client output (real lewm backend, in compose)

```
model          : lewm-pusht (rev 22b330c28c27..., backend lewm)
score request  : B=1, S=16, T=8
best_index     : 12
cost_statistics: min=22.1711, mean=24.3443, max=27.5269
plan horizon   : 8  best_action_sequence shape: [1, 8, 10]
plan best_cost : 34.7288
wrote /out/demo.html
```

## Telemetry observed (lewm run)

- Prometheus: `score backend=lewm`, `plan backend=lewm`, `wmcp_planner_iterations` present.
- Tempo: 4 traces, service `wmcp-jepa-service`, spans incl. `wmcp.request` / `wmcp.model.plan`.
- Grafana datasources: `prometheus`, `tempo`; dashboard: `WMCP-JEPA Serve`.

## Known limitations

- **Observations are synthetic** (random RGB frames). Real Push-T frames need the `stable-worldmodel`
  Push-T env (pymunk/pygame) to render — out of scope for the inference service. The real model runs
  genuinely on the provided pixels.
- **CPU latency**: `score-medium` (S=256) p50 ≈ 8.4 s on CPU. The `p95 < 700 ms` SLO is GPU-targeted
  (`make demo-gpu`). No dynamic batching/AMP/`torch.compile` yet (correctness-first; gated until #5).
- **Action scaler**: identity (candidates assumed pre-normalized); the upstream StandardScaler stats are
  not shipped in the checkpoint (RFC follow-up).
- The published WMCP envelope is a working draft (README "Source limitations").

## Follow-up RFC issues

Filed from implementation learnings (RFC-0005 acceptance criterion 7) — see the issue tracker
(`label:rfc`). Topics: action scaler/bounds packaging, URI tensor-resolution contract (+SSRF),
dynamic batching for the latency SLO, latent-output store contract, goal-frame/history-action
semantics, and real Push-T observation rendering.
