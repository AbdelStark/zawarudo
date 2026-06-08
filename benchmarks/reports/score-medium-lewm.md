# WMCP Push-T benchmark — real lewm backend (CPU)

- Date: 2026-06-08T12:06:33.440544+00:00
- Service commit: d4012e8
- Container image: n/a
- Model revision/checksum: lewm-pusht / n/a
- Backend: lewm
- Hardware: macOS-26.5-arm64-arm-64bit-Mach-O (GPU: none (cuda_available=False), count 0)
- Driver/CUDA/PyTorch: CUDA n/a / PyTorch 2.12.0
- Python: 3.13.11
- Encoding mode: base64  ·  dtype: float32
- Dynamic batching config: disabled

## Results

| Profile | Op | Conc | Reqs | RPS | p50 ms | p90 ms | p95 ms | p99 ms | Errors | Error rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke-score | score | 1 | 16 | 4.9 | 201.6 | 239.4 | 244.2 | 251.2 | 0 | 0.00% |
| score-medium | score | 8 | 16 | 0.8 | 8420.9 | 9640.5 | 9679.1 | 9707.4 | 0 | 0.00% |

### Candidate-count distribution

- smoke-score: S=4×16
- score-medium: S=256×16

## Findings

- Real `lewm` backend (vendored LeWorldModel, vanilla fp32, **CPU**), driven with **base64** real
  pixel + action payloads. **0 errors** across both profiles; `score` returned correctly-shaped
  `[B,S]` costs (S=4 and S=256).
- **score-medium** (S=256, T=16) ran end-to-end on CPU: p50 ≈ **8.4 s**, p95 ≈ **9.7 s**. The
  RFC-0005 acceptance threshold (`p95 < 700 ms`) is **GPU-targeted**; on CPU this is a *documented
  exception* — latency is dominated by the autoregressive rollout over 256 candidates × 16 steps
  with no batching/AMP/compile (correctness-first, per #3/#5).
- **smoke-score** (S=4) p50 ≈ 0.20 s — the per-request fixed cost (encode obs+goal + short rollout).

## Bottlenecks

- Candidate rollout dominates: cost scales with S×T. Single-request, no dynamic batching (Ray Serve
  is phase-2, ADR-0001), fp32, CPU. The ViT encode of obs+goal is amortised (encoded once per request).

## RFC implications

- Confirms RFC-0005 open issue #5: latency targets must be set **after** GPU benchmarking; CPU serving
  is viable for correctness/demo but not the 700 ms SLO.
- Suggests a candidate-batch cap / chunking for very large S (score-large S=1024) before enabling it
  in production limits.

## Next actions

- Re-run on a GPU host (`make demo-gpu` / HF Jobs) to capture representative latency + GPU util/mem and
  evaluate the `p95 < 700 ms` threshold.
- Enable the perf knobs gated until golden validation (now passed, #5): candidate-dim batching, AMP/fp16,
  `torch.compile`, goal-encoding reuse across a plan — re-validate numerics after each.
