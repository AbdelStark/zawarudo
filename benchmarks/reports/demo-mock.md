# WMCP Push-T benchmark — mock backend (demonstration)

- Date: 2026-06-08T10:53:22.565173+00:00
- Service commit: d01e46c
- Container image: n/a
- Model revision/checksum: lewm-pusht / n/a
- Backend: mock
- Hardware: macOS-26.5-arm64-arm-64bit-Mach-O (GPU: none (cuda_available=False), count 0)
- Driver/CUDA/PyTorch: CUDA n/a / PyTorch not installed
- Python: 3.13.11
- Encoding mode: uri  ·  dtype: float32
- Dynamic batching config: disabled

## Results

| Profile | Op | Conc | Reqs | RPS | p50 ms | p90 ms | p95 ms | p99 ms | Errors | Error rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| health | health | 16 | 200 | 1891.4 | 7.2 | 11.1 | 12.1 | 13.3 | 0 | 0.00% |
| smoke-score | score | 1 | 20 | 344.5 | 2.6 | 3.0 | 3.1 | 3.2 | 0 | 0.00% |
| score-small | score | 4 | 100 | 1371.7 | 2.5 | 3.7 | 4.4 | 4.8 | 0 | 0.00% |
| score-medium | score | 8 | 200 | 539.3 | 9.5 | 11.8 | 12.1 | 132.5 | 0 | 0.00% |
| score-large | score | 4 | 60 | 97.4 | 40.6 | 43.4 | 44.0 | 45.2 | 0 | 0.00% |
| rollout-medium | rollout | 8 | 120 | 855.0 | 9.0 | 10.9 | 11.6 | 12.8 | 0 | 0.00% |
| plan-medium | plan | 1 | 30 | 844.4 | 1.1 | 1.5 | 1.6 | 1.6 | 0 | 0.00% |
| plan-concurrent | plan | 4 | 60 | 2001.2 | 1.8 | 2.5 | 2.7 | 3.2 | 0 | 0.00% |

### Candidate-count distribution

- smoke-score: S=4×20
- score-small: S=16×100
- score-medium: S=256×200
- score-large: S=1024×60
- rollout-medium: S=256×120

## Findings

8 profile(s) run, 0 total errors. Backend `mock`; no GPU (CPU/mock run — latency is not representative of the real model).

## Bottlenecks

- Not analysed in this run.

## RFC implications

- None recorded.

## Next actions

- Re-run against the real `lewm` backend on GPU once #3 lands to capture representative latency, GPU util/mem, and queue wait.
