"""Fill the load-test-plan report template with metadata + aggregated profile results."""

from __future__ import annotations

from typing import Any, Optional

from .runner import ProfileResult


def _metadata_block(meta: dict[str, Any]) -> str:
    gpu = meta.get("gpu") or f"none (cuda_available={meta.get('cuda_available')})"
    driver = f"CUDA {meta.get('cuda') or 'n/a'} / PyTorch {meta.get('torch') or 'not installed'}"
    return "\n".join(
        [
            f"- Date: {meta.get('timestamp')}",
            f"- Service commit: {meta.get('service_commit')}",
            f"- Container image: {meta.get('image_digest')}",
            f"- Model revision/checksum: {meta.get('model_revision')} / {meta.get('model_checksum')}",
            f"- Backend: {meta.get('backend')}",
            f"- Hardware: {meta.get('hardware')} (GPU: {gpu}, count {meta.get('gpu_count')})",
            f"- Driver/CUDA/PyTorch: {driver}",
            f"- Python: {meta.get('python')}",
            f"- Encoding mode: {meta.get('encoding_mode')}  ·  dtype: {meta.get('dtype')}",
            f"- Dynamic batching config: {meta.get('dynamic_batching')}",
        ]
    )


def _results_table(results: list[ProfileResult]) -> str:
    header = (
        "| Profile | Op | Conc | Reqs | RPS | p50 ms | p90 ms | p95 ms | p99 ms | Errors | Error rate |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| {r.name} | {r.operation} | {r.concurrency} | {r.count} | {r.rps:.1f} | "
        f"{r.p50:.1f} | {r.p90:.1f} | {r.p95:.1f} | {r.p99:.1f} | {r.errors} | {r.error_rate * 100:.2f}% |"
        for r in results
    ]
    return "\n".join([header, *rows])


def _candidate_dist(results: list[ProfileResult]) -> str:
    lines = []
    for r in results:
        dist = r.candidate_distribution()
        if dist:
            pretty = ", ".join(f"S={k}×{v}" for k, v in sorted(dist.items()))
            lines.append(f"- {r.name}: {pretty}")
    return "\n".join(lines) or "- (no candidate-count diagnostics reported)"


def render_report(
    meta: dict[str, Any],
    results: list[ProfileResult],
    *,
    title: str = "Benchmark report",
    findings: Optional[str] = None,
    bottlenecks: Optional[str] = None,
    rfc_implications: Optional[str] = None,
    next_actions: Optional[str] = None,
) -> str:
    total_err = sum(r.errors for r in results)
    auto_findings = findings or (
        f"{len(results)} profile(s) run, {total_err} total errors. "
        f"Backend `{meta.get('backend')}`; "
        + ("GPU metrics available." if meta.get("cuda_available") else "no GPU (CPU/mock run — latency is not representative of the real model).")
    )
    return f"""# {title}

{_metadata_block(meta)}

## Results

{_results_table(results)}

### Candidate-count distribution

{_candidate_dist(results)}

## Findings

{auto_findings}

## Bottlenecks

{bottlenecks or "- Not analysed in this run."}

## RFC implications

{rfc_implications or "- None recorded."}

## Next actions

{next_actions or "- Re-run against the real `lewm` backend on GPU once #3 lands to capture representative latency, GPU util/mem, and queue wait."}
"""
