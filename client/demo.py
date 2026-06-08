"""Push-T demo entrypoint: drive a score + plan cycle against the WMCP service and print the result.

Usage::

    python -m client.demo                       # uses WMCP_BASE_URL (default http://localhost:8080)
    WMCP_DEMO_OUT=demo.html python -m client.demo   # also write a static HTML view

Runs end-to-end against the mock backend (inline action candidates) and, once the real LeWMRuntime is
wired (#3), against the real checkpoint by switching only the service's WMCP_BACKEND.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Optional

from . import payloads
from .render import render_html
from .wmcp_client import DEFAULT_BASE_URL, WMCPClient, WMCPError


def wait_until_ready(client: WMCPClient, *, attempts: int = 30, delay: float = 1.0) -> bool:
    for _ in range(attempts):
        try:
            client.readyz()
            return True
        except WMCPError:
            return True  # service answered (even an error envelope) -> it is up
        except OSError:
            time.sleep(delay)
    return False


def run_demo(
    client: WMCPClient, *, s: int = 16, t: int = 8, horizon: int = 8, iterations: int = 5,
    candidates: int = 64, seed: int = 0,
) -> dict[str, Any]:
    """Run one metadata -> score -> plan cycle and return the structured results.

    The real ``lewm`` backend materialises pixels, so we send real base64 frames to it; the mock reads
    only shapes, so URI placeholders suffice there.
    """
    metadata = client.metadata()
    backend = metadata.get("runtime", {}).get("backend", "mock")
    pixel_encoding = "base64" if backend == "lewm" else "uri"
    if backend == "lewm":  # keep the CPU demo snappy
        candidates = min(candidates, 24)
        iterations = min(iterations, 3)
    score = client.score(
        payloads.score_request("demo-score", s=s, t=t, inline_actions=True, pixel_encoding=pixel_encoding, seed=seed))
    plan = client.plan(
        payloads.plan_request("demo-plan", horizon=horizon, iterations=iterations, candidates=candidates,
                              pixel_encoding=pixel_encoding, seed=seed))
    return {"metadata": metadata, "score": score, "plan": plan, "backend": backend,
            "request_shape": {"S": s, "T": t, "horizon": horizon}}


def summarize(result: dict[str, Any]) -> str:
    meta = result.get("metadata", {})
    score_out = result.get("score", {}).get("outputs", {})
    plan_out = result.get("plan", {}).get("outputs", {})
    shape = result.get("request_shape", {})

    best_index = (score_out.get("best_index") or ["?"])[0]
    stats = score_out.get("cost_statistics", {})
    stat_line = ", ".join(f"{k}={v:.4f}" for k, v in stats.items() if isinstance(v, (int, float))) or "n/a"
    best_cost = (plan_out.get("best_cost") or ["?"])[0]
    plan_seq = plan_out.get("best_action_sequence", {})
    seq_shape = plan_seq.get("shape") if isinstance(plan_seq, dict) else None

    lines = [
        f"model          : {meta.get('model_id')} (rev {meta.get('model_revision')}, "
        f"backend {meta.get('runtime', {}).get('backend')})",
        f"score request  : B=1, S={shape.get('S')}, T={shape.get('T')}",
        f"best_index     : {best_index}",
        f"cost_statistics: {stat_line}",
        f"plan horizon   : {shape.get('horizon')}  best_action_sequence shape: {seq_shape}",
        f"plan best_cost : {best_cost if not isinstance(best_cost, float) else round(best_cost, 4)}",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Push-T WMCP demo client")
    parser.add_argument("--base-url", default=os.getenv("WMCP_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model-id", default=os.getenv("WMCP_MODEL_ID", "lewm-pusht"))
    parser.add_argument("--candidates", type=int, default=16, help="S (number of action candidates)")
    parser.add_argument("--horizon", type=int, default=8, help="T / plan horizon")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.getenv("WMCP_DEMO_OUT"), help="write a static HTML view to this path")
    args = parser.parse_args(argv)

    timeout = float(os.getenv("WMCP_TIMEOUT", "180"))
    client = WMCPClient(args.base_url, model_id=args.model_id, timeout=timeout)
    if not wait_until_ready(client):
        print(f"service at {args.base_url} did not become ready", flush=True)
        return 1

    result = run_demo(client, s=args.candidates, t=args.horizon, horizon=args.horizon, seed=args.seed)
    print(summarize(result), flush=True)
    if args.out:
        path = render_html(result, args.out)
        print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
