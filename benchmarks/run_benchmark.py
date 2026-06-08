#!/usr/bin/env python3
"""Run the WMCP benchmark harness against a live service and write a filled report.

Examples
--------
    # main demo target
    python benchmarks/run_benchmark.py --base-url http://localhost:8080 --profile score-medium

    # a set of profiles -> one report
    python benchmarks/run_benchmark.py --profiles smoke-score,score-small,score-medium \
        --out benchmarks/reports/demo.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.wmcp_bench import PROFILES, render_report, run_profile  # noqa: E402
from benchmarks.wmcp_bench.metadata import backend_from_readyz, collect_metadata  # noqa: E402
from benchmarks.wmcp_bench.runner import ProfileResult  # noqa: E402


def make_urllib_poster(base_url: str, timeout: float = 120.0):
    root = base_url.rstrip("/")

    def poster(method: str, path: str, body: Optional[dict]) -> "tuple[int, Optional[dict]]":
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"content-type": "application/json"} if data is not None else {}
        req = urllib.request.Request(root + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-provided URL
                raw = resp.read().decode("utf-8")
                return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            return exc.code, parsed

    return poster


def _readyz(base_url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/readyz", timeout=10) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError):
        return None


async def _run(base_url: str, names: list[str], requests: Optional[int], encoding: str) -> list[ProfileResult]:
    poster = make_urllib_poster(base_url)
    out: list[ProfileResult] = []
    for name in names:
        profile = PROFILES[name]
        out.append(await run_profile(poster, profile, total_requests=requests, encoding=encoding))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="WMCP benchmark harness")
    parser.add_argument("--base-url", default="http://localhost:8080")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile", help="single profile name")
    group.add_argument("--profiles", help="comma-separated profile names")
    group.add_argument("--all", action="store_true", help="run every profile")
    parser.add_argument("--requests", type=int, default=None, help="override request count per profile")
    parser.add_argument("--out", default=None, help="report path (default benchmarks/reports/<profile>-<backend>.md)")
    parser.add_argument("--title", default="WMCP Push-T benchmark report")
    parser.add_argument("--encoding", choices=["uri", "base64", "auto"], default="auto",
                        help="payload encoding; 'auto' uses base64 for the lewm backend, uri otherwise")
    args = parser.parse_args(argv)

    if args.all:
        names = list(PROFILES.keys())
    elif args.profiles:
        names = [n.strip() for n in args.profiles.split(",") if n.strip()]
    elif args.profile:
        names = [args.profile]
    else:
        names = ["score-medium"]
    unknown = [n for n in names if n not in PROFILES]
    if unknown:
        parser.error(f"unknown profiles: {unknown} (have {list(PROFILES)})")

    readyz = _readyz(args.base_url)
    backend = backend_from_readyz(readyz)
    encoding = ("base64" if backend == "lewm" else "uri") if args.encoding == "auto" else args.encoding
    meta = collect_metadata(base_url=args.base_url, backend=backend, model_revision=(readyz or {}).get("model", "lewm-pusht"))
    meta["encoding_mode"] = encoding

    results = asyncio.run(_run(args.base_url, names, args.requests, encoding))
    report = render_report(meta, results, title=args.title)

    out_path = Path(args.out) if args.out else Path("benchmarks/reports") / f"{names[0]}-{backend}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    for r in results:
        print(f"{r.name:<16} reqs={r.count:<5} rps={r.rps:7.1f} p50={r.p50:7.1f}ms p95={r.p95:7.1f}ms errors={r.errors}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
