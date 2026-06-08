#!/usr/bin/env python3
"""Resolve and pin upstream source commits + the HF artifact revision into ``sources.lock.json``.

Pinning needs network access. This resolves:
- ``git ls-remote`` HEAD for each upstream repo (default branch tip);
- the Hugging Face model ``main`` revision via the HF API.

Run against a package dir to overwrite its ``sources.lock.json`` with resolved shas::

    python scripts/pin_sources.py --package models/lewm-pusht

If a source cannot be resolved (offline / rate-limited), it is left as ``UNPINNED`` and a warning is
printed — pinning is required before a production build, so this fails non-zero if anything is unpinned
unless ``--allow-unpinned`` is given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPOS = {
    "le-wm": "https://github.com/lucas-maes/le-wm",
    "stable-worldmodel": "https://github.com/galilai-group/stable-worldmodel",
    "stable-pretraining": "https://github.com/rbalestr-lab/stable-pretraining",
}
HF_REPO = "quentinll/lewm-pusht"


def resolve_repo_head(url: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "ls-remote", url, "HEAD"], capture_output=True, text=True, timeout=30, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = out.stdout.strip().split("\n")[0] if out.stdout.strip() else ""
    return line.split("\t")[0] if line else None


def resolve_hf_revision(repo: str) -> str | None:
    url = f"https://huggingface.co/api/models/{repo}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted host)
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("sha")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pin upstream sources into a package sources.lock.json")
    parser.add_argument("--package", required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args(argv)

    lock_path = Path(args.package) / "sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {"sources": {}, "artifact": {}}
    lock.setdefault("sources", {})
    lock.setdefault("artifact", {})

    unpinned: list[str] = []
    for name, url in REPOS.items():
        sha = resolve_repo_head(url)
        lock["sources"][name] = {"url": url, "commit": sha or "UNPINNED"}
        if not sha:
            unpinned.append(name)
            print(f"warning: could not resolve {name} ({url})", file=sys.stderr)

    hf_sha = resolve_hf_revision(HF_REPO)
    lock["artifact"] = {"repo": HF_REPO, "revision": hf_sha or "UNPINNED"}
    if not hf_sha:
        unpinned.append(HF_REPO)
        print(f"warning: could not resolve HF revision for {HF_REPO}", file=sys.stderr)

    lock["note"] = "Resolved by scripts/pin_sources.py (git ls-remote + HF API)."
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {lock_path}")

    if unpinned and not args.allow_unpinned:
        print(f"error: still unpinned: {unpinned}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
