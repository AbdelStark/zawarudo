#!/usr/bin/env python3
"""Build a trusted WMCP model package.

Examples
--------
Synthetic package (stdlib only, no real weights) — useful for CI, the loader self-test, and demos::

    python scripts/build_model_package.py synthetic --out .artifacts/model-package/lewm-pusht

Real package from a downloaded HF checkpoint dir (needs the ``torch`` extra + ``safetensors``)::

    python scripts/build_model_package.py real \
        --source ~/.cache/lewm-pusht --out models/lewm-pusht \
        --source-revision <le-wm-commit> --artifact-revision <hf-revision>

Verify any package::

    python scripts/build_model_package.py verify --package models/lewm-pusht
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (``python scripts/build_model_package.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmcp_jepa_service import model_package as mp  # noqa: E402
from wmcp_jepa_service import packaging  # noqa: E402


def _cmd_synthetic(args: argparse.Namespace) -> int:
    out = packaging.build_synthetic_package(args.out, model_id=args.model_id)
    mp.verify_checksums(out)
    print(f"built + verified synthetic package at {out}")
    return 0


def _cmd_real(args: argparse.Namespace) -> int:
    out = packaging.build_real_package(
        args.source,
        args.out,
        model_id=args.model_id,
        source_revision=args.source_revision,
        artifact_revision=args.artifact_revision,
    )
    mp.verify_checksums(out)
    print(f"built + verified package at {out}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    mp.verify_checksums(args.package)
    manifest = mp.load_manifest(args.package)
    mp.validate_manifest(manifest)
    shapes = mp.read_state_shapes(args.package, manifest)
    mp.verify_shapes(shapes, manifest)
    print(f"OK: {args.package} verified ({len(shapes)} tensors, model_id={manifest['model_id']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build/verify a WMCP model package")
    sub = parser.add_subparsers(dest="command", required=True)

    p_syn = sub.add_parser("synthetic", help="build a tiny stdlib-only package (no real weights)")
    p_syn.add_argument("--out", required=True)
    p_syn.add_argument("--model-id", default="lewm-pusht")
    p_syn.set_defaults(func=_cmd_synthetic)

    p_real = sub.add_parser("real", help="convert an upstream checkpoint to a safetensors package")
    p_real.add_argument("--source", required=True, help="dir with config.json + weights.pt")
    p_real.add_argument("--out", required=True)
    p_real.add_argument("--model-id", default="lewm-pusht")
    p_real.add_argument("--source-revision", default="unpinned")
    p_real.add_argument("--artifact-revision", default="unpinned")
    p_real.set_defaults(func=_cmd_real)

    p_ver = sub.add_parser("verify", help="verify checksums + manifest + shapes of a package")
    p_ver.add_argument("--package", required=True)
    p_ver.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
