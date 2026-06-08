"""Collect the benchmark metadata the report MUST include (load-test-plan.md)."""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Optional


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _torch_info() -> dict[str, Any]:
    try:
        torch: Any = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        return {
            "torch": str(torch.__version__),
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_count": torch.cuda.device_count() if cuda else 0,
        }
    except Exception:  # noqa: BLE001 - torch is optional
        return {"torch": None, "cuda": None, "cuda_available": False, "gpu": None, "gpu_count": 0}


def collect_metadata(
    *,
    base_url: str,
    backend: str,
    model_revision: str = "lewm-pusht",
    model_checksum: str = "n/a",
    image_digest: str = "n/a",
    dtype: str = "float32",
    encoding: str = "uri",
    dynamic_batching: str = "disabled",
) -> dict[str, Any]:
    ti = _torch_info()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service_commit": _git_commit(),
        "image_digest": image_digest,
        "model_revision": model_revision,
        "model_checksum": model_checksum,
        "backend": backend,
        "hardware": platform.platform(),
        "gpu": ti["gpu"],
        "gpu_count": ti["gpu_count"],
        "python": sys.version.split()[0],
        "torch": ti["torch"],
        "cuda": ti["cuda"],
        "cuda_available": ti["cuda_available"],
        "dtype": dtype,
        "encoding_mode": encoding,
        "dynamic_batching": dynamic_batching,
        "base_url": base_url,
    }


def backend_from_readyz(readyz: Optional[dict]) -> str:
    if isinstance(readyz, dict) and readyz.get("backend"):
        return str(readyz["backend"])
    return "unknown"
