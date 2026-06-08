"""Make the repo root importable so tests can import top-level, non-installed packages.

Covers `client` (issue #6) and `benchmarks.wmcp_bench` (issue #8).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
