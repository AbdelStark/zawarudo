"""Push-T WMCP demo client — drives the inference service and visualizes a plan.

The client is intentionally dependency-free (stdlib only) so it runs anywhere a Python 3.10 image
exists; numpy/Pillow are optional and only needed to synthesize real Push-T image payloads.
"""

from __future__ import annotations

from .wmcp_client import DEFAULT_BASE_URL, WMCPClient, WMCPError

__all__ = ["WMCPClient", "WMCPError", "DEFAULT_BASE_URL"]
__version__ = "0.1.0"
