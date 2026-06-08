"""Stdlib WMCP HTTP client.

Transport-only: it speaks the WMCP envelope protocol over HTTP and returns parsed JSON. Request
envelopes are built by :mod:`client.payloads`. The ``requester`` seam lets tests drive a FastAPI
``TestClient`` in-process instead of a real socket.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

DEFAULT_BASE_URL = os.getenv("WMCP_BASE_URL", "http://localhost:8080")
DEFAULT_MODEL_ID = os.getenv("WMCP_MODEL_ID", "lewm-pusht")

# (method, path, json_body|None) -> parsed json dict
Requester = Callable[[str, str, Optional[dict]], dict]


class WMCPError(RuntimeError):
    """Raised when the service returns a non-2xx WMCP error envelope."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class WMCPClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        timeout: float = 30.0,
        requester: Optional[Requester] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self._requester = requester or self._urllib_request

    # --- transport ---------------------------------------------------------------------------

    def _urllib_request(self, method: str, path: str, body: Optional[dict]) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"content-type": "application/json"} if data is not None else {}
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - configured base URL
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            detail: dict[str, Any] = {}
            try:
                detail = json.loads(raw).get("detail", {}) or {}
            except (ValueError, AttributeError):
                detail = {}
            raise WMCPError(exc.code, str(detail.get("code", "HTTP_ERROR")), str(detail.get("message", raw))) from exc

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        return self._requester(method, path, body)

    # --- health / metadata -------------------------------------------------------------------

    def healthz(self) -> dict:
        return self._request("GET", "/healthz")

    def readyz(self) -> dict:
        return self._request("GET", "/readyz")

    def list_models(self) -> dict:
        return self._request("GET", "/wmcp/v1/models")

    def metadata(self) -> dict:
        return self._request("GET", f"/wmcp/v1/models/{self.model_id}")

    # --- operations --------------------------------------------------------------------------

    def _op(self, operation: str, request: dict) -> dict:
        body = {**request, "operation": operation, "model": self.model_id}
        return self._request("POST", f"/wmcp/v1/models/{self.model_id}:{operation}", body)

    def encode(self, request: dict) -> dict:
        return self._op("encode", request)

    def rollout(self, request: dict) -> dict:
        return self._op("rollout", request)

    def score(self, request: dict) -> dict:
        return self._op("score", request)

    def plan(self, request: dict) -> dict:
        return self._op("plan", request)
