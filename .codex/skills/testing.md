---
name: testing
description: Testing strategy for this service — Pydantic schema/contract tests (fast, no model) and golden numerical validation of a real runtime against upstream eval. Activate when adding tests, changing schemas (which require contract-test updates), validating a new backend, or before exposing a real runtime.
prerequisites: pytest + httpx (extra `dev`)
---

# Testing

<purpose>
Two test tiers: (1) contract tests guarantee the WMCP envelope shapes stay valid and stable; (2) golden
tests guarantee a real `LeWMRuntime` matches upstream model outputs before it is exposed.
</purpose>

<context>
- `tests/test_contracts.py` validates `RequestEnvelope.model_validate(...)` on canonical payloads — pure
  schema, no model, runs in <1s. Mirrors `examples/*.json`.
- Run: `pytest -q` (or `make test`). Lint/type gates: `ruff check . && mypy src`.
- The mock backend exists precisely so the API + observability are testable without weights/GPU.
- Golden validation (real runtime) is described in `LEWM_INTEGRATION_GUIDE.md` §4.
</context>

<procedure>
1. Schema change → update `tests/test_contracts.py` and `examples/*.json` in the same change.
2. New endpoint → add an httpx test against the FastAPI app (TestClient) asserting status + `outputs` keys.
3. Real backend → add golden tests:
   a. Run upstream `eval.py` (or minimal scoring path) on fixed tensors → save expected costs.
   b. Run `LeWMRuntime` on the same tensors.
   c. Assert cost SHAPE equal and values within tolerance (e.g. `atol/rtol`); assert no grads, correct device.
   d. Verify action normalizer + bounds round-trip.
4. Gate every change on `ruff check . && mypy src && pytest -q`.
</procedure>

<patterns>
<do>
— Keep contract tests fast and model-free; use the mock for API/integration tests.
— Pin golden fixtures (tensors + expected outputs) as committed artifacts with checksums.
</do>
<dont>
— Don't change `schemas.py` without updating contract tests in the same commit.
— Don't golden-test against a floating upstream commit — pin it.
</dont>
</patterns>

<examples>
Contract test shape (existing):
```python
from wmcp_jepa_service.schemas import RequestEnvelope
def test_minimal_score_request_contract() -> None:
    req = RequestEnvelope.model_validate({...})  # see tests/test_contracts.py
    assert req.operation == "score" and req.model == "lewm-pusht"
```
Endpoint test (add):
```python
from fastapi.testclient import TestClient
from wmcp_jepa_service.server import app
def test_score_endpoint():
    r = TestClient(app).post("/wmcp/v1/models/lewm-pusht:score", json={...})
    assert r.status_code == 200 and "costs" in r.json()["outputs"]
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| Contract test 422 after schema edit | example payload stale | update `examples/*.json` + test together |
| Golden values drift | upstream commit moved / AMP on | pin commit; validate fp32 first |
</troubleshooting>

<references>
— tests/test_contracts.py · examples/*.json · LEWM_INTEGRATION_GUIDE.md §4 (golden validation)
</references>
