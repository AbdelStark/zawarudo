# Push-T WMCP demo client

The other half of the demo: a dependency-free client that drives the WMCP inference API and visualizes
a plan. Runs end-to-end against the **mock** backend today and against the **real** `lewm-pusht`
checkpoint once #3 lands — no client change, just flip the service's `WMCP_BACKEND`.

## Run

```bash
# against a locally running service (make run, in another shell)
python -m client.demo

# point elsewhere / write a static HTML view
WMCP_BASE_URL=http://localhost:8080 WMCP_DEMO_OUT=demo.html python -m client.demo
python -m client.demo --candidates 64 --horizon 16 --out demo.html
```

It performs one `metadata → score → plan` cycle and prints `best_index`, `cost_statistics`, the plan
horizon/shape, and `best_cost`. With `--out` it also writes a self-contained HTML page with
candidate-cost bars and the chosen first action.

## In docker compose

The `client` service (added in #7) depends on the backend's healthcheck and runs this demo once on
`docker compose up`, with `WMCP_BASE_URL=http://wmcp-jepa-service:8080`.

The `traffic-generator` service runs `python -m client.traffic` continuously. It detects the active
backend, sends small base64-pixel requests to the real `lewm` runtime, varies score/rollout/plan
shapes, and periodically emits an expected validation error so error-rate panels have data.

## Layout

| Module | Role |
|---|---|
| `wmcp_client.py` | stdlib HTTP transport (`metadata`/`encode`/`rollout`/`score`/`plan`); `requester` seam for tests |
| `payloads.py` | build WMCP envelopes (URI-backed obs/goal, inline or URI action candidates `[B,S,T,10]`) |
| `demo.py` | `python -m client.demo` entrypoint (`run_demo` + `summarize`) |
| `traffic.py` | compose traffic stimulator for live Prometheus/Grafana data |
| `render.py` | static-HTML result view (no external deps) |

## Optional: real image payloads

`payloads.observation/goal` emit URI-backed `uint8` RGB tensors (`B,H,C,224,224`). To generate actual
Push-T frames and host them at a URI the service resolves, install the optional extras in
`requirements.txt` (numpy + Pillow).
