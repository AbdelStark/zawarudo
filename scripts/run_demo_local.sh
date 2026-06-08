#!/usr/bin/env bash
#
# Run the Push-T demo end-to-end LOCALLY, without docker compose.
#
# Starts the WMCP backend (uvicorn), waits for it to be ready, runs the client demo
# (metadata -> score -> plan) against it, writes an HTML view, then stops the backend.
#
# Usage:
#   scripts/run_demo_local.sh                 # mock backend (no weights needed)
#   scripts/run_demo_local.sh --backend lewm  # real LeWorldModel checkpoint (CPU)
#   scripts/run_demo_local.sh --keep          # leave the backend running afterwards
#
# Options:
#   --backend mock|lewm   backend to serve (default: mock)
#   --port N              backend port (default: 8080)
#   --candidates N        score candidates S (default: 16)
#   --horizon N           plan/rollout horizon T (default: 8)
#   --seed N              RNG seed (default: 0)
#   --out PATH            HTML output (default: .artifacts/demo/local-demo.html)
#   --package DIR         lewm model package (default: .artifacts/model-package/lewm-pusht)
#   --keep                keep the backend running after the demo (print how to stop)
#   -h, --help            show this help
#
set -euo pipefail

BACKEND="mock"; PORT="8080"; CANDIDATES="16"; HORIZON="8"; SEED="0"
OUT=".artifacts/demo/local-demo.html"
PACKAGE="${WMCP_MODEL_PACKAGE:-.artifacts/model-package/lewm-pusht}"
KEEP="0"

usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --candidates) CANDIDATES="$2"; shift 2;;
    --horizon) HORIZON="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --package) PACKAGE="$2"; shift 2;;
    --keep) KEEP="1"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage; exit 1;;
  esac
done

# Always operate from the repo root (this script lives in scripts/).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is required (https://docs.astral.sh/uv/). Install it, then re-run." >&2
  exit 1
fi

# Backend selection: uv extras + env.
UV_EXTRAS=()
BACKEND_ENV=(WMCP_MODEL_ID=lewm-pusht "WMCP_BACKEND=$BACKEND")
case "$BACKEND" in
  mock)
    ;;
  lewm)
    UV_EXTRAS=(--extra lewm)
    if [[ ! -f "$PACKAGE/weights.safetensors" && ! -f "$PACKAGE/weights.pt" ]]; then
      echo "error: lewm backend needs a model package at '$PACKAGE'." >&2
      echo "Build it from the HF checkpoint:" >&2
      echo "  python scripts/build_model_package.py real --source <hf-download-dir> --out $PACKAGE" >&2
      exit 1
    fi
    BACKEND_ENV+=("WMCP_MODEL_PACKAGE=$PACKAGE" "WMCP_HF_DEVICE=${WMCP_HF_DEVICE:-cpu}")
    ;;
  *)
    echo "error: --backend must be 'mock' or 'lewm' (got '$BACKEND')" >&2; exit 1;;
esac

BASE_URL="http://127.0.0.1:$PORT"
mkdir -p .artifacts
LOG=".artifacts/local-demo-backend.log"

echo "==> starting $BACKEND backend on $BASE_URL  (logs: $LOG)"
# ${UV_EXTRAS[@]+...} guards the empty-array case under `set -u` on bash 3.2 (macOS default).
env "${BACKEND_ENV[@]}" uv run ${UV_EXTRAS[@]+"${UV_EXTRAS[@]}"} \
  uvicorn wmcp_jepa_service.server:app --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
BACKEND_PID=$!

cleanup() {
  if [[ "$KEEP" == "1" ]]; then
    echo
    echo "==> backend still running: $BASE_URL  (pid $BACKEND_PID)"
    echo "    stop it with:  kill $BACKEND_PID"
  else
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
    echo "==> backend stopped"
  fi
}
trap cleanup EXIT INT TERM

echo "==> waiting for /readyz (lewm model load can take ~10-30s)"
if ! curl -fsS --retry 90 --retry-connrefused --retry-delay 2 --retry-max-time 200 "$BASE_URL/readyz" >/dev/null 2>&1; then
  echo "error: backend did not become ready. last log lines:" >&2
  tail -n 25 "$LOG" >&2
  exit 1
fi
echo "==> ready: $(curl -fsS "$BASE_URL/readyz")"

echo "==> running client demo (metadata -> score -> plan)"
echo
env "WMCP_BASE_URL=$BASE_URL" "WMCP_DEMO_OUT=$OUT" uv run ${UV_EXTRAS[@]+"${UV_EXTRAS[@]}"} \
  python -m client.demo --candidates "$CANDIDATES" --horizon "$HORIZON" --seed "$SEED"

echo
echo "==> done. open the result:  $OUT"
