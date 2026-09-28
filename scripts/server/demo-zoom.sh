#!/usr/bin/env bash
# The Zoom-integration demo: real transcript pipeline (zoom-transcript-vault's
# own mock datacenter + a real signed webhook), real classification and tool
# use in this backend (AGENT_PACK_MODE=mock — deterministic, no live model
# download required), asked through the workspace chat UI.
#
# Prerequisites, one-time:
#   cd .. && git clone https://github.com/Bedrock-Technical-Consulting/zoom-transcript-vault.git
#   cd zoom-transcript-vault && npm install
#   printf 'ZOOM_WEBHOOK_SECRET_TOKEN=demo-zoom-secret-123\nDATACENTER_BASE_URL=http://localhost:9100\nDATACENTER_SIGNING_SECRET=demo-datacenter-secret-456\nRETRIEVAL_API_KEY=demo-retrieval-key-789\n' > .dev.vars
#   npx drizzle-kit generate   # only if drizzle/ is empty
#   npx wrangler d1 migrations apply zoom-transcript-vault --local
#
# Override ZOOM_VAULT_DIR if the clone isn't a sibling of this repo.
set -euo pipefail
cd "$(dirname "$0")/.."

VAULT_DIR="${ZOOM_VAULT_DIR:-../zoom-transcript-vault}"
if [ ! -d "$VAULT_DIR" ]; then
  echo "Can't find zoom-transcript-vault at $VAULT_DIR."
  echo "Clone it as a sibling of this repo, or set ZOOM_VAULT_DIR. See this script's header for one-time setup."
  exit 1
fi
if [ ! -f "$VAULT_DIR/.dev.vars" ]; then
  echo "$VAULT_DIR/.dev.vars is missing — see this script's header for one-time setup."
  exit 1
fi

pkill -f "mock-datacenter.mjs" 2>/dev/null || true
pkill -f "wrangler dev" 2>/dev/null || true
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 1

echo "[1/4] Starting the mock datacenter + fake Zoom file host on :9100 ..."
( cd "$VAULT_DIR" && DATACENTER_SIGNING_SECRET=demo-datacenter-secret-456 node demo/mock-datacenter.mjs ) &
DATACENTER_PID=$!
sleep 1

echo "[2/4] Starting the zoom-transcript-vault Worker on :8787 ..."
( cd "$VAULT_DIR" && npx wrangler dev --port 8787 ) &
WORKER_PID=$!
sleep 5

echo "[3/4] Sending a real, signed Zoom webhook (pushes a sample transcript through the real pipeline) ..."
( cd "$VAULT_DIR" && ZOOM_WEBHOOK_SECRET_TOKEN=demo-zoom-secret-123 node demo/send-webhook.mjs http://localhost:8787 http://localhost:9100 )

echo "[4/4] Starting the workspace, pointed at the same mock datacenter, on http://localhost:8000 ..."
if [ ! -d .venv ]; then
  echo "No .venv found — run: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  exit 1
fi
source .venv/bin/activate
DEMO_MODE=false AGENT_PACK_MODE=mock AIRGAP_CHECK=false DATABASE_URL=sqlite:///./demo_zoom.db \
  DATACENTER_BASE_URL=http://localhost:9100 DATACENTER_SIGNING_SECRET=demo-datacenter-secret-456 \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

trap 'kill $DATACENTER_PID $WORKER_PID $SERVER_PID 2>/dev/null' EXIT
sleep 2
echo
echo "Ready. Open http://localhost:8000/, sign in with any name (no SSO configured — that's expected)"
echo "and ask: \"What happened in today's meeting about AlphaFold?\""
if command -v open >/dev/null; then open http://localhost:8000/; fi

echo "Running. Press Ctrl+C to stop everything."
wait $SERVER_PID
