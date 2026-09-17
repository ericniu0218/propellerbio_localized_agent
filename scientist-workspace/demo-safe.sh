#!/usr/bin/env bash
# The reliable demo: seeded fictional data, no live model, no external
# services. This is the one to lead with — start here.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "No .venv found — run: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  exit 1
fi
source .venv/bin/activate

echo "Starting Bedrock Workspace in demo mode on http://localhost:8000 ..."
DEMO_MODE=true uvicorn app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

sleep 2
if command -v open >/dev/null; then open http://localhost:8000/; fi

echo "Running. Press Ctrl+C to stop."
wait $SERVER_PID
