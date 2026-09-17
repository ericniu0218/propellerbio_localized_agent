#!/usr/bin/env bash
# Real (non-canned) answers using a local Ollama model instead of the
# company's GPU-served vLLM/SGLang endpoints, which don't exist yet. This is
# a stand-in for local development and demos only — Ollama's small models are
# nowhere near the quality of the production Director/Specialist/Runner
# tiers in config/models.yaml, but responses are genuinely generated, not
# templated.
#
# Prerequisites, one-time:
#   brew install ollama
#   brew services start ollama
#   ollama pull llama3.2
#
# Override OLLAMA_MODEL to use a different locally-pulled model for all three
# roles, or OLLAMA_URL if Ollama isn't on the default port.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "No .venv found — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
source .venv/bin/activate

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"

if ! curl -s -o /dev/null "${OLLAMA_URL%/v1}"; then
  echo "Can't reach Ollama at ${OLLAMA_URL%/v1} — run 'ollama serve' or 'brew services start ollama' first."
  exit 1
fi

echo "Starting Bedrock Workspace with $OLLAMA_MODEL (via Ollama) on http://localhost:8000 ..."
DEMO_MODE=false AGENT_PACK_MODE=real AIRGAP_CHECK=false \
  DATABASE_URL=sqlite:///./local_model.db \
  DIRECTOR_LLM_URL="$OLLAMA_URL" SPECIALIST_LLM_URL="$OLLAMA_URL" RUNNER_LLM_URL="$OLLAMA_URL" \
  DIRECTOR_MODEL_NAME="$OLLAMA_MODEL" SPECIALIST_MODEL_NAME="$OLLAMA_MODEL" RUNNER_MODEL_NAME="$OLLAMA_MODEL" \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

sleep 2
if command -v open >/dev/null; then open http://localhost:8000/; fi

echo "Ready. Sign in with any name (no SSO configured — that's expected)."
echo "Running. Press Ctrl+C to stop."
wait $SERVER_PID
