#!/usr/bin/env bash
# The one demo script: seeded fictional data, no external services required.
# Gives real, genuinely-generated answers (via a local Ollama model) when
# Ollama is available, and falls back to the canned "DEMO ANALYSIS" text
# when it isn't — either way, this always starts and never depends on
# anything outside this machine. This is the one to lead with.
#
# For real answers: brew install ollama && brew services start ollama &&
# ollama pull llama3.1:8b (one-time — llama3.2:3b runs but is too small to
# reliably follow this orchestrator's structured output: it tends to trail
# off into a leftover prompt-template fragment instead of answering
# open-ended questions. 8b follows it correctly). Override
# OLLAMA_MODEL/OLLAMA_URL to use a different local model or port.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "No .venv found — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
source .venv/bin/activate

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

if ! curl -s -o /dev/null "${OLLAMA_URL%/v1}"; then
  echo "Ollama not reachable at ${OLLAMA_URL%/v1} — using canned demo answers instead."
  echo "For real answers: brew install ollama && brew services start ollama && ollama pull llama3.1:8b"
  REAL_MODEL_ENV=()
elif ! curl -s "${OLLAMA_URL%/v1}/api/tags" | grep -q "\"$OLLAMA_MODEL\""; then
  echo "Ollama is running, but $OLLAMA_MODEL isn't pulled — using canned demo answers instead."
  echo "For real answers: ollama pull $OLLAMA_MODEL"
  REAL_MODEL_ENV=()
else
  echo "Ollama found — using $OLLAMA_MODEL for real answers."
  REAL_MODEL_ENV=(
    AGENT_PACK_MODE=real AIRGAP_CHECK=false
    DIRECTOR_LLM_URL="$OLLAMA_URL" SPECIALIST_LLM_URL="$OLLAMA_URL" RUNNER_LLM_URL="$OLLAMA_URL"
    DIRECTOR_MODEL_NAME="$OLLAMA_MODEL" SPECIALIST_MODEL_NAME="$OLLAMA_MODEL" RUNNER_MODEL_NAME="$OLLAMA_MODEL"
  )
fi

echo "Starting Bedrock Workspace in demo mode on http://localhost:8000 ..."
env DEMO_MODE=true "${REAL_MODEL_ENV[@]+"${REAL_MODEL_ENV[@]}"}" uvicorn app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

sleep 2
if command -v open >/dev/null; then open http://localhost:8000/; fi

echo "Running. Press Ctrl+C to stop."
wait $SERVER_PID
