#!/usr/bin/env bash
# Record a meeting on this device and ask the assistant about it afterwards.
#
# Starts the reference datacenter connector and the workspace, wired together,
# then opens the Meetings tab. Press Record, talk, press Stop; when the
# transcription finishes the meeting appears, and the assistant can answer
# questions about it by date in chat.
#
# Everything runs locally: whisper.cpp transcribes, Ollama writes the notes.
# No audio and no transcript leaves this machine.
#
# One-time setup:
#   brew install whisper-cpp ffmpeg ollama
#   brew services start ollama && ollama pull qwen2.5:14b   # notes; llama3.1:8b also works
#   mkdir -p ~/.cache/whisper-models && cd ~/.cache/whisper-models
#   curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
#   curl -LO https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin
#
# Note the VAD model is in a *different* Hugging Face repo from the speech
# model, and is not optional: without it every timestamp drifts by minutes.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-./.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="$(command -v python3)"; fi
PORT="${PORT:-8000}"
CONNECTOR_PORT="${CONNECTOR_PORT:-8080}"
DATA_DIR="${DATA_DIR:-data/demo-recorder}"

# Say what is wrong before anything is recorded. Finding out afterwards means
# the meeting is already gone.
MISSING=()
for tool in whisper-cli whisper-vad-speech-segments ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || MISSING+=("$tool")
done
MODEL_DIR="${MEETING_MODEL_DIR:-$HOME/.cache/whisper-models}"
# The multilingual model is the one to have; the English-only one from before
# the model change still runs, so it is accepted, and the server says which.
[ -f "$MODEL_DIR/ggml-large-v3-turbo.bin" ] || [ -f "$MODEL_DIR/ggml-small.en.bin" ] \
  || MISSING+=("$MODEL_DIR/ggml-large-v3-turbo.bin")
if [ ! -f "$MODEL_DIR/ggml-large-v3-turbo.bin" ] && [ -f "$MODEL_DIR/ggml-small.en.bin" ]; then
  echo "Note: only the English-only speech model is installed. Accented English and"
  echo "other languages need ggml-large-v3-turbo.bin — see this script's header."
  echo
fi
[ -f "$MODEL_DIR/ggml-silero-v5.1.2.bin" ] || MISSING+=("$MODEL_DIR/ggml-silero-v5.1.2.bin")
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "Cannot transcribe recordings yet. Missing:"
  printf '  - %s\n' "${MISSING[@]}"
  echo
  echo "See this script's header for the one-time setup."
  exit 1
fi

OLLAMA_URL="${MEETING_SUMMARY_ENDPOINT:-http://localhost:11434/v1}"
# The notes model. Qwen 2.5 14B when it is pulled: on a real recording its
# brief kept every specific and it extracts decisions and actions from a
# Mandarin transcript that llama3.1:8b returns nothing for. Otherwise the
# 8B llama, which every earlier install has.
if [ -z "${MEETING_SUMMARY_MODEL:-}" ] && curl -s -m 3 "${OLLAMA_URL%/v1}/api/tags" | grep -q '"qwen2.5:14b"'; then
  SUMMARY_MODEL="qwen2.5:14b"
else
  SUMMARY_MODEL="${MEETING_SUMMARY_MODEL:-llama3.1:8b}"
fi
if ! curl -s -m 3 "${OLLAMA_URL%/v1}/api/tags" | grep -q "\"$SUMMARY_MODEL\""; then
  echo "Warning: $SUMMARY_MODEL is not available at $OLLAMA_URL."
  echo "Recordings will still be transcribed; the notes will fall back to computed"
  echo "facts only (duration, word count). Run: ollama pull $SUMMARY_MODEL"
  echo
fi

pkill -f "contrib.datacenter_connector" 2>/dev/null || true
pkill -f "bedrock-server" 2>/dev/null || true
# And the work a previous run left behind. Killing the server does not kill the
# transcription it spawned: whisper and the VAD binary keep running, and the
# next recording then competes with them for every core. Two orphaned VAD
# processes made a forty-minute meeting look hung for the best part of an hour.
pkill -f "whisper-vad-speech-segments" 2>/dev/null || true
pkill -f "whisper-cli" 2>/dev/null || true
sleep 1

mkdir -p "$DATA_DIR"
SECRET="demo-datacenter-secret-456"

echo "[1/2] Starting the reference datacenter connector on :$CONNECTOR_PORT ..."
env DATACENTER_SIGNING_SECRET="$SECRET" TRANSCRIPT_DIR="$DATA_DIR/transcripts" \
  PYTHONPATH="$PWD" "$PYTHON" -m uvicorn contrib.datacenter_connector.main:app \
  --port "$CONNECTOR_PORT" --log-level warning &
CONNECTOR_PID=$!
trap 'kill $CONNECTOR_PID $SERVER_PID 2>/dev/null' EXIT
sleep 2

echo "[2/2] Starting Bedrock Workspace on http://localhost:$PORT ..."
env DEMO_MODE=true AGENT_PACK_MODE=mock AIRGAP_CHECK=false \
  WORKSPACE_DATA_DIR="$DATA_DIR/workspace" \
  MEETING_RECORDING_DIR="$DATA_DIR/recordings" \
  MEETING_SUMMARY_ENDPOINT="$OLLAMA_URL" \
  MEETING_SUMMARY_MODEL="$SUMMARY_MODEL" \
  MEETING_VOCABULARY="${MEETING_VOCABULARY-}" \
  DATACENTER_BASE_URL="http://127.0.0.1:$CONNECTOR_PORT" \
  DATACENTER_SIGNING_SECRET="$SECRET" \
  PYTHONPATH="$PWD" "$PYTHON" -m bedrock.server --port "$PORT" &
SERVER_PID=$!
sleep 3

if command -v open >/dev/null; then open "http://localhost:$PORT/"; fi

cat <<'NOTE'

Ready. Sign in as the demo user, open Meetings, and press Record.

  In a room            tick "This device's microphone".
  An online meeting    tick "Audio from an online meeting", then in Chrome's
                       picker choose the tab the call is in and tick
                       "Also share tab audio".

  On macOS Chrome cannot capture whole-screen audio, only a tab. So join the
  call in a browser tab, or pick a loopback device (BlackHole) as the
  microphone to capture a desktop app.

  Watch the level meter while you talk. If it stays flat, the wrong input is
  selected — that is the one failure you cannot recover afterwards.

  Names and jargon: whisper spells what it does not know phonetically, and
  writes "PREA" for "Priya". Put the words your team actually says into
  MEETING_VOCABULARY before a real meeting — it is the single highest-value
  setting here. It is empty by default on purpose: this script used to seed a
  list of gene names, and they turned up inside a recording of a club meeting
  that had never mentioned one. A wrong vocabulary is worse than none.

    MEETING_VOCABULARY="Priya, Ravi, KRAS, G12C, sotorasib" ./scripts/demo-recorder.sh

Press Ctrl+C to stop both services.
NOTE

wait $SERVER_PID
