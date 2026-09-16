#!/usr/bin/env bash
# Installs Bedrock Private AI with one command, no Python install or venv
# needed. See INSTALL.md for the full plain-language walkthrough — this
# comment block is for the curious, not required reading to use the script.
set -euo pipefail

REPO_URL="https://github.com/Bedrock-Technical-Consulting/bedrock-private-ai.git"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (a small tool that manages Python for you)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv's own installer puts it on PATH for *new* shells; add it to this
  # one too so the rest of this script doesn't need a terminal restart
  # partway through.
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing Bedrock Private AI..."
uv tool install --force "git+${REPO_URL}"

echo
echo "Done! Start it by typing:  bedrock"
echo "If your terminal says that command isn't found, close this window,"
echo "open a new terminal, and try again — that's normal the first time."
