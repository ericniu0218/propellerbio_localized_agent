# Bedrock Private AI

A terminal chat client for language models hosted inside your own company.
Nothing you type ever leaves your network: the client talks only to the
inference server you configure.

```
┌ Bedrock Private AI                       model: llama3.2:3b   ● connected ┐
│                                                                            │
│ ▌ Can you show me a Python function?                                       │
│                                                                            │
│ ▌ Sure! Here is a quick example:                                           │
│ ▌   def greet(name):                                                       │
│ ▌       return f"Hello, {name}"                                            │
│                                                                            │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ Type a message, or /help for commands                                 │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
└ ^c Quit  ^l Clear ─────────────────────────────────────────────────────────┘
```

## Quick start (local development)

1. Install [Ollama](https://ollama.com) and pull a small model:

   ```
   ollama pull llama3.2
   ```

2. Install and run the client:

   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   bedrock
   ```

The status bar shows `● connected` once Ollama is reachable. If it says
`● disconnected`, start Ollama (`ollama serve`) - the client re-checks every
10 seconds.

## Commands

| Command          | What it does                     |
|------------------|----------------------------------|
| `/model`         | List models on the server        |
| `/model <name>`  | Switch model                     |
| `/clear`         | Start a new conversation         |
| `/help`          | Show help                        |
| `/exit`          | Quit (also `Ctrl+C`)             |

`Esc` stops a response that is still streaming.

## Configuration

Settings are read from, highest priority first:

1. Environment variables
2. `~/.config/bedrock/config.toml`
3. Built-in defaults (local Ollama)

| Environment variable | Config key        | Default                      |
|----------------------|-------------------|------------------------------|
| `BEDROCK_API_BASE`   | `api_base`        | `http://localhost:11434/v1`  |
| `BEDROCK_MODEL`      | `model`           | first model the server lists |
| `BEDROCK_API_KEY`    | `api_key`         | none                         |
| `BEDROCK_TIMEOUT`    | `request_timeout` | `120` seconds                |
|                      | `system_prompt`   | none                         |

Example `~/.config/bedrock/config.toml` for a company inference server:

```toml
api_base = "https://inference.internal.example.com/v1"
model = "qwen2.5-32b"
api_key = "…"
system_prompt = "You are an assistant for Example Corp employees."
```

Any server that speaks the OpenAI chat-completions wire format works:
Ollama, vLLM, SGLang, llama.cpp. "OpenAI-compatible" only describes the
message format; no traffic goes to OpenAI.

## How the code is organised

```
src/bedrock/
  app.py                 Dispatches to BedrockApp or AgentPackApp based on settings.backend
  agentpack_app.py        The Textual app for the full agent-pack system (see below)
  ui/
    widgets.py           Status bar, message widgets
    agent_panel.py        The "what's happening now" strip (agent-pack mode)
    approval.py            Inline "waiting for you" approve/reject prompt
    commands.py           Slash-command parsing (pure functions)
    styles.tcss            Layout and colours (Textual CSS)
  inference/
    base.py              InferenceBackend protocol - the seam between UI and a plain model server
    openai_compatible.py The one backend for that today (Ollama/vLLM/SGLang direct)
    factory.py           Picks a backend from settings
  agentpack/
    client.py             REST + live-event-stream client for the agent-pack backend
    friendly.py            Plain-language labels/errors (pure functions, no networking)
    models.py              AgentEvent / AgentCardState data classes
  models/
    conversation.py      Message / Conversation data classes (openai mode only)
  config/
    settings.py          Loads env vars + config file
tests/                   pytest; drives the real app with a fake backend/client
```

The important design decision: the UI never imports a concrete backend. It
only knows the `InferenceBackend` protocol (`check_connection`, `list_models`,
`stream_chat`, `close`). Swapping Ollama for the production server is a
configuration change, not a code change.

## Two modes

This client can talk to two very different things, picked by one setting:

- **`openai` mode (default)** — a single OpenAI-compatible model server
  (Ollama/vLLM/SGLang). One message in, one streamed reply out. This is
  the original design above, unchanged.
- **`agentpack` mode** — the full biotech agent-pack backend: real
  sessions, multiple AI agents working a request (shown as a quiet
  "what's happening now" strip), and inline approval prompts for
  high-stakes answers. Point it at the company's agent-pack server:

  ```
  export BEDROCK_BACKEND=agentpack
  export AGENT_PACK_API_BASE=http://ai.company.internal:8000   # or http://localhost:8000 for local dev
  export BEDROCK_USER=your-username                             # stand-in until SSO login is wired in
  bedrock
  ```

  The screen looks the same either way — one input box, a scrolling chat —
  the agent-pack mode just adds a small status strip above it while
  something is running, and an inline Approve/Needs-changes prompt when
  the AI system needs a human decision before finishing. See the
  agent-pack repo's `docs/UX.md` for the full plain-language rules this
  follows, and `docs/WORKFLOW.md` for what the backend behind it does.

## Running the tests

```
pytest
```

No model server or agent-pack backend is needed; tests use in-memory fakes
for both modes.
