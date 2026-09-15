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
| `/digest <path>` | Compress a file into the store   |
| `/digest`        | List / search what you stored    |
| `/clear`         | Start a new conversation         |
| `/help`          | Show help                        |
| `/exit`          | Quit (also `Ctrl+C`)             |

`Esc` stops a response that is still streaming.

## AlphaFold 2 and PyMOL

The first biotech workflow is implemented as private local tool adapters:
validate a one-protein FASTA, run the official AlphaFold 2 Docker launcher,
preserve its structure and confidence artifacts, render a PNG and session with
open-source PyMOL, and record hashes and provenance in a per-run manifest.

After operator provisioning, check it with `bedrock-biotech status` and run it
with `bedrock-biotech fold candidate.fasta`. In agent-pack mode the equivalent
commands are `/tools`, `/fold <fasta-path>`, and `/pymol [structure-path]`.
Direct requests work too: “Run AlphaFold 2 on `candidate.fasta`” or “Please
open the latest structure interactively in PyMOL.” Questions about the tools
remain normal agent conversations.
See [the integration guide](docs/BIOTECH_INTEGRATIONS.md) for the required
database, container, GPU, privacy, and configuration steps.

## Meeting notes

`bedrock.meetings` turns a meeting transcript into notes, against a model you
host. Give it the rough lines you typed during the call and it keeps your
headings and fills in the detail underneath, citing timestamps — and it never
cites a timestamp that was not said. Works on a laptop with Ollama; degrades to
computed facts rather than failing when inference is unavailable. See
[the meeting notes guide](docs/MEETINGS.md).

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
| `BEDROCK_CONTEXT_DIR`| `context_dir`     | `~/.local/share/bedrock/context` |
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
  context/
    ingest.py             `/digest`: compress any data into a small digest + chunks
    adapters/             One file per data type; SourceAdapter is the seam
    store.py              Compressed, content-addressed store on local disk
    digest.py             Rules-based summary, optionally improved by a model
    search.py             BM25 keyword ranking over stored chunks
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

## Compressing data for the model (`/digest`)

A 2 MB spreadsheet is roughly half a million tokens. No local model will read
that, and gzipping it does not help - decompressing just gives you half a
million tokens back. Fewer bytes on disk and fewer tokens in a prompt are two
different problems, and `/digest` goes after both.

```
/digest ./assay.csv
  Compressed `assay.csv` (tabular) -> a3f1c02d9b4e5f77
  - Storage: 2.1 MB -> 447 kB (4.6x smaller)
  - Context: ~259,000 tokens -> ~140 in the digest (1,864x cheaper to read)
  - 801 chunks indexed for exact lookup

/digest search contamination
  assay.csv - rows 17226-17250 (chunk 689)
  S017233,P42,9.91,adesai,CONTAMINATION suspected in plate P42
```

Each source is stored twice over: a **digest** small enough to hand a model on
every turn, and the **full content**, compressed and keyword-indexed, for when
an exact answer is needed. Nothing is lost - the rows are all still there.

The generalisation over data types is `SourceAdapter`, the same kind of seam as
`InferenceBackend`. Each adapter knows how its own data compresses best, which
matters more than it sounds: a table's real schema and column statistics beat
any prose description of them, and 30,000 log lines collapse to the four
patterns they actually are.

| Data | What the digest becomes |
|---|---|
| CSV / TSV | Schema, per-column type, ranges, distinct values, missing counts |
| Log files | Repeated patterns with counts, level breakdown, time span |
| JSON / JSONL | Field paths and types, with arrays collapsed to one entry |
| Source code | Signatures and docstrings (Python parsed with `ast`) |
| Text / Markdown | Size, heading outline, plus a model-written summary |

Only the last row involves a language model, and only because prose has no
structure to read. Everything else is computed, so the store works with no
inference server running at all - and if the model is unreachable when it
*would* have helped, ingest still succeeds with the computed digest.

Adding a data type is one file in `src/bedrock/context/adapters/` plus a line
in `registry.py`. Nothing in `ui/`, `app.py` or the store changes.

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
