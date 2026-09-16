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
| `/sources`       | List company data sources        |
| `/digest <src>://<id>` | Pull a record from a source and compress it |
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
Both `/fold` and `bedrock-biotech fold` also accept a company source reference
such as `registry://SEQ-00123` - see "Company data sources" below.
Direct requests work too: “Run AlphaFold 2 on `candidate.fasta`” or “Please
open the latest structure interactively in PyMOL.” Questions about the tools
remain normal agent conversations.
See [the integration guide](docs/BIOTECH_INTEGRATIONS.md) for the required
database, container, GPU, privacy, and configuration steps.

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
  connectors/
    base.py               SourceConnector protocol, source references, provenance
    rest.py               Config-shaped connector for internal HTTP/JSON APIs
    share.py              Connector for a controlled export folder
    hub.py                Builds connectors from config; fetch, search, stage
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

## Company data sources (`/sources`)

Without connectors, data reaches Bedrock only after someone exports a file and
types its path. With them, an employee names the record in the company's own
system instead:

```
/sources                              what is configured, and whether it is set up
/sources search registry kinase       find a record without knowing its id
/digest lims://plates/P42.csv         pull a record and compress it for the model
/fold registry://SEQ-00123            pull a sequence and run AlphaFold 2 (agent-pack mode)
bedrock-biotech fold registry://SEQ-00123
```

Natural requests work too: "Run AlphaFold 2 on registry://SEQ-00123".

Each system is one table in `~/.config/bedrock/config.toml`. Two general types
cover most internal systems without product-specific code:

```toml
# Anything with an HTTP/JSON API: registry, LIMS, ELN
[connectors.registry]
type = "rest"
base_url = "https://registry.internal.example.com/api/v1"
record_path = "/sequences/{id}"      # {id} is URL-encoded into one path segment
content_field = "data.fasta"         # omit to use the whole response body
filename_field = "data.name"
version_field = "data.revision"
suffix = ".fasta"                    # tells /digest what kind of data it is
search_path = "/sequences"           # optional; enables /sources search
results_field = "results"
token_env = "REGISTRY_TOKEN"         # names the env var; the token stays out of the file

# Systems that export into a controlled folder
[connectors.lims]
type = "share"
root = "/mnt/lims-exports"
extensions = [".csv", ".fasta"]      # optional allow-list
```

How it behaves:

- **Access is decided by the source system.** A 401/403 from the API, or a
  file-permission error on the share, reaches the employee as a plain "access
  refused" message. Bedrock keeps no copy of anyone's permission rules.
- **Everything fetched carries provenance** - system, record, version, who
  fetched it, when, and the SHA-256 of the exact bytes. It is saved as a
  `.provenance.json` next to the staged file in `<biotech_work_dir>/sources/`
  (owner-only permissions) and written into the AlphaFold run manifest under
  `source.system_of_record`.
- **References cannot escape.** REST ids are URL-encoded into a single path
  segment; share ids are resolved, symlinks included, and must stay under
  `root`.
- **One broken table does not break the rest.** `/sources` reports the problem
  and every other connector keeps working.

Identity caveat: the employee name sent to a REST source (`X-Bedrock-User`)
comes from `BEDROCK_USER` until company sign-in reaches this client, so it
identifies but does not authenticate. Point `rest` connectors at APIs that
check the `token_env` token, not ones that trust the header alone.

A system that needs more than configuration (paging, request signing, its own
query language) gets a class in `src/bedrock/connectors/` and one entry in
`hub.CONNECTOR_TYPES`. Nothing in `ui/`, the digest pipeline or the AlphaFold
workflow changes.

### Not built yet

- **Not tested against a real company system.** Coverage is unit tests, a
  temporary folder, and a local stand-in HTTP server. The first real
  connection is the acceptance test.
- **No product-specific connectors** (Benchling, LabWare, and so on). Whether the
  generic `rest` settings fit a given product is unknown until it is tried.
- **No direct database connection.** Sources are read through the system's own
  API or its exports. Reading its database directly would bypass the
  application's permission checks, so it is not offered.
- **Read-only.** Nothing is written back to the source system - for example, a
  predicted structure is not attached to the registry record.
- **Identity is not verified** until company sign-in reaches this client (see
  the caveat above and `NEXT_STEPS.md`).
- **Configuration and provenance live on each employee's machine.** IT has to
  distribute `config.toml`; there is no central list of connectors and no
  central audit log of fetches.
- **Staged copies are never cleaned up**, and there is no retention policy for
  `<biotech_work_dir>/sources/`.
- **Limited search.** REST search reads one page of results; share search
  matches file paths, not contents, and stops after 20,000 files.
- **`/pymol` does not accept references.** Use `/sources fetch` and open the
  saved file. `/fold` exists only in agent-pack mode; `/digest <ref>` and
  `/sources` work in both modes.

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
