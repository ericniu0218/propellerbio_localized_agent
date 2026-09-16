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
| `/runs`          | Prediction runs on this machine  |
| `/confidence <run-id>` | pLDDT bands, low regions, PAE |
| `/audit`         | What this client did here        |
| `/workspace`     | List private team workspaces     |
| `/workspace help`| Show collaboration actions       |
| `/clear`         | Start a new conversation         |
| `/help`          | Show help                        |
| `/exit`          | Quit (also `Ctrl+C`)             |

`Esc` stops a response that is still streaming.

## Private shared workspaces

The Gmail reference's reusable-results feature is available here as a native
terminal workflow. It has no browser UI, Gmail connection, hosted database, or
external AI dependency. `/digest` prepares a compact local result;
`/workspace publish` shows the exact digest and bounded source extract that
would be shared; `/workspace confirm` writes that reviewed result to the
private SQLite vault. Full compressed source chunks stay in the publisher's
context store.

```text
/digest ./assay.csv
/workspace create Research team
/workspace publish a3f1c02d --title "Assay review"
/workspace confirm
/workspace items
```

Workspaces preserve owner/editor/viewer roles, single-use invitations,
immutable artifacts and parent lineage, replay-safe publication, expiring
hashed integration keys, deterministic folds, exports, and a resumable event
feed. Use `/workspace help` for the full command set; `/team` is an alias.

The default vault is local to one OS account. For several app instances to see
the same records, they must use the same `BEDROCK_WORKSPACE_DIR` on the private
host. The operator must provide a filesystem with reliable SQLite/POSIX
locking and appropriate access controls. `BEDROCK_USER` is still a temporary
identity stand-in, so workspace roles are collaboration guardrails rather than
an authentication boundary until company SSO is wired in. See
[the workspace guide](docs/SHARED_WORKSPACES.md) before provisioning a shared
vault or integration.

## AlphaFold 2 and PyMOL

The first biotech workflow is implemented as private local tool adapters:
validate a one-protein FASTA, run the official AlphaFold 2 Docker launcher,
preserve its structure and confidence artifacts, render a PNG and session with
open-source PyMOL, and record hashes and provenance in a per-run manifest.

After operator provisioning, check it with `bedrock-biotech status` and run it
with `bedrock-biotech fold candidate.fasta --project X`. In agent-pack mode the
equivalent commands are `/tools`, `/fold <fasta-path> [--project X]`, and
`/pymol [structure-path]`.
Direct requests work too: “Run AlphaFold 2 on `candidate.fasta`” or “Please
open the latest structure interactively in PyMOL.” Questions about the tools
remain normal agent conversations.
`/pymol` opens the structure in a desktop PyMOL window with the same four
pLDDT bands the saved preview uses — one fixed script and one palette serve
both, so the window and the PNG cannot disagree. A structure you point at
yourself is styled neutrally: an experimental file's B-factor column is
crystallographic displacement, not model confidence.

See [the integration guide](docs/BIOTECH_INTEGRATIONS.md) for the required
database, container, GPU, privacy, and configuration steps.

## What this machine did (`/runs`, `/confidence`, `/audit`)

Every prediction already wrote a manifest; until now nothing could read one
back. These three commands answer the questions people actually ask about
past work, from local files only — no model is consulted and no network is
needed, so they keep working when the inference server is down.

```text
/runs --project X --since 2026-07-01 --before 2026-10-01   # this quarter
/runs show 20260715T090000Z-3f2a91cc                       # settings, lineage, integrity
/runs verify 20260715T090000Z-3f2a91cc                     # re-hash every recorded artifact
/runs tag 20260715T090000Z-3f2a91cc --project X            # label an older run
/confidence 20260715T090000Z-3f2a91cc                      # pLDDT bands, low regions, PAE
/audit                                                     # every recorded action
/audit export "/approved/review/actions.jsonl"             # metadata-only JSONL
```

The same subcommands exist on the command line: `bedrock-biotech runs`,
`bedrock-biotech confidence RUN_ID`, `bedrock-biotech audit`.

Dates are UTC; `--since` includes its day and `--before` excludes it. Pages
hold 25 runs by default and 100 at most. Tagging never edits a manifest —
the new assignment is a recorded event, and a project label organises results
without granting anybody access to anything.

**`/confidence` is deliberately cautious.** It shows the four published pLDDT
bands with counts and shares, the contiguous stretches below 70 and below 50
as *input sequence positions*, and which model was actually scored — `ranked_0`
is a rank, not `model_1`. Predicted aligned error is drawn as a bounded text
heatmap of the **maximum** error per block, rows aligned and columns scored,
with a numeric ångström legend so colour is never the only signal; the full
matrix stays in its original JSON, whose path is printed. pLDDT estimates
local model confidence: it is not experimental validation or a stability
measurement, and low scores can also reflect disorder.

**`/audit` records attempts, not just successes.** Each tracked action writes
an intent before it starts and an outcome after it finishes. An intent with no
outcome means the result was never confirmed — the work may still be running —
and this client will never invent a failure to fill that gap. Approvals are
sent exactly once: an ambiguous error after dispatch is recorded as *unknown*
and the prompt stays visibly unresolved rather than claiming a decision that
may not have been recorded.

This is local history for one OS account, kept in SQLite under
`BEDROCK_HISTORY_DIR`. It is not an authenticated approval ledger and not
tamper-proof storage; the account owner can alter the file. It stores
identifiers, hashes, decisions and bounded error codes — never file contents,
prompts, approval previews or credentials.

## Meeting notes

`bedrock.meetings` turns a meeting transcript into notes, against a model you
host. Give it the rough lines you typed during the call and it keeps your
headings and fills in the detail underneath, citing timestamps — and it never
cites a timestamp that was not said. Works on a laptop with Ollama; degrades to
computed facts rather than failing when inference is unavailable. Run `bedrock-meeting` after a
Zoom local recording and it writes the transcript and the notes next to it. See
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
| `BEDROCK_WORKSPACE_DIR`| `workspace_dir` | `~/.local/share/bedrock/workspaces` |
| `BEDROCK_HISTORY_DIR`| `history_dir`     | `~/.local/share/bedrock/history` |
| `BEDROCK_BIOTECH_WORK_DIR`| `biotech_work_dir` | `~/.local/share/bedrock/biotech` |
| `BEDROCK_USER`       | `user_id`         | current local username       |
| `BEDROCK_DISPLAY_NAME`| `display_name`   | `user_id`                    |
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
    runs_command.py        `/runs`: list, detail, verification, project tagging
    confidence_view.py     `/confidence`: pLDDT bands and the PAE heatmap
    audit_command.py       `/audit`: the local action journal, and its export
    journal.py             Intent-before / outcome-after history writing
    services.py            Lazy, injectable HistoryStore + RunCatalog
    argparsing.py          argparse that reports errors instead of exiting
    rendering.py           Escaped, control-character-free terminal text
    workspace_command.py  Native picker, preview, publish, roles and export flow
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
  provenance/
    history.py            Append-only local action journal (SQLite)
    runs.py               Read-only catalogue over per-run manifests
  workspaces/
    store.py              SQLite authorization, persistence, idempotency and events
    processors.py         `/digest` corpus adapter and deterministic source fold
    validation.py         Versioned artifact contract, size bounds and checksums
    migrations/           Applied-once SQLite schema
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

No model server, agent-pack backend, AlphaFold installation or PyMOL is
needed; tests use in-memory fakes and temporary directories for all of them,
and drive both real Textual apps headlessly.

## Releasing

[The release note](docs/RELEASE.md) records the built artifacts, their
hashes, the checks run against the *installed* wheel, and how to roll back
without deleting history or rewriting manifests.
[The smoke checklist](docs/SMOKE_CHECKLIST.md) is the part that needs a real
AlphaFold and PyMOL workstation and a person to run it — nothing automated
here demonstrates a real prediction.
