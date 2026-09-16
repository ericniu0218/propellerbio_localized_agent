# Bedrock Private AI

## Product

Bedrock Private AI is a lightweight employee-facing terminal application for interacting with privately hosted open-weight language models.

The user experience should resemble Claude Code: the employee launches the application from a terminal, receives a clean full-screen interactive interface, types messages, sees streaming responses, can switch models, and can attach or reference files.

## CRITICAL ARCHITECTURE RULE

THIS IS NOT A WEBSITE.

Do not use React, Next.js, Vue, Svelte, HTML, browser-based UI frameworks, Electron, localhost web pages, or frontend JavaScript frameworks. Do not create a web server to render the employee interface.

The employee application is a native Python terminal application.

## Technology Stack

- Language: Python 3.11+
- Terminal UI: Textual
- Terminal rendering: Rich
- HTTP client: httpx

Installed as the `bedrock` command (see `[project.scripts]` in `pyproject.toml`).

## Architecture

```
Employee TUI                  src/bedrock/app.py + src/bedrock/ui/
    ↓
Model API abstraction         src/bedrock/inference/base.py  (InferenceBackend protocol)
    ↓
OpenAI-compatible backend     src/bedrock/inference/openai_compatible.py
    ↓
Ollama (dev) / vLLM / SGLang (production)
    ↓
Open-weight language model
```

The UI only ever touches `InferenceBackend`. To support a new server protocol, add a class in `src/bedrock/inference/` and select it in `factory.py`; nothing in `ui/` or `app.py` changes.

## Privacy

The client must never send prompts to external AI APIs, depend on OpenAI/Anthropic/Gemini, or include analytics, tracking or telemetry. All endpoints come from configuration (`src/bedrock/config/settings.py`): environment variables override `~/.config/bedrock/config.toml`, which overrides defaults.

## Development

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest              # no server needed
bedrock             # run the app (expects Ollama at http://localhost:11434 by default)
```

Tests drive the real Textual app headlessly with `FakeBackend` (`tests/conftest.py`). Add tests there when changing behaviour. An autouse fixture redirects `XDG_DATA_HOME` to a temporary directory, so tests never write conversations into a developer's real home directory - keep it that way.

## Current status

MVP complete: chat UI, streaming Markdown responses, `/model`, `/clear`, `/help`, `/exit`, model indicator, connection indicator.

Conversation persistence: conversations are saved to `~/.local/share/bedrock/conversations/` as one JSON file each, the most recent reopens on startup, and `/history`, `/resume`, `/new` and `/forget` manage them. See `src/bedrock/storage.py`. Saving is local-only and can be switched off with `BEDROCK_SAVE_HISTORY=0`.

History loading is demand-driven: keep only the open transcript and lightweight sidebar metadata. Never load other transcripts for title backfill. Legacy files require a metadata extraction scan off the UI thread; no message snapshots remain in the store. Large chats render in 50-message pages with a button to reveal earlier messages. Idle polling checks the directory signature, with a file-fingerprint scan once per minute or on explicit history refresh. Unchanged transcripts are not saved when leaving a chat. Writers merge title/timestamp-only changes, but preserve divergent transcripts as linked copies. `/forget` deletes the whole family and retains empty markers to block stale saves; new IDs contain no start timestamp. A filesystem without advisory locking uses the atomic-directory fallback, and UI saves fail promptly on contention; see README for stale-lock recovery. Files from newer schemas open read-only.

Also built: company data-source connectors (`src/bedrock/connectors/`) -
`/digest`, `/fold` and `bedrock-biotech fold` accept `<source>://<id>` and pull
the record from a system of record instead of a manually exported file.
`/sources` lists, searches and fetches. Two generic types, `rest` (HTTP/JSON)
and `share` (export folder), configured as `[connectors.<name>]` tables.
Authorisation is left to the source system; every fetch records provenance
into a sidecar and the AlphaFold manifest.

Not yet built (deliberately): file attachments in chat, embedding-based
retrieval (`/digest` search is keyword/BM25), authentication, and
product-specific connectors (only the generic `rest` and `share` types exist).

Conversation sidebar: a clickable list on the left (`ConversationList` in `ui/widgets.py`, toggled with Ctrl+B) showing short summary titles. Titles come from `src/bedrock/titles.py`, which asks the *configured* model through `InferenceBackend` - never any other service. A conversation stays "Untitled" until it has a concrete topic: a code check for real words, a YES/NO topic request, then the title request, whose result is rejected if it is NONE or only generic words (`VAGUE_WORDS`). The module docstring explains why the YES/NO step exists (tested against llama3.2). `FakeBackend` answers these separately (`topic_checks`, `title_requests`), so `requests` in tests holds only chat turns.

Also built: local compliance and provenance. `bedrock.provenance` keeps an
append-only SQLite action journal (`BEDROCK_HISTORY_DIR`) and a read-only
catalogue over the per-run manifests; `bedrock/biotech/confidence.py`
normalises AlphaFold 2 confidence. The employee-facing half is `/runs`,
`/confidence` and `/audit` in both app modes and as `bedrock-biotech`
subcommands, plus confidence-coloured interactive PyMOL. These read local
files only, never consult a model, and work with every server switched off.
Local history is *not* an authenticated ledger or tamper-proof storage, and a
project label grants no access; read `docs/COMPLIANCE_MVP.md` and
`docs/compliance_mvp/CONTRACT.md` before changing the journal, the run
catalogue, the confidence contract or the approval flow.

Also built: native shared workspaces (`src/bedrock/workspaces/` and
`src/bedrock/ui/workspace_command.py`). Read `docs/SHARED_WORKSPACES.md` before
changing workspace storage, artifact contracts, roles, invitations,
integration keys, publish previews, folds, or events. The current vault is
local SQLite and `BEDROCK_USER` is not authentication; preserve that boundary
until the workspace store moves behind verified company identity.

Also built: local meeting transcription and model-assisted meeting notes in
`src/bedrock/meetings/`, exposed by the `bedrock-meeting` command.

Also built: `/digest` context compression, the optional biotech agent-pack backend,
and local AlphaFold 2/PyMOL adapters. The web reference client includes the real
Keycloak login button; the terminal client still uses `BEDROCK_USER` until OIDC is
wired into it.

Recall of past chats (`src/bedrock/recall.py`): transcripts are never loaded unless the employee asks, via `/recall <words>` (alias `/context`) or plain English matched by `wants_recall` ("do you remember when..."). Keyword search reads files one at a time off the UI thread and skips files without the words before parsing. Results are bounded excerpts kept in `Conversation.recalled_context`, sent as system text by `to_api`, and never written to disk - keep it that way so `/forget` stays complete.

Also built: a one-command install with no Python/venv required —
`uv tool install "git+https://github.com/Bedrock-Technical-Consulting/bedrock-private-ai.git"`,
or `scripts/install.sh`/`scripts/install.ps1` if `uv` itself isn't
installed yet either. See `INSTALL.md` for the non-engineer-facing
walkthrough. A first-run check (`src/bedrock/doctor.py`) prints
plain-language guidance — unreachable default backend, missing
AlphaFold/PyMOL config — the first time `bedrock` runs with no config
file yet; never blocks the app from starting.

Not yet built (deliberately): file attachments, embedding-based retrieval,
terminal-client authentication.

## Development Philosophy

Build the smallest working vertical slice first. Prioritise understandable, maintainable Python over abstraction. Explain important architectural decisions in plain English in docstrings, because some contributors are learning software development.
