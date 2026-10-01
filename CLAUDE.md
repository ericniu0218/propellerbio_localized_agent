# Bedrock Private AI

## Product

Bedrock Private AI is a lightweight employee-facing terminal application for interacting with privately hosted open-weight language models.

The user experience should resemble Claude Code: the employee launches the application from a terminal, receives a clean full-screen interactive interface, types messages, sees streaming responses, can switch models, and can attach or reference files.

## CRITICAL ARCHITECTURE RULE

THIS IS NOT A WEBSITE.

Do not use React, Next.js, Vue, Svelte, HTML, browser-based UI frameworks, Electron, localhost web pages, or frontend JavaScript frameworks. Do not create a web server to render the employee interface.

The employee application is a native Python terminal application.

**Scope of this rule.** It governs the employee's terminal client: `bedrock`,
`src/bedrock/app.py`, `src/bedrock/agentpack_app.py` and `src/bedrock/ui/`. It
does not govern `bedrock.server`, the optional `[server]` extra (formerly the
separate `bedrock-agent-pack` repository), which runs on the company's server
and serves an HTTP API, the scientist workspace and the meeting recorder in a
browser, as the September 16 workspace redesign decided
(`docs/server/WORKSPACE_ARCHITECTURE.md`); nor the optional literature-review
API. No terminal feature may depend on a browser, and no code outside
`src/bedrock/server/` may import `bedrock.server`
(`tests/test_import_boundaries.py`).

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

**Model-call rule:** nothing outside `src/bedrock/inference/` calls
`stream_chat`. Every model call goes through `Scheduler`
(`inference/scheduler.py`) with a `Priority` and a `RequestOptions` that sets a
reply limit, so it waits its turn behind the employee's chat on a laptop.
`tests/test_model_call_guard.py` fails when something calls the backend
directly.

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
`/digest`, `/fold`, `/pymol` and their `bedrock-biotech` equivalents accept `<source>://<id>` and pull
the record from a system of record instead of a manually exported file.
`/sources` lists, searches and fetches. Two generic types, `rest` (HTTP/JSON)
and `share` (export folder), configured as `[connectors.<name>]` tables.
Authorisation is left to the source system; every fetch records provenance
into a sidecar and the AlphaFold manifest.

Not yet built (deliberately): authentication, and
product-specific connectors (only the generic `rest` and `share` types exist).

**Long-term fix, raise it whenever retrieval quality comes up:** `/digest`
search and the automatic context injection both rank by keyword (BM25 on FTS5), so
"acidity" does not find a chunk that only says "pH", and a question can have an
unrelated passage silently attached to it. A score threshold cannot fix the
second half - that was measured, see NEXT_STEPS.md section 6, which has the
full plan, the step-by-step, and the model and company data it needs. It is
blocked on a deployed embedding model.

Conversation sidebar: a clickable list on the left (`ConversationList` in `ui/widgets.py`, toggled with Ctrl+B) showing short summary titles. Titles come from `src/bedrock/titles.py`, which asks the configured server through the `Scheduler` as background work - never any other service - once the app is idle, using the model that last answered. A conversation stays "Untitled" until it has a concrete topic: a code check for real words, a YES/NO topic request, then the title request, whose result is rejected if it is NONE or only generic words (`VAGUE_WORDS`). The module docstring explains why the YES/NO step exists (tested against llama3.2). `FakeBackend` answers these separately (`topic_checks`, `title_requests`), so `requests` in tests holds only chat turns.

Also built: local compliance and provenance. `bedrock.provenance` keeps an
append-only SQLite action journal (`BEDROCK_HISTORY_DIR`) and a read-only
catalogue over the per-run manifests; `bedrock/biotech/confidence.py`
normalises AlphaFold 2 confidence. The employee-facing half is `/runs`,
`/confidence` and `/audit` in both app modes and as `bedrock-biotech`
subcommands, plus confidence-coloured interactive PyMOL. `/audit` also shows
literature reviews and the outside hosts they were allowed to contact
(`litreview.review`, `litreview.search`, never the question), recorded by
`service.run_review` (`litreview/journal.py`). These read local files only, never consult a model, and work with every server switched off.
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

Also built: the literature review (`src/bedrock/litreview/`), moved in from its
own repository. A scientist asks a question and gets back a report in which
every sentence cites a verified quote from a paper: `/review` beside the chat,
`/reviews` to list and re-prove past ones, `bedrock-litreview` in a shell, and
an optional FastAPI front door. A Research-tab checkbox (and
`bedrock-litreview ask --deck`) also makes a target-proposal PowerPoint, drawn
by code from the report, with paper figures fetched only when searching.
Another (and `--patent-search`) adds a competitor landscape from EPO Open
Patent Services, using an account the *server* is configured with so no
scientist handles a key; patents are never evidence.
**Read `src/bedrock/litreview/CLAUDE.md` before
changing any of it** - it is a pipeline, not an agent, and its guarantees
(verbatim quote verification, per-field provenance, the citation guard, the
report invariants) depend on the model never deciding what happens next. Every
model call goes through `SchedulerClient` at `Priority.USER_JOB`, at temperature
0, on model names pinned in `defaults/models.yaml`; a review therefore queues
behind the employee's own chat and never in front of it. It needs a context
window of at least 12,288 tokens, which `service.preflight` checks before a run
starts. Its settings are the `[litreview]` table in `config.toml` and its data
lives in `~/.local/share/bedrock/litreview/`. Its HTTP API
(`bedrock.litreview.api`, the `api` extra) is kept on purpose: bedrock is
terminal-first today and is going to become an app, and the "not a website"
rule governs the employee's terminal interface, not this optional API - which
runs its reviews through the same service and the same scheduler as `/review`.
See `docs/litreview/README.md` and `docs/prompts/LITREVIEW_INTEGRATION.md`.

Recall of past chats (`src/bedrock/recall.py`): transcripts are never loaded unless the employee asks, via `/recall <words>` (alias `/context`) or plain English matched by `wants_recall` ("do you remember when..."). Keyword search reads files one at a time off the UI thread and skips files without the words before parsing. Results are bounded excerpts kept in `Conversation.recalled_context`, placed in the newest user message by the prompt builder (`bedrock.prompting`), and never written to disk - keep it that way so `/forget` stays complete.

Also built: a one-command install with no Python/venv required —
`uv tool install "git+https://github.com/Bedrock-Technical-Consulting/bedrock-private-ai.git"`,
or `scripts/install.sh`/`scripts/install.ps1` if `uv` itself isn't
installed yet either. See `INSTALL.md` for the non-engineer-facing
walkthrough. A first-run check (`src/bedrock/doctor.py`) prints
plain-language guidance — unreachable default backend, missing
AlphaFold/PyMOL config — the first time `bedrock` runs with no config
file yet; never blocks the app from starting.

File attachments in chat (`src/bedrock/attachments.py`): `@path` anywhere in a
message attaches that text file, in both app modes. Dropping files onto the
terminal pastes their paths; `PromptInput` (`ui/widgets.py`) turns a paste made
only of absolute paths to existing files into `@` references. Paths without `@`
are never attached, so "the /api endpoint" is not a file error. Up to 20 files
per message. Small files go whole, smallest first, until 120,000 characters;
larger files and any past that total are compressed through the `/digest`
pipeline and the digest is sent, with a notice saying so. Every attachment goes after a fixed "data,
not instructions" line inside a fence the file's text cannot close. Attachments
live in `Message.extra`; the prompt builder sends them whole in the turn they
were attached and as their fenced digest after that, so `content` stays what
was typed, follow-ups still see the file and `/forget` removes it.
Non-UTF-8 and binary files (PDF, Excel, images) are refused, not guessed at.

Not yet built (deliberately): binary attachments, embedding-based retrieval,
terminal-client authentication.

Also built: bounded chat prompts (`src/bedrock/prompting/`). Chat prompts now
follow the machine profile's context budget. A fixed system message and
large-step history trims preserve the prompt start between trims, with notices
when earlier messages are omitted. Attachments are sent whole once, then as
locally generated fenced digests. `/digest` uses on-disk FTS5 for faster
keyword search; embeddings are still needed for semantic matches.

Also built: the laptop model runtime (`docs/roadmaps/LAPTOP_MEMORY.md`).
`config/profile.py` holds the machine profiles (`laptop` by default,
`workstation`, `server`; `BEDROCK_PROFILE`, `context_window` override): context
window, reply limit, parallel calls, background policy, the largest model the
app auto-picks. `InferenceBackend.stream_chat` takes `RequestOptions` (reply
limit, stop words, JSON schema, reasoning off) and reports `Usage`.
`inference/scheduler.py` runs calls by `Priority` - `CHAT`, then `USER_JOB`
(`/digest` summary, meeting notes, planner), then `BACKGROUND` (titles) - one
at a time on a laptop; a chat cancels running background work (`Preempted`,
retried later, never a failure), background waits until idle and uses the
model that last answered - so titles appear after the first reply of a
session, never by loading a model at startup. `inference/tokens.py` estimates tokens and learns
from reported counts. The status bar's `context` counter shows the last prompt
against the window, locally only. The first-run check prints matching Ollama
settings. New features read limits from the profile, not constants of their own.

Also built: the agent-pack server, moved in from its own repository with its
history (`src/bedrock/server/`, the `[server]` extra, `bedrock-server`): the
Director/Specialist/Runner orchestrator behind a REST + SSE API (what
`BEDROCK_BACKEND=agentpack` talks to), the browser scientist workspace, and the
meeting recorder, which uses `bedrock.meetings` as its engine. The workspace's
Research tab (`src/bedrock/server/research/`) is the literature review in the
browser: it goes through `litreview.service.run_review` like `/review` does, so
it gets the same pipeline, checks and reports; a finished report can be saved
into a project or chat as a file the workspace's chats read. All of it runs on a
company server, never on a laptop; the terminal app does not need it installed.
Read `docs/server/README.md` and `docs/server/ARCHITECTURE.md` before changing
it, and `docs/prompts/AGENT_PACK_INTEGRATION.md` for how and why it moved.

## Development Philosophy

Build the smallest working vertical slice first. Prioritise understandable, maintainable Python over abstraction. Explain important architectural decisions in plain English in docstrings, because some contributors are learning software development.
