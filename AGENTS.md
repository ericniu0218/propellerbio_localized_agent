# Bedrock Private AI

## Product

Bedrock Private AI is a lightweight employee-facing terminal application for interacting with privately hosted open-weight language models.

The user experience should resemble Codex: the employee launches the application from a terminal, receives a clean full-screen interactive interface, types messages, sees streaming responses, can switch models, and can attach or reference files.

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
pytest              # no model or agent-pack server needed
bedrock             # run the app (expects Ollama at http://localhost:11434 by default)
```

Tests drive the real Textual app headlessly with `FakeBackend` (`tests/conftest.py`). Add tests there when changing behaviour.

## Git Workflow

- **Never push directly to `main`**: Automated agents and tools must never push commits directly to `main`.
- **Feature branches**: Always create and work on a dedicated branch (e.g., `feature/<name>` or `fix/<name>`) before committing changes.
- **Pull requests**: Pushes must target feature branches only. Changes enter `main` exclusively through reviewed pull requests.

## Current status

MVP complete: chat UI, streaming Markdown responses, `/model`, `/clear`, `/help`, `/exit`, model indicator, connection indicator.

Also built: `/digest` (`src/bedrock/context/`) - compresses any data into a
small digest plus compressed, keyword-indexed chunks, so a local model can
read a large file cheaply and still get exact detail on request. Generalised
over data types by the `SourceAdapter` protocol, the same seam idea as
`InferenceBackend`. See README's "Compressing data for the model" section.

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
local SQLite and `BEDROCK_USER` is not authentication; preserve that explicit
boundary until the workspace store moves behind verified company identity.

Not yet built (deliberately): file attachments in chat, embedding-based
retrieval (`/digest` search is keyword ranking on SQLite FTS5), authentication, conversation
persistence, packaging as a standalone executable.

## Agent-pack mode

This client can now also connect to the biotech "agent-pack" server, which
lives in this repository as `src/bedrock/server/` (a multi-agent system:
Director/Specialist/Runner, sessions, human-approval gates — see
`docs/server/ARCHITECTURE.md`; started with `bedrock-server`).
`app.py::main` dispatches to `agentpack_app.AgentPackApp` instead of the
plain `BedrockApp` when `settings.backend == "agentpack"`
(`BEDROCK_BACKEND=agentpack`). This is still the same terminal app, same
"not a website" rule, same one-input-box-and-a-scrolling-chat screen — it
just talks to a richer backend over REST + a live event stream instead of
a single OpenAI-compatible model server, and adds two small pieces of UI
for that: a quiet "what's happening now" strip and an inline
approve/reject prompt for high-stakes answers. See `README.md`'s "Two
modes" section for how to point it at the agent-pack server.

Login is still a stand-in (a `BEDROCK_USER` env var, not real SSO) —
authentication remains genuinely not built, per the line above; the
agent-pack backend's `docs/WORKFLOW.md`/`docs/DEPLOYMENT.md` describe the
intended Keycloak/OIDC login this client would eventually use instead.

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

Also built: the literature review (`src/bedrock/litreview/`), moved in from its
own repository. A scientist asks a question and gets back a report in which
every sentence cites a verified quote from a paper: `/review` beside the chat,
`/reviews` to list and re-prove past ones, `bedrock-litreview` in a shell, and
an optional FastAPI front door. **Read `src/bedrock/litreview/CLAUDE.md` before
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

Also built: the agent-pack server, moved in from its own repository with its
history (`src/bedrock/server/`, the `[server]` extra, `bedrock-server`): the
Director/Specialist/Runner orchestrator behind a REST + SSE API (what
`BEDROCK_BACKEND=agentpack` talks to), the browser scientist workspace, and the
meeting recorder, which uses `bedrock.meetings` as its engine. It runs on a
company server, never on a laptop; the terminal app does not need it installed.
Read `docs/server/README.md` and `docs/server/ARCHITECTURE.md` before changing
it, and `docs/prompts/AGENT_PACK_INTEGRATION.md` for how and why it moved.

## Development Philosophy

Build the smallest working vertical slice first. Prioritise understandable, maintainable Python over abstraction. Explain important architectural decisions in plain English in docstrings, because some contributors are learning software development.
