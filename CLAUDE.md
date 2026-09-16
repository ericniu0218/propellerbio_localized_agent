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
pytest              # no model or agent-pack server needed
bedrock             # run the app (expects Ollama at http://localhost:11434 by default)
```

Tests drive the real Textual app headlessly with `FakeBackend` (`tests/conftest.py`). Add tests there when changing behaviour.

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
retrieval (`/digest` search is keyword/BM25), authentication, conversation
persistence, packaging as a standalone executable.

## Agent-pack mode

This client can now also connect to a separate project, the biotech
"agent-pack" backend (a multi-agent system: Director/Specialist/Runner,
sessions, human-approval gates — see that repo's `docs/ARCHITECTURE.md`).
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

## Development Philosophy

Build the smallest working vertical slice first. Prioritise understandable, maintainable Python over abstraction. Explain important architectural decisions in plain English in docstrings, because some contributors are learning software development.
