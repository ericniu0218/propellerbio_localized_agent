# The agent pack into bedrock-private-ai: analysis, prompt and plan

This file is three things at once:

1. **Part A** is a top-down account of what `bedrock-agent-pack` actually does
   today. It is based on reading the code, not the docs, and cites `file:line`
   wherever the two disagree.
2. **Part B** is a short account of `bedrock-private-ai`, the repository the
   agent pack is moving into.
3. **Part C onward** is the prompt for whoever carries out the merge, a person
   or a coding agent, and the plan the team reviews that work against. Give
   the executor the whole file.

- **Source:** `Bedrock-Technical-Consulting/bedrock-agent-pack`, cloned at
  `/Users/li/agent pack/bedrock-agent-pack`, `main` at `f126e57` (82 commits;
  every remote branch already merged).
- **Target:** `Bedrock-Technical-Consulting/bedrock-private-ai`, cloned at
  `/Users/li/Bedrock private AI`, `origin/main` at `d3e3342`. This is the same
  commit the source pins in `requirements.txt`, so the two agree today.
- **Result:** one repository. The agent pack becomes `bedrock.server`, an
  optional `[server]` extra started with `bedrock-server`. The employee's
  terminal app is unchanged by default, and gains what its agent-pack mode was
  missing. The agent pack's full git history is kept. The merge has no textual
  conflicts by construction, and every semantic conflict is decided in §2.
- **Branch and worktree:** `feature/agent-pack-server` at
  `/Users/li/bedrock-agentpack-merge`. Do not reuse the name
  `agent-pack-integration`: that is an old branch, already merged, that built
  the terminal client's side.

Written for: the executor and the reviewing team. Short on prose, long on
precision.

---

# Part A: what the agent pack actually does

## A.1 In one paragraph

It is a FastAPI server built for a biotech company and meant to run on one
air-gapped, on-prem GPU box. It has 8,785 lines of Python under `app/`, 3,116
lines of HTML/JS/CSS, and 188 test functions. Three products share one process:

1. **The agent API:** REST plus a Server-Sent Events stream. A
   Director/Specialist/Runner pipeline answers a message, streams progress
   events, and, for high-stakes answers, waits for a human approval. The
   terminal app's agent-pack mode (`BEDROCK_BACKEND=agentpack`) and a small
   legacy web page (`/legacy`) use it.
2. **The scientist workspace:** a browser single-page app served at `/`, with
   its own cookie-authenticated API under `/workspace/api`. It covers chats,
   projects, memory, files, meetings, tasks, agent runs and follow-ups.
3. **The meeting recorder:** the browser captures a meeting's audio (and the
   call window, to read participants' names). The server transcribes it with
   whisper.cpp, separates speakers, writes notes with a local model, and files
   the transcript with the customer's datacenter connector, where the
   `meeting_transcripts` tool can find it.

The server starts in **mock mode** by default. No model is called; replies come
from fixtures that recognise each agent's prompt text.

## A.2 The pieces, as built

| Area | Lines | What it really does | Status |
|---|---:|---|---|
| `app/orchestrator/` | 373 | `handle_message` (`loop.py:47`): Runner classifies, then routes by tier; review/retry loop; human gate | Real |
| `app/agents/` | 564 | `BaseAgent.call_model` streams tokens as `agent.token` events, records usage and enforces per-task budgets (`base.py:115-183`). The Director's `plan` is a **fixed two-node template** (`director.py:51-85`) and its `review` is a **deterministic** citation/emptiness check (`director.py:87-107`); neither calls a model | Planning is a placeholder |
| `app/models/` | 460 | `ModelClient` protocol; `HTTPModelClient` (OpenAI wire format, a new httpx client per call, never run against a real server); `MockModelClient` plus prompt-matching fixtures | Real mode untested |
| `app/events/` | 144 | In-process bus: per-session `seq`, a 500-event ring buffer, a terminal event kept outside the buffer, `session.resync` replay (`bus.py`). Every event also goes to the audit sink | Real, single process only |
| `app/api/` | 463 | `/sessions`, `/messages`, `/tasks/cancel`, `/results/{ref}`, the SSE stream, `/sessions/{id}/agents|history|audit-log`, `/approvals`, `/agent-tiers`, `/auth/*` | Real, **no authentication** |
| `app/compliance/` | 260 | Audit sink (every event becomes a row), regex PHI redaction, RBAC with an in-memory grant table, approval futures, a startup check that the internet is unreachable | Partly wired (A.7) |
| `app/db/`, `app/task_store/` | 451 | Synchronous SQLAlchemy over SQLite (Postgres by `DATABASE_URL`); `create_all`, no migrations; tables `sessions`, `history_turns`, `tasks`, `results`, `audit_log`, `token_usage`, `result_cache` | Real |
| `app/tools/` | 1,355 | Eight tools in a registry; only **three** can be reached from a chat (A.6) | Mixed |
| `app/workspace/` | 770 | Owner-scoped JSON records in a separate SQLite file (`data/workspace/{demo,live}/workspace.sqlite`), a cookie session table, an append-only audit table (triggers), uploads with text extraction, runs through a Mock or LocalLLM provider | Real, **untested** |
| `app/meetings/` | 3,043 | The recording API and pipeline (A.5) | Real |
| `app/auth/` | 177 | Keycloak OIDC: authorize, exchange the code, verify the id token against JWKS (`python-jose`) | Real, but see F8 |
| `app/static/` | 3,116 | The workspace SPA (`workspace/app.js` 1,504 lines, `recorder.js` 400) and the legacy page (`index.html`) | Real |

## A.3 One chat message, end to end

1. `POST /messages` (`api/routes_session.py:34`) generates the root task id and
   calls `asyncio.create_task(handle_message(...))` **without keeping a
   reference or handling errors** (`:47`). It returns at once.
2. `handle_message` appends the user turn to `history_turns`. The **Runner
   classifies** the message into complexity (simple/normal/hard) and one of ten
   domains (`runner.py:36`).
3. `is_high_stakes` (`router.py:30`) forces HARD for the regulatory and clinical
   domains or high-stakes keywords. It overrides both the classifier and
   `force_tier`.
4. The tiers, and the model calls each makes:

| Tier | Path | Model calls |
|---|---|---|
| simple | Runner answers, Runner formats | classify + answer + format = **3** |
| normal | Specialist runs the domain tool (if any) and drafts; Runner formats. **No citation review** (ARCHITECTURE.md §9 q8) | classify + (date-range extraction for meetings) + draft + format = **3–4** |
| hard | Fixed plan: Specialist research, reviewed by the deterministic check (up to 2 retries, then `blocked_on_human`); Director synthesis with thinking on; review again; **human gate if high-stakes**; Runner formats | classify + 1–3 drafts + synthesis + format = **4–6** |

5. Answers are stored in `results`, and clients fetch them by `result_ref`
   (`GET /results/{ref}`). Events stream on
   `GET /sessions/{id}/events` with `Last-Event-ID` resume.
6. **No conversation memory.** No prompt includes earlier turns. The Runner
   classifies `message` alone, and the Specialist's task description is
   `message`. The terminal client relies on the server for history
   (`agentpack_app.py:375`: *"The server owns session history"*). The server
   stores history but never uses it. The web workspace works around this by
   packing the last 8 messages into every message (`workspace/providers.py:57-67`).

## A.4 The scientist workspace (browser)

- Views: chats, projects, memory, meetings, settings (plus disabled knowledge
  and tools placeholders). There is an activity panel that can be resized,
  collapsed and popped out, a **Create Agent** picker (modes
  `research`/`agent` map to the Director, `analyze` to the Specialist,
  `ask`/`auto` to the classifier; `providers.py:12-15`), per-run follow-ups,
  voice enrolment, speaker renaming, live preview while recording, and retry.
- Sign-in: `/workspace/api/demo-login` (demo mode), `/dev-login` (real mode
  without SSO), or Keycloak. The session is the `bedrock_workspace` cookie
  (`routes.py:43-56`).
- A run (`POST /runs`, `routes.py:384`) builds context (`build_context`,
  `routes.py:265`): up to 8 files, 12 memories and the last 8 messages. It then
  runs `MockAIProvider` in demo mode, otherwise `LocalLLMProvider`, which
  **starts a new orchestrator session for every run**. Quick replies and agent
  runs are separate "lanes" per chat, so one of each may run at once.
- The browser **polls `GET /workspace/api/state` every 1.8 s**
  (`app.js:158`). Each call returns all of the user's records, runs (each with
  up to 80 stored events), artifacts metadata, and 200 audit rows.
- A regex quarantine (`files.py:11` `SENSITIVE`) refuses prompts and memory that
  contain identifiers, and keeps flagged uploads out of the model context.

## A.5 The meeting recorder

Browser `MediaRecorder` (plus `getDisplayMedia` for online calls) uploads to
`POST /workspace/api/meetings/recordings`. A durable job (`job_store.py`,
resumed on startup) then runs `pipeline.py`:

ffmpeg → whisper.cpp with a VAD model → diarisation (`bedrock.meetings.diarize`)
→ voice-print identification (`bedrock.meetings.enrollment`) → names read from
the call window by OCR (`screen.py`, rapidocr) → vocabulary repair
(`bedrock.meetings.vocabulary`, seeded by `vocabulary_source.py` from workspace
records) → notes (`bedrock.meetings.summarize`) → PHI redaction, access groups
and consent (`governance.py`) → delivery to the datacenter connector, signed with
HMAC and refused on public hosts (`delivery.py`, `tools/datacenter_client.py`).

Retention runs once per start (`retention.py`). Live preview is a second
recorder uploading clips (`live.py`). Semantic search uses model2vec
(`semantic.py`).

**Already moved to the target** (source commit b6565d0, target commit 89a8d6f):
`diarize`, `enrollment`, `vocabulary`, next to the target's `transcribe`,
`transcript` and `summarize`. The agent pack imports them from the pinned
package (`pipeline.py:35-51`, `routes.py:39,53`, `live.py:148`,
`tools/meeting_transcripts.py:74`). The "browser half" stayed behind on
purpose, pending a team decision on the target's "not a website" rule. **This
plan is that decision** (D3).

## A.6 Which tools a chat can actually reach

`Specialist._DOMAIN_TOOL` (`specialist.py:32-43`) is the only way a tool is
chosen.

| Tool | Reachable | Notes |
|---|---|---|
| `rag_search` | yes (internal_docs, regulatory, clinical) | keyword overlap over `data/internal_docs/*.txt`, RBAC-filtered |
| `literature_search` | yes (literature) | same index over `data/literature/*.txt` |
| `meeting_transcripts` | yes (meetings) | datacenter fetch, hybrid keyword + model2vec ranking |
| `sequence_structure` | **no** | Biopython ops; BLAST; structure prediction is a stub |
| `chemistry` | **no** | RDKit properties and similarity |
| `eln_lims` | **no** | ELN/LIMS stub plus a real instrument-CSV parser |
| `doc_template` | **no** | SOP, study report, deviation and regulatory templates |
| `data_analysis_sandbox` | **no** | subprocess with rlimits; not network-isolated |

The sandbox and BLAST call blocking `subprocess.run` inside `async def`. That
is harmless while they cannot be reached, and a stall of the whole server once
they can.

## A.7 Documented versus built

- ARCHITECTURE §8 says *"Runner-based redaction runs before anything is logged
  or handed to another agent"*. In the code, `Runner.redact` is never called,
  and redaction runs only on meeting records (`meetings/governance.py:92,105`).
  The audit log stores raw payloads.
- §6 says the Runner summarises large tool output and compacts history, and
  that a result cache exists. In the code, `Runner.summarize`,
  `extract_entities` and `compact_history`, `TaskStore.ready_tasks`,
  `Database.cache_get` and `cache_set`, and the `summarization` and
  `context_compaction` budgets are never used.
- Per-task budgets exist, but `HTTPModelClient.stream` reports **0 tokens**
  (`http_client.py:104-112`: it never asks for `stream_options.include_usage`).
  **Budgets are never enforced in real mode.**
- pgvector, Redis and `scripts/gen_ts_types.py` are described in the docs but
  do not exist. `contracts/types.ts` is maintained by hand.
- A pywebview desktop shell (`WORKSPACE_ARCHITECTURE.md:39`) does not exist.

## A.8 How the two repositories relate today

- Each depends on the other. The agent pack pins
  `bedrock-private-ai[speakers]` at `d3e3342` (CI clones it with a private
  token). The target's `bedrock.agentpack` is the client for the agent pack's
  REST/SSE API.
- `bedrock-private-ai/web-client/index.html` is a **byte-identical copy** of
  `app/static/index.html`, kept "for visibility". This is the same kind of copy
  the target deleted in 89a8d6f (`scientist-workspace/`).
- The Docker image never installs the pinned package (`Dockerfile:10-11`). So in
  Docker, `meeting_transcripts` always reports its transcript reader missing,
  and the meeting pipeline is unavailable.

## A.9 Defects found (IDs are referenced by the phases)

| ID | Severity | Defect | Where | Fixed in |
|---|---|---|---|---|
| F1 | High | Background orchestration tasks have no reference kept and no error handling. An exception (budget, server down) never produces `task.failed`, and clients wait forever | `api/routes_session.py:47` | Phase 8 |
| F2 | Medium | `/tasks/cancel` marks the task cancelled but never stops the coroutine, which can later publish `task.completed` | `api/routes_session.py:53-74` | Phase 8 |
| F3 | High | Budgets are never enforced in real mode, because streaming usage is always 0 | `models/http_client.py:104-112` | Phase 7 |
| F4 | High | The audit sink runs one synchronous SELECT plus INSERT/COMMIT **per streamed token**, on the event loop | `compliance/audit.py:15,25-41` | Phase 9 |
| F5 | High | PHI is never redacted before logging, contrary to ARCHITECTURE §8 | `compliance/audit.py`, `agents/runner.py:72` | Phase 9 |
| F6 | High | Web approvals skip the signature record (no `approval.resolved`, so no signer in `audit_log`). They also resolve an asyncio future from a worker thread (sync `def` handler); the cancel handler has the same thread problem | `workspace/routes.py:411-438` | Phases 8, 9 |
| F7 | High | After a human approves, a model **rewrites** the approved text (`format_output`). What was signed is not what is delivered | `orchestrator/loop.py:239` (also `:116,:155`) | Phase 9 |
| F8 | High | Keycloak sign-in cannot reach the workspace. The callback never issues the workspace cookie, and with SSO on, dev and demo login are refused | `api/routes_auth.py:57-77` | Phase 8 |
| F9 | High | No conversation memory in agent-pack mode (A.3 step 6) | `orchestrator/loop.py` | Phase 10 |
| F10 | Med-high | Evidence (files, memory, attachments) is packed **into the message**, so the Runner classifies up to ~128k characters; this can overflow a 32k Runner window | `workspace/providers.py:57-67`; target `agentpack_app.py:375-385` | Phase 10 |
| F11 | Medium | The normal tier skips the citation guard | `orchestrator/loop.py:131` | Phase 9 |
| F12 | Medium | `DEMO_MODE` defaults to true, so real deployments seed fictional data unless told not to | `workspace/store.py:24` | Phase 8 |
| F13 | Medium | Runtime Hugging Face download of the model2vec model breaks the air gap | `meetings/semantic.py:59,105` | Phase 8 |
| F14 | Medium | Settings are fixed at import (`config.py:100-131`); the DB engine is built at import (`db/base.py:28`); the job store opens `data/recordings/jobs.sqlite` at import (`meetings/routes.py:68`); data directories are relative to the working directory (`workspace/store.py:30`, `meetings/config.py:103`) | as listed | Phase 5 |
| F15 | Medium | Meeting notes build a new backend per meeting (never closed) and a **laptop-profile** scheduler (8k window, uncoordinated with other traffic) | `meetings/pipeline.py:420`; target `meetings/summarize.py:804-807` | Phase 7 |
| F16 | Medium | The whole run is rewritten to SQLite on **every event, including every token**; the 1.8 s `/state` poll returns everything | `workspace/providers.py:77-90`; `routes.py:146`; `app.js:158` | Phase 10 |
| F17 | Low now | Five of eight tools cannot be reached; two block the event loop once they can | A.6 | Phase 12 |
| F18 | Medium | Usernames from SSO become file-system path components | `meetings/routes.py:155,252,368,425` | Phase 8 |
| F19 | High (known) | The legacy REST API has no authentication; `approved_by` is free text | `api/*` | Not in this change (§9.1) |
| F20 | Medium | The Docker image lacks the meeting engine | `Dockerfile` | Phase 6 |

---

# Part B: the target in one page

`bedrock-private-ai` is a Textual terminal app (`bedrock`), plus the CLIs
`bedrock-biotech`, `bedrock-meeting` and `bedrock-litreview`. What matters for
this merge:

- **Model-call rule:** only `src/bedrock/inference/` calls `stream_chat`.
  Every call goes through `Scheduler` (`inference/scheduler.py`) with a
  `Priority` (`CHAT` > `USER_JOB` > `BACKGROUND`) and `RequestOptions`.
  `tests/test_model_call_guard.py` enforces it with `ast`.
  `OpenAICompatibleBackend` asks for usage (`stream_options.include_usage`),
  handles the reasoning switches for Ollama and vLLM, and reuses one httpx
  client.
- **Machine profiles** (`config/profile.py`): `laptop` (8k window, 1 call at a
  time), `workstation`, and `server` (32k window, 4 in parallel, background
  work concurrent).
- **Settings** (`config/settings.py`): environment, then
  `~/.config/bedrock/config.toml`, then defaults. Nested tables
  (`[connectors.*]`, `[litreview]`, `[agents]`) are carried as raw dicts. The
  literature review reads its own at call time, with YAML defaults shipped as
  package data and a whole-file override directory (`litreview/config.py`).
  **This merge copies that pattern.**
- **Its own "agents":** `src/bedrock/agents/` is a *local* kernel of evidence,
  plans and grants. A planner writes an inert plan, a person grants it, and a
  supervisor runs capabilities (files, shell, web, mail, calendar, sources)
  within hard budgets, journalled. It has nothing to do with
  Director/Specialist/Runner, and the two are not merged (D22).
- **Agent-pack client:** `agentpack_app.py` and `agentpack/client.py`. They use
  only `/health`, `/sessions`, `/messages` (without `force_tier`),
  `/results/{ref}`, `/approvals` and the SSE stream. The mode has no
  `/history`, `/resume`, `/recall`, `/new` or `/model`, refuses `/review`, and
  saves no conversations.
- **Everything else:** the `/digest` context store (FTS5 BM25),
  `rest`/`share` connectors, AlphaFold/PyMOL, a local action journal (`/audit`,
  intent-before and outcome-after, never content), shared workspaces (a local
  SQLite vault), the meeting engine, and the literature review. The review has
  its own optional FastAPI app (`[api]` extra), which is the precedent for an
  HTTP component in this repository.
- **Tests:** about 1,320 test functions. Autouse fixtures redirect
  `XDG_DATA_HOME` and all default data directories per test. `tests/` is a
  package.
- **CLAUDE.md, CRITICAL RULE:** "THIS IS NOT A WEBSITE" (amended by D3).

---

# Part C: the prompt

You are moving `bedrock-agent-pack` into `bedrock-private-ai`. Every design
decision is already made in §2–§4. Carry them out, and do not redesign them.
Where both codebases had a version of the same thing, §2 says which wins and
why.

**Where to work.**
- Target work happens in a git worktree at `/Users/li/bedrock-agentpack-merge`
  on the new branch `feature/agent-pack-server`, cut from `origin/main`
  (Phase 0).
- Source-side work (characterisation tests, renames, the import rewrite)
  happens in a **staging clone** at `/Users/li/bedrock-agentpack-staging`.
- Never edit, stash, reset or check out anything in
  `/Users/li/Bedrock private AI`: it holds someone's untracked files
  (`docs/prompts/LAPTOP_AGENT_*.md`).
- Never modify `/Users/li/agent pack/bedrock-agent-pack`: you only read it.

**Read these first, in this order:**
1. the target's `CLAUDE.md` and `AGENTS.md`;
2. `src/bedrock/litreview/CLAUDE.md` (for how an HTTP component is allowed);
3. the source's `docs/ARCHITECTURE.md`;
4. §2–§6 of this file.

Read a file before you edit it.

**Spend tokens on new code, not on the move.**
- Move files with the Phase 2 script (`git mv`) and rewrite imports with the
  Phase 3 `perl` script. Never retype a file that a script moves.
- Hand-edit only the files each phase lists under **Edit**.
- Do not open files you only move (the static JS/CSS, the sample corpora,
  `TEAM_PLAN.md`, `team-plan-page.html`) unless a phase says to.

**Rules that do not bend.**
- Every model call on the server goes through bedrock's `Scheduler` (Phase 7).
  `tests/test_model_call_guard.py` stays green and gets stricter.
- Every API or event change is **additive**. Laptops running an older `bedrock`
  talk to a newer server, and the reverse. New behaviour is advertised on
  `/health` and detected by the client (§6 T10).
- `contracts/types.ts` changes in the same commit as any schema change.
- Mock-mode fixtures (`bedrock/server/models/mock_fixtures.py`) match on
  prompt text. Any prompt edit updates them in the same commit (§6 T6).
- No test may read or write a real home directory or the current working
  directory.
- The terminal app keeps working with no server extras installed. Nothing
  outside `src/bedrock/server/` imports `bedrock.server` (Phase 6 adds the
  test that enforces this).
- The server runs as **one** process (§6 T9).

**Commits.** Make one commit per phase (two in Phase 0/1 staging), each after
that phase's checks pass, using the given messages. Phases 4 and 5 form one
test gate: Phase 4's merge commit is expected to fail `tests/server` until
Phase 5. Do not push or open a pull request unless the person running you asks;
the target's `AGENTS.md` forbids pushing to `main`. If the team wants smaller
PRs, cut them after Phase 6 (the behaviour-preserving move), after Phase 11,
and after Phase 12.

**When the code disagrees with this file** (a line number moved, a count
differs, a function is missing), stop and report the difference rather than
working around it. You may fix obvious mechanical slips the plan clearly meant
to cover, such as an import the rewrite missed; list them in the hand-off note.

Finish with the hand-off note in §8.

---

## 1. What moves, and what is deleted

**Moves (history kept):**
- all of `app/`, including `static/`;
- `config/*.yaml`, `data/` (sample corpora), `tests/`, `docs/`;
- `contracts/types.ts` and `contrib/datacenter_connector/` (the reference
  connector customers replace);
- the demo scripts and the meeting eval harness;
- the Dockerfile and compose file.

**Deleted during the move:**

| File | Why |
|---|---|
| `pytest.ini` | The target's `pyproject.toml` already sets `asyncio_mode = "auto"` |
| The source `.gitignore` | Its lines are merged into the target's in Phase 5 |
| `.github/workflows/test.yml` | Rewritten in Phase 6 |
| `requirements.txt` | Parked, then folded into `pyproject.toml` extras in Phase 5 and deleted |
| The accidental WAV (history only) | Removed in Phase 0; see T25 |
| `tests/test_meeting_transcripts.py::test_server_starts_when_bedrock_private_ai_is_not_installed` | The scenario can no longer happen |
| The target's `web-client/index.html` | The second copy of the legacy page (D14) |

**Behaviour that changes on purpose:** only the fixes in Phases 7–12, each its
own commit with tests. Phases 1–6 change no behaviour, except where data lives
by default (D10), which is documented for IT.

## 2. Decisions: where the codebases overlap, and which version wins

⚑ marks the five decisions confirmed by the owner on 2026-09-28:

- **D3:** the terminal-only rule governs the terminal client; the optional
  server may serve the browser workspace and recorder.
- **D10:** server data defaults to `$XDG_DATA_HOME/bedrock/server` (normally
  `~/.local/share/bedrock/server`), with private directories. Existing
  deployments retain their paths through `DATABASE_URL`, `WORKSPACE_DATA_DIR`
  and `MEETING_RECORDING_DIR`; Docker uses `/var/lib/bedrock`.
- **D16:** format answers deterministically in code and deliver approved text
  byte for byte, without a later model rewrite.
- **D21:** apply deterministic citation review to normal-tier answers in
  citation-required domains.
- **D28:** demo data follows server mode (`mock` → demo, `real` → live), while
  an explicit `DEMO_MODE` override still wins.

The owner also authorized pushing the feature branch and opening its PR.
See [verification and handoff](AGENT_PACK_INTEGRATION_VERIFICATION.md) for
validation results and remaining checks.

| # | Topic | Agent pack | Private AI | Decision | Why |
|---|---|---|---|---|---|
| D1 | Where the code lives | top-level package `app` | features live at `src/bedrock/<feature>/` | **`src/bedrock/server/`**, structure unchanged; one prefix rewrite `app.` → `bedrock.server.` | `bedrock.{agents,config,meetings,models}` already exist, so flattening would collide. A namespace keeps the rewrite mechanical, and the model-call guard walks `src/bedrock`, so it covers the server |
| D2 | History | 82 commits whose messages carry the reasoning | – | **Keep it:** renames on a staging branch, then `git merge --allow-unrelated-histories` | `git log --follow` and blame survive. Every file lands on a path the target does not have, so there are **zero textual conflicts** |
| D3 ⚑ | "THIS IS NOT A WEBSITE" | browser workspace and recorder | CRITICAL rule for the employee interface | **Scope the rule, do not drop it.** It keeps governing the terminal client. `bedrock.server` may serve the workspace, the legacy page and the recorder. Text in §4.13 | The Sept 16 redesign brief (`WORKSPACE_ARCHITECTURE.md:5-7`) chose a browser workspace for scientists. Recording needs `getUserMedia`/`getDisplayMedia`. The owners asked for *all* agent-pack features. The litreview API is the precedent. 89a8d6f said the rule should be revisited deliberately, and this is that revisit |
| D4 | Install shape | `requirements.txt`, Docker | `pip install`/`uv tool install`; extras | **Optional extras**: `server`, `postgres`, `recorder`, `chem` (plus the existing `speakers`); entry point `bedrock-server` | Laptops never pull FastAPI, SQLAlchemy or ONNX. One version number serves both |
| D5 | Talking to models | `HTTPModelClient` (no usage, a client per call, untested) | `OpenAICompatibleBackend` behind `Scheduler` | **Private AI's transport.** The agent pack's `ModelClient` protocol stays as the seam agents see, served by a new `SchedulerModelClient`. `HTTPModelClient` is deleted | Fixes F3 (budgets); tested against Ollama and vLLM behaviour; follows the model-call rule. Same pattern as litreview D2 |
| D6 | Scheduling on a shared server | none (every call runs at once) | one `Scheduler` per backend, profiles | **One `Scheduler` per distinct endpoint**, shared by every role, user and meeting job using it. Profile = `server` with that endpoint's window and `max_parallel` (§4.3). Agent calls use `CHAT`, meeting notes `USER_JOB`. The server never uses `BACKGROUND` | One GPU endpoint should be one queue. Fixes F15. Per-call usage comes from a new `on_usage` hook, never from `last_usage`, which is shared between users |
| D7 | Mock mode | `MockModelClient` bypasses any queue | `FakeBackend` in tests | **`MockBackend` (an `InferenceBackend`) behind the same scheduler path**; fixtures unchanged | Mock mode, and the demos, then exercise the production path |
| D8 | Settings | about 40 env vars; `Settings` evaluated at import | env > `config.toml` > defaults, read at call time | **Keep every agent-pack env var name.** Add a `[server]` table as a lower-precedence source, `BEDROCK_SERVER_*` names for new keys only, and load lazily. Meeting env vars stay call-time reads, as today | No deployment, script or doc breaks. Fixes the import-time half of F14 |
| D9 | Model and budget config | `config/*.yaml` at the repo root | litreview: YAML as package data plus an override directory | **`bedrock/server/defaults/{models,budgets}.yaml`**, overridable as whole files in `BEDROCK_SERVER_CONFIG_DIR` (default `~/.config/bedrock/server`) | A wheel has no repo root. Copies litreview D7 |
| D10 ⚑ | Where data lives | `./agent_pack.db`, `./data/workspace`, `./data/recordings` (relative to the working directory) | `~/.local/share/bedrock/...`, 0700 | **`BEDROCK_SERVER_DATA_DIR`** (default `$XDG_DATA_HOME/bedrock/server`; `/var/lib/bedrock` in Docker). `DATABASE_URL`, `WORKSPACE_DATA_DIR` and `MEETING_RECORDING_DIR` still override. Directory 0700 | Wheel installs and test isolation. **Existing deployments set those three variables to keep their data where it is** (DEPLOYMENT.md) |
| D11 | Meetings | the server half, pinned to the engine | the engine (`bedrock.meetings`) | Engine stays in `bedrock.meetings`. The server half moves to `bedrock.server.meetings`. The pin, and guards that only protect against bedrock being absent, go. Guards for optional third-party libraries stay | Continues 89a8d6f. There is no version skew between the two halves anymore |
| D12 | Reference datacenter connector | `contrib/datacenter_connector/` | – | **Repo-root `contrib/`, not packaged** | Customers replace it; it is not product code |
| D13 | Terminal client for the server | – | `bedrock.agentpack`, `BEDROCK_BACKEND=agentpack`, `AGENT_PACK_API_BASE` | **Names unchanged** | Installed laptops keep working |
| D14 | Legacy web page | `app/static/index.html` | an identical copy in `web-client/` | **One copy**, served by the server; `web-client/README.md` becomes a pointer | 89a8d6f already set this precedent |
| D15 | Approvals | `/approvals` writes a signature record; the web path does not (F6) | TUI `ApprovalPrompt`, journalled exactly once | **One `resolve_approval` service** used by both routes. The record is written before release, and the signer comes from the authenticated identity when there is one. Handlers are async | Two paths with different guarantees is how F6 happened |
| D16 ⚑ | Formatting the answer | a Runner model call rewrites every answer, even after human approval | – | **Deterministic formatting** (a Sources list appended in code), all tiers. The approved text is stored before the gate, and delivered byte for byte | Fixes F7 (what was signed is what is delivered) and removes one model call per request |
| D17 | Audit | a server table of every event (multi-user) | a local journal per OS account (identifiers, never content) | **Keep both**; they make different guarantees. Server change: one `agent.output` row per model call instead of one row per token, written off the event loop, with PHI redacted in `detail` | Fixes F4 and F5 without losing any record |
| D18 | PHI patterns | `files.py` `SENSITIVE` (quarantine) and `phi_redaction` `_PATTERNS` (redaction) | – | **Not merged in this change** | `_PATTERNS["DOB"]` matches any mm/dd/yyyy date; using it for quarantine would block most lab CSVs. Follow-up §9.11 |
| D19 | Conversation memory | stored, never used | `PromptBuilder` budgets history | **The server supplies history**, bounded by the drafting role's window. The workspace passes its chat turns explicitly and keeps one session per run | Fixes F9. One session per run avoids cross-lane event mix-ups (T30) |
| D20 | Evidence | packed into `message` | `attachments.prompt_block` fences files as "data, not instructions" | **Optional `context` blocks on `/messages`.** Classify `message` only; at least the normal tier when context is present; fence with bedrock's helper; cap at the profile's `attachment_share` | Fixes F10 and reuses the better fencing |
| D21 ⚑ | Citation guard on the normal tier | skipped | – | **Run the deterministic review on the normal tier** for citation-required domains | ARCHITECTURE §9 q8, the cheap option (no model call) |
| D22 | "Agents" | Director/Specialist/Runner tools act *as the server*, under RBAC | capabilities act *as the employee*, under grants | **Not unified.** Engines are shared underneath (connectors, context store, biotech) | Different trust models; merging them would weaken both |
| D23 | Internal-document search | naive keyword overlap | FTS5 BM25 (`ContextStore`) | **`ContextStore` behind `rag_search`'s `VectorIndex` seam**, plus an access-group manifest; keyword index kept for the packaged samples | Better ranking at zero new dependencies. pgvector and embeddings stay the long-term plan (§9.6) |
| D24 | ELN/LIMS | a stub | `rest`/`share` connectors | **`eln_lims` fetches through `SourceHub`** when `[server.connectors.*]` is configured; the CSV parser is kept | Turns a stub into a feature |
| D25 | Reaching tools | domain map only | – | **Deterministic detectors** (sequence, SMILES, `src://id`, template requests). The sandbox stays unreachable until it is isolated (§9.8) | Makes F17's tools usable without letting a model pick arbitrary tools |
| D26 | Sign-in | Keycloak for the legacy page only (F8) | `BEDROCK_USER` | **The callback issues the workspace cookie**; the legacy page is still supported (`next=/legacy`). Terminal OIDC stays a follow-up | Unblocks real deployments |
| D27 | Legacy API authentication | none | – | **Unchanged in this change**, documented | Requiring it would lock out every laptop until terminal sign-in exists (§9.1) |
| D28 ⚑ | Demo default | `DEMO_MODE` true unless set | – | **Default follows the server mode**: `mock` → demo, `real` → live. An explicit `DEMO_MODE` still wins | Real mode must never seed fiction (`WORKSPACE_ARCHITECTURE.md:25-26`) |
| D29 | Bundled models | model2vec downloads at runtime | – | **`bedrock-server` sets `HF_HUB_OFFLINE=1` in real mode**; the model is staged locally (DEPLOYMENT.md) | Fixes F13 |
| D30 | Processes | in-memory bus, gate, jobs | – | **One worker.** `bedrock-server` refuses `--workers` | More workers would silently break approvals and SSE (T9) |
| D31 | Python | Docker 3.12 | `>=3.11`, CI 3.12 | **CI on 3.12.** `rapidocr-onnxruntime` gets the marker `python_version < "3.13"`; screen reading reports itself unavailable on 3.13 and later | The package declares `requires_python <3.13` (checked on PyPI) |
| D32 | Tests | `tests/`, env set at import | `tests/` package with autouse isolation | **`tests/server/` subpackage** with its own conftest; collected only when FastAPI is installed; characterisation tests for the workspace API written *before* the move | The workspace API has none today |
| D33 | CI | full/minimal matrix plus a private-repo token | one job, `[dev,api]` | **Three jobs**: `core` (`dev,api`), `server-full`, `server-minimal`; no token | Proves the terminal install stays free of server dependencies, and that optional imports degrade |
| D34 | Contracts | `contracts/types.ts` by hand | – | **Stays at the repo root, by hand**, updated in the same commit as schemas | A generator is §9.13 |
| D35 | Docker | an image without the engine (F20) | – | **`docker/server/Dockerfile`** installing `.[server,postgres,recorder,speakers]` from the repo | Fixes F20 |
| D36 | Target side branches | – | 6 unmerged | **None merged here.** `meeting-notes` and `prompt-queue` are already in `main` (`git cherry` shows `-`); `wire-digest-auto-inject…` has an empty diff; `fix-chat-file-attachments` is superseded by `attachments.py`; `fix-meeting-timestamp-verification` is superseded by `meetings/citations.py`; `litreview/deployment-and-it` is **in flight** (T18) | Nothing is silently dropped or double-applied |
| D37 | Docs | `docs/` | `docs/` (with its own `MEETINGS.md`) | **`docs/server/`**; paths updated; `TEAM_PLAN.md` kept as history with a banner | No name collision; the history stays readable |

## 3. Where everything goes

```
src/bedrock/server/                 <- app/                    (structure unchanged)
  defaults/models.yaml, budgets.yaml <- config/                 (package data; §4.4 adds keys)
  sample_data/internal_docs/, literature/ <- data/              (package data; mock/demo only after Phase 12)
  static/                           <- app/static/             (package data)
  cli.py                            NEW  bedrock-server (§4.11)
  __main__.py                       NEW  python -m bedrock.server
  config.py                         EDIT lazy settings, data/config dirs (§4.2)
  db/base.py, db/database.py        EDIT lazy engine
  models/scheduled.py               NEW  SchedulerModelClient (§4.3)
  models/hub.py                     NEW  ModelHub: one Scheduler per endpoint (§4.3)
  models/mock_backend.py            NEW  MockBackend (replaces mock_client.py)
  models/http_client.py             DELETED
  orchestrator/runs.py              NEW  supervised message tasks, real cancel (§4.5)
  orchestrator/context.py           NEW  history and context rendering (§4.9)
  compliance/approvals.py           NEW  resolve_approval (§4.6)
  compliance/audit.py               EDIT aggregated, off-loop, redacted (§4.7)
  answers.py                        NEW  format_answer (§4.8)
  paths.py                          NEW  safe_component (§4.10)
  tools/intent.py                   NEW  deterministic tool detectors (Phase 12)
  tools/knowledge_index.py          NEW  ContextStoreIndex (Phase 12)
tests/server/                       <- tests/   (+ __init__.py, conftest rewritten, new tests)
docs/server/                        <- docs/ + README.md   (+ DEPLOYMENT.md NEW)
docs/prompts/AGENT_PACK_INTEGRATION.md  <- this file
scripts/server/                     <- scripts/demo-*.sh, scripts/eval/
docker/server/Dockerfile, compose.yml, env.example  <- Dockerfile, docker-compose.yml, .env.example
contracts/types.ts, contrib/datacenter_connector/   (same paths)
src/bedrock/inference/scheduler.py        EDIT on_usage (§4.1)
src/bedrock/inference/openai_compatible.py EDIT reasoning=True (§4.1)
src/bedrock/config/settings.py            EDIT carry the [server] table
src/bedrock/attachments.py                EDIT factor out fence_block() (no behaviour change)
src/bedrock/agentpack/client.py, agentpack_app.py  EDIT Phase 11
web-client/                         index.html deleted; README pointer
CLAUDE.md, AGENTS.md, README.md, INSTALL.md, NEXT_STEPS.md  EDIT
.github/workflows/test.yml          EDIT (§4.12)
tests/test_model_call_guard.py      EDIT stricter; tests/test_import_boundaries.py, tests/test_package_data.py NEW
```

## 4. New interfaces

### 4.1 Two additive changes in `bedrock.inference`

```python
# scheduler.py: Scheduler.stream and Scheduler.complete gain one keyword
async def stream(self, model, messages, *, priority, options=None,
                 on_usage: Callable[[Usage], None] | None = None) -> AsyncIterator[str]:
    ...  # unchanged, until the reply is complete:
    if on_usage is not None and usage:
        on_usage(usage[-1])      # this call's own counts; last_usage stays chat-only
```
`complete` passes `on_usage` through. The docstring should say why `last_usage`
must not be used for anything per-caller.

```python
# openai_compatible.py _payload, next to the `reasoning is False` branch
if options.reasoning is True:
    payload["chat_template_kwargs"] = {"enable_thinking": True}
```
Update `RequestOptions.reasoning`'s docstring: `True` asks for thinking where
the server supports the switch (vLLM/SGLang); Ollama ignores it. Tests go in
`tests/test_scheduler.py` and `tests/test_openai_backend.py`, following their
existing patterns.

### 4.2 Server settings (`bedrock/server/config.py`)

- `get_settings()` stays the accessor, keeps `lru_cache`, and builds on first
  call. Precedence for each field: the agent-pack env name, then the
  `BEDROCK_SERVER_*` alias, then `[server]` in `config.toml` (carried as
  `Settings.server: dict` in `bedrock.config.settings`, added to `known`),
  then the default.
- New fields:
  - `data_dir`: `BEDROCK_SERVER_DATA_DIR`, else
    `$XDG_DATA_HOME/bedrock/server`, else `~/.local/share/bedrock/server`;
    created 0700.
  - `config_dir`: `BEDROCK_SERVER_CONFIG_DIR`, else
    `~/.config/bedrock/server`.
  - `database_url`: `DATABASE_URL`, else `sqlite:///{data_dir}/agent_pack.db`.
  - `host` and `port` (CLI defaults `127.0.0.1:8000`).
- `config_file(name)` returns `config_dir/name` if it exists, else
  `importlib.resources.files("bedrock.server") / "defaults" / name`.
  `load_models_config` and `load_budgets_config` use it.
- `reset_server_state()` clears every `lru_cache` singleton, for tests:
  `get_settings`, `load_models_config`, `load_budgets_config`, `get_database`
  (and disposes the engine), `get_event_bus`, `get_task_store`,
  `get_approval_gate`, `get_model_registry`, `get_model_hub`,
  `get_tool_registry`, `get_rbac_filter`, `get_state_store`, and the meetings
  `job_store`.

### 4.3 Models: hub, scheduled client, mock backend

```python
# models/hub.py
class ModelHub:
    """One Scheduler per distinct (endpoint, api_key), shared by every role, user and job on it."""
    def scheduler_for(self, role: str) -> Scheduler: ...
    def client_for(self, role: str, *, priority: Priority = Priority.CHAT) -> SchedulerModelClient: ...
    async def aclose(self) -> None: ...   # closes every backend (lifespan shutdown)
```
- The backend is `OpenAICompatibleBackend(base_url=endpoint, api_key=...,
  timeout=defaults.request_timeout_s)` in real mode, and
  `MockBackend(token_delay_s=settings.mock_token_delay_s)` in mock mode.
- Profile: `dataclasses.replace(PROFILES["server"], name=f"server:{endpoint}",
  context_window=min(context_length of roles on this endpoint),
  max_parallel=max(max_parallel of those roles))`.
- `models.yaml` gains per-role `max_parallel` (default 8) and a `notes` role for
  meeting summaries (§4.4).

```python
# models/scheduled.py
class SchedulerModelClient:            # implements app.models.client.ModelClient unchanged
    def __init__(self, scheduler: Scheduler, config: ModelConfig, *, priority: Priority) -> None: ...
    async def stream(self, messages, *, max_tokens, thinking=False, temperature=0.2):
        options = RequestOptions(max_tokens=max_tokens, temperature=temperature,
                                 reasoning=thinking if self._config.supports_thinking_mode else None)
        # yield StreamChunk per fragment; the final chunk carries CompletionResult with the
        # server-reported counts (on_usage), or estimate_tokens() of prompt/reply if none were sent.
    async def complete(self, messages, *, max_tokens, thinking=False, temperature=0.2): ...
```
`MockBackend` implements `check_connection`, `list_models`,
`stream_chat(model, messages, options=None, on_usage=None)` (using
`mock_fixtures.fake_reply` on the last user message, word-by-word, reporting
`Usage`) and `close`. Defining `stream_chat` outside `bedrock/inference` is
allowed; only *calling* it is not. `ModelRegistry.client_for(role)` returns
`get_model_hub().client_for(role)`. `BaseAgent`'s `model_client=` injection
(used by `test_token_budgets.py`) is unchanged.

### 4.4 `defaults/models.yaml` additions

```yaml
notes:                      # meeting summaries; the old env names still work
  role: notes
  model_name: ${MEETING_SUMMARY_MODEL:-llama3.1:8b}
  endpoint: ${MEETING_SUMMARY_ENDPOINT:-http://localhost:11434/v1}
  context_length: ${MEETING_SUMMARY_CONTEXT:-8192}   # today's laptop-profile window, so notes are unchanged by default
  supports_thinking_mode: false
# and on director/specialist/runner:  max_parallel: ${DIRECTOR_MAX_PARALLEL:-8}  (etc.)
```
`ModelsConfig` gains `notes: ModelConfig | None = None`. `for_role("notes")`
falls back to building the config from the two env vars when the key is absent
from an overriding file. `_expand_env` returns strings, so coerce ints with
pydantic, which already happens for `context_length`.

### 4.5 Supervised message runs (`orchestrator/runs.py`)

```python
class MessageRuns:
    def start(self, *, session_id: str, root_task_id: str, coro) -> None
    async def cancel(self, task_id: str) -> bool      # root id, or any plan node id (via the task store's session)
```
- Keeps a reference to every task.
- On an uncaught exception it logs with `logger.exception`, publishes **one**
  `task.failed` for the root task, and marks the node FAILED if it exists.
  Messages: `BudgetExceededError` → *"This request used up its token budget
  before it finished. Try a narrower question."*; `InferenceError` with
  `unavailable=True` → *"The language model server did not answer. Try again in
  a minute."*; anything else → *"Something went wrong on the server. The error
  was logged."*
- On `CancelledError` it publishes `task.failed` with *"Cancelled by user."*
  exactly once.
- `ApprovalGate.wait_for_decision` removes its future in a `finally`.
- `/messages` uses `get_message_runs().start(...)`. `/tasks/cancel` calls
  `cancel()`, and falls back to today's mark-and-publish only when no run is
  live.

### 4.6 One approval service (`compliance/approvals.py`)

```python
async def resolve_approval(task_id: str, decision: ApprovalDecision, *,
                           signer: str, signer_verified: bool, comment: str | None) -> None
# raises NotPending (409) / UnknownTask (404); publishes approval.resolved FIRST, then gate.resolve()
```
- `ApprovalResolvedPayload` gains `signer_verified: bool = False`.
- `/approvals` calls it with `signer=req.approved_by, signer_verified=False`.
- `/workspace/api/runs/{id}/decision` becomes `async def` and calls it with
  `signer=user["id"]` and
  `signer_verified=settings.sso_enabled and not demo_mode()`. When SSO is on,
  every workspace cookie comes from Keycloak (after Phase 8), because dev login
  is refused.
- `/runs/{id}/cancel` becomes `async def`.

### 4.7 Audit sink v2 (`compliance/audit.py`)

- `agent.token` events are buffered per `(session_id, agent_id, task_id)`. On
  `done: true`, write **one** row with action `agent.output` and detail
  `{agent_id, role, task_id, text, chars, sha256}`. Terminal events flush any
  leftover buffers of that session with `incomplete: true`.
- Every other event is written as today.
- Before writing, every string value in `detail` (recursively) passes through
  `redact_phi`. When anything is redacted, `detail["redacted"] = [labels]`.
  `results` and `history_turns` keep the original text; ARCHITECTURE §8 scopes
  raw PHI to the task.
- The session's user is looked up once per session (a small dict), and the
  write runs through `await asyncio.to_thread(db.write_audit, ...)`. Order is
  preserved because `EventBus.publish` awaits its sinks one after another.

### 4.8 Deterministic answers (`answers.py`)

```python
def format_answer(content: str, citations: list[str]) -> str:
    body = content.strip()
    return body + ("\n\n**Sources**\n" + "\n".join(f"- {c}" for c in citations) if citations else "")
```
- It replaces all three `runner.format_output` calls (`loop.py:116,155,239`).
- In the hard path, the formatted candidate is stored *before* the gate
  (`candidate_ref = store.store_result(...)`), `ApprovalRequiredPayload` gains
  `result_ref: str | None = None`, and the completed task reuses
  `candidate_ref`. What was approved is what is delivered.
- `Runner.format_output` and the `_fake_format` fixture are removed.

### 4.9 History and context (`orchestrator/context.py`, API)

- `SendMessageRequest.context: list[ContextBlock] = []`, where
  `ContextBlock = {id: str, title: str, text: str, kind:
  "attachment"|"passage"|"memory"|"artifact"|"other"}`.
- `handle_message(..., context: list[ContextBlock] | None = None, history:
  list[HistoryTurn] | None = None)`. When `history` is `None`, it reads the
  session's own turns with a new bounded query,
  `Database.recent_history(session_id, limit=40)`, **excluding** the turn just
  added.
- `render_history(turns, budget_tokens)` walks newest to oldest, rendered oldest
  first, and stops at 25% of the drafting role's `context_window`, capped at
  8,000 tokens (`estimate_tokens`).
- `render_context(blocks, budget_tokens)` fences with
  `bedrock.attachments.fence_block`, a pure refactor of `prompt_block`'s loop,
  under `DATA_NOT_INSTRUCTIONS`. The budget is `int(window *
  profile.attachment_share)`, largest blocks truncated last, with a note.
- The Runner's `classify` sees the message plus the previous user turn (up to
  500 characters). It never sees context.
- If context is present and the tier would be SIMPLE, use NORMAL.
- The Specialist's `_draft` and the Director's `synthesize` append
  `"\n\nConversation so far:\n…"` and the fenced context **after** the existing
  `Task:` / `Synthesize a final answer for:` lines, so the fixture prefixes
  still match.
- Citations gain every `[source:ID]` from the context that the draft actually
  cites.
- `/health` returns `{"status","mode","features":["context_blocks",
  "server_history","force_tier","cancel","approval_result_ref"]}`.

### 4.10 Safe path components (`paths.py`)

`safe_component(user_id) -> str` returns the id unchanged if it matches
`[A-Za-z0-9._@-]{1,64}` and is not `.` or `..`. Otherwise it returns
`"u-" + sha256(id)[:16]`. Use it wherever `storage_dir / user["id"]` appears
(`meetings/routes.py:155,252,368,425`), and at matching sites found by
`grep -n 'user\["id"\]' src/bedrock/server/meetings/`.

### 4.11 `bedrock-server` (`cli.py`, also `python -m bedrock.server`)

```
bedrock-server [--host H] [--port P] [--reload]   run (always one worker; --workers is refused with a reason)
bedrock-server check                              config, data dir, model endpoints (hub check_connection),
                                                  meeting readiness_issues, SSO, air-gap result; exit 1 on problems
bedrock-server index                              (Phase 12) build the knowledge index
```
- In real mode it sets `os.environ.setdefault("HF_HUB_OFFLINE", "1")` unless
  `BEDROCK_SERVER_ALLOW_MODEL_DOWNLOAD=true`.
- It calls `uvicorn.run("bedrock.server.main:app", workers=1, ...)`.
- `main.py` moves from `on_event("startup")` to `lifespan`: resume meetings,
  run retention, the air-gap check, register the audit sink, then construct
  the hub. Shutdown calls `get_model_hub().aclose()`.

### 4.12 `pyproject.toml`

```toml
api = ["fastapi>=0.115", "uvicorn>=0.30"]                       # raised from 0.110/0.29; one FastAPI floor
server = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "sse-starlette>=2.1", "sqlalchemy>=2.0",
          "python-multipart>=0.0.9", "pypdf>=4.0", "python-jose[cryptography]>=3.3", "pydantic>=2.8"]
postgres = ["psycopg2-binary>=2.9"]
recorder = ["model2vec>=0.9", "rapidocr-onnxruntime>=1.2; python_version < '3.13'", "pillow>=10.0"]
chem = ["biopython>=1.83", "rdkit>=2024.3.1"]
server-all = ["bedrock-private-ai[server,postgres,recorder,speakers,chem]"]
# [project.scripts]  bedrock-server = "bedrock.server.cli:main"
# [tool.setuptools.package-data] bedrock += "server/static/*.html", "server/static/workspace/*",
#   "server/defaults/*.yaml", "server/sample_data/*/*.txt"
```

### 4.13 The rule amendment (CLAUDE.md and AGENTS.md, directly under the CRITICAL rule)

> **Scope.** This rule governs the employee's terminal client: `bedrock`,
> `src/bedrock/app.py`, `src/bedrock/agentpack_app.py` and `src/bedrock/ui/`.
> It does not govern `bedrock.server`, the optional `[server]` extra
> (formerly the separate `bedrock-agent-pack` repository). That component runs
> on the company's server and serves an HTTP API, the scientist workspace, and
> the meeting recorder in a browser, as the September 16 workspace redesign
> decided (`docs/server/WORKSPACE_ARCHITECTURE.md`). Nor does it govern the
> optional literature-review API. No terminal feature may depend on a browser,
> and no code outside `src/bedrock/server/` may import `bedrock.server`
> (`tests/test_import_boundaries.py`).

---

## 5. Steps

Each Bash call starts a fresh shell, so the scripts set their own variables and
use full paths.

### Phase 0: worktree, staging clone, baselines

```bash
set -euo pipefail
SRC="/Users/li/agent pack/bedrock-agent-pack"; TGT="/Users/li/Bedrock private AI"
WT=/Users/li/bedrock-agentpack-merge; STAGE=/Users/li/bedrock-agentpack-staging
PY=$(command -v python3.12 || command -v python3.11 || command -v python3); "$PY" --version
test -z "$(git -C "$SRC" status --porcelain --untracked-files=no)"
git -C "$SRC" fetch -q origin && test "$(git -C "$SRC" rev-parse main)" = "$(git -C "$SRC" rev-parse origin/main)"
git -C "$SRC" rev-parse --short main                  # plan written against f126e57
git -C "$TGT" fetch -q origin && git -C "$TGT" rev-parse --short origin/main   # written against d3e3342
git -C "$TGT" merge-base --is-ancestor origin/litreview/deployment-and-it origin/main \
  && echo "litreview/deployment-and-it: merged" || echo "litreview/deployment-and-it: NOT merged (T18)"
git -C "$TGT" worktree add -b feature/agent-pack-server "$WT" origin/main
cd "$WT" && "$PY" -m venv .venv && . .venv/bin/activate && pip install -q -e ".[dev,api]" && pytest -q | tail -1
mkdir -p docs/prompts && cp "$SRC/docs/PRIVATE_AI_INTEGRATION.md" docs/prompts/AGENT_PACK_INTEGRATION.md
git clone -q "$SRC" "$STAGE" && git -C "$STAGE" switch -q -c import/agent-pack
```

- Record `TARGET_BASELINE`.
- If `$WT` already exists on `feature/agent-pack-server` because the person
  running you created it, skip the `worktree add`.
- Use Python 3.12 if it exists: CI runs 3.12, and `rapidocr-onnxruntime` cannot
  install on 3.13. The target's `.venv` is stale (3.13, no `pydantic`); do not
  use it.

**Strip the accidental WAV from the staging history** (T25; skip only if the
owner says it may stay):

```bash
set -euo pipefail; STAGE=/Users/li/bedrock-agentpack-staging
cd "$STAGE" && python3 -m venv /tmp/gfr && /tmp/gfr/bin/pip -q install git-filter-repo
/tmp/gfr/bin/git-filter-repo --force --invert-paths --path app/static/workspace/_testaudio.wav
git switch -q import/agent-pack 2>/dev/null || git switch -q -c import/agent-pack
git rev-list --objects --all | grep -c _testaudio.wav || true      # expect 0
cp .git/filter-repo/commit-map /tmp/agentpack-commit-map.txt       # keep for the hand-off note
```

**Source baseline** (in the staging clone, against the target engine):

```bash
set -euo pipefail; STAGE=/Users/li/bedrock-agentpack-staging; WT=/Users/li/bedrock-agentpack-merge
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
cd "$STAGE" && "$PY" -m venv .venv && . .venv/bin/activate
REQ=$(mktemp); grep -vE '^(bedrock-private-ai|rapidocr-onnxruntime)' requirements.txt > "$REQ"
python -c 'import sys; sys.exit(sys.version_info >= (3, 13))' && echo 'rapidocr-onnxruntime>=1.2' >> "$REQ"
pip install -q -r "$REQ" && pip install -q -e "$WT[speakers]"
PYTHONPATH=. pytest -q | tail -1
```

Record `SOURCE_BASELINE`. If anything fails here, stop and report: the move
must start from green.

### Phase 1 (staging): characterise the workspace API before moving it

**New:** `tests/test_workspace_api.py`, in the staging clone, against the
**unmodified** code.
- Use `fastapi.testclient.TestClient(app.main.app)`.
- Set `monkeypatch.setenv("WORKSPACE_DATA_DIR", str(tmp_path))` per test.
  `DEMO_MODE` is read per call, so toggle it per test.
- Make the waits instant: `monkeypatch.setattr(app.workspace.providers.asyncio,
  "sleep", _instant)`.
- Poll `/state` with a 10-second deadline for background runs.

Cover each of these:
1. `/config` defaults (demo true, dev login false).
2. `/state` without a cookie returns 401.
3. Demo login seeds the demo projects and chats.
4. Dev login is refused in demo mode; with `DEMO_MODE=false` it works and seeds
   nothing.
5. **Owner isolation**: user B gets 404 on A's chat for GET-by-list, PATCH and
   DELETE.
6. Records: an unknown kind returns 400, sensitive memory returns 422, a record
   over 300 KB returns 413.
7. Uploads:
   - CSV is parsed and profiled;
   - invalid base64 returns 422;
   - over 20 MB returns 413;
   - sensitive text gives `scanStatus: review` and is excluded from
     `build_context`;
   - `meeting=True` creates a meeting record.
8. `/artifacts/{id}/download` returns the original bytes; `/redact` replaces
   the sensitive text.
9. A demo run completes and appends an assistant message; a second run in the
   same lane while the first is running returns 409.
10. A follow-up before completion returns 409; after completion it completes.
11. Cancel sets the run to `cancelled`.
12. A decision on a run that is not waiting returns 409.
13. Deleting a chat with an active run returns 409.
14. **The real-orchestrator path**:
    - with `DEMO_MODE=false` in mock mode, the message *"This is a final report
      for regulatory submission on our safety findings."* (it reaches the gate:
      `tests/test_force_tier.py:51-60`) makes the run go to `waiting`;
    - `POST /runs/{id}/decision` approve makes it go to `completed`.

The tests must pass unchanged against the original code.

**Commit (staging):** `Characterise the workspace API before the move`.

### Phase 2 (staging): renames only

```bash
set -euo pipefail; cd /Users/li/bedrock-agentpack-staging
mkdir -p src/bedrock && git mv app src/bedrock/server
mkdir -p src/bedrock/server/defaults src/bedrock/server/sample_data
git mv config/models.yaml config/budgets.yaml src/bedrock/server/defaults/
git mv data/internal_docs data/literature src/bedrock/server/sample_data/
git mv tests tests-server && mkdir tests && git mv tests-server tests/server
git mv docs docs-server && mkdir docs && git mv docs-server docs/server
git mv README.md docs/server/README.md
mkdir -p scripts/server docker/server
git mv scripts/demo-safe.sh scripts/demo-zoom.sh scripts/demo-recorder.sh scripts/server/
git mv scripts/eval scripts/server/eval
git mv Dockerfile docker/server/Dockerfile
git mv docker-compose.yml docker/server/compose.yml
git mv .env.example docker/server/env.example
git mv requirements.txt docker/server/requirements.agent-pack.txt
git rm -q pytest.ini .gitignore .github/workflows/test.yml
! git status --porcelain | grep -vE '^(R |D )'           # renames and the three deletions only
git commit -qm "Move agent-pack files to their bedrock-private-ai paths (renames only)"
```

### Phase 3 (staging): the import rewrite

```bash
set -euo pipefail; cd /Users/li/bedrock-agentpack-staging
find src/bedrock/server tests/server scripts/server -name '*.py' -print0 | xargs -0 perl -pi -e '
  s/^(\s*)from app(\.|\s+import\b)/$1from bedrock.server$2/;
  s/^(\s*)import app\./$1import bedrock.server./;
  s/\bapp\.main\.app\b/bedrock.server.main.app/g;
  s/(["\x27])app\.(?=(?:agents|api|auth|compliance|config|db|events|main|meetings|models|orchestrator|schemas|task_store|tools|workspace)\b)/$1bedrock.server./g;'
! grep -rnE '^\s*(from app(\.| import)|import app\.)' src tests scripts
! grep -rnE "[\"']app\.(agents|api|auth|compliance|config|db|events|main|meetings|models|orchestrator|schemas|task_store|tools|workspace)\b" src tests scripts
git commit -qam "Rewrite app.* imports as bedrock.server.* (mechanical)"
```

There are 288 import lines, 20 of them function-local. The special forms are
`import app.main`, `import app.workspace.store as store_module` and
`import app.meetings.routes as routes` (`test_meeting_delete.py:27-29,34`,
`test_meeting_recorder.py:351`), and the monkeypatch strings
`"app.meetings.delivery.DatacenterTranscriptClient"`
(`test_meeting_recorder.py:224,283,376`). All are covered. Paths in `.sh`, `.md`
and Docker files are fixed in Phases 5, 6 and 13, not here.

### Phase 4: merge into the target (no conflicts by construction)

```bash
set -euo pipefail; cd /Users/li/bedrock-agentpack-merge
git fetch -q /Users/li/bedrock-agentpack-staging import/agent-pack
git merge --no-ff --no-commit --allow-unrelated-histories FETCH_HEAD
test -z "$(git diff --name-only --diff-filter=U)"                     # no conflicts
test -z "$(git diff --cached --name-only --diff-filter=MD)"           # nothing existing modified or deleted
git commit -qm "Merge the bedrock-agent-pack history into bedrock-private-ai" \
  -m "Files arrive at their new paths (src/bedrock/server, tests/server, docs/server, ...). tests/server is importable from the next commit."
git log --follow --format=%h -- src/bedrock/server/orchestrator/loop.py | tail -1   # an early source commit
```

### Phase 5: install, run, test (behaviour unchanged except D10)

**Edit:**
- `pyproject.toml` as §4.12. Then delete
  `docker/server/requirements.agent-pack.txt`; every line of it must now be
  covered by an extra or be obsolete (the bedrock pin). List any exception.
- `src/bedrock/config/settings.py`: add `server: dict` to `Settings`, add
  `"server"` to `known`, and set `server=file_values.get("server") or {}`.
- `bedrock/server/config.py` as §4.2 (lazy; `config_file`;
  `reset_server_state`).
- `bedrock/server/db/base.py` and `database.py`: create the engine and
  `SessionLocal` in `get_engine()` on first use. `Database.__init__` calls
  `init_db()`.
- `bedrock/server/workspace/store.py:30` and `meetings/config.py:103`: the
  defaults become `get_settings().data_dir / "workspace"` and `… /
  "recordings"`. The env overrides are unchanged.
- `bedrock/server/meetings/routes.py:68`: `_job_store` becomes an
  `lru_cache`d `job_store()` (part of `reset_server_state`). Update its users.
- `tools/rag_search.py:30` and `literature_search.py:23`: default to
  `importlib.resources.files("bedrock.server") / "sample_data" / …`.
- `tools/meeting_transcripts.py:74-80` and `meetings/pipeline.py:44-51`: remove
  the guards that only protect against `bedrock` being absent. First check
  `python -c "import bedrock.meetings.summarize, bedrock.meetings.transcribe,
  bedrock.meetings.transcript"` in a `[dev]`-only venv. Keep any guard that
  protects a third-party import.
- Delete `tests/server/test_meeting_transcripts.py::test_server_starts_when_bedrock_private_ai_is_not_installed`
  (`:289-305`).
- Rewrite `tests/server/conftest.py`:

  ```python
  import importlib.util
  if importlib.util.find_spec("fastapi") is None:
      collect_ignore_glob = ["*"]          # core install: the server is not there to test
  else:
      import atexit, os, shutil, tempfile
      _data = tempfile.mkdtemp(prefix="bedrock-server-tests-"); atexit.register(shutil.rmtree, _data, True)
      for name in [n for n in os.environ if n.startswith(("KEYCLOAK_", "DATACENTER_", "MEETING_"))] + [
              "DEMO_MODE", "WORKSPACE_DATA_DIR", "DATABASE_URL", "BEDROCK_SERVER_CONFIG_DIR"]:
          os.environ.pop(name, None)       # a developer's shell must not leak into the suite
      os.environ.update(AGENT_PACK_MODE="mock", AIRGAP_CHECK="false", BEDROCK_SERVER_DATA_DIR=_data)
      # ...then the original fixtures (session_id, db), importing bedrock.server lazily inside them.
  ```

  Also add an empty `tests/server/__init__.py`.
- `.gitignore`: add `data/workspace/`, `data/recordings/`, `data/demo-recorder/`,
  `scripts/server/eval/build/`, and `agent_pack*.db`.
- Delete `web-client/index.html`. Replace `web-client/README.md` with three lines
  pointing to `src/bedrock/server/static/index.html` and `bedrock-server`.
- `CLAUDE.md` and `AGENTS.md`: the §4.13 scope paragraph, plus one line under
  "Also built" pointing at `docs/server/`.
- `bedrock/server/cli.py` and `__main__.py` (§4.11, without `index`), and
  `main.py` changed to use lifespan.

**Check:**
- `pip install -e ".[dev,api,server,recorder,speakers,chem]"` succeeds.
- `pytest -q` gives `TARGET_BASELINE + SOURCE_BASELINE − 1 + Phase 1 tests`,
  all passing.
- `bedrock-server check` runs.
- In a separate `[dev,api]` venv, `pytest -q` equals `TARGET_BASELINE`
  (server tests ignored).

**Commit:** `Package the agent-pack server as bedrock.server (the [server] extra); no behaviour change`.

### Phase 6: CI, Docker, boundaries, package data

**Edit:**
- `.github/workflows/test.yml`: a matrix of `core` (`dev,api`), `server-full`
  (`dev,api,server,recorder,speakers,chem`) and `server-minimal`
  (`dev,server`), on Python 3.12, with no secrets. For server jobs, add a step
  running `python -c "import bedrock.server.main"`, so server tests can never
  be skipped silently.
- `docker/server/Dockerfile`:
  - `FROM python:3.12-slim`;
  - apt `ffmpeg`;
  - `COPY pyproject.toml README.md ./` and `COPY src ./src`;
  - `pip install ".[server,postgres,recorder,speakers]"` (add `chem` through a
    build argument `EXTRAS`);
  - `ENV BEDROCK_SERVER_DATA_DIR=/var/lib/bedrock HF_HUB_OFFLINE=1`,
    `VOLUME /var/lib/bedrock`;
  - `CMD ["bedrock-server","--host","0.0.0.0","--port","8000"]`.

  Say in a comment that whisper.cpp binaries and models are mounted, exactly
  as before. `docker/server/compose.yml`: `build: {context: ../.., dockerfile:
  docker/server/Dockerfile}`; everything else unchanged.
- `tests/test_import_boundaries.py` (new, `ast`): no module outside
  `src/bedrock/server/` imports `bedrock.server`, and nothing under
  `src/bedrock/server/` imports `bedrock.app`, `bedrock.agentpack_app` or
  `bedrock.ui`.
- `tests/test_package_data.py` (new; if the in-flight litreview branch already
  added this file, extend it): every non-`.py` file under
  `src/bedrock/server/{static,defaults,sample_data}` matches a package-data
  glob.

**Check:** the full suite, plus `docker build -f docker/server/Dockerfile .` if
Docker is available (skip and say so if not).

**Commit:** `CI, Docker and boundary tests for the merged server`.

### Phase 7: server model calls through bedrock's inference layer (D5–D7, F3, F15)

**Edit:**
- `inference/scheduler.py` and `inference/openai_compatible.py` (§4.1).
- `server/models/`: add `hub.py`, `scheduled.py` and `mock_backend.py` (§4.3).
  `registry.py` now uses the hub. Delete `http_client.py` and `mock_client.py`,
  and update `models/__init__.py`.
- `defaults/models.yaml`, and `ModelsConfig` in `config.py` (§4.4).
- `meetings/pipeline.py:420`: pass
  `scheduler=get_model_hub().scheduler_for("notes")` and
  `model=load_models_config().for_role("notes").model_name` to
  `summarize_meeting`. Keep the error text, naming the endpoint.
- `main.py` lifespan: close the hub.
- `tests/test_model_call_guard.py`: add a second check. No `ast.Constant`
  string containing `/chat/completions` may appear outside
  `src/bedrock/inference/` (docstrings excepted). Run it and fix or justify
  every offender.

**Tests** (`tests/server/test_model_hub.py`, `tests/test_scheduler.py`,
`tests/test_openai_backend.py`):
- `on_usage` is called once with the server's counts; `last_usage` is unchanged
  for non-chat calls.
- `reasoning=True` sends `enable_thinking: true`.
- Two roles on one endpoint share one scheduler; different endpoints get
  different schedulers.
- `BudgetExceededError` fires when a fake backend reports usage above budget.
  This is the regression test for F3.
- Mock mode streams `agent.token` events through the scheduler.
- Meeting notes use the `notes` scheduler's window.

All previously passing tests still pass.

**Commit:** `Server model calls go through bedrock's Scheduler (fixes unenforced budgets)`.

### Phase 8: runtime correctness and security (F1, F2, F6 threads, F8, F12, F13, F18)

**Edit:**
- `orchestrator/runs.py` (§4.5), wired into `routes_session.py`.
- `workspace/routes.py`: `cancel` and `decide` become `async def`.
- `api/routes_auth.py` callback:
  - build the `RedirectResponse` to `/`, call `issue_session(response,
    user_id, user.display_name)`, and return it;
  - `/auth/login?next=/legacy` stores `next` alongside the state
    (`StateStore.issue(state, next)`), and then the callback redirects to
    `/legacy?session_id=…&user_id=…&user_name=…` exactly as today.
- `workspace/store.py:23-24`: `demo_mode()` reads `DEMO_MODE` if set, else
  returns `get_settings().mode != "real"` (D28).
- `cli.py`: `HF_HUB_OFFLINE` (D29). Also stop `semantic.load()` from reaching
  the network in real mode: if `MEETING_TEXT_MODEL` is not a local directory
  and `HF_HUB_OFFLINE=1`, the load fails fast and falls back to keyword
  ranking, as its guard already does.
- `paths.py` (§4.10), applied in `meetings/routes.py`.

**Tests:**
- An agent that raises produces exactly one `task.failed`, with the friendly
  text.
- Cancel stops a run that is waiting at the approval gate, and no
  `task.completed` follows.
- The workspace decide/cancel handlers run on the event loop: assert
  `asyncio.get_running_loop()` inside a patched `gate.resolve`.
- SSO callback, with a fake `OIDCClient` (monkeypatch
  `_require_oidc_client`): the cookie is set and `/workspace/api/state`
  returns 200; with `next=/legacy`, the old query parameters are kept.
- `demo_mode()` defaults by mode.
- `safe_component("../x")` never escapes the folder.

**Commit:** `Fix server runtime defects found in the merge review (F1, F2, F6, F8, F12, F13, F18)`.

### Phase 9: compliance fixes (F4, F5, F6 signature, F7, F11)

**Edit:**
- `compliance/approvals.py` (§4.6), used by `routes_approval.py` and
  `workspace/routes.py::decide`.
- `compliance/audit.py` (§4.7).
- `answers.py` (§4.8), with `loop.py:116,155,239` updated and the gate storing
  the candidate first.
- `loop.py::_handle_normal`: for domains in
  `director._CITATION_REQUIRED_DOMAINS`, use `_run_with_review_loop(...)` with a
  `Director(session_id=...)` (D21). Other domains are unchanged.
- `mock_fixtures.py`: remove `_fake_format`. If the normal-tier guard now blocks
  a demo question, make the mock draft cite, rather than weakening the guard.
- `schemas/events.py` gets the additive fields; mirror them in
  `contracts/types.ts`.

**Tests:**
- A workspace approval writes an `approval.resolved` audit row with
  `approved_by=<user id>`.
- One streamed answer gives exactly one `agent.output` row whose `sha256`
  matches the text.
- An SSN in a prompt is `[REDACTED:SSN]` in `audit_log.detail` and intact in
  `results`.
- The approved text equals the delivered text byte for byte.
- A normal-tier literature answer without citations is retried, then blocked.

Update `test_audit_logging.py` to assert that tokens are aggregated. Its
method-set test is unchanged.

**Commit:** `Approvals, audit and answer integrity (F4, F5, F6, F7, F11)`.

### Phase 10: conversation memory and evidence (F9, F10, F16)

**Edit:**
- `orchestrator/context.py`, `schemas/api.py` (`ContextBlock`, the `context`
  field), `loop.py`, `agents/specialist.py`, `agents/director.py`,
  `runner.classify`, and `/health` features (§4.9).
- `database.py::recent_history`.
- `bedrock/attachments.py`: factor out `fence_block`, with no behaviour change.
- `workspace/providers.py` and `workspace/routes.py::execute_followup`:
  - pass the artifacts and memory as `context` and the chat messages as
    `history`; `message` becomes `run["prompt"]` only;
  - keep one orchestrator session per run (T30);
  - persist the run only on non-token events;
  - stop storing `events` on runs.
- `workspace/routes.py::state`: drop any `events` still present on old runs
  from the response.

**Tests:**
- A follow-up question sees the previous answer (the fixture echoes the
  history block's presence).
- The classify prompt contains no context text.
- Context with a SIMPLE classification runs NORMAL.
- Over-budget context is truncated with a note.
- A run is written O(events without tokens) times (count `store.put` calls).
- `/health` lists the features.
- An old-style `/messages` (no `context`) behaves exactly as before.

**Commit:** `Conversation memory and separate evidence for the agent server`.

### Phase 11: the terminal client catches up (agent-pack mode)

**Edit:**
- `bedrock/agentpack/client.py`:
  - `features()` reads `/health` once per connection;
  - `send_message(session_id, text, *, force_tier=None, context=None)` sends
    only the fields that are set;
  - add `cancel_task(task_id)`.
- `agentpack_app.py`:
  - send attachments and auto-added passages as `context` when the server
    advertises `context_blocks`; otherwise send exactly what is sent today;
  - add `/tier [quick|standard|deep|auto]` (names from `/agent-tiers`; shown in
    the status bar; per window);
  - add `/cancel` (and Esc while an agent task runs, if Esc is not already
    bound in this mode) → `cancel_task`;
  - the approval prompt offers "full text" through `result_ref` when present;
  - update `AGENTPACK_HELP_TEXT`.
- `README.md` "Two modes": mention `/tier` and `/cancel`.

**Tests** (in `tests/test_agentpack_app.py` and `test_agentpack_reconnect.py`,
with the fake client):
- Feature detection falls back against an old server.
- `force_tier` is sent.
- Cancel is sent once.
- Context blocks carry the attachment text, and the message no longer does.
- The full approval text is fetched by reference.

**Commit:** `Agent-pack mode: tier choice, cancel, structured context, full approval text`.

### Phase 12 (separable PR): Private AI engines behind the agent-pack tools (D23–D25, F17)

**Edit:**
- `tools/knowledge_index.py`: `ContextStoreIndex(VectorIndex)`.
  - It uses a `ContextStore(data_dir / "knowledge-index")` and ingests
    `knowledge_dir` (`BEDROCK_SERVER_KNOWLEDGE_DIR` / `[server].knowledge_dir`)
    through `bedrock.context.ingest`.
  - `access.toml` in the corpus maps globs to `access_groups` (default
    `["general"]`).
  - It maps `SearchHit` to `DocChunk(doc_id=stem, section=f"chunk-{n}")`.
  - `rag_search` uses it when an index exists; otherwise it uses the keyword
    index. In real mode it never reads the packaged samples.
  - `bedrock-server index` builds or refreshes the index.
- `tools/lab_connectors.py`: `get_experiment` and `get_sample` accept an
  optional `source` (`name://id`) and fetch through
  `SourceHub(build_connectors(server_table["connectors"]))`. The citation is
  `system://record@version`. With no connectors, it keeps today's stub text.
- `tools/intent.py` — detectors return `(tool, args)` or `None`:
  - a configured `name://id` → `eln_lims`;
  - a FASTA header, or a run of ≥ 30 nucleotides or ≥ 20 amino acids **plus**
    an operation word (translate, transcribe, reverse complement, blast) →
    `sequence_structure`;
  - SMILES tokens that RDKit parses → `chemistry` (similarity when two are
    found);
  - "draft/write a … SOP | study report | deviation report | regulatory
    section" → `doc_template`.

  `Specialist._maybe_call_tool` consults the detectors when the domain map
  gives `None`. Never the sandbox.
- `sequence_tools.py` and `sandbox.py`: wrap `subprocess.run` in
  `await asyncio.to_thread(...)`.

**Tests:** each detector, positive and negative; RBAC filtering through the new
index; connector fetch through a fake connector; the event loop is not blocked
(a concurrent coroutine keeps running during a slow fake BLAST).

**Commit:** `Agent tools reach bedrock's context store, connectors and the unused biotech tools`.

### Phase 13: documents

**Edit:**
- `docs/server/*.md`: every `app/…` path becomes `src/bedrock/server/…`;
  `uvicorn app.main:app` becomes `bedrock-server`; `requirements.txt` becomes
  `pip install "bedrock-private-ai[server]"`.
- `docs/server/README.md`: its quick start becomes `bedrock-server`.
- `docs/server/TEAM_PLAN.md`: a banner saying it is the pre-merge plan, and that
  repo URLs and paths refer to the old layout.
- `scripts/server/*.sh`: `cd` to the repo root, use `.venv` of this repo,
  `bedrock-server --port`, and `pkill -f bedrock-server`.
- `scripts/server/eval/score.py`: update its usage docstring.
- **New `docs/server/DEPLOYMENT.md`**, for IT:
  - install (`pip`/Docker) and the extras;
  - every env var (the Appendix table), the `[server]` table, and the data
    directory;
  - **migrating an existing deployment**: set `DATABASE_URL`,
    `WORKSPACE_DATA_DIR`, `MEETING_RECORDING_DIR` and `DEMO_MODE` to the old
    values;
  - one worker; Keycloak redirect URIs (`/auth/callback`);
  - staging models offline (whisper and VAD in `MEETING_MODEL_DIR`;
    `potion-base-8M` downloaded once elsewhere and pointed to by
    `MEETING_TEXT_MODEL`);
  - the air-gap check; the network placement (the legacy API is
    unauthenticated, F19);
  - `bedrock-server check`.
- Root `README.md`: a short "The server (`bedrock-server`)" section, and the
  project layout.
- `AGENTS.md` "Agent-pack mode": the server now lives here.
- `NEXT_STEPS.md`: replace "Where the actual AI system lives" (it is here now);
  mark items 1–2 against the new paths.
- `INSTALL.md`: one line saying employees never install the server.
- `docs/server/ARCHITECTURE.md` §6/§8/§9: correct the claims A.7 lists as not
  built, or mark them as planned.

**Check:** `grep -rn "app/\|app\.main\|requirements\.txt\|bedrock-agent-pack" docs/server README.md AGENTS.md CLAUDE.md NEXT_STEPS.md scripts/server docker/server`
matches only deliberate historical mentions.

**Commit:** `Docs: the server in bedrock-private-ai`.

### Phase 14: verify

1. Run `pytest -q` in full, in the full venv, in a `[dev,server]` venv, and in a
   `[dev,api]` venv. Report all three counts.
2. These greps return nothing in `src/` and `tests/`:
   - `^\s*(from|import) app\b`
   - `http_client`
   - `HTTPModelClient`
   - `requirements.txt`
3. **Untouched-file check.** Every file that differs from the staging Phase 3
   commit must be named in some phase's **Edit** list:

   ```bash
   cd /Users/li/bedrock-agentpack-merge && git diff --name-only "$(git -C /Users/li/bedrock-agentpack-staging rev-parse import/agent-pack)" HEAD -- src/bedrock/server tests/server
   ```

   Explain each file in the hand-off note.
4. Build the wheel into a temporary directory and install it (`[server]`) into a
   throwaway venv, with `HOME` set to a scratch directory. Then:
   - `bedrock-server check`;
   - start `bedrock-server --port 8765` in mock mode;
   - `curl /health` lists the features;
   - `POST /sessions`, then `POST /messages` with a regulatory message;
   - read the SSE stream until `approval.required`, then `POST /approvals`, and
     see `task.completed`;
   - the `/results/{ref}` text equals the stored candidate.
5. Point the terminal client at it (`BEDROCK_BACKEND=agentpack
   AGENT_PACK_API_BASE=http://127.0.0.1:8765 bedrock`). Ask a question, then a
   follow-up that needs the first answer, then use `/tier deep`, then approve
   one high-stakes answer.
6. In a browser, check `http://127.0.0.1:8765/`:
   - demo login;
   - upload a CSV;
   - one Ask and one Create Agent in parallel;
   - approve a regulatory run;
   - `/state` responses carry no `events`;
   - check the recorder's readiness message.

   If no browser can be driven, list these checks for a person.
7. If Ollama is available: set `AGENT_PACK_MODE=real` with all three role URLs
   at `http://localhost:11434/v1` and small models (`qwen3:8b` / `llama3.2:3b`).
   Then check:
   - one answer per tier;
   - `usage.updated` shows non-zero tokens;
   - a tiny `max_tokens_per_task` triggers the budget message.

   Skip this and say so if Ollama is not available.

## 6. Tricky parts, collected

| # | Trap | Handled |
|---|---|---|
| T1 | Unrelated-history merge: any path overlap would conflict. The only overlaps were `README.md`, `.gitignore`, the CI file and `docs/MEETINGS.md`; all are renamed or deleted in staging | Phases 2, 4 |
| T2 | `bedrock.{agents,config,meetings,models}` collide with `app.*` | D1 |
| T3 | Import-time state: settings defaults, the engine, the job store and `lru_cache` singletons. Env must be set **before** import, and `reset_server_state()` is needed for tests that change modes | Phase 5, §4.2 |
| T4 | Working-directory-relative data paths wrote into the repo during tests; the target's autouse fixtures do not cover module singletons | Phase 5 conftest, D10 |
| T5 | Guards against "bedrock not installed" are dead after the merge; guards for third-party libraries are not | Phase 5 |
| T6 | Mock fixtures match prompt prefixes (`Respond with JSON: {"complexity"`, `Task:`, `Synthesize a final answer for:`, `Extract the date/time range`). Keep them first; update fixtures in the same commit | §4.9, Phase 9 |
| T7 | `Scheduler.last_usage` is per scheduler, not per caller. On a shared server it belongs to whoever finished last, so never use it for budgets | §4.1 |
| T8 | Async hazards: sync `def` handlers touching asyncio objects; blocking subprocesses; sync DB writes in the audit sink; tasks without references | Phases 8, 9, 12 |
| T9 | More than one uvicorn worker splits the bus, the gate and the jobs across processes. Approvals then hang and SSE misses events, silently | D30, cli.py |
| T10 | Version skew: laptop TUIs update on their own schedule, and the server is deployed by IT. Changes are additive, advertised on `/health`, and the client falls back | Phases 10, 11 |
| T11 | `contracts/types.ts` drifts silently | Rule in Part C |
| T12 | Air gap: model2vec downloads at runtime; the air-gap check only runs in real mode; datacenter host guard; whisper models must be staged | D29, DEPLOYMENT.md |
| T13 | Python 3.13: `rapidocr-onnxruntime` requires <3.13; the target's dev venv is 3.13 | D31 |
| T14 | The SSO redirect lands on the workspace with no cookie. The cookie is `SameSite=Lax`, which is still set on top-level redirects, so the fix works | Phase 8 |
| T15 | Usernames from SSO become folder names | §4.10 |
| T16 | Changing the `DEMO_MODE` default changes which store a deployment reads (`demo/` vs `live/`) | D28, DEPLOYMENT.md |
| T17 | Changing the `DATABASE_URL` default loses an existing SQLite file unless it is set explicitly | D10, DEPLOYMENT.md |
| T18 | `origin/litreview/deployment-and-it` (Eric Niu, 26 Sep, +2,047 lines) is unmerged and edits `scheduler.py` (the idle clock), `profile.py`, `app.py`, `local_commands.py` and `litreview/*`, and adds `tests/test_package_data.py`. If it lands first, rebase onto it. On conflict, keep **both**: its `_last_foreground_finished` idle clock and this plan's `on_usage` touch different hunks. Extend its package-data test rather than adding a second one | Phase 0, Phase 6 |
| T19 | The user's target checkout holds untracked files; never touch it | Part C |
| T20 | Missing package data only fails after a wheel install | Phase 6 test, Phase 14 step 4 |
| T21 | `TestClient` plus background tasks: runs continue on the client's portal loop. Poll with deadlines, and make provider sleeps instant | Phase 1 |
| T22 | Behaviour changes a demo may notice: no model formatting pass; the normal-tier citation guard; the demo default | D16, D21, D28 |
| T23 | Part 11 wording: the audit trail is better, but the legacy API is still unauthenticated. Docs must not claim compliance | Phase 13 |
| T24 | Two "workspaces" (`bedrock.workspaces` is the local vault; `bedrock.server.workspace` is the browser app) and two "agents" (the kernel and the orchestrator). Name them explicitly in the docs; do not merge them here | D22, §9.7 |
| T25 | Importing history imports every blob ever committed. The only large one is a 900 KB WAV recorded for development (removed in c2dd48d), stripped in staging. SHAs of imported commits then differ from the GitHub repo; keep the commit map | Phase 0 |
| T26 | Loggers become `bedrock.server.*` (`__name__`); `main.py` keeps `"agent_pack"`. No test depends on logger names | none needed |
| T27 | The import regex must not touch the variable `app` in `main.py`. It is anchored to import lines and quoted known subpackages, and Phase 3's greps verify it | Phase 3 |
| T28 | The Docker image grows: it now includes the ONNX speaker and OCR libraries that it silently lacked | D35 |
| T29 | `Database.get_history` loads every turn; long sessions need the bounded query | §4.9 |
| T30 | The workspace runs an ask lane and an agent lane on one chat at once. Sharing one orchestrator session would attribute one lane's `task.completed` to the other, so keep one session per run and pass history explicitly | D19 |

## 7. Done means

- The full suite passes with `TARGET_BASELINE + SOURCE_BASELINE − 1 + new
  tests`. Report every number. The `[dev,api]` venv passes with the server
  tests ignored.
- `tests/test_model_call_guard.py` (stricter), `tests/test_import_boundaries.py`
  and `tests/test_package_data.py` pass.
- `git log --follow` on a moved file reaches the source's history.
- Phase 4's merge had no conflicts and modified no existing file.
- A wheel install serves mock-mode chat, SSE and approvals end to end, and
  `bedrock-server check` passes.
- The terminal client in agent-pack mode keeps a conversation, chooses a tier,
  cancels, and shows a full approval text. Against an old server it behaves
  exactly as before.
- F1–F18 and F20 each have a regression test.
- The untouched-file check is explained.
- CLAUDE.md and AGENTS.md carry the scope paragraph, and DEPLOYMENT.md exists.

## 8. Hand-off note

End with:
- the files added, changed, moved and deleted (a summary of each phase's
  `git diff --stat`);
- the test counts: both baselines, after Phase 5, and final, in all three
  venvs;
- the WAV commit map location, or "not stripped" with the owner's reason;
- every deviation from this plan and why;
- every decision the plan did not cover;
- the Phase 14 step 5–7 results, or "not run" with the reason;
- the ⚑ decisions (D3, D10, D16, D21, D28), restated in one line each, for the
  owners, with their confirmation status;
- a 3–5 line "Current status" paragraph for CLAUDE.md, if not already added.

## 9. Not in this change (follow-ups, in order)

1. **Terminal sign-in** (OIDC loopback or device flow; NEXT_STEPS §2). Then
   require authentication on the legacy API, derive `user_id` and
   `approved_by` from the token (F19), and set `signer_verified` on
   `/approvals`.
2. **Durable orchestration:** Postgres plus Alembic; persist events,
   approvals, and workspace runs so they survive a restart
   (NEXT_STEPS §5 limitation).
3. **Multiple workers:** a Redis-backed bus and gate behind the same
   interfaces.
4. **Workspace push instead of polling:** a per-user SSE channel and `?since=`
   deltas on `/state`.
5. **Literature review on the server:**
   - mount `bedrock.litreview.api` under `/litreview` behind server auth;
   - `/review` in agent-pack mode;
   - a read-only tool that can start the pipeline but never steer it
     (`litreview/CLAUDE.md`).
6. **Semantic retrieval:** embeddings (NEXT_STEPS §6) shared by `rag_search`,
   `/digest`, recall and meeting search. model2vec is already on the server.
7. **One workspace concept:** move the local vault behind the server
   (NEXT_STEPS §3).
8. **Real Director planning** (Workstream 4) with the fixed plan as the
   fallback. A container sandbox, then make `data_analysis_sandbox`
   reachable.
9. **Structure prediction as a server job** (AlphaFold on the GPU box),
   reached through `sequence_structure`.
10. **Screen reading** on the `rapidocr` 3.x API (Python 3.13+).
11. **One PHI pattern module** with separate pattern sets for quarantine and
    redaction (D18).
12. **Workstream 5** (sharing, reviews, recipes, scheduling, notifications)
    and the admin metrics.
13. **`scripts/gen_ts_types.py`** for `contracts/types.ts`.
14. **Owner actions, outward-facing:**
    - archive `bedrock-agent-pack` on GitHub with a pointer here;
    - remove the superseded target branches (D36) and the old
      `agent-pack-integration` branch;
    - set the Keycloak redirect URIs for the new deployment.

---

## Appendix: server settings reference (names unchanged unless marked NEW)

| Env var | `[server]` key | Default | Read |
|---|---|---|---|
| `AGENT_PACK_MODE` | `mode` | `mock` | lazy (Phase 5) |
| `DATABASE_URL` | `database_url` | `sqlite:///<data_dir>/agent_pack.db` (was `./agent_pack.db`) | lazy |
| `BEDROCK_SERVER_DATA_DIR` (NEW) | `data_dir` | `$XDG_DATA_HOME/bedrock/server` | lazy |
| `BEDROCK_SERVER_CONFIG_DIR` (NEW) | `config_dir` | `~/.config/bedrock/server` | lazy |
| `BEDROCK_SERVER_KNOWLEDGE_DIR` (NEW, Phase 12) | `knowledge_dir` | none (samples in mock only) | call time |
| `BEDROCK_SERVER_ALLOW_MODEL_DOWNLOAD` (NEW) | – | `false` | CLI |
| `AIRGAP_CHECK` | `airgap_check` | `true` (real mode only) | lazy |
| `MOCK_TOKEN_DELAY_S` | `mock_token_delay_s` | `0` | lazy |
| `REDIS_URL` | – | unused today; keep documenting it as reserved | – |
| `APP_BASE_URL`, `KEYCLOAK_BASE_URL`, `KEYCLOAK_REALM`, `KEYCLOAK_CLIENT_ID`, `KEYCLOAK_CLIENT_SECRET` | `app_base_url`, `keycloak_*` | as today | lazy |
| `DIRECTOR_/SPECIALIST_/RUNNER_LLM_URL`, `_MODEL_NAME`, `_LLM_API_KEY`, `_MAX_PARALLEL` (NEW) | via `models.yaml` | as today | `models.yaml` |
| `MEETING_SUMMARY_MODEL`, `MEETING_SUMMARY_ENDPOINT`, `MEETING_SUMMARY_CONTEXT` (NEW) | via `models.yaml` `notes` | `llama3.1:8b`, `http://localhost:11434/v1`, `8192` | `models.yaml` |
| `DEMO_MODE` | – | follows mode (was `true`) | call time |
| `WORKSPACE_DATA_DIR`, `MEETING_RECORDING_DIR` | – | under `data_dir` (was `./data/...`) | call time |
| `DATACENTER_BASE_URL`, `DATACENTER_SIGNING_SECRET`, `DATACENTER_TIMEOUT_S`, `AGENT_PACK_ALLOW_PUBLIC_DATACENTER` | – | as today | call time |
| `MEETING_MODEL_DIR`, `_SPEECH_MODEL`, `_VAD_MODEL`, `_SEGMENTATION_MODEL`, `_EMBEDDING_MODEL`, `_WHISPER_THREADS`, `_LANGUAGE`, `_VOCABULARY`, `_VOCABULARY_AUTO`, `_MAX_CONCURRENT`, `_READ_SCREEN`, `_SEMANTIC`, `_TEXT_MODEL`, `_AUDIO_RETENTION_DAYS`, `_TRANSCRIPT_RETENTION_DAYS`, `_REDACT_PHI`, `_ACCESS_GROUPS`, `_REQUIRE_CONSENT` | – | as today | call time |
| `HF_HUB_OFFLINE` | – | set to `1` by `bedrock-server` in real mode | process |
