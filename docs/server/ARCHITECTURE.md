# Architecture — Biotech Agent Pack

Status: **approved plan, implementation in progress**

> **As built, after the move into bedrock-private-ai (2026-09):** model calls go
> through bedrock's `Scheduler` (one queue per model server, real token counts,
> budgets enforced); answers are formatted in code, and on the high-stakes path
> the approved text is the delivered text; the audit log has one row per model
> reply and PHI is redacted in it (identity fields excepted); the normal tier
> runs the citation check for literature, internal-document and meeting
> answers; agents see the conversation so far and material sent with a message
> (`ContextBlock`). **Not built, despite what sections 5–8 describe:** the
> Director's planning is a fixed two-step template and its review a
> deterministic check (no model call); Runner summarising, entity extraction,
> history compaction and the result cache are unused; pgvector, Redis and
> `scripts/gen_ts_types.py` do not exist (`contracts/types.ts` is kept by
> hand); the REST API has no authentication. See
> `docs/prompts/AGENT_PACK_INTEGRATION.md` (Part A, section 9).

## 1. Goal and constraints

Backend for a self-hosted, three-agent AI system (Director / Specialist /
Runner) for a biotech company, exposed to an already-in-progress
Claude-Code-style frontend via a clean HTTP + event-stream API. Hard
constraints:

- No cloud LLM APIs, no telemetry, no outbound calls at runtime. Air-gapped.
- Open-source stack only.
- Runs on a single ~$300K on-prem GPU server.
- 21 CFR Part 11 / GxP-style audit traceability.
- Frontend is separate and not built here; this repo ships the API, event
  stream, TypeScript contracts, and a mock mode so the frontend can be built
  and demoed without the GPU server.

## 2. Three agents and routing

| Agent | Model role | Called | Job |
|---|---|---|---|
| Director | DeepSeek reasoning model | Rarely, only for hard/high-stakes tasks | Plan → assign → review/approve/reject (max 2 retries, then human escalation) → handle hardest reasoning itself |
| Specialist | Qwen large model | Most "real work" tasks | Research, bioinformatics/data analysis via sandboxed Python, drafting documents |
| Runner | Small Qwen model | Constantly, on every request/turn | Classify request difficulty/domain, summarize/trim tool output, extract structured entities, redact PHI, compress history, format final output |

Routing rules (implemented in `src/bedrock/server/orchestrator/router.py`, decided by a
Runner classification call, not hardcoded regexes):

- **simple** → Runner answers directly, Director never invoked.
- **normal** → Specialist executes, Runner cleans up; Director invoked only
  if the output gets flagged high-stakes after the fact.
- **hard / multi-step / high-stakes** → Director plans the task graph,
  Specialist/Runner execute steps, Director reviews before completion.
- High-stakes is a fixed predicate (not model-decided alone): anything
  touching regulatory submissions, clinical/patient data, safety
  conclusions, or a "final report" step type always routes through Director
  review, regardless of what the classifier says. This is enforced in code
  (`src/bedrock/server/orchestrator/router.py::is_high_stakes`) as a safety net over the
  model's own judgment.

Retry/escalation: a rejected Specialist/Director-reviewed step gets at most
2 retries with the Director's rejection reason fed back as a fix
instruction; on a 3rd failure the task is marked `blocked_on_human` and an
`approval.required`-style event surfaces it in the frontend chat.

## 3. Model selection

All model identifiers, endpoints, precision, context length, "thinking
mode" flags, and per-agent token budgets live in **`src/bedrock/server/defaults/models.yaml`**
and **`src/bedrock/server/defaults/budgets.yaml`** only — nothing else in the codebase hardcodes
a model name, so any model can be swapped by editing YAML.

Current defaults, chosen by checking Hugging Face + vendor announcements as
of this build (see caveat below):

- **Director** → `deepseek-ai/DeepSeek-V4-Flash-0731` (MoE, ~284B total /
  ~13B active params, 1M context, MIT license, has a "thinking" mode used
  only for tasks the Director marks hard). `DeepSeek-V4-Pro` (~1.6T total /
  ~49B active) is documented as an optional upgrade if the company adds a
  second node — it is very unlikely to fit a single ~$300K server (see
  `docs/server/HARDWARE.md`).
- **Specialist** → `Qwen/Qwen3.8-27B` (dense, deployment-friendly single- or
  dual-GPU size). `Qwen/Qwen3.8-2.4T-A95B` (MoE) is listed as an optional
  higher-end swap if GPU budget allows; both are wired through the same
  `src/bedrock/server/defaults/models.yaml` entry so switching is a one-line change.
- **Runner** → `Qwen/Qwen3.6-35B-A3B` (MoE, ~3B active — same shape class as
  the `Qwen3-30B-A3B` referenced in the original spec, just the current
  generation), configurable down to a smaller dense model (e.g. an ~8B
  point, or `Qwen/Qwen3.5-0.8B` for extreme-low-latency classification only)
  if the company wants an even cheaper Runner tier.

**Caveat, flagged explicitly per the instruction to flag anything
unverifiable:** these are very recent (2026) releases, found via web search
and Hugging Face listings rather than my own training data. Treat the exact
parameter counts, context lengths, and — especially — **licenses** in the
table above as "believed correct, re-verify against the official model
card before procurement or download," not as guaranteed fact. If any of
these turn out to be wrong or renamed by the time of purchase,
`src/bedrock/server/defaults/models.yaml` is the only file that needs to change.

## 4. Serving

One OpenAI-compatible endpoint per model via vLLM (or SGLang — both speak
the same HTTP API so this is a config choice, not a code choice):

```
Director   → http://director-llm:8001/v1   (DeepSeek-V4-Flash)
Specialist → http://specialist-llm:8002/v1  (Qwen3.8-27B)
Runner     → http://runner-llm:8003/v1      (Qwen3.6-35B-A3B)
```

`src/bedrock/server/models/client.py` defines a `ModelClient` protocol (chat-completion +
streaming) with two implementations: `HTTPModelClient` (real, talks to the
above endpoints) and `MockModelClient` (fixture-driven, used when
`AGENT_PACK_MODE=mock`). Agents only ever depend on the protocol, never on
which implementation is active.

## 5. Service architecture

```
                         ┌──────────────────────┐
 frontend  ── REST/SSE ──▶      FastAPI app       │
                         │  (src/bedrock/server/api)            │
                         └──────────┬────────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
             Orchestrator      Event bus         Compliance
          (session, router,   (SSE + audit      (audit log,
           task graph,         subscribers)      PHI redact,
           retry/escalate)                        RBAC, gates)
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
    Director    Specialist     Runner
   (agents/)    (agents/)    (agents/)
        │           │            │
        └─────┬─────┴─────┬──────┘
              ▼           ▼
        ModelClient    Tool registry
      (vLLM/SGLang or   (RAG, lit search,
         mock)           BLAST/structure,
                         RDKit, sandbox,
                         ELN/LIMS stubs,
                         doc templates,
                         meeting transcripts)

  Postgres: sessions, task graph, audit log, token usage, RAG cache table
  pgvector: RAG embeddings (default; Qdrant documented as a swap-in)
  Redis: task queue + pub/sub feeding the SSE stream (in-process queue
         used automatically in mock mode / single-worker dev runs)
```

Directory layout:

```
src/bedrock/server/   (formerly app/)
  main.py            FastAPI app factory, mounts routers, startup air-gap check
  config.py          loads config/*.yaml, env vars
  schemas/           Pydantic models — the single source of truth for the
                     API, events, and contracts/types.ts
  events/            typed event bus (publish/subscribe), SSE formatting
  orchestrator/      session state machine, router, task graph, retry loop
  agents/            director.py, specialist.py, runner.py (thin, share base.py)
  models/            ModelClient protocol, HTTP client, mock client, registry
  task_store/        task graph persistence (Postgres in real mode, in-memory
                     in mock/test mode) — always accessed through references
  tools/             one module per biotech tool, Pydantic-typed, registered
                     in a registry the agents call through
  compliance/        audit log, PHI redaction, RBAC, approval gates, air-gap check
  api/               REST routers + SSE endpoint
  db/                SQLAlchemy models + session/init helpers
contracts/types.ts   hand-maintained TS mirror of src/bedrock/server/schemas (see §7)
config/              models.yaml, budgets.yaml
docker/              per-service Dockerfiles/configs (vLLM entrypoints etc.)
tests/               routing, retry/escalation, PHI redaction, audit log,
                     token budgets
```

## 6. Token efficiency mechanisms

- System prompts: static, unchanging prefix first (role + rules), variable
  task context appended after, so vLLM/SGLang prefix caching is effective.
  Enforced by convention in `src/bedrock/server/agents/base.py::build_prompt`.
- Agents pass **references** through the task store (doc IDs, tool-result
  IDs, file paths), never full content, across agent boundaries.
- Runner summarizes any tool output over `budgets.yaml: summarize_over_tokens`
  before it reaches another agent.
- Inter-agent messages are compact JSON (Pydantic models), never prose.
- Per-task-per-agent token budgets in `src/bedrock/server/defaults/budgets.yaml`; a task that
  exceeds budget stops and reports to the Director instead of continuing.
- Runner compacts old conversation turns into short summaries once history
  exceeds a configured turn/token count.
- RAG returns only the top-k chunks (default k in `budgets.yaml`), never
  whole documents.
- A `(query_hash, doc_set_hash) → result` cache table avoids recomputing
  repeated RAG/tool calls.
- Every model call logs prompt/completion token counts; aggregated per
  agent per task and pushed via `usage.updated` events for the frontend
  panel.

## 7. Frontend contract

- REST: start session, send message, cancel task, approve/reject a human
  gate, fetch history, fetch audit log (`src/bedrock/server/api/`).
- SSE stream (`GET /sessions/{id}/events`) with the event types the spec
  requires: `session.started`, `task.planned`, `task.assigned`,
  `agent.started`, `agent.status`, `agent.token`, `tool.called`,
  `tool.result`, `review.approved`, `review.rejected`, `approval.required`,
  `task.completed`, `task.failed`, `usage.updated`. Every event carries
  enough for an agent card: agent name, role, model, status, current task
  title, tokens used, elapsed time.
- `contracts/types.ts` mirrors every Pydantic request/response/event model.
  Kept in sync by convention + a checked generation script
  (`scripts/gen_ts_types.py`, JSON-schema based) rather than a build-time
  dependency, so the frontend team can just import the file without running
  Python tooling.
- Mock mode (`AGENT_PACK_MODE=mock`): the entire API runs against
  `MockModelClient` + fixture tool results with simulated latencies, so the
  frontend can be built/demoed with zero GPUs.

## 8. Compliance & security

- **Audit trail**: append-only `audit_log` table (Postgres in real mode);
  every prompt, agent decision, tool call, approval/rejection, and output
  writes a row with timestamp, user, agent, model+version, before/alongside
  the action. Rows are never updated or deleted by the app (enforced by
  granting the app DB role INSERT/SELECT only on that table).
- **PHI protection**: Runner-based redaction step runs before anything is
  logged or handed to another agent; raw PHI is scoped to the task that
  needed it and never persisted outside that task's working memory.
- **RBAC**: RAG/document tools take the requesting user's permission set and
  filter results server-side before any content reaches a model; agents
  never get broader access than the user who started the session.
- **Human approval gates**: any tool marked `writes_to_external_system=True`
  or any task marked high-stakes blocks on an `approval.required` event
  until a human approve/reject call resolves it.
- **Air-gapped by default**: a startup check (`src/bedrock/server/compliance/airgap_check.py`)
  attempts a handful of outbound connections/DNS lookups and refuses to
  start the app if any succeed, on the theory that a properly air-gapped
  network makes them fail — this catches misconfiguration, not a
  determined bypass, and is documented as such.
- **Hallucination guardrails**: the Director's review step rejects any
  Specialist output containing a scientific claim without an attached
  citation or tool-result reference.

## 9. Open questions / assumptions (recorded, not blocking)

1. **ELN/LIMS**: no specific product named, so `src/bedrock/server/tools/lab_connectors.py`
   ships as typed stub interfaces only (loosely shaped like common
   open-source ELN/LIMS REST APIs) — not a working integration. Tell me the
   actual system and I'll wire a real client against this interface.
2. **Literature mirror**: assumed to be a directory of parsed
   records (e.g. a PubMed baseline/OA subset export) that an indexer job
   can ingest; the actual source/ETL is out of scope until specified.
3. **Event transport**: SSE chosen over WebSocket (one-way is sufficient,
   simpler through proxies/load balancers); swappable later if the frontend
   needs bidirectional push.
4. **Vector store**: pgvector chosen as the default (one fewer service than
   Qdrant); Qdrant remains a documented drop-in via the same tool interface.
5. **Auth/identity**: RBAC assumes a pluggable identity provider (e.g. OIDC)
   with a local stub user table for mock mode/dev, since no company IdP was
   given.
6. **Model versions**: see §3 caveat — recent releases, verify before
   procurement.
7. **Meeting transcript RBAC**: `src/bedrock/server/tools/meeting_transcripts.py` (added
   after this doc's initial approval — see zoom-transcript-vault repo for
   the ingestion side) does not yet filter results by the requesting
   user's permissions the way `rag_search` does; the datacenter connector
   it queries has no per-meeting access-group concept today. Worth a real
   decision before this handles anything sensitive: either the datacenter
   connector starts tagging transcripts with `access_groups` (same shape
   `rag_search`'s documents use) or this tool grows its own filtering.
8. **Citation guardrail only runs on the hard/Director path**: this
   predates the meeting_transcripts addition and isn't specific to it —
   `src/bedrock/server/orchestrator/loop.py::_handle_normal` (used for
   `TaskComplexity.NORMAL`) never calls `Director.review()`, so a
   citation-required domain (`literature`, `internal_docs`, `meetings`;
   `regulatory`/`clinical` are safe because they're forced to `HARD` via
   `_ALWAYS_HIGH_STAKES_DOMAINS` in `src/bedrock/server/orchestrator/router.py`) can
   complete with zero citations whenever the Runner's classifier — or the
   mock heuristic — calls it merely "normal" rather than "hard". Confirmed
   via `tests/test_meeting_domain_routing.py`'s
   `test_a_raw_chat_message_reaches_the_real_tool_through_handle_message`.
   Two ways to close it: run the citation check in `_handle_normal` too
   (cheap, no model call, `Director.review`'s guardrail check is already
   deterministic), or add every citation-required domain to
   `_ALWAYS_HIGH_STAKES_DOMAINS` (more expensive — forces the full
   Director path for what might be a simple, cheap lookup). Flagging for
   Workstream 4 (Agent Intelligence & Compliance) rather than fixing
   unilaterally here, since it changes shared orchestrator behavior for
   `literature`/`internal_docs` too, not just meetings.

## 10. Build order (in progress)

1. ~~Architecture doc (this file)~~ — done, approved.
2. Contracts (Pydantic schemas + `contracts/types.ts`).
3. Mock mode end-to-end.
4. Real model clients + router + Director/Specialist/Runner loop.
5. Tools, then compliance layer.
6. Token optimizations + usage tracking.
7. Tests, `docs/server/HARDWARE.md`, Docker Compose, `README.md`.
