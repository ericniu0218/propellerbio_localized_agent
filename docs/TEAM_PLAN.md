# Team Plan — What's Left, Who Owns It, How to Start

**Browsable version with one-click copy on every Claude briefing:**
https://claude.ai/code/artifact/a7273d50-adee-45e0-8ce9-11c9e4ea7d2e
(private by default — share it with the team from the page's share menu).
This file is the version-controlled source of truth; the page is a copy
for day-to-day use.

Status: current as of 2026-09-11. This is the handoff checklist from
"working prototype" to "ready for a real biotech client" turned into six
parallel workstreams for a 6-person team. Each section is self-contained:
what's built, what's missing, how to build it, how to know it's done, and
a ready-to-paste briefing for a fresh Claude Code session working on
exactly that piece.

**How to use this doc:** pick a workstream (or split one across two
people, or combine two if your team is smaller in that area). Whoever
owns it clones the relevant repo(s), pastes that workstream's "Claude
briefing" as their first message to Claude Code, and goes. Cross-team
dependencies are called out explicitly — mostly, they're not blocking:
this system was built so most of these can proceed against **mock mode**
without waiting on each other.

**Repos:**
- Backend: `https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack`
- Terminal frontend: `https://github.com/Bedrock-Technical-Consulting/bedrock-private-ai`
- Zoom capture/datacenter handoff (independent service, not part of the
  six workstreams below): `https://github.com/Bedrock-Technical-Consulting/zoom-transcript-vault`

> **Not covered by any workstream below — added after this plan was
> written:** a `meeting_transcripts` tool letting the AI answer
> "what happened in meeting X" questions, in
> [PR #1](https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack/pull/1)
> on the backend repo (open, not merged). It queries the datacenter
> connector `zoom-transcript-vault` pushes transcripts into, directly over
> the internal network (never the public internet — this system is
> air-gapped). Read the PR description and its follow-up comment before
> touching related code. Two things it explicitly left as open questions
> that overlap existing workstreams rather than deciding unilaterally:
> per-meeting RBAC (overlaps **Workstream 2**'s real-permissions work —
> see `docs/ARCHITECTURE.md` open question 7) and a citation-guardrail gap
> in `_handle_normal` affecting `literature`/`internal_docs` too, not just
> meetings (overlaps **Workstream 4** — see open question 8). Whoever picks
> up either of those workstreams should read those two open questions first.

**Read first, regardless of workstream:** `docs/ARCHITECTURE.md`,
`docs/WORKFLOW.md`, `docs/UX.md` in the backend repo. They're the design
source of truth — if your workstream's plan below conflicts with them,
the docs win (or flag the conflict, don't silently pick one).

---

## Workstream 1 — Platform & Deployment

**Owns:** getting this off mock mode and onto real infrastructure.

### What's built
Mock mode (fake model responses, SQLite), `config/models.yaml` naming the
three target models, `app/models/http_client.py` (untested against a real
server), `docker-compose.yml` with a `real` profile, `docs/HARDWARE.md`
with a GPU memory budget.

### What's missing and how to build it

1. **GPU server + real models.** Confirm what hardware is actually
   available against the budget in `docs/HARDWARE.md`. Download the model
   weights named in `config/models.yaml` onto that machine (no internet
   access afterward — air-gap rule). Serve each with vLLM or SGLang using
   `docker-compose.yml`'s `real` profile as a starting point — you'll need
   to fix the volume paths and `--model` args to match where weights
   actually land.
2. **Verify `HTTPModelClient` against a live server.** It's never been
   run against anything real. Start one vLLM instance, point
   `DIRECTOR_LLM_URL` at it, set `AGENT_PACK_MODE=real`, and send one
   message through the full API. Fix whatever breaks — likely candidates:
   streaming response format differences between vLLM versions, and
   whether `chat_template_kwargs` is actually how your vLLM version wants
   thinking-mode toggled (check its docs; that's a guess in the code today).
3. **Real Postgres + migrations.** Point `DATABASE_URL` at a real Postgres
   instance. Add Alembic (`pip install alembic`, `alembic init`), generate
   a migration matching `app/db/models.py`, and from here on, schema
   changes go through migrations, not the current
   `Base.metadata.create_all` auto-create.
4. **CI.** Add `.github/workflows/test.yml` to both repos: install deps,
   run `pytest`. Nothing currently runs tests except a human typing the
   command.
5. **`docs/DEPLOYMENT.md`.** Write it for someone on the IT team who has
   never seen this project — Docker Compose, environment variables
   (`.env.example` is the checklist), the air-gap check, Postgres setup.

### Done when
`AGENT_PACK_MODE=real` answers a real question with a real model,
end-to-end, over the real API. CI is green on a PR. A new IT person could
follow `docs/DEPLOYMENT.md` without asking you anything.

### Claude briefing — paste this to start
```
I'm working on Workstream 1 (Platform & Deployment) for the biotech
agent-pack project. Clone
https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack and
read docs/ARCHITECTURE.md and docs/HARDWARE.md first.

Today everything runs in AGENT_PACK_MODE=mock (fake model responses,
SQLite). My job: get it running for real. Specifically:
1. Help me stand up vLLM (or SGLang) serving the three models named in
   config/models.yaml, using docker-compose.yml's `real` profile as a
   starting point — I'll tell you what GPU hardware I actually have.
2. Once one model server is up, help me verify app/models/http_client.py
   actually works against it (it's never been tested against a real
   server) — start with the Director model and one plain chat message
   through the full API, fix whatever the response-format mismatch turns
   out to be.
3. Help me add Alembic migrations for the schema in app/db/models.py, and
   switch DATABASE_URL to a real Postgres instance I'll provision.
4. Add a GitHub Actions workflow that runs `pytest` on every push/PR in
   this repo (and I'll paste you the frontend repo's structure separately
   for the same treatment there).
5. Once 1-4 work, help me write docs/DEPLOYMENT.md from scratch, assuming
   the reader has never seen this project — Docker Compose, env vars,
   the air-gap check, Postgres.

Don't touch anything under app/sharing/, app/recipes/, app/uploads/, or
app/auth/ (other teams own those) — flag it to me if my changes would
affect their interfaces. Ask me before making assumptions about our
actual GPU hardware or hosting environment; I'll tell you what we have.
```

---

## Workstream 2 — Identity & Access

**Owns:** real login, real permissions, security review.

### What's built
A trivial `user_id` string with no verification anywhere. A toy
`app/compliance/rbac.py::StaticPermissionSource` (in-memory, not
connected to anything real). The decision (already made with the client):
departments/reviewers are managed in-app, not pulled from Keycloak.

### What's missing and how to build it

1. **Keycloak.** Stand it up (official Docker image), create a realm and
   an OIDC client for this app. Decide with IT how company accounts get
   into it (sync from an existing directory, or manual provisioning).
2. **JWT verification in the backend.** New `app/auth/` module: a
   `verify_token(token) -> User` function checking signature/expiry/
   audience against Keycloak's JWKS endpoint (`authlib` or `python-jose`).
   Add a FastAPI dependency, `get_current_user()`, and put it on every
   route — this replaces the plain `user_id` fields currently trusted at
   face value in request bodies. **This changes the API contract** —
   coordinate with Workstream 6 (frontends) before shipping it.
3. **Department/reviewer admin.** New `app/org/`: a `Department` model,
   endpoints to assign a user to a department and set a department's
   reviewer(s) — per the decision already recorded in
   `docs/WORKFLOW.md` §1. This is what Workstream 5's review flow depends on.
4. **Real workspaces + RBAC.** New `app/workspaces/`: `Workspace`
   (personal/team/company) and `WorkspaceMembership`. Update
   `app/tools/rag_search.py` to filter by real membership via a SQL join
   — the file's docstring already says this is the intended extension
   point; today it uses the toy `StaticPermissionSource`.
5. **Security review.** Once the above exists, do a real pass: sandbox
   escape vectors (`app/tools/sandbox.py`), injection surface (should be
   fine via the ORM — verify), whether the audit log is genuinely
   append-only under real Postgres grants (`docs/ARCHITECTURE.md` §8
   describes the intent; nothing enforces it yet on SQLite).

### Done when
Logging in requires a real company account. A user genuinely cannot pull
up another team's documents through any path, including via an agent —
test this adversarially, don't just check the UI hides it.

### Claude briefing — paste this to start
```
I'm working on Workstream 2 (Identity & Access) for the biotech
agent-pack project. Clone
https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack and
read docs/ARCHITECTURE.md section 8 and docs/WORKFLOW.md sections 1-2
first — that's the compliance/permissions design and the already-made
decision that departments/reviewers are admin-managed in our own app, not
pulled from Keycloak.

Today there is no real authentication — any request just trusts a plain
user_id string. My job:
1. Help me stand up Keycloak (I'll handle the org-specific realm/client
   config) and add an app/auth/ module that verifies its JWTs against the
   JWKS endpoint, plus a get_current_user() FastAPI dependency.
2. Help me figure out which existing routes (see app/api/) need that
   dependency added, and what changes in the request/response shapes —
   flag every contract change clearly, since Workstream 6 owns both
   frontends and needs to know.
3. Help me build app/org/ (Department model + admin endpoints for
   department assignment and reviewer assignment) and app/workspaces/
   (Workspace + WorkspaceMembership models).
4. Help me rewrite app/tools/rag_search.py's permission filtering to use
   real workspace membership via a SQL join instead of the current
   StaticPermissionSource stub — its docstring already explains the
   intended design.
5. Once that's solid, help me do a security pass: check
   app/tools/sandbox.py for escape vectors, check for injection surface
   outside the ORM, and check what's needed to make the audit log
   genuinely append-only on real Postgres (see ARCHITECTURE.md section 8).

Don't change the agent/orchestrator logic (app/orchestrator/,
app/agents/) beyond what's needed to thread the current user through —
that's other teams' territory. Ask me before assuming how our company
directory/SSO is actually organized.
```

---

## Workstream 3 — Knowledge Base & Ingestion

**Owns:** making the RAG/literature search real, and file uploads.

### What's built
`app/tools/rag_search.py` and `literature_search.py` — real tool
interfaces, but backed by a `KeywordOverlapIndex` reading 3 sample text
files. No vector DB, no embeddings, no real document pipeline. No file
upload capability anywhere.

### What's missing and how to build it

1. **A real vector store.** `pgvector` is the documented default (see
   `docs/ARCHITECTURE.md` §9) — add the extension to Postgres, create a
   real `content_chunks` table with an embedding column (shape sketched
   in `docs/WORKFLOW.md` §8). Qdrant is the documented alternative if the
   team prefers a dedicated service.
2. **A self-hosted embedding model.** Must run locally — no external API
   calls (the whole point of this system). An open-weight
   sentence-embedding model served alongside the LLMs, or run via
   `sentence-transformers` on CPU/GPU, works.
3. **Swap the search backend.** Replace `KeywordOverlapIndex` in
   `rag_search.py`/`literature_search.py` with a real similarity query —
   both files already define a `VectorIndex` protocol specifically so
   this swap doesn't touch calling code.
4. **Document ingestion pipeline.** A script/job that walks the company's
   real SOPs/study reports/protocols (wherever they actually live — ask
   the client, don't assume), chunks them, embeds them, and stores them
   with `doc_id`/`section` preserved for citations. This is the real
   "Company Knowledge" corpus, replacing `data/internal_docs/`.
5. **File uploads.** New `app/uploads/` per `docs/WORKFLOW.md` §5: accept
   a file, store on local disk, extract text (`pypdf` for PDF,
   `python-docx` for Word, `openpyxl`/`pandas` for Excel/CSV,
   `python-pptx` for PPTX — the existing CSV parser in
   `app/tools/lab_connectors.py` covers instrument exports already),
   chunk into the same content index, and have the Runner produce a
   summary card immediately. Build the `Brief` schema
   (`app/schemas/brief.py`, shape in `docs/UX.md` §5) now if it doesn't
   exist yet — summary cards and chat answers should share it.

### Done when
Asking about a real company topic returns real company documents with
correct, clickable citations. Dragging in an actual SOP PDF produces a
sensible summary card in a reasonable time.

### Claude briefing — paste this to start
```
I'm working on Workstream 3 (Knowledge Base & Ingestion) for the biotech
agent-pack project. Clone
https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack and
read app/tools/rag_search.py and app/tools/literature_search.py first —
both already define the exact seam I need to build behind (a
`VectorIndex` protocol), currently implemented by a toy keyword-matching
class over 3 sample files in data/internal_docs/. Also skim
docs/WORKFLOW.md section 8 for the planned content_chunks shape and
docs/ARCHITECTURE.md section 9 for why pgvector is the default choice
over Qdrant.

My job:
1. Help me add pgvector to our Postgres instance and design a real
   content_chunks table (workspace_id/owner_id columns matter here even
   though Workstream 2 owns the actual permission enforcement — leave
   room for them).
2. Help me pick and stand up a self-hosted embedding model — must run
   locally, no external API calls allowed anywhere in this system.
3. Help me replace KeywordOverlapIndex in both tool files with a real
   pgvector similarity query, implementing the same VectorIndex protocol
   so I don't have to touch app/agents/specialist.py.
4. Help me write an ingestion script that chunks and embeds real
   documents into that table, preserving doc_id/section for citations —
   I'll tell you where our actual SOPs/study reports live once I know.
5. Help me build app/uploads/: file upload endpoint, text extraction per
   format (PDF/Word/Excel/CSV/PPTX), chunking into the same index, and a
   Runner-generated summary card. If app/schemas/brief.py doesn't exist
   yet, help me create it per docs/UX.md section 5's Bottom
   line/Key findings/Key numbers/Open questions/Sources shape.

Don't touch app/workspaces/ or app/auth/ (Workstream 2 owns those) —
just leave workspace_id/owner_id columns for them to wire up permission
checks against. Ask me before assuming where our real documents live or
which embedding model to use if there's a licensing/procurement question.
```

---

## Workstream 4 — Agent Intelligence & Compliance

**Owns:** making the Director actually plan, sandbox hardening, compliance readiness.

### What's built
A working reject/retry/escalate loop, but the Director's plan is always
the same hardcoded 2-step shape (research → synthesis) — see
`app/agents/director.py::plan`. The sandbox (`app/tools/sandbox.py`) is a
subprocess with basic resource limits, explicitly documented as not
properly network-isolated. The audit log is designed with 21 CFR Part 11
in mind but never reviewed by a compliance professional.

### What's missing and how to build it

1. **Real dynamic planning.** In `director.py::plan`, replace the
   hardcoded two-node template with an actual model call asking the
   Director to produce a JSON task graph (nodes, dependencies,
   assigned_to, domain, complexity). Real models don't always produce
   valid JSON on the first try — add a parse-and-retry loop, and fall
   back to today's 2-node template if it still fails (keep the current
   behavior as the safety net, don't remove it).
2. **Sandbox hardening.** Replace the subprocess approach with real
   isolation — a locked-down Docker container per execution
   (`--network=none`, cgroup limits, a throwaway tmp mount, nothing else)
   is the minimum bar; gVisor or Firecracker if the team wants more.
   Depends on Workstream 1 for where this actually runs (same GPU box or
   a sibling host).
3. **Prompt tuning against real models.** This can only really happen
   after Workstream 1 has a live model — the JSON-producing prompts
   (`Runner.classify`, `Runner.extract_entities`, the new planning prompt
   above) were only ever validated against a deterministic mock. Expect
   real models to need tighter instructions or few-shot examples.
4. **Compliance-readiness document.** Not a compliance sign-off (that's a
   professional's job, not this team's) — but a clear mapping from each
   relevant 21 CFR Part 11 requirement to how `app/compliance/audit.py`
   and `app/db/models.py::AuditLogRow` address it, so an actual QA/
   compliance reviewer has something concrete to evaluate instead of
   reverse-engineering the code themselves.

### Done when
Two genuinely different requests produce two genuinely different task
graphs (not just different content in the same 2-node shape). The
sandbox can't reach the network or escape its filesystem — try to break
it. A compliance reviewer can read one document and know what to check.

### Claude briefing — paste this to start
```
I'm working on Workstream 4 (Agent Intelligence & Compliance) for the
biotech agent-pack project. Clone
https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack and
read app/agents/director.py, app/orchestrator/loop.py, and
docs/ARCHITECTURE.md section 8 first.

Today the Director's plan() method always returns the same hardcoded
2-node graph (a research step then a synthesis step) regardless of the
request — it's a placeholder for real planning. My job:
1. Help me replace that with a real model call: ask the Director's model
   for a JSON task graph (nodes/dependencies/assigned_to/domain/
   complexity), parse it, retry on invalid JSON, and fall back to
   today's 2-node template if it still fails after retries — I want to
   keep that fallback as a safety net, not remove it.
2. Help me harden app/tools/sandbox.py — it currently runs untrusted
   Python in a plain subprocess with resource limits only, no real
   network isolation. I want to move it to a locked-down Docker container
   per execution (--network=none, cgroup limits, throwaway tmp mount).
3. Once Workstream 1 has a real model server up, help me test
   app/agents/runner.py's JSON-producing prompts (classify,
   extract_entities) against it and tighten them if the real model's
   output doesn't parse as reliably as the mock's always did.
4. Help me write a compliance-readiness document mapping
   app/compliance/audit.py and app/db/models.py::AuditLogRow to specific
   21 CFR Part 11 requirements — this is prep material for an actual
   compliance reviewer, not a claim that we're compliant, so be precise
   about what's actually implemented versus just designed-for.

Don't touch app/tools/rag_search.py or anything under app/workspaces/ —
other teams own those. Ask me before changing the TaskNode/TaskPlan
schemas in a way that would break contracts/types.ts — coordinate any
schema change with Workstream 6.
```

---

## Workstream 5 — Collaboration Features

**Owns:** sharing, reviews, recipes, scheduling, notifications — the whole
product layer from `docs/WORKFLOW.md` that doesn't exist in code yet.

### What's built
Nothing yet — this entire workstream is `docs/WORKFLOW.md` turned into
code. The event types it needs (`share.created`, `review.requested`,
`review.decided`, `item.updated`, `recipe.started`, `recipe.completed`,
`notification.created`) are documented but not yet added to
`app/schemas/enums.py::EventType`.

### What's missing and how to build it

1. **Sharing.** New `app/sharing/`: `SharedItem` (points at a chat task,
   upload, summary card, or recipe; target = a person, a team workspace,
   or Company Knowledge), `SharedItemVersion` (append-only, for "Updated
   by X on Y"), `Review` (pending/approved/needs_changes). **PHI check
   before any share completes** — call
   `app/compliance/phi_redaction.py::redact_phi` on the content and
   surface a plain warning if it finds something; this is a hard
   requirement from the client, not optional polish.
2. **Review queue.** Depends on Workstream 2's department/reviewer
   config. A reviewer sees pending Company Knowledge submissions for
   their department and can approve or reject with a comment.
3. **Recipes.** New `app/recipes/`: a `Recipe` (template string with
   `{{field}}` placeholders + a simple input schema), a run endpoint that
   renders the template and feeds it into the existing
   `handle_message` path. Ship the 10 starter recipes from
   `docs/WORKFLOW.md` §6 as seed data (literature summary, study report
   draft, SOP draft, assay data summary, compare two protocols, meeting
   notes → action items, regulatory section draft, deviation report
   draft, weekly assay results summary, catch-me-up).
4. **Scheduling.** APScheduler (in-process, per the decision already made
   with the client — no new external job system), a `Schedule` model, a
   background job checking due schedules and running the linked recipe.
5. **Notifications.** New `app/notifications/`: a per-user SSE channel
   (separate from the existing per-session one in `app/api/sse.py`) plus
   SMTP email via the company's internal relay (ask Workstream 1/IT for
   the actual relay address — none exists yet in any config).
6. **Compare / Catch-me-up.** New orchestrator entrypoints reusing
   Specialist + Director, similar shape to the existing "hard" path in
   `app/orchestrator/loop.py`. Output shape is in `docs/WORKFLOW.md` §5.

### Done when
Sharing something containing what looks like patient data gets flagged
before it completes. A Company Knowledge submission is genuinely
invisible company-wide until the right reviewer approves it. A starter
recipe runs end-to-end with real inputs.

### Claude briefing — paste this to start
```
I'm working on Workstream 5 (Collaboration Features) for the biotech
agent-pack project. Clone
https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack and
read docs/WORKFLOW.md in full first — sections 3, 4, 6, and 7 are
specifically what I'm building; the whole document hasn't been turned
into code yet, I'm doing that now.

My job, roughly in this order:
1. Help me add the new event types this needs (share.created,
   review.requested, review.decided, item.updated, recipe.started,
   recipe.completed, notification.created) to
   app/schemas/enums.py::EventType, additively — don't touch existing
   event types.
2. Help me build app/sharing/ (SharedItem, SharedItemVersion, Review) per
   docs/WORKFLOW.md section 3. Critical requirement: before any share
   completes, run app/compliance/phi_redaction.py::redact_phi on the
   content and block/warn if it finds something — this came directly
   from the client as a hard requirement, not something to skip for a
   first pass.
3. Help me build app/recipes/ (Recipe model with template + input schema,
   a run endpoint reusing app/orchestrator/loop.py::handle_message) and
   seed the 10 starter recipes listed in docs/WORKFLOW.md section 6.
4. Help me add APScheduler-based scheduling (a Schedule model + a
   background job) — no new external job infrastructure, that's already
   been decided with the client.
5. Help me build app/notifications/ — a new per-user SSE endpoint
   (separate from the existing per-session one in app/api/sse.py) plus
   an SMTP send; I'll get the actual relay address from IT.
6. Help me add Compare and Catch-me-up as new orchestrator entrypoints,
   modeled on the existing "hard" path in app/orchestrator/loop.py.

This depends on Workstream 2's Department/Workspace models for the
review-routing and sharing-target logic — if those aren't built yet,
help me stub them clearly (e.g. a single hardcoded department) so I can
keep moving, and flag the stub plainly in the code for later replacement.
Ask me before assuming what the 10 starter recipes' exact wording should
be — draft them, but check before finalizing.
```

---

## Workstream 6 — Frontend & UX Polish

**Owns:** both frontends (terminal + web reference client), accessibility,
copy consistency, the admin dashboard.

### What's built
Both clients work end-to-end against mock mode: session creation, live
agent status, approvals, a tier picker and "Create Agent" shortcut (web
client only), a right-hand running-agents panel. Neither streams
token-by-token into the final answer bubble. Neither reconnects if the
live connection drops. No trust badges, no "Show technical details"
toggle, no accessibility pass, no `docs/COPY_GUIDE.md`, no admin dashboard.

### What's missing and how to build it

1. **Token streaming.** The backend already emits `agent.token` events
   with live text — both clients currently wait for `task.completed` and
   fetch the stored result instead (see the reasoning in
   `bedrock-private-ai`'s `agentpack_app.py` docstring: a request can
   involve multiple agents drafting in sequence, and naively streaming
   all of them looks like the answer rewriting itself). Fix: stream only
   the *last* agent's tokens, using the `task.planned` event's node list
   (both clients already parse it) to know when you're on the final step.
2. **Reconnection.** Wrap the SSE consumption (`EventSource` in the web
   client, `AgentPackClient.events()` in the terminal client) in a
   retry-with-backoff loop; on reconnect, re-fetch
   `GET /sessions/{id}/agents` and `GET /sessions/{id}/history` to
   recover state instead of assuming nothing happened while disconnected.
3. **Trust badges + technical-details toggle.** Add `friendly_label`/
   `trust_status` fields to `AgentCard`/`AgentStatusPayload` in the
   backend (additive — coordinate with whoever's touched those schemas
   recently), compute `trust_status` from whether Director review
   actually happened, and render a badge in both clients. Add a
   client-side "Show technical details" toggle per `docs/UX.md` section 1
   — model names/tokens/etc. stay hidden by default either way.
4. **Accessibility pass.** Run a real keyboard-only and screen-reader pass
   on `app/static/index.html` (the web client). Fix tab order, add ARIA
   labels on the custom tier dropdown and approval buttons, and check
   contrast ratios for the earthy palette against WCAG AA — some of the
   muted/dim tones may not actually pass; verify, don't assume they do.
5. **`docs/COPY_GUIDE.md`.** A table mapping every internal technical
   term (agent roles, status enum values, error types) to the plain word
   shown in the UI, plus tone rules. The logic mostly already exists in
   `bedrock-private-ai`'s `agentpack/friendly.py` — the doc should explain
   the mapping and reasoning, not just restate the code.
6. **Admin dashboard.** New `app/metrics/` in the backend — counters only
   (tasks completed, docs condensed, recipes run, estimated minutes
   saved via a configurable table), admin-only endpoints, and a page
   (can be a new section of the web client) showing adoption, most-used
   recipes, and review-queue size. Explicitly no content tracking — counts
   only, per the client's requirement.

### Done when
The chat feels responsive even before real models exist. A dropped
network connection recovers on its own. A screen-reader-only user can
complete the 5-minute usability checklist in `docs/UX.md` §9.

### Claude briefing — paste this to start
```
I'm working on Workstream 6 (Frontend & UX Polish) for the biotech
agent-pack project. There are two repos:
https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack
(backend + a minimal reference web client at app/static/index.html) and
https://github.com/Bedrock-Technical-Consulting/bedrock-private-ai (the
terminal client). Clone both. Read docs/UX.md in the backend repo first —
it's the plain-language design spec both frontends are built against —
and skim bedrock-private-ai's agentpack_app.py and its NEXT_STEPS.md,
which already documents most of what I'm about to ask for.

My job:
1. Help me add real token-by-token streaming into the final answer bubble
   in both clients, using the agent.token events the backend already
   sends. Read agentpack_app.py's docstring on handle_event first — there's
   a specific reason it currently waits for task.completed instead
   (multiple agents can draft in sequence), so the fix needs to detect
   "this is the last agent" via the task.planned event's node list, not
   just naively stream everything.
2. Help me add reconnect-with-backoff to the SSE consumption in both
   clients, re-fetching /sessions/{id}/agents and /sessions/{id}/history
   on reconnect to recover state.
3. Help me add friendly_label/trust_status fields to the backend's
   AgentCard/AgentStatusPayload schemas (additive only — check
   contracts/types.ts needs the same fields added) and render trust
   badges plus a "Show technical details" toggle in both clients.
4. Help me do an actual accessibility pass on app/static/index.html —
   keyboard-only navigation, ARIA labels on the custom dropdown/approval
   buttons, and check the earthy color palette's contrast ratios against
   WCAG AA (don't assume they pass).
5. Help me write docs/COPY_GUIDE.md in the backend repo, based on the
   mapping logic already in bedrock-private-ai's
   src/bedrock/agentpack/friendly.py.
6. Help me build a small admin metrics dashboard — counters only (tasks
   completed, docs condensed, recipes run, estimated time saved), no
   content tracking, per docs/WORKFLOW.md section 6.

Don't change the orchestrator/agent logic — only the event/schema fields
needed for trust badges, and coordinate that specific change since other
teams may be mid-edit on the same schema files. Ask me before deciding on
exact copy/wording for the plain-language toggle or badges — draft
options, then check with me.
```

---

## Cross-team notes

- **Mock mode stays the default** for local development in every
  workstream except 1 and 4 (which specifically need real models to
  finish). Don't require a GPU server to run tests.
- **Contract changes are additive, not breaking**, per the pattern this
  whole project has followed — new fields, new event types, new
  endpoints; never rename or remove something another team's code depends
  on without a heads-up.
- **Update `contracts/types.ts`** in the same change as any backend
  schema/event addition — it's the frontend teams' source of truth and
  drifts silently otherwise.
- **Run the existing test suites before opening a PR** — `pytest` in the
  backend repo (21 tests today), `pytest` in the frontend repo (34 tests
  today). If your change breaks one, fix it or explain why in the PR, not
  after merge.
- **When in doubt about a company-specific detail** (real document
  locations, real SSO setup, real SMTP relay, actual GPU hardware) — ask
  the client rather than guessing. Every stub in this codebase exists
  because that information wasn't available yet, not because it doesn't
  matter.
