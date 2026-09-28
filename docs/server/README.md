# Biotech Agent Pack

A self-hosted, air-gapped backend for a three-agent AI system (Director /
Specialist / Runner) built for a biotech company, exposed via a REST + SSE
API for a separate frontend to consume. See `docs/ARCHITECTURE.md` for the
full design, `docs/HARDWARE.md` for GPU sizing, and `contracts/types.ts`
for the frontend's TypeScript contract.

## Quick start — mock mode (no GPUs required)

Mock mode runs the entire system — routing, the Director/Specialist/Runner
loop, tools, compliance layer, event stream — against fake model responses
with simulated latency, so the frontend can be built and demoed without the
GPU server.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export AGENT_PACK_MODE=mock          # default if unset
export DATABASE_URL=sqlite:///./agent_pack.db
export AIRGAP_CHECK=false            # the air-gap check is only meaningful in real mode

uvicorn app.main:app --reload
```

Then:

```bash
curl -s -X POST localhost:8000/sessions -d '{"user_id":"alice"}' -H 'content-type: application/json'
curl -s -X POST localhost:8000/messages -d '{"session_id":"<id>","message":"What is the boiling point of water?"}' -H 'content-type: application/json'
curl -N localhost:8000/sessions/<id>/events   # SSE stream
```

Or with Docker:

```bash
docker compose up --build
```

## Real mode

Real mode requires the on-prem GPU server with vLLM (or SGLang) serving
the three models declared in `config/models.yaml`, and Postgres/Redis
reachable per `.env.example`.

```bash
cp .env.example .env
# edit .env: AGENT_PACK_MODE=real, DATABASE_URL=postgresql://..., model endpoints
docker compose --profile real up --build
```

Before going live, read `docs/HARDWARE.md` for whether the three models
fit your GPU server and `docs/ARCHITECTURE.md` section 3 for the model
version caveats (recent releases — re-verify against vendor model cards).

## Project layout

```
app/            FastAPI app: schemas, events, orchestrator, agents, models,
                task_store, tools, compliance, api, db
config/         models.yaml, budgets.yaml — the only place model names and
                token budgets are defined
contrib/        reference implementations that a customer replaces — currently
                the datacenter transcript connector (see docs/MEETINGS.md)
contracts/      types.ts — TypeScript mirror of app/schemas for the frontend
data/           sample internal-docs/literature corpora for the RAG tools
                (replace with the company's real corpus)
docs/           ARCHITECTURE.md, HARDWARE.md
tests/          pytest suite: routing, retry/escalation, PHI redaction,
                audit logging, token budgets
docker-compose.yml, Dockerfile
```

## Tests

```bash
pip install pytest pytest-asyncio
PYTHONPATH=. pytest
```

Tests run entirely in mock mode against a temporary SQLite database — no
external services required.

## Key design points (see docs/ARCHITECTURE.md for the full version)

- **Routing**: simple → Runner only; normal → Specialist + Runner; hard or
  high-stakes (regulatory/clinical/safety/final-report) → Director plans
  and reviews, with a max-2-retries-then-human-escalation loop.
- **Token efficiency**: static system-prompt prefixes, reference-passing
  through the task store, per-agent-per-task budgets
  (`config/budgets.yaml`), Runner-driven summarization/compaction, RAG
  top-k chunking, and a query/doc-hash result cache.
- **Compliance**: append-only audit log (every event on the bus is
  logged), Runner-based PHI redaction before logging/handoff, RBAC-filtered
  document search, human approval gates on high-stakes final output, and a
  startup air-gap check.
- **Meeting transcripts**: `meeting_transcripts` answers questions about the
  company's own Zoom meetings, citing meeting, date, and timestamp. It reads
  from the customer's datacenter connector over the internal network — never
  through the public Zoom relay, which would defeat the air gap. Unconfigured
  deployments get a clear "not configured" result rather than an error.
  Transcripts are parsed by the reader in bedrock-private-ai, pinned in
  `requirements.txt`, rather than a copy kept here. See `docs/MEETINGS.md`.
- **Swappable everything**: model names/endpoints only ever live in
  `config/models.yaml`; the `ModelClient` protocol swaps mock/real/serving
  engine without touching agent code; tools are stubbed with a clear
  interface wherever a real company system (ELN/LIMS, structure
  prediction, local BLAST) wasn't available to wire up.

## Known simplifications (flagged, not hidden)

- The task store / audit log use synchronous SQLAlchemy against SQLite by
  default; point `DATABASE_URL` at Postgres for real deployment, and
  consider async SQLAlchemy + asyncpg if request volume becomes a
  bottleneck.
- The event bus is in-process (single worker); swap for Redis pub/sub
  behind the same interface (`app/events/bus.py`) for a multi-worker
  deployment.
- `app/tools/lab_connectors.py` and the structure-prediction path in
  `app/tools/sequence_tools.py` are stub interfaces — no specific
  ELN/LIMS/structure-tool product was named; wire a real client behind the
  existing interface once one is chosen.
