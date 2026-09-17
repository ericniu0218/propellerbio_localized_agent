# Roadmap: agents that can actually do things

How an employee gets several agents that reach the internet, their apps and
their machine — without this becoming a website, a cloud service, or a program
that emails somebody by surprise.

## The constraint that shapes everything

This product's promise is that nothing leaves the machine. Giving an agent a
browser and a mail client is the single largest threat to that promise, so the
architecture is built around *evidence, plans and grants*, not around tools:

- An agent proposes. A person disposes. `Plan` is inert; only a `Grant`
  executes it. This mirrors the approval gate the agent-pack mode already has.
- Every capability is configured, never discovered. Endpoints come from
  `config/settings.py` exactly as connectors do. No agent may invent a host.
- Every run writes to the existing append-only journal in `bedrock.provenance`.
  An agent that touched the outside world and left no record is the failure
  mode that ends the product.
- `BEDROCK_USER` is not authentication, and this makes that gap load-bearing
  rather than cosmetic. **Multi-agent execution does not ship to a customer
  before OIDC does.** Say so in the release notes, not in a footnote.

## The shape

```
Ask / user request
      ↓
Planner          agents/planner.py     Ask → Plan (model, behind InferenceBackend)
      ↓
Approval         ui/ + cli             a person reads the Plan
      ↓
Grant            agents/grants.py      one plan, one person, once
      ↓
Supervisor       agents/supervisor.py  runs Steps, concurrently, with a budget
      ↓
Capabilities     agents/capabilities/  the only code that touches the world
      ↓
Journal          bedrock.provenance    what actually happened
```

`agents/contract.py` is written and frozen. Everything else is new.

## Phase 1 — the kernel, with one fake capability

Build `Capability` as a protocol in the shape `InferenceBackend` already
established: a name, a `risk`, a `describe(arguments) -> str` used to render
the plan, and an async `run(arguments) -> Outcome`.

Ship it with exactly one implementation — `echo`, which does nothing — plus the
supervisor, the grant store and the approval prompt. The whole point is that
the dangerous part is finished and tested before anything can reach the
network.

## Phase 2 — read-only capabilities

`Risk.READ_ONLY` steps may run before a person looks, which makes them the
right second step and the most immediately useful ("have the resources ready
after the summary"):

- **files.search / files.read** — rooted at configured directories. A path
  outside the root is a hard error, not a warning.
- **sources.fetch** — reuse `bedrock/connectors/` as-is. It already does
  `<source>://<id>`, provenance sidecars and configured endpoints.
- **digest.query** — reuse `bedrock/context/`. Asking an agent to read a large
  document is a solved problem here.

Note that two of the three are wiring, not new systems. That is deliberate.

## Phase 3 — the internet, narrowly

- **web.fetch** — one URL, through a configured egress proxy, allowlisted by
  domain, with the response size capped and the body stored locally before any
  model sees it.
- **web.search** — a self-hosted SearxNG instance named in config. Not a
  commercial search API: the query text is the leak.

Both are `READ_ONLY` but they are the first code that transmits anything, so
they get their own review and their own journal event type. A fetch records
the URL, the hash and the byte count.

## Phase 4 — the apps that change things

Everything here is `REACHES_PEOPLE` or `CHANGES_THINGS` and therefore never
runs without an explicit grant:

- **calendar.create** — CalDAV against the company server. This is what turns
  "let's meet next Tuesday" into a hold.
- **mail.draft** then, separately, **mail.send** — two capabilities on purpose.
  Drafting is `CHANGES_THINGS`; sending is `REACHES_PEOPLE` and is the single
  riskiest verb in this document. Default configuration ships with `mail.send`
  disabled.
- **files.write** — new files under a configured output root only. Never
  overwrite; write `name (2).md` and say so.

## Phase 5 — the computer, and several agents at once

- **shell.run** — the one capability that cannot be made safe by allowlisting
  arguments. It runs in a sandbox with an explicit working directory, no
  network by default, a wall-clock budget, and full command text in the plan.
  It is last for a reason, and it is reasonable to ship without it.
- **Concurrency.** The supervisor runs independent steps together with a
  per-agent budget (steps, seconds, bytes). Agents share nothing but the
  journal; there is no inter-agent messaging in v1, because an agent that can
  instruct another agent turns one approved grant into an unbounded one.
- **A roster.** `~/.config/bedrock/agents.toml` names each agent and the
  capabilities it may use. A capability absent from the roster is not merely
  denied at runtime — it is never offered to the planner, so it cannot appear
  in a plan and be approved by a tired person at 6pm.

## What is deliberately not here

Agent-to-agent delegation, standing grants, autonomous background loops,
learned permissions, and anything that runs while the terminal is closed.
Each is a reasonable v2 and each breaks the "a person read this plan" property
that makes the rest defensible.

---

## Splitting this between two models

The seam is **the kernel vs. the outside world**. One model builds the thing
that decides and records; the other builds the adapters that touch reality.
They share only `contract.py`, which is already written, and a protocol file
that A1 writes on day one and then freezes.

### Model A1 — "nothing runs that was not approved"

Owns Phases 1 and 5's concurrency. Files (all new):

- `src/bedrock/agents/protocol.py` — the `Capability` protocol. **Written
  first, then frozen.** A2 codes against it and never edits it.
- `src/bedrock/agents/planner.py`
- `src/bedrock/agents/supervisor.py`
- `src/bedrock/agents/grants.py`
- `src/bedrock/agents/roster.py`
- `src/bedrock/agents/capabilities/echo.py` — the fake one, for tests
- `src/bedrock/ui/agent_command.py` — the approval prompt
- `tests/test_agents_kernel.py`

**Must not touch:** `agents/capabilities/` beyond `echo.py`, anything under
`meetings/`, `connectors/`, `context/`.

### Model A2 — "the world, one adapter at a time"

Owns Phases 2, 3, 4 and `shell.run`. Files (all new, one per capability):

- `src/bedrock/agents/capabilities/files.py`
- `src/bedrock/agents/capabilities/web.py`
- `src/bedrock/agents/capabilities/sources.py`
- `src/bedrock/agents/capabilities/calendar.py`
- `src/bedrock/agents/capabilities/mail.py`
- `src/bedrock/agents/capabilities/shell.py`
- `tests/test_capabilities_*.py` — one file per capability
- `docs/AGENT_CAPABILITIES.md`

**Must not touch:** the kernel files above, `contract.py`, `protocol.py`.

### The two places they would otherwise collide

1. **Capability registration.** Do not have A2 edit a registry A1 owns. A1
   writes `roster.py` to discover capabilities by importing
   `agents/capabilities/<name>.py` and reading a module-level `CAPABILITY`.
   Adding an adapter then means adding a file, never editing one.
2. **Configuration.** Both need `config/settings.py`. A1 adds the whole
   `[agents]` block — roster path, budgets, and one `[agents.capabilities.*]`
   sub-table per planned adapter, empty — in its first commit. A2 fills values
   in its own TOML fixtures and never edits `settings.py`.

### How they join up

Model B2's `Ask` (see `MEETING_INTELLIGENCE.md`) is A1's planner input. Until
B2 exists, A1 builds `Ask` values by hand in tests. Until A1 exists, B2 renders
asks as a `## Requests` section and nothing executes. Neither workstream blocks
the other at any point, and both are independently useful if the other is
cancelled.
