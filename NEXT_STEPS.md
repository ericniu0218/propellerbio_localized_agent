# Next steps for the agent-pack integration

This documents where the "agentpack" mode (added in this branch) stands
today and exactly what's left to make it production-ready. Read this
before picking up more work here.

## Where the actual AI system lives

This repo is the **client only**. The three-agent backend it talks to
(Director/Specialist/Runner — planning, research, review, approvals,
compliance/audit logging) is a **separate project**, currently only local
to the machine that built it (not yet pushed to GitHub — ask whoever ran
this integration for it, or push it and update this section with the
repo link). Its docs, which this client's behavior is built to match, are:

- `docs/ARCHITECTURE.md` — the backend's overall design
- `docs/WORKFLOW.md` — the sharing/uploads/recipes product plan (**mostly
  not built yet** — see below)
- `docs/UX.md` — the plain-language rules this client's `friendly.py`
  implements

## What's actually working right now

- `BEDROCK_BACKEND=agentpack` mode: real sessions, real live event stream,
  a plain-language "what's happening now" strip, and an inline
  Approve/Needs-changes prompt for high-stakes answers.
- All of it runs against **mock mode** on the backend — no real language
  model anywhere yet. Every reply comes from a fixture generator that
  fakes plausible-looking text.
- A minimal reference *web* client also exists on the backend itself
  (`app/static/index.html`, served at `/`) — same idea, same API, browser
  instead of terminal. Not a replacement for this app; built so someone
  without a terminal could see the system working.

## What's next, in priority order

### 1. Real models
The backend's `config/models.yaml` names DeepSeek/Qwen models and knows
how to call a vLLM/SGLang OpenAI-compatible endpoint
(`app/models/http_client.py` on the backend) — but nothing is deployed.
Once the company's GPU server is up and `AGENT_PACK_MODE=real` on the
backend, this client needs **no changes** — same REST/SSE contract either
way. Just point `AGENT_PACK_API_BASE` at the real server.

### 2. Real login (replace `BEDROCK_USER`) — backend + web client done, this one isn't
As of 2026-09-14, the backend (`bedrock-agent-pack`, `app/auth/`) and the
web reference client (`web-client/index.html`, and the original it's
copied from) do real Keycloak login — a "Log in with company account"
button, a verified redirect round-trip, sessions tied to the real
Keycloak username. Tested end-to-end against a local Keycloak instance.

**This terminal client still doesn't have it.** `settings.user_id` here
is still just a plain environment variable — a stand-in, not
authentication. To bring it up to par with the web client:
- Add an OIDC browser-redirect flow: shell out to the system browser for
  the Keycloak login step, and run a short-lived local HTTP server
  (`http.server` or similar) on a loopback port to catch the redirect
  back with the authorization code — Textual apps can do this fine, it's
  just not built yet.
- Reuse the backend's existing `/auth/login` and `/auth/callback` routes
  rather than reimplementing the OIDC handshake here — point the local
  callback server's redirect at wherever the backend already listens, or
  add a client-specific redirect URI to the Keycloak client if a
  different local port is needed.
- Once a session is established this way, keep using `AgentPackClient` as
  today (it already talks to sessions by id) — the only change is *how*
  `user_id`/the session gets created, not the ongoing request shape.

### 3. Centralize shared workspaces after SSO
The local workspace MVP is now implemented in this client. `/workspace`
provides selection, invitations, roles, preview-before-publish from `/digest`,
immutable results, folds, integration keys, export, and events in a private
SQLite vault. See `docs/SHARED_WORKSPACES.md` for its contract and limits.

It deliberately does not pretend that a file on one machine is a cross-device
service. After OIDC identity is available, move `WorkspaceStore` behind the
private agent-pack backend and retain the same artifact interface. That is the
right point to add My Space/Company Knowledge, uploads, recipes, scheduling,
notifications, a blob store for results over 256 KiB, and mutually untrusted
multi-host access. The terminal UX should continue to use paths and explicit
previews rather than browser drag-and-drop.

### 4. Streaming into the chat bubble
Right now the chat bubble shows "Working on it…" and then the finished
answer — it does not stream token-by-token like the original `BedrockApp`
does for a plain model server. The backend's `agent.token` events carry
the live text; `agentpack_app.py::handle_event` currently ignores that
event type on purpose (see the module docstring) because a request can
involve *multiple* agents drafting in sequence (Specialist drafts, then
Runner reformats) and naively streaming both would look like the answer
being rewritten mid-stream. If this is worth fixing, the likely approach
is: stream only the *last* agent's tokens once a `task.completed` for this
turn's task tree is imminent, which needs a bit more state than exists
today.

### 5. Reconnection — done 2026-09-16
`AgentPackClient.events()` now reconnects automatically on a dropped
connection (including the server just closing the stream cleanly, e.g. a
restart — that's still a drop from this client's perspective, never
successful completion) with exponential backoff and jitter (1s → 30s
ceiling, equal jitter), sending back the last `id:` it saw as
`Last-Event-ID` so the backend resumes rather than replays everything.
A 4xx response (e.g. the session no longer exists) is *not* retried —
that's a permanent failure, not a transient drop, and raises
`InferenceError` same as before.

Two synthetic event types the backend never sends, only this client:
`client.reconnecting` (payload: `attempt`, `delay_s`) each time a drop is
detected, `client.reconnected` once a new connection is established. The
UI (`agentpack_app.py`) shows this as a `StatusBar.reconnecting` indicator
plus one system notice per outage (not one per retry attempt) — see
`_handle_reconnecting`/`_handle_reconnected`.

**Server side** (see the backend's own PR/commit for detail): every event
now carries a monotonically increasing per-session `seq`, sent as the SSE
`id:`; the backend keeps a bounded per-session buffer (default 500) rather
than unbounded history. If a reconnect's `Last-Event-ID` is older than
what survives in that buffer, the backend sends an explicit
`session.resync` event instead of silently replaying a partial window —
handled here by `_handle_resync`: reset the agent panel, replay whatever
window the backend still has, then check its `terminal` field (the
session's most recent completion/failure, kept outside the bounded buffer
specifically for this) so a pending answer resolves even if the exact
event that completed it was evicted before reconnecting.

A disconnect never cancels the underlying job — that was already true
(the backend runs `handle_message` as an independent task, decoupled from
any SSE connection) and remains true; this is purely about the *client*
catching back up, never about the job's lifecycle.

Tests: `tests/test_agentpack_reconnect.py` (backoff/jitter, Last-Event-ID
propagation, 4xx-vs-5xx/connection-error retry distinction, attempt
counter reset after a successful reconnect) and the `session.resync`/
`client.reconnecting`/`client.reconnected` cases in
`tests/test_agentpack_app.py`.

**Known limitation, not fixed here, and not fixable from this repo**: the
backend's buffer and terminal-event tracking are in-memory. A backend
*process restart* still loses all of it — a reconnecting client gets
`session.resync` with `reason: "no_history"` and starts fresh rather than
hanging, which is the best available behavior without durable job storage
on the backend (out of scope for this task; see that repo's Workstream 1
notes on Postgres/durability).

## Testing

`pytest` covers the client's REST/SSE parsing
against a fake server and the full Textual app against a fake
`AgentPackClient`. It does **not** cover the real backend's behavior —
that's the other repo's test suite. When testing changes here against a
live backend, see the README's "Two modes" section.
