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

### 2. Real login (replace `BEDROCK_USER`)
Right now `settings.user_id` is a plain environment variable — a
stand-in, not authentication. The backend's plan (`docs/WORKFLOW.md` §1)
is company SSO via Keycloak/OIDC. To wire that up here:
- Add an OIDC device-code or browser-redirect flow (Textual can shell out
  to the system browser and listen on a local port for the callback).
- Send the resulting token as an `Authorization: Bearer <token>` header on
  every `AgentPackClient` request instead of a plain `user_id` field.
- The backend doesn't enforce auth yet either — this is a two-sided piece
  of work, not just a client change.

### 3. Sharing / uploads / recipes / workspaces
`docs/WORKFLOW.md` on the backend specs a whole product layer — My
Space/Team Spaces/Company Knowledge, drag-and-drop file summarization, a
standard "brief" format with a length slider, reusable "recipes,"
scheduling, notifications. **None of it is implemented in this client**,
and most of it isn't implemented on the backend yet either (check the
backend's current state before assuming). A terminal app is a bad fit for
drag-and-drop file upload specifically — that will likely need either a
file-path prompt (`/attach <path>`) here, or push people toward the web
client for that particular workflow. Worth a real product decision before
building it either place.

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

### 5. Reconnection
`AgentPackClient.events()` opens one SSE connection for the life of the
app. If the backend restarts or a network blip drops it, this client does
not currently retry — it just reports the error and goes quiet. Worth a
retry-with-backoff loop before this goes in front of real users on a real
network.

## Testing

`pytest` (93 tests) covers the client's REST/SSE parsing
against a fake server and the full Textual app against a fake
`AgentPackClient`. It does **not** cover the real backend's behavior —
that's the other repo's test suite. When testing changes here against a
live backend, see the README's "Two modes" section.
