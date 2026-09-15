# Web client (reference copy)

`index.html` here is a copy of the minimal browser-based reference client
that actually lives (and is served from) the backend repo:
`bedrock-agent-pack`, at `app/static/index.html`. It's included here too
so it's easy to find alongside the terminal client, since people keep
looking for it in this repo.

**This file does nothing on its own.** It's a static page that calls a
running agent-pack backend's REST + event-stream API (session creation,
messages, live agent status, approvals) — opening it directly in a
browser (`file://...`) will not work; it needs to be served by that
backend, which also needs to be running, e.g.:

```
# in the bedrock-agent-pack repo
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

then open `http://localhost:8000` (the backend serves this same file at
its own `/` route). If you edit this copy, also update the original in
`bedrock-agent-pack` — this is a duplicate for visibility, not the source
of truth.

**Login (as of 2026-09-14):** this page now has a real "Log in with
company account" button, not just a typed name — but that only does
anything if the backend it's talking to has Keycloak configured
(`KEYCLOAK_BASE_URL`/`REALM`/`CLIENT_ID`/`CLIENT_SECRET` — see
`bedrock-agent-pack`'s `.env.example`). Without those set, the page falls
back to the old "type a name" box automatically — nothing to configure
here, the page just asks the backend at `/auth/config` which mode it's in.
All of the actual login logic (`app/auth/`) lives in `bedrock-agent-pack`,
not here — there's nothing to port into this repo for that part.

**The terminal client (`bedrock` command) does not have this yet** — it
still uses the `BEDROCK_USER` environment variable as a stand-in. Giving
it real login would mean opening the system browser for the Keycloak
step and running a short-lived local callback server to catch the
redirect — see `NEXT_STEPS.md`.

**This is not the primary product surface.** The terminal client
(`bedrock` command, this repo's main content) is — see the root
`README.md` and `NEXT_STEPS.md`. This page exists as a quick, no-install
way to look at the same backend from a browser instead of a terminal.
