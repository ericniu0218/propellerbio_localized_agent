# Scientist Workspace UI (mirrored copy)

This is a copy of the new browser-based workspace (chat, projects,
memory, meetings, agent activity, Zoom meeting-transcript integration)
built for the redesign brief. It's mirrored into this repo for
visibility, but it doesn't run from here.

**Where it actually runs:** `bedrock-agent-pack` (the FastAPI backend),
on branch `feature/scientist-workspace`
([PR #6](https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack/pull/6)).
That server serves `app/static/workspace/*` at `http://localhost:8000/`
and exposes the API those pages call (`app/workspace/routes.py` etc.).
This Textual terminal client repo has no web server, so these files are
inert here — they're included as-is for reference.

To actually run the workspace: clone `bedrock-agent-pack`, check out
`feature/scientist-workspace`, and run `./scripts/demo-safe.sh` (or
`./scripts/demo-zoom.sh` for the real Zoom-transcript integration — see
that script's header for one-time setup). Both scripts are duplicated
here for convenience but are meant to be run from that repo's root.

## Contents

- `app/static/workspace/` — the UI itself (HTML/CSS/JS, no build step)
- `app/workspace/` — the FastAPI backend it talks to
- `WORKSPACE_ARCHITECTURE.md` — the implementation plan this was built from
- `demo-safe.sh` / `demo-zoom.sh` — one-command demo launchers
