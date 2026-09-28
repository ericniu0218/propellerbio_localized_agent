# Scientist workspace implementation

## Direction and existing surfaces

The September 16 redesign brief supersedes the terminal-first UI guidance for
this work. Existing Textual commands remain supported. Work is on feature
branches in both repositories; no direct main push or merge.

The backend is FastAPI with an existing three-agent orchestrator, REST/SSE,
Keycloak redirect flow, and SQLAlchemy task store. The client repo contains
Textual chat, file digestion, sequence tools, meeting transcription, local
workspaces, provenance, and a separate reference HTML page. These features
are not all exposed by server APIs; importing the terminal UI into a web
request would not provide authorization or a safe execution boundary.

## Implementation plan

- Modular browser components in `frontend/web-client/workspace/`, served by
  FastAPI at `/`. No CDN, external fonts, analytics, or frontend build required.
- Server workspace APIs under `/workspace/api/` coexist with existing APIs.
  SQLite JSON records store metadata with stable IDs; artifact bytes live
  separately on disk. The store always queries with the authenticated owner.
  Interfaces isolate the future company database implementation.
- Explicit demo environment has its own database/blob directory and seeded
  fictional project, meetings, and tasks. Real mode never seeds examples or
  silently falls back to demo answers. Demo identities are clearly labeled.
- Cookie identity for the new API; the existing Keycloak callback issues the
  cookie after verification. The legacy API security gap remains separate.
- Chat context is rebuilt server-side from saved messages, attached artifacts,
  and relevant personal/project memory. References use IDs, not local paths.
- Demo AI produces labeled deterministic output and real file statistics;
  the real provider adapts the existing orchestrator and events. Runs persist
  with steps, outputs, errors, and unknown progress where appropriate.
- Graphical file picker/drop, preview, context removal, memory creation,
  project organization, meeting import, source-linked tasks, search, and audit.
- Project membership/invites and company/team scope have explicit demo-only
  boundaries until directory-backed authorization is implemented. No UI grant
  permits another user to read server records.
- Lightweight optional Python/pywebview desktop shell opens a configured
  company URL. It contains no model runtime and no privileged JavaScript bridge.
  This is a runnable development shell, not a signed distributable installer.

## Command to visual workflow mapping

| Existing capability | Workspace action |
| --- | --- |
| /digest | Attach file; persisted parsed content automatically accompanies chat |
| /history, /new, /resume | Searchable/pinnable chat sidebar and new chat |
| /workspace | Project tabs, membership preview, files and memory |
| meeting transcripts | Meetings import using one transcript extraction service |
| /fold, /pymol, /confidence | Structure tool screen; unavailable tools labeled |
| /audit, /runs | Read-only activity and agent-run cards |
| model selection | Automatic routing; optional composer mode |

## Boundaries

PDF parsing uses optional pypdf; Office XML extracts text with ZIP expansion
limits. Unsupported binaries stay downloadable and explicitly unparsed. No
claim of OCR, semantic retrieval, HIPAA compliance, production retention,
cross-device directory membership, or real AlphaFold execution is made.
Sensitive-looking text is quarantined from model context until reviewed.
Demo meeting action extraction is labeled and requires user review.

## Verification and handoff

Test owner isolation, file persistence/parse, sensitive quarantine, run context,
and demo separation. Drive real browser upload/chat/project/memory flow,
refresh persistence, panel resize/popout, and narrow viewport. Preserve the
existing database and running service configuration on restart. Publish changes
only through reviewed branches/PRs. Package signing, hardened deployment,
central ACLs and legacy API authentication remain prerequisites for production.
