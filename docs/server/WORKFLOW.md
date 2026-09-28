# Workflow — Employee Sharing, Reuse, and Condensing Layer

Status: **draft, awaiting approval before implementation** (this document
and `docs/server/UX.md` are the deliverable for build-order step 1; nothing else
gets written until both are approved).

This extends `docs/server/ARCHITECTURE.md` (the Director/Specialist/Runner
backend) with the product layer scientists, lab staff, and engineers
actually touch: login, workspaces, sharing/review, drag-and-drop
condensing, and reusable recipes. Nothing in the existing API or event
contract is removed or renamed — everything below is additive.

## 1. Login and access

Employees open `https://ai.company.internal` in a browser — nothing
installed. Login is the company's existing work account via SSO
(Keycloak, OIDC). No new password. On first login, we create a `User` row
keyed by the identity provider's subject ID, with email and display name
from the token, and a **My Space** workspace for them automatically —
there's no "create your workspace" step.

Department and Company-Knowledge-reviewer assignment is **managed inside
this app**, not read from Keycloak (see decision log below): an admin
screen lets someone set "Jane = Chemistry department" and "Chemistry's
reviewer = Bob," independent of how the company's SSO happens to be
configured. This unblocks launch without needing IT to restructure
Keycloak first.

The frontend is a normal web app today; nothing here assumes a specific
JS framework or blocks a later Tauri desktop wrapper — the backend only
cares that requests carry a bearer token and a session ID, which works
identically from a browser tab or a wrapped native window.

## 2. Workspaces: My Space, Team Spaces, Company Knowledge

Three workspace types, same shape in the data model:

- **Personal** — one per user, auto-created, owner-only visibility. The
  default destination for every new chat, upload, and result.
- **Team** — created for a lab/project/department; members see and reuse
  everything in it. Anyone can be added by a team admin.
- **Company** — one singleton, company-wide. Anything here is visible to
  every employee by default (still subject to whatever document-level
  access rules a specific item carries, if any are added later).

A workspace membership table records who's in which team and at what role
(member or admin). Every piece of content — a chat/task, an upload, a
summary card, a recipe — is tagged with exactly one workspace at any given
time (its current "home"); sharing (below) is how it moves or gets copied
to another one.

**Permission enforcement happens at the data layer, not the UI.** The RAG
search tool (and every other read path into shared content) is scoped in
SQL by "workspaces this user belongs to, plus their own personal
workspace" — an agent acting on a user's behalf can only ever retrieve
rows that query already excludes, so there's no code path where an agent
"forgets" to check a permission the UI would have hidden. This is the same
principle as the RBAC filter already built for internal-doc search;
that filter is generalized to real workspace membership.

## 3. Pushing: Share

Any chat, answer, report, uploaded file, or saved recipe has one **Share**
action with three destinations:

1. **A person** — shows up in their Shared-with-me.
2. **A Team Space** — visible to every member immediately.
3. **Company Knowledge** — visible to everyone, but only after review.

**Review gate for Company Knowledge only.** Sharing to a Team Space or a
person is immediate. Sharing to Company Knowledge creates a review request
routed to that department's reviewer(s) (set by an admin — see §1), with a
plain status shown to the submitter: **Pending review → Approved / Needs
changes**. A "needs changes" decision carries the reviewer's comment back
to the submitter; nothing silently disappears.

**PHI check before any share completes.** Before the share is created
(any destination — a share to one teammate can still leak patient data),
the Runner's existing redaction/detection pass runs over the content. If
it finds something that looks like patient data, the share is paused with
a plain-language warning ("This contains what looks like patient names.
Remove them before sharing?") and a one-click "share anyway" only for
destinations the user already has legitimate access to redact/edit — the
share does not silently proceed with PHI attached.

**Versions.** A shared item can be updated by its owner after the fact;
each update is a new version, and everyone who has it (via a direct share,
a team, or Company Knowledge) sees "Updated by [name] on [date]" with a
one-click diff view rather than being silently out of sync.

**Staleness flag.** If a shared item hasn't been reviewed or updated in
longer than a configurable threshold (`config/freshness.yaml`), it's
flagged as possibly outdated wherever it's shown — a hint to check it, not
a hard block.

## 4. Pulling: reuse what's already been done

- **Shared with me** — a searchable list of everything shared to the user
  directly, plus everything visible via their team memberships.
- **Add to chat** — on any shared item (a doc, a past answer, a recipe
  output), pulls it into the current conversation as context the agents
  can use — this is just handing the task store a reference, the same
  reference-passing mechanism the agents already use internally, so no
  new "how do agents see this" mechanism is needed.
- **`@` mention** — typing `@` in the chat box opens a picker over
  teammates' shared items, Team Spaces, and documents by name. Purely a
  faster path to the same "Add to chat" action; every `@` capability has an
  equivalent button, per the no-commands-required rule in `docs/server/UX.md`.

## 5. Condensing data (the core feature)

- **Drag-and-drop**, single or multiple files at once: PDF, Word, Excel,
  CSV, PowerPoint, and instrument/plate-reader CSV exports (the last of
  these reuses the existing `eln_lims` CSV parser). Each file is stored on
  local disk on the GPU server and its text extracted per format.
- **Automatic summary card**, produced by the Runner the moment extraction
  finishes: what the file is, key findings, key numbers, anything unusual.
  Small files finish inline; large files process in the background with a
  progress indicator (`upload.processing` events carrying percent-complete,
  then `upload.summarized` when the card is ready).
- **One brief format everywhere**: Bottom line → Key findings → Key
  numbers (table) → Open questions/risks → Sources (clickable, opens the
  exact passage/row highlighted). Every claim traces to a citation or a
  tool result; if something can't be backed by one, the brief says so
  instead of guessing — this reuses the Director's existing
  citation-required review rule.
- **Length slider** (One line · Short · One page · Full detail) re-renders
  the *same* underlying structured brief — nothing is recomputed, so
  moving the slider is instant.
- **Compare**: pick 2+ documents/datasets, get agreements / conflicts /
  gaps, each claim still cited back to its source.
- **Extract tables**: pull tabular data out of PDFs/reports into a
  downloadable Excel/CSV. (Scanned/image-only PDFs with no text layer are
  a known limitation — flagged plainly rather than silently producing an
  empty table.)
- **Catch me up**: one button on a Team Space or project; summarizes what
  changed since the user's last visit, in brief format.
- **Search everything in plain English**: one search box answers
  cross-document questions ("what did we find about compound X stability
  last year?") using the same permission-scoped index as everything else,
  in brief format with citations.

## 6. Recipes — reusable one-click workflows

- **Save as Recipe** on any completed chat turns its request into a
  reusable template with simple input fields (e.g. "Drop your plate-reader
  files here") — no prompt writing. Under the hood this is a template
  string with `{{field}}` placeholders plus a small input schema; running
  it fills the template and hands it to the same orchestrator loop as a
  normal chat message.
- Recipes share exactly like any other item (My Space → Team → Company,
  same review gate for Company).
- **10 starter recipes** ship pre-approved in Company Knowledge: literature
  summary, study report draft, SOP draft, assay data summary, compare two
  protocols, meeting notes → action items, regulatory section first draft,
  deviation report draft, weekly assay results summary, catch me up on a
  project.
- **Scheduling**: a recipe can run on a cron schedule (e.g. every Monday
  8am), delivering its result to the owner's space plus a notification —
  runs via an in-process scheduler (no new external job system).

## 7. Notifications

In-app (a notification list + live updates over a new per-user event
channel — additive, the existing per-session stream is untouched) and
email via the company's internal SMTP relay, for: a scheduled recipe
finishing, a review decision, and (optionally, per-user setting) a
long-running task completing while the user has navigated away.

## 8. Data model additions (Postgres, alongside the existing tables)

`users`, `departments`, `department_reviewers`, `workspaces`,
`workspace_memberships`, `uploads`, `summary_cards`, `content_chunks`
(the permission-scoped searchable index), `shared_items`,
`shared_item_versions`, `reviews`, `recipes`, `schedules`,
`notifications`, `usage_metrics`. Full field-level detail lives in the
code (each gets a short docstring, same convention as `src/bedrock/server/db/models.py`
today) rather than duplicated here.

## 9. New REST endpoints (grouped; exact paths finalized in code)

- Auth: session bootstrap from an SSO token (mock mode: a header).
- Org admin: set a user's department; set a department's reviewers and
  starter recipes.
- Workspaces: create/list team workspaces, manage membership.
- Uploads: upload file(s), get upload status/summary card, extract tables.
- Sharing: share an item, list Shared-with-me, get/decide a review,
  list versions/diff.
- Recipes: create (incl. "save as recipe"), list, run, schedule/unschedule.
- Notifications: list, mark read.
- Metrics: admin-only aggregates (adoption, time saved, review queue size).
- Compare: submit 2+ item references, get a comparison brief.

## 10. New event types (appended to the existing `EventType` enum)

`upload.processing`, `upload.summarized`, `share.created`,
`review.requested`, `review.decided`, `item.updated`, `recipe.started`,
`recipe.completed`, `notification.created`. All existing event types are
unchanged. `AgentCard`/`AgentStatusPayload` gain two additive fields
(`friendly_label`, `trust_status`) — see `docs/server/UX.md` for what populates
them.

## 11. Open items already resolved with the user

- Departments/reviewers: admin-managed in-app, not pulled from Keycloak.
- File storage: local disk on the GPU server.
- Notifications: in-app + SMTP email.
- Scheduling: in-process (APScheduler), no external job system.

## 12. Still out of scope

Training/fine-tuning models (explicitly excluded). Building the actual
frontend UI (a separate team's work — this phase ships the backend,
contracts, and the UX/copy rules they build against). OCR for scanned
PDFs (flagged as a known limitation of table/text extraction, not solved
here).
