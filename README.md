# Bedrock Private AI

A terminal chat client for language models hosted inside your own company.
Nothing you type ever leaves your network: the client talks only to the
inference server you configure.

```
┌ Bedrock Private AI                       model: llama3.2:3b   ● connected ┐
│                                                                            │
│ ▌ Can you show me a Python function?                                       │
│                                                                            │
│ ▌ Sure! Here is a quick example:                                           │
│ ▌   def greet(name):                                                       │
│ ▌       return f"Hello, {name}"                                            │
│                                                                            │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ Type a message, or /help for commands                                 │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
└ ^c Quit  ^l Clear ─────────────────────────────────────────────────────────┘
```

## Installing (not developing)

Just want to run this, not work on its code? See **[INSTALL.md](./INSTALL.md)**
for a no-Python-experience-assumed walkthrough — one command, no virtual
environment. The rest of this README is for contributors.

## Quick start (local development)

1. Install [Ollama](https://ollama.com) and pull a small model:

   ```
   ollama pull llama3.2
   ```

2. Install and run the client:

   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   bedrock
   ```

The status bar shows `● connected` once Ollama is reachable. If it says
`● disconnected`, start Ollama (`ollama serve`) - the client re-checks every
10 seconds.

## Commands

| Command          | What it does                     |
|------------------|----------------------------------|
| `/model`         | List models on the server        |
| `/model <name>`  | Switch model                     |
| `/digest <path>` | Compress a file into local context storage |
| `/digest`        | List/search locally stored context |
| `/sources`       | List company data sources        |
| `/digest <src>://<id>` | Pull a record from a source and compress it |
| `/runs`          | Prediction runs recorded on this machine |
| `/confidence <run-id>` | pLDDT bands, low-confidence regions, PAE |
| `/audit`         | What this client has done on this machine |
| `/workspace`     | List private team workspaces     |
| `/workspace help`| Show collaboration actions       |
| `/history`       | List saved conversations         |
| `/resume <n or name>` | Reopen a saved conversation by number or title words |
| `/new`           | Start a new conversation         |
| `/forget`        | Delete this conversation from disk |
| `/recall <words>`| Pull matching parts of past chats into this one (also `/context`); `/recall clear` removes them |
| `/help`          | Show help                        |
| `/exit`          | Quit (also `Ctrl+C`)             |

`Esc` stops a response that is still streaming. `/clear` is an alias for `/new`.
`Ctrl+B` shows or hides the conversation sidebar.

### Attaching a file

Drag a file onto the terminal window, or write `@` before its path anywhere in
a message:

`What are the last 10 letters of candidate_3_control in @~/Downloads/candidate_proteins.fasta?`

A dropped file appears in the prompt as an `@` reference, so you can see what
will be attached and type your question around it. Quote a path that contains
spaces (`@"My Data.csv"`); dropped files are quoted for you. Up to five files
can go with one message. A path written *without* `@` is ordinary text, so
"check the /api endpoint" is never mistaken for a file.

The client reads the file **on this computer** and sends it with the question
to the configured model or agent-pack server, which cannot open a path on your
laptop itself.

- A text file up to about 24,000 characters is sent whole.
- A larger one is compressed the way `/digest` does it, and the model receives
  the digest (structure and statistics, not every line). A notice says so, and
  `/digest search` finds exact passages in it afterwards.
- UTF-8 text, FASTA, CSV, JSON, code and logs work. Files that are not UTF-8,
  and binary formats such as PDF, Word and Excel, are refused rather than
  guessed at. Nothing is sent when a file cannot be attached, and your draft
  stays in the input.

The model is told to treat an attached file as data, not as instructions, and
the file is fenced so its text cannot pose as your message. That lowers the
risk from a file that contains instructions; it cannot remove it, so attach
files you would be comfortable pasting.

The chat shows your question with a 📎 line per file. The file's contents are
saved with that message in the conversation, so follow-up questions still see
it after a restart, and `/forget` deletes it along with the chat.

## Private shared workspaces

`/workspace` provides native terminal collaboration backed by a private SQLite
vault. `/digest` prepares a compact local result, `/workspace publish` previews
the exact digest and bounded source extract, and `/workspace confirm` writes
the reviewed artifact. It supports owner/editor/viewer roles, single-use
invitations, immutable artifacts and lineage, replay-safe publication,
integration keys, deterministic folds, exports, and a resumable event feed.
`/team` is an alias. See [the workspace guide](docs/SHARED_WORKSPACES.md).

`BEDROCK_USER` remains a local identity stand-in; workspace roles are not an
authentication boundary until company SSO is connected.

## AlphaFold 2 and PyMOL

The first biotech workflow is implemented as private local tool adapters:
validate a one-protein FASTA, run the official AlphaFold 2 Docker launcher,
preserve its structure and confidence artifacts, render a PNG and session with
open-source PyMOL, and record hashes and provenance in a per-run manifest.

After operator provisioning, check it with `bedrock-biotech status` and run it
with `bedrock-biotech fold candidate.fasta --project X`. In agent-pack mode the
equivalent commands are `/tools`, `/fold <fasta-path> [--project X]`, and
`/pymol [structure-path]`. Both `/fold` and `bedrock-biotech fold` also accept
a company source reference such as `registry://SEQ-00123` — see "Company data
sources" below.
Direct requests work too: “Run AlphaFold 2 on `candidate.fasta`” or “Please
open the latest structure interactively in PyMOL.” Questions about the tools
remain normal agent conversations.
`/pymol` opens the structure in a desktop PyMOL window with the same four
pLDDT bands the saved preview uses — one fixed script and one palette serve
both, so the window and the PNG cannot disagree. A structure you point at
yourself is styled neutrally: an experimental file's B-factor column is
crystallographic displacement, not model confidence.

See [the integration guide](docs/BIOTECH_INTEGRATIONS.md) for the required
database, container, GPU, privacy, and configuration steps.

## What this machine did (`/runs`, `/confidence`, `/audit`)

Every prediction already wrote a manifest; until now nothing could read one
back. These three commands answer the questions people actually ask about
past work, from local files only — no model is consulted and no network is
needed, so they keep working when the inference server is down.

```text
/runs --project X --since 2026-07-01 --before 2026-10-01   # this quarter
/runs show 20260715T090000Z-3f2a91cc                       # settings, lineage, integrity
/runs verify 20260715T090000Z-3f2a91cc                     # re-hash every recorded artifact
/runs tag 20260715T090000Z-3f2a91cc --project X            # label an older run
/confidence 20260715T090000Z-3f2a91cc                      # pLDDT bands, low regions, PAE
/audit                                                     # every recorded action
/audit export "/approved/review/actions.jsonl"             # metadata-only JSONL
```

The same subcommands exist on the command line: `bedrock-biotech runs`,
`bedrock-biotech confidence RUN_ID`, `bedrock-biotech audit`.

Dates are UTC; `--since` includes its day and `--before` excludes it. Pages
hold 25 runs by default and 100 at most. Tagging never edits a manifest —
the new assignment is a recorded event, and a project label organises results
without granting anybody access to anything.

**`/confidence` is deliberately cautious.** It shows the four published pLDDT
bands with counts and shares, the contiguous stretches below 70 and below 50
as *input sequence positions*, and which model was actually scored — `ranked_0`
is a rank, not `model_1`. Predicted aligned error is drawn as a bounded text
heatmap of the **maximum** error per block, rows aligned and columns scored,
with a numeric ångström legend so colour is never the only signal; the full
matrix stays in its original JSON, whose path is printed. pLDDT estimates
local model confidence: it is not experimental validation or a stability
measurement, and low scores can also reflect disorder.

**`/audit` records attempts, not just successes.** Each tracked action writes
an intent before it starts and an outcome after it finishes. An intent with no
outcome means the result was never confirmed — the work may still be running —
and this client will never invent a failure to fill that gap. Approvals are
sent exactly once: an ambiguous error after dispatch is recorded as *unknown*
and the prompt stays visibly unresolved rather than claiming a decision that
may not have been recorded.

This is local history for one OS account, kept in SQLite under
`BEDROCK_ACTION_HISTORY_DIR` (not `BEDROCK_HISTORY_DIR`, which is where saved
conversations go). It is not an authenticated approval ledger and not
tamper-proof storage; the account owner can alter the file. It stores
identifiers, hashes, decisions and bounded error codes — never file contents,
prompts, approval previews or credentials.

A sequence pulled from a company source is journalled the same way a local
file is, and the run manifest records which registry record and version it
came from — so "where did this prediction's input come from" has an answer
that does not depend on remembering.

## Meeting notes

`bedrock.meetings` turns a meeting transcript into notes using a model you
host. It preserves the user's headings, cites only timestamps that occur in
the transcript, and falls back to computed facts when inference is unavailable.
Run `bedrock-meeting` after a Zoom local recording; see
[the meeting notes guide](docs/MEETINGS.md).

## Saved conversations

Conversations are saved as you go, and the most recent one reopens the next
time you start `bedrock`. The sidebar on the left lists them, most recently
used first: click one to open it, or "+ New conversation" to start fresh.

A conversation gets a short title such as "Sea Animals" once it has a
concrete topic. Until then - "hi", "1", "can you help me" - it is listed as
*Untitled*, and it is checked again after each new reply. Titles are written
by your configured model server, the same one you chat with: a quick YES/NO
"is there a topic yet?" request, then the title itself. If the server is
unreachable, the conversation stays Untitled and is retried next launch. Use `/history` to see the rest and `/resume 2` to
reopen one (or `/resume platypus` to find it by title); `/new` puts the current
one away and starts fresh.

Only the open conversation is loaded for chatting and title generation; old
chats are not opened in the background to fill in titles. The sidebar caches
small summaries. Every two seconds it checks the directory timestamp. It scans
file metadata when that changes, once per minute to catch edits made in place,
or when you explicitly refresh history with `/history`.
Legacy JSON files are parsed once to extract their summaries; message bodies
are discarded after that scan. File reads for listing and opening run off the
UI thread. Long chats initially display the last 50 messages; use **Load earlier
messages** above them to reveal more, 50 at a time.

Title and timestamp changes from another window merge into the existing chat.
Leaving or closing a chat without editing its messages does not save a copy.
If both windows change the transcript, a competing save keeps the local
conversation as a linked copy and displays a notice. Replies stay paired
with their original questions. `/forget` removes the whole linked family,
including copies made before links existed when their original creation and
first-message timestamps are available. Chats saved by a newer app version open
read-only: use `/new` before sending a message.

They are stored as one JSON file per conversation in
`~/.local/share/bedrock/conversations/` (or `$XDG_DATA_HOME/bedrock/…`):

- Files are created with mode `0600` and the directory is restricted to `0700`
  where the filesystem supports POSIX permissions. Network shares may apply
  their own access rules.
- Files are written atomically, so a crash mid-save cannot corrupt one.
- This is a local file write. Nothing is uploaded anywhere; the only network
  traffic the client makes is to your own inference server.
- `/forget` deletes the open conversation. To remove everything, delete the
  directory. To save nothing at all, set `BEDROCK_SAVE_HISTORY=0`.
- An empty deletion marker prevents an old window from recreating a forgotten
  chat or any of its linked copies. It contains no message text. A stale window
  blocks further messages, keeps the draft, and asks you to start with `/new`.
  New chats use random IDs without start timestamps, including in marker names.
  Existing IDs and markers remain compatible with older windows. Markers are
  retained because deleting them could let an old window recreate a linked copy.

On filesystems without advisory file locks, writes use an atomic directory
lock. If a writer is killed while holding that fallback lock, the next save
reports the stale `.history-writer` directory without waiting on the UI; remove it only after all writers
have stopped. Reading history remains available.

### Recalling past chats

Past chats stay on disk and are not loaded while you work. They are searched
only when you ask:

- `/recall deploy workflow` (or `/context deploy workflow`) searches saved chats
  for those words and pulls the best matches into the open chat. `/recall`
  shows what is pulled in; `/recall clear` removes it.
- Plain English works too: "do you remember when we set up the deploy workflow?"
  or "in a previous chat you wrote a SQL query for me" searches before answering.
  If nothing matches, the model is told so, instead of guessing.

The search is local keyword matching (`src/bedrock/recall.py`). At most three
chats and about 6,000 characters of excerpts are sent. The excerpts go to your
model with each message while that chat stays open, but are never saved into
it, so `/forget` on the original chat still removes that text from the machine.

## Configuration

Settings are read from, highest priority first:

1. Environment variables
2. `~/.config/bedrock/config.toml`
3. Built-in defaults (local Ollama)

| Environment variable | Config key        | Default                      |
|----------------------|-------------------|------------------------------|
| `BEDROCK_API_BASE`   | `api_base`        | `http://localhost:11434/v1`  |
| `BEDROCK_MODEL`      | `model`           | first model the server lists |
| `BEDROCK_API_KEY`    | `api_key`         | none                         |
| `BEDROCK_TIMEOUT`    | `request_timeout` | `120` seconds                |
| `BEDROCK_HISTORY_DIR`| `history_dir`     | `~/.local/share/bedrock/conversations` |
| `BEDROCK_SAVE_HISTORY` | `save_history`  | `true`                       |
| `BEDROCK_CONTEXT_DIR`| `context_dir`     | `~/.local/share/bedrock/context` |
| `BEDROCK_WORKSPACE_DIR`| `workspace_dir` | `~/.local/share/bedrock/workspaces` |
| `BEDROCK_ACTION_HISTORY_DIR`| `action_history_dir` | `~/.local/share/bedrock/history` |
| `BEDROCK_BIOTECH_WORK_DIR`| `biotech_work_dir` | `~/.local/share/bedrock/biotech` |
| `BEDROCK_USER`       | `user_id`         | current local username       |
| `BEDROCK_DISPLAY_NAME`| `display_name`   | `user_id`                    |
|                      | `system_prompt`   | none                         |

Two of those hold different kinds of history and are deliberately separate.
`history_dir` holds **saved conversations** — the employee's own prompts and
answers. `action_history_dir` holds the **action journal** behind `/audit` —
identifiers, hashes, decisions and error codes, never any content. One setting
for both would mean moving the audit trail whenever someone relocated their
chat history.

Example `~/.config/bedrock/config.toml` for a company inference server:

```toml
api_base = "https://inference.internal.example.com/v1"
model = "qwen2.5-32b"
api_key = "…"
system_prompt = "You are an assistant for Example Corp employees."
```

Any server that speaks the OpenAI chat-completions wire format works:
Ollama, vLLM, SGLang, llama.cpp. "OpenAI-compatible" only describes the
message format; no traffic goes to OpenAI.

## How the code is organised

```
src/bedrock/
  app.py                 The Textual application; wires UI to backend
  agentpack_app.py       Optional biotech multi-agent client
  ui/
    widgets.py           Status bar, message widgets, conversation sidebar
    agent_panel.py       Live agent activity in agent-pack mode
    approval.py          Human approval prompts in agent-pack mode
    commands.py          Slash-command parsing (pure functions)
    runs_command.py      `/runs`: list, detail, verification, project tagging
    confidence_view.py   `/confidence`: pLDDT bands and the PAE heatmap
    audit_command.py     `/audit`: the local action journal, and its export
    journal.py           Intent-before / outcome-after history writing
    services.py          Lazy, injectable HistoryStore + RunCatalog
    argparsing.py        argparse that reports errors instead of exiting
    rendering.py         Escaped, control-character-free terminal text
    workspace_command.py Native picker, preview, publish, roles and export flow
    styles.tcss          Layout and colours (Textual CSS)
  inference/
    base.py              InferenceBackend protocol - the seam between UI and server
    openai_compatible.py The one backend we need today
    factory.py           Picks a backend from settings
  models/
    conversation.py      Message / Conversation data classes, and their JSON form
  storage.py             Saves conversations to disk, one JSON file each
  titles.py              Asks the model for short conversation titles
  context/               Local context compression, indexing, and search
  connectors/            Company data sources (rest + share), fetch and provenance
  provenance/            Local action journal and the run catalogue over manifests
  biotech/               AlphaFold 2 and PyMOL adapters, confidence, workflow
  workspaces/            Private SQLite collaboration vault and validation
  meetings/              Local transcription and model-assisted meeting notes
  config/
    settings.py          Loads env vars + config file
tests/                   pytest; drives the real app with a fake backend
```

The important design decision: the UI never imports a concrete backend. It
only knows the `InferenceBackend` protocol (`check_connection`, `list_models`,
`stream_chat`, `close`). Swapping Ollama for the production server is a
configuration change, not a code change.

## Compressing data for the model (`/digest`)

`/digest` stores both a compact, model-friendly digest and compressed full
content for exact local lookup. Structured inputs such as CSV, logs, JSON, and
source code are summarized deterministically; prose may use the configured
private model, but ingest still succeeds when inference is unavailable.

Adding a data type is one adapter in `src/bedrock/context/adapters/` plus its
registry entry. Nothing in the UI or storage layer needs to change.

## Company data sources (`/sources`)

Without connectors, data reaches Bedrock only after someone exports a file and
types its path. With them, an employee names the record in the company's own
system instead:

```
/sources                              what is configured, and whether it is set up
/sources search registry kinase       find a record without knowing its id
/digest lims://plates/P42.csv         pull a record and compress it for the model
/fold registry://SEQ-00123            pull a sequence and run AlphaFold 2 (agent-pack mode)
bedrock-biotech fold registry://SEQ-00123
```

Natural requests work too: "Run AlphaFold 2 on registry://SEQ-00123".

Each system is one table in `~/.config/bedrock/config.toml`. Two general types
cover most internal systems without product-specific code:

```toml
# Anything with an HTTP/JSON API: registry, LIMS, ELN
[connectors.registry]
type = "rest"
base_url = "https://registry.internal.example.com/api/v1"
record_path = "/sequences/{id}"      # {id} is URL-encoded into one path segment
content_field = "data.fasta"         # omit to use the whole response body
filename_field = "data.name"
version_field = "data.revision"
suffix = ".fasta"                    # tells /digest what kind of data it is
search_path = "/sequences"           # optional; enables /sources search
results_field = "results"
token_env = "REGISTRY_TOKEN"         # names the env var; the token stays out of the file

# Systems that export into a controlled folder
[connectors.lims]
type = "share"
root = "/mnt/lims-exports"
extensions = [".csv", ".fasta"]      # optional allow-list
```

How it behaves:

- **Access is decided by the source system.** A 401/403 from the API, or a
  file-permission error on the share, reaches the employee as a plain "access
  refused" message. Bedrock keeps no copy of anyone's permission rules.
- **Everything fetched carries provenance** - system, record, version, who
  fetched it, when, and the SHA-256 of the exact bytes. It is saved as a
  `.provenance.json` next to the staged file in `<biotech_work_dir>/sources/`
  (owner-only permissions) and written into the AlphaFold run manifest under
  `source.system_of_record`.
- **References cannot escape.** REST ids are URL-encoded into a single path
  segment; share ids are resolved, symlinks included, and must stay under
  `root`.
- **One broken table does not break the rest.** `/sources` reports the problem
  and every other connector keeps working.

Identity caveat: the employee name sent to a REST source (`X-Bedrock-User`)
comes from `BEDROCK_USER` until company sign-in reaches this client, so it
identifies but does not authenticate. Point `rest` connectors at APIs that
check the `token_env` token, not ones that trust the header alone.

A system that needs more than configuration (paging, request signing, its own
query language) gets a class in `src/bedrock/connectors/` and one entry in
`hub.CONNECTOR_TYPES`. Nothing in `ui/`, the digest pipeline or the AlphaFold
workflow changes.

### Not built yet

- **Not tested against a real company system.** Coverage is unit tests, a
  temporary folder, and a local stand-in HTTP server. The first real
  connection is the acceptance test.
- **No product-specific connectors** (Benchling, LabWare, and so on). Whether the
  generic `rest` settings fit a given product is unknown until it is tried.
- **No direct database connection.** Sources are read through the system's own
  API or its exports. Reading its database directly would bypass the
  application's permission checks, so it is not offered.
- **Read-only.** Nothing is written back to the source system - for example, a
  predicted structure is not attached to the registry record.
- **Identity is not verified** until company sign-in reaches this client (see
  the caveat above and `NEXT_STEPS.md`).
- **Configuration and provenance live on each employee's machine.** IT has to
  distribute `config.toml`; there is no central list of connectors and no
  central audit log of fetches.
- **Staged copies are never cleaned up**, and there is no retention policy for
  `<biotech_work_dir>/sources/`.
- **Limited search.** REST search reads one page of results; share search
  matches file paths, not contents, and stops after 20,000 files.
- **`/pymol` does not accept references.** Use `/sources fetch` and open the
  saved file. `/fold` exists only in agent-pack mode; `/digest <ref>` and
  `/sources` work in both modes.

## Two modes

The default `openai` mode talks directly to one OpenAI-compatible private model
server. Setting `BEDROCK_BACKEND=agentpack` uses the biotech agent-pack backend,
with multi-agent activity and inline human approval prompts, while keeping the
same terminal interface. Configure its endpoint with `AGENT_PACK_API_BASE`.

## Running the tests

```
pytest
```

No model server, agent-pack backend, AlphaFold installation or PyMOL is
needed; tests use in-memory fakes and temporary directories for all of them,
and drive both real Textual apps headlessly.

## Releasing

[The release note](docs/RELEASE.md) records the built artifacts, their
hashes, the checks run against the *installed* wheel, and how to roll back
without deleting history or rewriting manifests.
[The smoke checklist](docs/SMOKE_CHECKLIST.md) is the part that needs a real
AlphaFold and PyMOL workstation and a person to run it — nothing automated
here demonstrates a real prediction.
