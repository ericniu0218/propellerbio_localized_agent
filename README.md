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
with `bedrock-biotech fold candidate.fasta`. In agent-pack mode the equivalent
commands are `/tools`, `/fold <fasta-path>`, and `/pymol [structure-path]`.
See [the integration guide](docs/BIOTECH_INTEGRATIONS.md) for provisioning and
privacy requirements.

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
| `BEDROCK_USER`       | `user_id`         | current local username       |
| `BEDROCK_DISPLAY_NAME`| `display_name`   | `user_id`                    |
|                      | `system_prompt`   | none                         |

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

## Two modes

The default `openai` mode talks directly to one OpenAI-compatible private model
server. Setting `BEDROCK_BACKEND=agentpack` uses the biotech agent-pack backend,
with multi-agent activity and inline human approval prompts, while keeping the
same terminal interface. Configure its endpoint with `AGENT_PACK_API_BASE`.

## Running the tests

```
pytest
```

No model server is needed; tests use an in-memory fake backend.
