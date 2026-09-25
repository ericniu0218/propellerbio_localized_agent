# Running Bedrock on an 8 GB laptop

The Python client isn't what's stopping an 8 GB laptop. Importing the whole app
uses 44 MB of RAM and takes 0.24 s. What decides it is the model's memory: its
weights plus the KV cache, which is the memory the server holds for the prompt
it is working on. Ollama sets aside that cache for the whole context window when
it loads the model, so the window size is the real memory setting. No part of
Bedrock knows how big that window is, and several parts keep making the server
redo its work. The fixes that will last are in how prompts are built and sent,
not in the UI.

The two build prompts for this work are
[`docs/prompts/LAPTOP_AGENT_A.md`](../prompts/LAPTOP_AGENT_A.md) (prompt
content) and [`docs/prompts/LAPTOP_AGENT_B.md`](../prompts/LAPTOP_AGENT_B.md)
(model runtime). The split, the ownership rules and the shared contract are in
[Parallel build plan](#parallel-build-plan) below.

## Where 8 GB goes

This is `llama3.2:3b` at 4-bit, the model the README suggests:

| | Memory |
|---|---|
| Weights | ~2.0 GB |
| KV cache at 4k / 8k / 16k / 32k tokens | 0.5 / 0.9 / 1.9 / 3.8 GB |

Once macOS and other apps take their share, a laptop has to run with a window of
about 4k–8k tokens. The code assumes it doesn't have to.

## Faults

### 1. Nothing knows the context window, and each feature sets its own limit

- [conversation.py:234](../../src/bedrock/models/conversation.py#L234) sends the whole history on every turn.
- [conversation.py:117](../../src/bedrock/models/conversation.py#L117) re-adds every earlier attachment on every turn, up to 120,000 characters (~30k tokens) per message ([attachments.py:78](../../src/bedrock/attachments.py#L78)).
- The four other places that call the model each hard-code their own character limit, and none knows the window or the others.

Measured with the real classes:

- One 24 KB file makes the first prompt ~6,100 tokens, and it reaches ~19,000 by turn 30.
- With no file at all, a chat passes 4k tokens around turn 10 and 8k around turn 19.

When a prompt doesn't fit, Ollama cuts it down and only logs a warning on its
side; vLLM rejects the request. Either way the employee isn't told. The meeting
summariser checks citations only against the lines it sent the model
([summarize.py:242](../../src/bedrock/meetings/summarize.py#L242)), but on a
small window the server has already cut some of those lines.

### 2. Each turn's extra text goes at the top of the prompt, so the server can't reuse its earlier work

Ollama, llama.cpp and vLLM reuse their work for whatever part of the new prompt
starts the same as the last one. The one-turn `note` goes into the system
message, which is the first message
([conversation.py:243](../../src/bedrock/models/conversation.py#L243)).
Auto-inject is on by default and fills that note with `/digest` passages
whenever a keyword matches ([app.py:1186](../../src/bedrock/app.py#L1186)).

- On such a turn the first message has changed, so all ~19,800 tokens are recomputed, and again on the next turn when the note is gone.
- On a laptop that is tens of seconds (Apple silicon) to minutes (CPU only) before the first word.
- Recalled excerpts also move to the newest message each turn ([conversation.py:252](../../src/bedrock/models/conversation.py#L252)).
- Agent-pack mode puts the same passages in the user message instead ([agentpack_app.py:354](../../src/bedrock/agentpack_app.py#L354)), so no single piece of code decides where prompt text goes.
- Plain mode gives text from files the authority of a system instruction, which [recall.py:32](../../src/bedrock/recall.py#L32) says must never happen.

### 3. Background model calls compete with the chat and have no reply limit

- After every reply until a chat has a title, [app.py:1252](../../src/bedrock/app.py#L1252) makes two more model calls.
- A laptop should serve one request at a time, because each extra slot costs another full KV cache. So these calls delay the employee's next message and overwrite the server's saved work, and the next turn is recomputed from the start.
- The only cap is 32,000 characters ([titles.py:259](../../src/bedrock/titles.py#L259)). With a reasoning model that is up to ~8k tokens, several minutes on a laptop, for a YES/NO answer.
- `/digest` summaries, meeting notes and the agent planner have no cap at all.
- Titles use the currently selected model, not the one that just answered ([app.py:723](../../src/bedrock/app.py#L723)). After a `/model` switch with prompts still queued, the laptop swaps gigabytes of weights in and out between turns.

### 4. The backend interface can't express any of this

`stream_chat(model, messages)` ([base.py:46](../../src/bedrock/inference/base.py#L46))
can't ask for a reply limit, stop words, JSON output or reasoning off, and
returns no token counts. `ModelInfo` holds only an id
([base.py:30](../../src/bedrock/inference/base.py#L30)). With no model
configured, the app loads whatever the server lists first
([app.py:200](../../src/bedrock/app.py#L200)), even an 8B model that won't fit.

### 5. The `/digest` index is loaded whole and scanned on every message, on the UI thread

[store.py:235](../../src/bedrock/context/store.py#L235) keeps every manifest in
memory for good, re-reads every `digest.json`, and scores every chunk.
Auto-inject runs it on nearly every message in both app modes, without moving it
off the UI thread. Large attachments are added to the store automatically
([attachments.py:328](../../src/bedrock/attachments.py#L328)), so it grows even
if nobody runs `/digest`. On 20 MB of synthetic text (14k chunks):

- the first message froze the UI for 340 ms;
- every later message froze it for 40 ms;
- about 58 MB stayed in RAM.

All three grow in step with the store.

**Not worth the effort right now:** the client's own memory, saving (about 2 ms
per turn for an 86 KB chat), and the 2-second timers that check the connection,
history and queue.

## Framework: one path for every model call

```
chat, titles, /digest, meetings, planner, and whatever comes next
    │  add text to the prompt or submit a job; never call stream_chat directly
    ▼
PromptBuilder     token budget from the machine profile; unchanged text first, this turn's extras last
    ▼
Scheduler         priorities; one call at a time on a laptop; background work only when idle
    ▼
InferenceBackend  options in (reply limit, reasoning off, JSON schema); text and token counts out
```

- **Widen the interface** in `base.py`:
  - Pass request options in: reply limit, temperature, stop words, JSON schema, reasoning off.
  - Get token counts back where the server supports it.
  - Add a `context_window` setting.
  - The token counts can correct the 4-characters-per-token estimate and drive a local context meter in the status bar. It must stay local, because CLAUDE.md rules out telemetry.
- **A PromptBuilder with a token budget**, replacing `Conversation.to_api`:
  - Every source of prompt text registers with a priority and a share of the window: history, attachments, recall, digest passages, notes, and later embeddings or tool results.
  - It trims the oldest history first and tells the employee what it left out, as the attachment notice already does.
  - Attachments go whole in the turn they're attached and as their digest after that. `ingest` makes digests without a model, so this costs nothing.
  - A test drives a scripted 50-turn chat with files and checks the prompt never exceeds the laptop window.
- **Unchanged prompt start as a tested rule:**
  - Order: system prompt, then history exactly as last turn, then this turn's extras inside the newest user message.
  - Trim in large steps, for example from 90% of the window down to 60%. Dropping one message per turn would make every turn after the window fills recompute everything.
  - A test checks that each prompt starts with the previous one.
- **Scheduler:**
  - Chat comes first, then jobs the employee started, then background work.
  - On a laptop, one call at a time.
  - Background calls wait until the app is idle, are cancelled when the employee sends a message, and always use the model that answered last, which is the one already loaded.
  - Keep the separate YES/NO title step ([titles.py:18](../../src/bedrock/titles.py#L18) explains why it exists); run it later rather than merging it.
- **A machine profile setting** (`profile = "laptop"`):
  - One setting fixes the window, reply limit, attachment share, number of auto-inject passages, background policy, and the largest model the app will pick on its own.
  - The first-run check prints the matching Ollama settings.
  - New features read limits from here instead of adding their own constants.
- **`/digest` on SQLite FTS5**, SQLite's built-in full-text search:
  - It is in the standard library here (SQLite 3.53); check the Python that `uv` installs too.
  - Ranking happens in the database and the index stays on disk. Searches only touch matching words and run off the UI thread.
  - This fixes speed and memory, not search quality: "acidity" still won't find "pH". NEXT_STEPS §6 (embeddings) is still the real fix, and it is still blocked on a deployed embedding model. On a laptop that model stays loaded next to the chat model (130–270 MB), and the profile has to leave room for it.

To keep this from coming back as features are added, add one rule to CLAUDE.md,
"nothing outside `inference/` calls `stream_chat`", plus a test that fails when
something does. Today five places do.

## Where to start

0. **No code, today:** use a 3–4B model at 4-bit and set these on the Ollama server:
   ```
   OLLAMA_CONTEXT_LENGTH=8192   # sets the KV-cache size; use 4096 if memory is still tight
   OLLAMA_NUM_PARALLEL=1        # each extra slot is another full KV cache
   OLLAMA_MAX_LOADED_MODELS=1
   OLLAMA_FLASH_ATTENTION=1
   OLLAMA_KV_CACHE_TYPE=q8_0    # roughly halves the cache
   ```
1. Widen the interface and add `context_window`. It's small and unblocks everything else.
2. Build the PromptBuilder with the budget and the prompt-start rule. This is the biggest win, because it fixes both the silent cutting and the wasted recomputation.
3. Add the scheduler and move the five calls onto it.
4. Move `/digest` to FTS5.

## Decisions that change what employees see

The prompts are written with the defaults below. Change a default here and in
the prompt that owns it before handing the prompts out.

| # | Decision | Default in the prompts | Owner |
|---|---|---|---|
| D1 | Earlier attachments drop to their digest after the turn they were attached in | **Yes** | A |
| D2 | Titles appear later, once the app is idle, instead of right after each reply | **Yes** | B |
| D3 | `/digest` passages and recall notes leave the system message and go inside the newest user message | **Yes** (also fixes the authority problem) | A |
| D4 | The default profile when none is configured | **`laptop`**, because the default endpoint is Ollama on `localhost` | B |

## Parallel build plan

Two models build this at the same time. The split is not by effort. It follows
the most distinct seam in the design:

- **Model A — prompt content.** *What text* goes into a request: the
  PromptBuilder, the token budget, the unchanged-prompt-start rule, attachments,
  recall, auto-inject placement, and the `/digest` store it searches (FTS5).
- **Model B — model runtime.** *How, when, and with which limits* a request
  reaches the server: the widened backend interface, token counts, the machine
  profile, the scheduler, every model call except the chat turn, the first-run
  check, and the guard rule.

The dependency runs one way. A reads the profile, the token estimator and the
scheduler's method signature; B never needs anything from A. So B lands a small
contract commit first (Phase 0, below) and then freezes it. A starts at once on
the parts that don't need it: FTS5 and the PromptBuilder as a pure module.

### Working copies

Each model works in its **own git worktree on its own branch**, both cut from the
same base commit:

| Model | Branch |
|---|---|
| A | `laptop/prompt` |
| B | `laptop/runtime` |

B's Phase 0 is one commit on `laptop/runtime`. A merges that commit into
`laptop/prompt` before touching any file listed under "A, after Phase 0" below.
Neither model runs Git commands in the other's worktree.

### File ownership

A model edits only what it owns. "Nobody" means neither model edits it in this
work.

| Path | Owner |
|---|---|
| `src/bedrock/prompting/` (new package) | A |
| `src/bedrock/models/conversation.py` | A |
| `src/bedrock/attachments.py` | A |
| `src/bedrock/recall.py` | A |
| `src/bedrock/context/` — every file **except** `digest.py` | A |
| `src/bedrock/agentpack_app.py` | A |
| `NEXT_STEPS.md` | A |
| `src/bedrock/inference/` (all of it, including new `scheduler.py`, `tokens.py`) | B |
| `src/bedrock/config/` (including new `profile.py`) | B |
| `src/bedrock/context/digest.py` | B |
| `src/bedrock/titles.py` | B |
| `src/bedrock/meetings/` | B |
| `src/bedrock/agents/planner.py` | B |
| `src/bedrock/doctor.py` | B |
| `src/bedrock/ui/digest_command.py`, `src/bedrock/ui/local_commands.py` | B |
| `src/bedrock/ui/widgets.py` — the `StatusBar` class only | B |
| `tests/conftest.py` | B |
| `CLAUDE.md`, `AGENTS.md`, `README.md`, `INSTALL.md`, `pyproject.toml` | B |
| Everything else in `ui/`, `storage.py`, `prompt_queue.py`, `workspaces/`, `biotech/`, `provenance/`, `connectors/` | Nobody |

**`src/bedrock/app.py` is split by method.** Methods not listed belong to nobody.

| Owner | Methods and lines in `app.py` |
|---|---|
| B | the import block **in Phase 0 only**; `__init__`; `check_connection`; `pick_default_model`; `command_model`; `write_missing_titles`; `_next_untitled`; `apply_title`; `action_quit` |
| A | the import block **after Phase 0**; `send_message`; `dispatch_prompt`; `stream_reply`; `command_recall`; `recall_into` |

B adds every import either model needs in `app.py` during Phase 0 (listed in the
contract), so after Phase 0 B never edits the import block again.

**Tests.** Each model adds its own new test files and edits the existing test
files for the modules it owns:

- A: `test_conversation.py`, `test_attachments.py`, `test_recall.py`, `test_auto_inject.py`, `test_context_store.py`, `test_agentpack_app.py`.
- B: `conftest.py`, `test_titles.py`, `test_sidebar.py`, `test_openai_backend.py`, `test_settings.py`, `test_doctor.py`, `test_meetings.py`, `test_context_digest.py`, `test_digest_command.py`, `test_agents_kernel.py`.

In any other test file (for example `test_app.py`), a model may edit only a test
function that fails because of its own change, and must name each such function
in its hand-off note. If both models need the same test function changed, the
second one stops and reports it instead of editing.

### Phase 0 contract (B writes it first, then it is frozen)

B lands exactly these names in one commit, with a pass-through `Scheduler`, and
then does not change their signatures. B may add fields with defaults later; it
may not rename or remove anything.

```python
# src/bedrock/inference/base.py
class Priority(IntEnum):
    CHAT = 0          # the employee's own turn
    USER_JOB = 1      # something the employee asked for: /digest summary, meeting notes, planner
    BACKGROUND = 2    # titles and anything else nobody is waiting for

@dataclass(frozen=True)
class RequestOptions:
    max_tokens: int | None = None
    temperature: float | None = None
    stop: tuple[str, ...] = ()
    json_schema: dict | None = None
    reasoning: bool | None = None     # False asks the server to skip thinking; None = server default

@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None

@dataclass(frozen=True)
class ModelInfo:
    id: str
    size_bytes: int | None = None
    context_length: int | None = None

class InferenceBackend(Protocol):
    # check_connection, list_models, close: unchanged
    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: RequestOptions | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]: ...

# src/bedrock/inference/scheduler.py
class Preempted(InferenceError):
    """A lower-priority call was cancelled because the employee sent a message. Retry later; not a failure."""

class Scheduler:
    def __init__(self, backend: InferenceBackend, profile: MachineProfile) -> None: ...
    def stream(
        self,
        model: str | None,            # None only for BACKGROUND: use the model that last answered a CHAT call
        messages: list[dict[str, str]],
        *,
        priority: Priority,
        options: RequestOptions | None = None,
    ) -> AsyncIterator[str]: ...
    async def complete(self, model, messages, *, priority, options=None) -> str: ...
    @property
    def last_usage(self) -> Usage | None: ...    # from the last CHAT call
    @property
    def loaded_model(self) -> str | None: ...

# src/bedrock/inference/tokens.py
def estimate_tokens(text: str) -> int: ...         # ~4 characters per token until corrected
def observe(characters: int, usage: Usage) -> None: ...   # B uses real counts to correct the ratio

# src/bedrock/config/profile.py
@dataclass(frozen=True)
class MachineProfile:
    name: str
    context_window: int              # tokens; must match OLLAMA_CONTEXT_LENGTH / vLLM --max-model-len
    reply_limit: int                 # max_tokens for a chat reply; the builder reserves this much
    attachment_share: float          # fraction of the window attachments may use in their own turn
    auto_inject_passages: int        # /digest passages added per question; 0 turns auto-inject off
    background: Literal["idle", "concurrent"]
    max_parallel: int
    max_auto_model_bytes: int | None # largest model the app picks on its own; None = no limit

PROFILES: dict[str, MachineProfile]  # "laptop", "workstation", "server"
# laptop:      8192 window, 1024 reply, 0.35 attachments, 2 passages, idle,       1 parallel, 3.5 GB
# workstation: 32768,       2048,       0.40,             3,          idle,       1,          12 GB
# server:      32768,       4096,       0.50,             3,          concurrent, 4,          None

# src/bedrock/config/settings.py
Settings.profile: MachineProfile   # `profile = "laptop"` in config.toml, BEDROCK_PROFILE; D4 default
# `context_window = N` in config.toml / BEDROCK_CONTEXT_WINDOW overrides profile.context_window

# src/bedrock/app.py  (B, Phase 0)
from bedrock.inference import Preempted, Priority, RequestOptions, Scheduler
self.scheduler = Scheduler(self.backend, self.settings.profile)   # in __init__, after self.backend
# self.backend stays the raw backend, used only for check_connection, list_models and close.
```

`FakeBackend` in `conftest.py` accepts `options` and `on_usage`, records the
options it received in `request_options`, and reports a `Usage` when asked.
Everything in `bedrock.inference` and `bedrock.config` above is exported from
the package `__init__`.

### How the halves join

1. Both branches pass the full suite on their own (baseline before this work: 963 passing).
2. B merges `laptop/prompt` into `laptop/runtime`. File ownership is disjoint, so there should be no conflicts. If there is one, stop and report it rather than resolving it by hand.
3. B then does the join step:
   - Adds the guard test `tests/test_model_call_guard.py`: it walks `src/bedrock` with `ast` and fails on any `.stream_chat(` call outside `src/bedrock/inference/`.
   - Adds the CLAUDE.md rule "nothing outside `inference/` calls `stream_chat`; every model call goes through `Scheduler`", plus A's status paragraph from A's hand-off note.
   - Adds an app-level test that a title request waits while a chat reply is streaming and is preempted by a new message. This needs both halves, which is why it waits until now.
   - Runs the full suite.

**Each half is useful alone.** If A stops after Phase 0 is merged, B's work still
caps every background call, runs titles only when idle, and picks a model that
fits. The only gap is that the chat turn doesn't go through the scheduler yet; B
then changes that one call in `stream_reply` during the join. If B stops after
Phase 0, A's work still keeps every chat prompt inside the window, keeps the
prompt start unchanged between turns, and makes `/digest` fast. It just runs
against the pass-through scheduler.
