"""The Textual application for talking to the full biotech agent-pack
system, instead of a single OpenAI-compatible model server.

Same look and feel as `bedrock.app.BedrockApp` (status bar, scrolling chat,
one input box at the bottom) plus one small addition: a quiet strip above
the chat showing what the AI system is doing right now, in plain language,
and an inline prompt when it needs a human decision. Everything else about
the screen is deliberately unchanged — this is still "the terminal thing,"
not a dashboard.

Why a separate App class rather than another `InferenceBackend`: the
agent-pack backend isn't a single-shot "send messages, get a reply" model
server — it has sessions, multiple agents working one request, and
human-approval gates the UI must be able to show and respond to. Forcing
that into the one-fragment-at-a-time `stream_chat` shape would hide
information the UI actually needs to display. `bedrock/app.py::main`
picks this class instead of `BedrockApp` when `settings.backend ==
"agentpack"` — nothing about how the app is launched changes for IT or
the employee.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Footer, Input

from bedrock import attachments
from bedrock.agentpack.client import AgentPackClient
from bedrock.agentpack.friendly import friendly_error
from bedrock.agentpack.models import AgentCardState, AgentEvent
from bedrock.biotech import ProteinWorkflow
from bedrock.config import Settings, load_settings
from bedrock.connectors import SourceHub, SourceRef
from bedrock.context import find_relevant_context, with_injected_context
from bedrock.inference.base import InferenceError
from bedrock.prompt_queue import PromptItem, PromptQueue, QueueChanged, QueueSnapshot, QueueFull
from bedrock.ui.agent_panel import AgentPanel
from bedrock.ui.approval import ApprovalPrompt, approval_fingerprint
from bedrock.ui.argparsing import CommandUsageError
from bedrock.ui.commands import Command, parse_command, parse_scientific_request
from bedrock.ui.digest_command import AUTO_CONTEXT_OFF_HINT
from bedrock.ui.prompt_queue import PromptQueuePanel
from bedrock.ui.fold_command import (
    parse_fold_request,
    parse_structure_argument,
    run_fold,
    run_pymol,
    tool_status_text,
)
from bedrock.ui.journal import JournalledAction
from bedrock.ui.local_commands import LocalCommandsMixin
from bedrock.ui.rendering import safe_line
from bedrock.ui.services import ComplianceServices
from bedrock.ui.widgets import AssistantMessage, PromptInput, StatusBar, SystemNotice, UserMessage

CONNECTION_CHECK_SECONDS = 10

# Streamed into the answer bubble the moment the drafting agent's first
# token arrives, so it's visually unmistakable that this text has not yet
# passed the Director's citation/hallucination review — never remove this
# without also removing the streaming itself; showing unreviewed text with
# no such marker would be presenting it as final when it isn't.
PENDING_REVIEW_PREFIX = "_⏳ Drafting — not yet compliance-reviewed_\n\n---\n\n"
# Shown once the draft finishes streaming but before task.completed/failed
# resolves the turn — review itself is synchronous/near-instant in this
# codebase (see app/agents/director.py::review), but formatting the
# approved answer (or, for a high-stakes request, a human approval gate)
# still takes real time after that.
REVIEWING_PREFIX = "_🔍 Reviewing before this is shown as final…_\n\n---\n\n"

AGENTPACK_HELP_TEXT = """\
**Commands**

| Command | What it does |
|---|---|
| `/fold <fasta-path or source://id> [--project X]` | Predict one monomer with AlphaFold 2 and render it in PyMOL |
| `/pymol [structure-path or source://id]` | Open a structure, or the latest prediction, in PyMOL |
| `/tools` | Check AlphaFold 2 and PyMOL configuration |
| `/runs` | Prediction runs recorded on this machine |
| `/runs show <run-id>` | Settings, lineage and integrity of one run |
| `/runs tag <run-id> --project X` | Assign an existing run to a project |
| `/confidence <run-id>` | pLDDT bands, low-confidence regions and PAE |
| `/audit` | What this client has done on this machine |
| `@<path>` in a message | Attach a text file (or drag the file onto the terminal) |
| `/digest <path>` | Compress a file so a model can read it cheaply |
| `/digest` | List and search what you have compressed |
| `/digest auto off` | Stop adding matching stored content to your messages in this window |
| `/sources` | List the company data sources you can pull from |
| `/workspace` | List private team workspaces |
| `/workspace help` | Show workspace actions |
| `/clear` | Clear the conversation |
| `/help` | Show this help |
| `/exit` | Quit (also `Ctrl+C`) |

You can also ask naturally, for example: “Run AlphaFold 2 on `candidate.fasta`”,
“Run AlphaFold 2 on `registry://SEQ-00123`”, or “Open the latest structure
interactively in PyMOL.”

**Examples**

- Runs for project X this quarter:
  `/runs --project X --since 2026-07-01 --before 2026-10-01`
- Export the action history: `/audit export "/approved/review/actions.jsonl"`

`/runs`, `/confidence` and `/audit` read local files only. They keep working
when the AI system is unreachable, and nothing they show is sent to a model.
"""


class AgentPackApp(LocalCommandsMixin, App[None]):
    TITLE = "Bedrock Private AI"
    CSS_PATH = "ui/styles.tcss"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear"),
    ]

    def __init__(
        self,
        settings: Settings | None = None,
        client: AgentPackClient | None = None,
        biotech_workflow: ProteinWorkflow | None = None,
        sources: SourceHub | None = None,
        services: ComplianceServices | None = None,
    ) -> None:
        super().__init__()
        # Explicit, not just relying on Textual's default: this must always
        # be a dark, black-background terminal look, never a light theme.
        self.theme = "textual-dark"
        self.settings = settings or load_settings()
        # Tests pass a fake client; real runs build one from settings.
        self.client = client or AgentPackClient(self.settings.agent_pack_api_base)
        self.biotech_workflow = biotech_workflow or ProteinWorkflow.from_settings(self.settings)
        self.session_id: str | None = None
        self._waiting_for: str | None = None  # task_id we're showing an answer bubble for
        self._seen_events: list[str] = []  # event ids already acted on, newest last
        self._answer_widget: AssistantMessage | None = None
        self._agents: dict[str, AgentCardState] = {}
        self._init_local_commands(sources=sources, services=services)
        self._scientific_job_running = False
        # Pending approvals by request fingerprint, so a re-delivered event
        # reuses its prompt instead of asking the same question twice. In
        # memory only - nothing here belongs in durable history.
        self._pending_approvals: dict[str, ApprovalPrompt] = {}
        # Only announce a dropped live-event connection once per outage,
        # not on every retry attempt — see _handle_reconnecting.
        self._reconnect_notice_shown = False
        # Token-streaming state for the current turn's answer bubble — see
        # _handle_agent_token. Reset alongside _waiting_for/_answer_widget
        # in send_message, action_clear_chat, and _finish.
        self._plan_seen = False  # a task.planned event means hard/high-stakes tier
        self._draft_role: str | None = None  # which agent's tokens are "the draft"
        self._draft_stream = None  # the open MarkdownStream, while one is active
        self._draft_text = ""  # accumulated so far, for the post-stream "reviewing" swap
        # The same queue rules as the chat app: one request at a time, later
        # prompts wait. Scientific jobs count as requests; see prompt_queue.py.
        self.prompt_queue = PromptQueue()
        self._running_item: PromptItem | None = None
        self._preparing_prompts: set[str] = set()
        self._queue_update_scheduled = False
        self._quit_confirmation = (0, 0.0)
        # Keep terminal event identities for this window, not answer bodies.
        # A replay must not finish whichever prompt happens to be running now.
        # Two requests can fail in the same words; the stream's own event id
        # keeps a genuinely new failure from being mistaken for a replay.
        self._seen_completions: set[tuple[str, str, str | None, str | None, str | None]] = set()

    # ---- Screen layout ------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield StatusBar()
        yield AgentPanel(id="agent-panel")
        yield VerticalScroll(id="chat")
        yield PromptQueuePanel(id="queue-panel")
        yield PromptInput(placeholder="Type a message, or /help for commands", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one(StatusBar).model_name = "AI system"
        self.query_one("#agent-panel", AgentPanel).clear()
        self.query_one(PromptQueuePanel).show_queue(self.queue_snapshot())
        self.query_one("#prompt", Input).focus()
        self.set_interval(CONNECTION_CHECK_SECONDS, self.check_connection)
        self.set_interval(2, self.recover_queue)
        self.check_connection()
        self.start_session()

    @work(exclusive=True, group="connection")
    async def check_connection(self) -> None:
        connected = await self.client.check_connection()
        self.query_one(StatusBar).connected = connected

    @work(exclusive=True, group="session")
    async def start_session(self) -> None:
        try:
            self.session_id = await self.client.create_session(self.settings.user_id)
        except InferenceError as exc:
            await self.post_notice(str(exc), kind="error")
            return
        self.listen_for_events(self.session_id)

    @work(exclusive=True, group="events")
    async def listen_for_events(self, session_id: str) -> None:
        """Runs for the lifetime of the app: the one place that reads the
        live event stream and fans updates out to the agent panel, the
        answer bubble, and approval prompts."""
        try:
            async for event in self.client.events(session_id):
                await self.handle_event(event)
        except InferenceError as exc:
            await self.post_notice(str(exc), kind="error")

    # ---- Input handling -------------------------------------------------------

    @on(Input.Submitted, "#prompt")
    async def handle_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        command = parse_command(text)
        if command is not None and command.name not in {"fold", "pymol"}:
            await self.run_command(command)
        else:
            # Chat messages and scientific requests share one queue: a running
            # fold is a request, so the next prompt waits for it.
            await self.queue_prompt(text)

    async def run_command(self, command: Command) -> None:
        if await self.run_local_command(command):
            return
        match command.name:
            case "help":
                await self.post_notice(AGENTPACK_HELP_TEXT)
            case "clear":
                await self.action_clear_chat()
            case "exit" | "quit":
                await self.action_quit()
            case "tools":
                await self.show_tool_status()
            case "fold":
                was_running = self._scientific_job_running
                await self.start_fold(command.args)
                # A refused fold (bad arguments, or one already running) never
                # reports completion, so release the queue slot here instead.
                if was_running or not self._scientific_job_running:
                    self.finish_running_prompt(ok=False)
            case "pymol":
                try:
                    structure = parse_structure_argument(command.args)
                except CommandUsageError as exc:
                    await self.post_notice(str(exc), kind="error")
                else:
                    self.open_pymol(structure)
            case _:
                await self.post_notice(
                    f"`/{command.name}` isn't available in this mode. Type `/help` for what is.",
                    kind="error",
                )

    async def show_tool_status(self) -> None:
        await self.post_notice(tool_status_text(self.biotech_workflow))

    async def start_fold(self, args: str) -> None:
        """Validate the arguments here, so a typo never starts a GPU job."""
        try:
            target, project_id = parse_fold_request(args)
        except CommandUsageError as exc:
            await self.post_notice(str(exc), kind="error")
            return
        if self._scientific_job_running:
            await self.post_notice("A scientific job is already running.", kind="error")
            return
        self._scientific_job_running = True
        self.fold_protein(target, project_id)

    @work(exclusive=True, group="scientific")
    async def fold_protein(self, target: Path | SourceRef, project_id: str | None = None) -> None:
        """Run AlphaFold 2 on a local FASTA, or on a record pulled from a company source.

        The fetching, folding and reporting are shared with the plain model
        mode (``bedrock.ui.fold_command.run_fold``). What is particular to this
        mode is the queue: a fold is a request here, so the next prompt waits
        until this one has really finished.
        """
        completed = False  # the queue may only advance on a finished prediction
        try:
            completed = await run_fold(
                target,
                project_id,
                workflow=self.biotech_workflow,
                sources=self.sources,
                services=self.services,
                notify=self.post_notice,
            )
        finally:
            self._scientific_job_running = False
            # A fold counts as finished only here, never when it starts.
            self.finish_running_prompt(ok=completed)

    @work(exclusive=True, group="pymol-open")
    async def open_pymol(self, target: Path | SourceRef | None) -> None:
        opened_ok = False
        try:
            opened_ok = await run_pymol(
                target, workflow=self.biotech_workflow, sources=self.sources, notify=self.post_notice
            )
        finally:
            self.finish_running_prompt(ok=opened_ok)

    async def send_message(self, text: str) -> None:
        if not self.session_id:
            await self.post_notice("Still connecting to the AI system — try again in a moment.", kind="error")
            self.finish_running_prompt(ok=False)
            return

        # @files are read when the prompt starts, off the UI thread, because a
        # large one is compressed first. See bedrock/attachments.py.
        try:
            attached = await asyncio.to_thread(attachments.load, attachments.resolve(text), self.context_store)
        except attachments.AttachmentError as exc:
            await self.post_notice(str(exc) + " Your draft is still in the input; nothing was sent.", kind="error")
            self.query_one("#prompt", Input).value = text
            self.finish_running_prompt(ok=False)
            return

        await self.mount_message(UserMessage(attachments.display_text(text, attached)))
        if notice := attachments.summary(attached):
            await self.post_notice(notice)
        self._agents.clear()
        self.query_one("#agent-panel", AgentPanel).clear()

        # Pull in anything relevant that was compressed earlier with
        # /digest, so using the store doesn't require remembering it
        # exists and pasting a search result in by hand — see
        # bedrock.context.auto_inject for what this can and can't do.
        # The bubble above shows exactly what the user typed; only the
        # text actually sent to the model gets the extra context, and the
        # notice below says so out loud rather than doing it silently.
        # Read here, at send time, so a prompt that waited in the queue
        # follows `/digest auto off` typed while it waited. When off, the
        # store is not even searched.
        injected = (
            find_relevant_context(text, self.context_store) if self.auto_context.enabled else None
        )
        outgoing_text = with_injected_context(text, injected)
        if injected is not None:
            await self.post_notice(f"{injected.summary} {AUTO_CONTEXT_OFF_HINT}")

        # Attached files follow the typed question, after any stored context.
        outgoing_text = attachments.with_attachments(outgoing_text, attached)
        try:
            task_id = await self.client.send_message(self.session_id, outgoing_text)
        except InferenceError as exc:
            await self.post_notice(str(exc), kind="error")
            if self._running_item is not None:
                self.prompt_queue.requeue_front(self._running_item)
                self._running_item = None
                self.queue_changed()
                self.notify("Your unsent prompt was kept. The queue is paused; resume to retry.", timeout=8)
            else:
                self.query_one("#prompt", Input).value = text
            return

        self._waiting_for = task_id
        self._answer_widget = AssistantMessage("AI system")
        await self.mount_message(self._answer_widget)
        self._answer_widget.update("_Working on it…_")
        self._set_input_busy(True)
        self._plan_seen = False
        self._draft_role = None
        self._draft_stream = None
        self._draft_text = ""

    # ---- Prompt queue -----------------------------------------------------------

    @property
    def queue_key(self) -> str:
        """Agent-pack has one session per window, so the session is the queue."""
        return self.session_id or "session"

    def queue_snapshot(self) -> QueueSnapshot:
        # "Stop & run next" stays off here: the agent-pack backend has no
        # cancel call, no way to confirm a stop was recorded, and (per
        # _finish's own docstring) no correlation id to tell a stale event
        # from this session's next request. Pretending to stop by just
        # ignoring the backend's eventual event would be lying about what
        # happened, so this always reports can_skip=False.
        return self.prompt_queue.snapshot(self.queue_key)

    def remove_queued_prompt(self, item_id: str) -> bool:
        removed = self.prompt_queue.remove(item_id)
        if removed is not None:
            self.queue_changed()
        return removed is not None

    def pause_queue(self) -> None:
        if self.prompt_queue.pause(self.queue_key):
            self.queue_changed()

    def resume_queue(self) -> None:
        if self.prompt_queue.resume(self.queue_key):
            self.queue_changed()
        if not self.prompt_queue.busy:
            self.advance_queue_soon(self.queue_key)

    def clear_queue(self) -> None:
        self.prompt_queue.clear(self.queue_key)
        self.queue_changed()

    def queue_changed(self) -> None:
        if not self._queue_update_scheduled:
            self._queue_update_scheduled = True
            # Batch nearby lifecycle changes into one display update.
            self.set_timer(1 / 60, self._emit_queue_changed)

    def _emit_queue_changed(self) -> None:
        self._queue_update_scheduled = False
        self.post_message(QueueChanged(self.queue_key, self.queue_snapshot()))

    def recover_queue(self) -> None:
        item = self.prompt_queue.running
        if item is None or item.id in self._preparing_prompts or self._waiting_for is not None:
            return
        if any(w.group in {"queue", "scientific", "pymol-open"} and not w.is_finished for w in self.workers):
            return
        if self.prompt_queue.requeue_front(item):
            self._running_item = None
            self.queue_changed()
            self.notify("The previous request stopped unexpectedly. Its prompt was kept and the queue is paused; resume to retry.", severity="error", timeout=10)

    async def queue_prompt(self, text: str) -> None:
        """Queue a chat message or scientific request and run it when it can."""
        if not self.session_id:
            await self.post_notice("Still connecting to the AI system — try again in a moment.", kind="error")
            return
        try:
            # Only checks that each @file exists, so a typo is reported before
            # the prompt waits in the queue.
            attachments.resolve(text)
        except attachments.AttachmentError as exc:
            self.query_one("#prompt", Input).value = text
            await self.post_notice(str(exc) + " Your draft is still in the input.", kind="error")
            return
        try:
            self.prompt_queue.add(text, self.queue_key, "AI system")
        except QueueFull as exc:
            self.query_one("#prompt", Input).value = text
            await self.post_notice(str(exc) + " Your draft is still in the input.", kind="error")
            return
        self.queue_changed()
        await self.advance_queue(self.queue_key)

    async def advance_queue(self, key: str | None = None) -> None:
        key = key or self.queue_key
        item = self.prompt_queue.start_next(key)
        if item is None:
            return
        # Busy from here on, before any awaiting preparation below.
        self._running_item = item
        self.queue_changed()
        self._preparing_prompts.add(item.id)
        try:
            await self.dispatch_prompt(item)
        except asyncio.CancelledError:
            if self.prompt_queue.requeue_front(item):
                self._running_item = None
                self.queue_changed()
            raise
        except Exception as exc:
            if self.prompt_queue.requeue_front(item):
                self._running_item = None
                self.queue_changed()
            self._waiting_for = None
            self._answer_widget = None
            self._set_input_busy(False)
            await self.post_notice("Could not start the prompt. Its queue is paused; resume to retry. " + friendly_error(str(exc)), kind="error")
        finally:
            self._preparing_prompts.discard(item.id)

    @work(group="queue")
    async def advance_queue_soon(self, key: str) -> None:
        await self.advance_queue(key)

    async def dispatch_prompt(self, item: PromptItem) -> None:
        """Run one queued prompt: a slash command, a scientific request, or chat."""
        command = parse_command(item.text)
        if command is not None:
            await self.run_command(command)
            return
        scientific = parse_scientific_request(item.text)
        if scientific is not None:
            await self.mount_message(UserMessage(item.text))
            await self.run_command(scientific)
            return
        await self.send_message(item.text)

    def finish_running_prompt(self, *, ok: bool) -> None:
        """End the running request, then start the next one if it went well.

        Replayed completion events are filtered in _finish. Approval prompts
        do not come through here, so waiting for a human is not completion.
        Identifying previously unseen, unrelated terminal events within this
        session still requires request correlation from the backend: a final
        step may have a different task_id than send_message returned.
        """
        item = self._running_item
        if item is None or not self.prompt_queue.finish(item.id):
            return
        self._running_item = None
        if not ok and self.prompt_queue.pending_for(item.conversation_id):
            if self.prompt_queue.pause(item.conversation_id):
                self.notify("That request did not finish, so the rest of the queue is paused. Resume it when you are ready.", timeout=8)
        self.queue_changed()
        if ok:
            self.advance_queue_soon(item.conversation_id)

    @on(QueueChanged)
    def show_queue(self, message: QueueChanged) -> None:
        try:
            self.query_one(PromptQueuePanel).show_queue(message.snapshot)
        except NoMatches:
            pass  # A queued notification may arrive during teardown.

    @on(PromptQueuePanel.RemoveRequested)
    def on_queue_remove(self, message: PromptQueuePanel.RemoveRequested) -> None:
        self.remove_queued_prompt(message.item_id)

    @on(PromptQueuePanel.PauseRequested)
    def on_queue_pause(self) -> None:
        self.pause_queue()

    @on(PromptQueuePanel.ResumeRequested)
    def on_queue_resume(self) -> None:
        self.resume_queue()

    @on(PromptQueuePanel.ClearRequested)
    def on_queue_clear(self) -> None:
        self.clear_queue()

    @on(PromptQueuePanel.SkipRequested)
    def on_queue_skip(self) -> None:
        # can_skip is always False here (see queue_snapshot), so the button
        # is never enabled; this only guards a stale or forced message.
        pass

    # ---- Live events ------------------------------------------------------------

    async def handle_event(self, event: AgentEvent) -> None:
        # A reconnecting stream can redeliver an event. Acting on the copy
        # would end whichever request is running *now* - see the prompt queue:
        # completions carry no reference to the request they belong to, so a
        # repeat is otherwise indistinguishable from the next answer arriving.
        if event.event_id:
            if event.event_id in self._seen_events:
                return
            self._seen_events.append(event.event_id)
            if len(self._seen_events) > 500:
                del self._seen_events[:250]
        if event.type in ("agent.started", "agent.status"):
            self._update_agent(event)
        elif event.type == "task.planned":
            self._plan_seen = True
        elif event.type == "agent.token":
            await self._handle_agent_token(event)
        elif event.type == "approval.required":
            await self._show_approval(event)
        elif event.type == "task.completed":
            await self._finish(event, failed=False)
        elif event.type == "task.failed":
            await self._finish(event, failed=True)
        elif event.type == "client.reconnecting":
            await self._handle_reconnecting()
        elif event.type == "client.reconnected":
            await self._handle_reconnected()
        elif event.type == "session.resync":
            await self._handle_resync(event)

    def _update_agent(self, event: AgentEvent) -> None:
        payload = event.payload
        card = payload.get("agent") if event.type == "agent.started" else payload
        if not card or "agent_id" not in card:
            return
        self._agents[card["agent_id"]] = AgentCardState(
            agent_id=card["agent_id"],
            role=card["role"],
            status=card["status"],
            task_title=card.get("current_task_title") or card.get("task_title"),
        )
        self.query_one("#agent-panel", AgentPanel).update_agents(self._agents)

    async def _show_approval(self, event: AgentEvent) -> None:
        """Mount a prompt for a pending decision, once per distinct request.

        The backend may re-deliver ``approval.required`` - a reconnect, a
        replayed stream - and the employee must not be asked the same
        question twice. The same task coming back with a *changed* preview is
        a different matter: that is a revised answer and a genuinely new
        decision, which the fingerprint distinguishes.
        """
        payload = event.payload
        task_id = str(payload.get("task_id", ""))
        reason = str(payload.get("reason", "This needs a human check before it's finished."))
        preview = str(payload.get("preview", ""))
        fingerprint = approval_fingerprint(event.session_id, task_id, reason, preview)
        if fingerprint in self._pending_approvals:
            return
        prompt = ApprovalPrompt(
            task_id=task_id,
            reason=reason,
            preview=preview,
            session_id=event.session_id,
            fingerprint=fingerprint,
        )
        self._pending_approvals[fingerprint] = prompt
        await self.mount_message(prompt)

    @on(ApprovalPrompt.Decided)
    async def on_approval_decided(self, message: ApprovalPrompt.Decided) -> None:
        """Send one decision, exactly once, and be honest about the result.

        The ordering is the whole design: intent is committed before the
        request leaves, the request is sent once, and the prompt disappears
        only after the backend has confirmed it. The prompt disabled its own
        buttons synchronously before this coroutine started, so a second
        click cannot reach here.

        This backend has no way to ask what it already recorded and no
        idempotency key, so an ambiguous failure after dispatch cannot be
        retried safely and is not retried. It is left visibly unresolved
        instead, which is the honest state and the one a person can act on.
        """
        prompt = message.prompt
        decision = "approve" if message.approved else "reject"
        details = {
            "decision": decision,
            "task_id": prompt.task_id,
            "session_id": prompt.session_id or None,
        }

        if not prompt.task_id:
            prompt.release("Not sent: this request arrived without a task ID.")
            await self.post_notice(
                "That request arrived without a task ID, so nothing was sent.", kind="error"
            )
            return

        history = await self.services.optional_history()
        action = JournalledAction(
            history, self.settings.user_id, "approval", subject_id=prompt.task_id
        )
        # Before dispatch, a failure is known not-sent: the decision is still
        # the employee's to make, so the buttons come back.
        if not await action.begin(self.post_notice, **details):
            prompt.release("Not sent — nothing reached the AI system. You can decide again.")
            return

        try:
            await self.client.approve(
                prompt.task_id, decision, approved_by=self.settings.user_id
            )
        except InferenceError as exc:
            # `approve()` raises this for an HTTP status error and for a
            # network failure alike. After dispatch, both mean the same thing:
            # we do not know. Reading the message to guess "rejected" would be
            # inventing a decision out of an error string.
            await action.record(
                self.post_notice, "unknown", error_code="approval_result_unknown", **details
            )
            prompt.mark_unresolved(
                "Unknown — this decision was sent but never confirmed. Not re-sent."
            )
            await self.post_notice(
                f"Your decision was sent but the AI system never confirmed it "
                f"({safe_line(exc, limit=200)}). It may or may not have been recorded, so "
                "this request stays open and was not sent again. Check with the AI "
                "system before deciding again.",
                kind="error",
            )
            return
        except asyncio.CancelledError:
            await action.record(
                self.post_notice,
                "unknown",
                error_code="approval_cancelled_after_dispatch",
                **details,
            )
            prompt.mark_unresolved("Unknown — interrupted after the decision was sent.")
            raise

        # Confirmed. If the outcome write fails the decision still stands and
        # is never repeated; ``record`` reports the gap.
        await action.record(self.post_notice, "succeeded", already_done=True, **details)
        self._pending_approvals.pop(prompt.fingerprint, None)
        await prompt.remove()
        await self.post_notice("Approved." if message.approved else "Sent back for changes.")

    async def _handle_reconnecting(self) -> None:
        # A dropped live-event connection never means the job stopped —
        # AgentPackClient retries with backoff on its own (see that
        # module); this only reflects that state in the UI. Whatever
        # answer we're waiting on keeps showing "Working on it…" until a
        # real completion arrives, either live or via a resync catch-up.
        self.query_one(StatusBar).reconnecting = True
        if not self._reconnect_notice_shown:
            self._reconnect_notice_shown = True
            await self.post_notice(
                "Lost the live connection to the AI system — reconnecting…", kind="error"
            )

    async def _handle_reconnected(self) -> None:
        self.query_one(StatusBar).reconnecting = False
        if self._reconnect_notice_shown:
            self._reconnect_notice_shown = False
            await self.post_notice("Reconnected.")

    async def _handle_resync(self, event: AgentEvent) -> None:
        """The backend's buffer didn't hold everything we missed while
        disconnected (see app/events/bus.py on the backend) — reset the
        agent panel rather than assume continuity, replay whatever window
        the backend still has, and check its authoritative `terminal`
        snapshot so a pending answer resolves even if the specific
        completing event itself was evicted before we reconnected.
        Re-dispatches through handle_event, so this is naturally
        idempotent if the terminal event is also present in `events`.
        """
        payload = event.payload
        self._agents.clear()
        self.query_one("#agent-panel", AgentPanel).clear()
        for inner in payload.get("events") or []:
            await self.handle_event(
                AgentEvent(
                    type=inner["type"],
                    session_id=inner.get("session_id", event.session_id),
                    payload=inner.get("payload", {}),
                )
            )
        terminal = payload.get("terminal")
        if terminal is not None:
            await self.handle_event(
                AgentEvent(
                    type=terminal["type"],
                    session_id=terminal.get("session_id", event.session_id),
                    payload=terminal.get("payload", {}),
                )
            )

    async def _handle_agent_token(self, event: AgentEvent) -> None:
        """Streams the drafting agent's tokens into the answer bubble as
        they arrive, clearly marked as not-yet-reviewed, then swaps in the
        confirmed final text once _finish resolves the turn — see
        NEXT_STEPS.md §4 for why this is the design (review here is
        synchronous/deterministic, so there's no meaningful review *delay*
        to hide; the latency worth showing progress on is the model calls
        themselves).

        Never streams: the Specialist's research-step tokens in the
        hard/high-stakes tier (that content is upstream material for the
        Director's synthesis, not the answer itself — distinguished by
        whether a task.planned event has been seen this turn, since only
        that tier gets a Director plan), or the Runner's tokens at any
        point (classification, or the post-approval reformatting pass —
        neither is ever the visible draft).
        """
        if self._waiting_for is None or self._answer_widget is None:
            return
        payload = event.payload
        role = payload.get("role")
        text = payload.get("text") or ""
        done = bool(payload.get("done"))

        if self._draft_role is None:
            if role == "director" or (role == "specialist" and not self._plan_seen):
                self._draft_role = role
                self._draft_text = ""
                self._draft_stream = self._answer_widget.begin_stream()
                await self._draft_stream.write(PENDING_REVIEW_PREFIX)
            else:
                return  # research-step or runner tokens -- never the visible draft

        if role != self._draft_role or self._draft_stream is None:
            return  # a different call's tokens, e.g. the runner's later reformatting pass

        if text:
            self._draft_text += text
            await self._draft_stream.write(text)
            self.scroll_chat_to_end()

        if done:
            await self._answer_widget.end_stream()
            self._draft_stream = None
            self._answer_widget.update(REVIEWING_PREFIX + self._draft_text)

    async def _finish(self, event: AgentEvent, failed: bool) -> None:
        # A multi-step request's *final* step carries a different task_id
        # than the one `send_message` returned (see
        # app/api/routes_session.py) — we treat whatever completes/fails
        # next as the answer to this turn rather than matching task ids,
        # since that's what a person watching the chat would expect too.
        payload = event.payload
        if event.session_id != self.session_id:
            return
        task_id, result_ref = payload.get("task_id"), payload.get("result_ref")
        key = (
            event.session_id, event.type,
            task_id if isinstance(task_id, str) else None,
            result_ref if isinstance(result_ref, str) else None,
            # Two requests can fail in exactly the same words. The stream's own
            # id keeps them apart; without it, the second failure would look
            # like a replay of the first and leave the queue running for ever.
            event.event_id or None,
        )
        if key in self._seen_completions:
            return
        # Record before awaiting the result, including events received idle.
        # Otherwise a replay arriving during the next prompt can release it.
        self._seen_completions.add(key)
        if self._waiting_for is None or self._answer_widget is None:
            return
        answer_widget = self._answer_widget
        if self._draft_stream is not None:
            # Defensive only: call_model always emits a done=True token
            # before a task can complete/fail, so this shouldn't normally
            # still be open — but never leave a stream dangling under a
            # widget we're about to overwrite outright.
            await self._answer_widget.end_stream()
            self._draft_stream = None
        if failed:
            answer_widget.update(friendly_error(payload.get("error", "")))
        else:
            try:
                content = await self.client.get_result(payload["result_ref"])
            except InferenceError as exc:
                content = friendly_error(str(exc))
                failed = True
            # /clear may have replaced this request during the result fetch.
            if self._answer_widget is not answer_widget:
                return
            answer_widget.update(content or "_(no answer produced)_")
        self._waiting_for = None
        self._answer_widget = None
        self._plan_seen = False
        self._draft_role = None
        self._draft_text = ""
        self._set_input_busy(False)
        self.scroll_chat_to_end()
        self.finish_running_prompt(ok=not failed)

    # ---- Helpers ------------------------------------------------------------------

    def _set_input_busy(self, busy: bool) -> None:
        prompt_box = self.query_one("#prompt", Input)
        prompt_box.placeholder = "Working…" if busy else "Type a message, or /help for commands"

    async def mount_message(self, widget) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(widget)
        self.scroll_chat_to_end()

    async def post_notice(self, text: str, kind: str = "info") -> None:
        await self.mount_message(SystemNotice(text, kind=kind))

    def scroll_chat_to_end(self) -> None:
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    # ---- Actions --------------------------------------------------------------------

    async def action_clear_chat(self) -> None:
        """Clear the screen only.

        Recorded history is untouched: `/clear` is a display action, and a
        journal you can erase by tidying up your screen would not be one.
        """
        waiting = self.prompt_queue.pending_count
        await self.query_one("#chat", VerticalScroll).remove_children()
        self._pending_approvals.clear()
        # Nothing will report completion for a request whose answer bubble has
        # just been removed, so release the queue rather than wedging it.
        self.prompt_queue.clear(self.queue_key)
        self.finish_running_prompt(ok=False)
        self.queue_changed()
        self._waiting_for = None
        self._answer_widget = None
        self._agents.clear()
        self.query_one("#agent-panel", AgentPanel).clear()
        self._set_input_busy(False)
        self._plan_seen = False
        self._draft_role = None
        self._draft_stream = None
        self._draft_text = ""
        if waiting:
            await self.post_notice(f"Cleared the chat and {waiting} waiting prompt(s).")

    async def action_quit(self) -> None:
        waiting = self.prompt_queue.pending_count
        count, deadline = self._quit_confirmation
        if waiting and (waiting != count or time.monotonic() > deadline):
            self._quit_confirmation = (waiting, time.monotonic() + 5)
            self.notify(f"Quitting will discard {waiting} waiting prompt(s). Press Ctrl+C again within 5 seconds to confirm, or keep working.", severity="warning", timeout=5)
            return
        if self._workspace_session is not None:
            self._workspace_session.close()
        await self.client.close()
        self.exit()


def main() -> None:
    AgentPackApp().run()


if __name__ == "__main__":
    main()
