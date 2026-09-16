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
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input

from bedrock.agentpack.client import AgentPackClient
from bedrock.agentpack.friendly import friendly_error
from bedrock.agentpack.models import AgentCardState, AgentEvent
from bedrock.biotech import ProteinWorkflow, ToolIntegrationError
from bedrock.config import Settings, load_settings
from bedrock.context import ContextStore, find_relevant_context, with_injected_context
from bedrock.inference.base import InferenceError
from bedrock.ui.agent_panel import AgentPanel
from bedrock.ui.approval import ApprovalPrompt
from bedrock.ui.commands import Command, parse_command, parse_scientific_request
from bedrock.ui.digest_command import run_digest_command
from bedrock.ui.workspace_command import WorkspaceCommandSession
from bedrock.ui.widgets import AssistantMessage, StatusBar, SystemNotice, UserMessage
from bedrock.workspaces import WorkspaceStore

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
| `/fold <fasta-path>` | Predict one monomer with AlphaFold 2 and render it in PyMOL |
| `/pymol [structure-path]` | Open a structure, or the latest prediction, in PyMOL |
| `/tools` | Check AlphaFold 2 and PyMOL configuration |
| `/digest <path>` | Compress a file so a model can read it cheaply |
| `/digest` | List and search what you have compressed |
| `/workspace` | List private team workspaces |
| `/workspace help` | Show workspace actions |
| `/clear` | Clear the conversation |
| `/help` | Show this help |
| `/exit` | Quit (also `Ctrl+C`) |

You can also ask naturally, for example: “Run AlphaFold 2 on `candidate.fasta`”
or “Open the latest structure interactively in PyMOL.”
"""


class AgentPackApp(App[None]):
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
        self._answer_widget: AssistantMessage | None = None
        self._agents: dict[str, AgentCardState] = {}
        self._context_store: ContextStore | None = None
        self._workspace_session: WorkspaceCommandSession | None = None
        self._scientific_job_running = False
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

    # ---- Screen layout ------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield StatusBar()
        yield AgentPanel(id="agent-panel")
        yield VerticalScroll(id="chat")
        yield Input(placeholder="Type a message, or /help for commands", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one(StatusBar).model_name = "AI system"
        self.query_one("#agent-panel", AgentPanel).clear()
        self.query_one("#prompt", Input).focus()
        self.set_interval(CONNECTION_CHECK_SECONDS, self.check_connection)
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
        if command is not None:
            await self.run_command(command)
        else:
            scientific_request = parse_scientific_request(text)
            if scientific_request is not None:
                await self.mount_message(UserMessage(text))
                await self.run_command(scientific_request)
            else:
                await self.send_message(text)

    async def run_command(self, command: Command) -> None:
        match command.name:
            case "help":
                await self.post_notice(AGENTPACK_HELP_TEXT)
            case "clear":
                await self.action_clear_chat()
            case "exit" | "quit":
                await self.action_quit()
            case "digest":
                self.command_digest(command.args)
            case "workspace" | "team":
                self.command_workspace(command.args)
            case "tools":
                await self.show_tool_status()
            case "fold":
                if not command.args:
                    await self.post_notice(
                        "Tell me which one-protein FASTA file to fold, or use `/fold <path>`.",
                        kind="error",
                    )
                elif self._scientific_job_running:
                    await self.post_notice("A scientific job is already running.", kind="error")
                else:
                    self._scientific_job_running = True
                    self.fold_protein(_path_argument(command.args))
            case "pymol":
                self.open_pymol(_path_argument(command.args) if command.args else None)
            case _:
                await self.post_notice(
                    f"`/{command.name}` isn't available in this mode. Type `/help` for what is.",
                    kind="error",
                )

    async def show_tool_status(self) -> None:
        lines = ["**Scientific tools**", ""]
        for status in self.biotech_workflow.status():
            lines.append(f"- {'Ready' if status.ready else 'Not ready'}: **{status.name}**")
            lines.extend(f"  - {issue}" for issue in status.issues)
        await self.post_notice("\n".join(lines))

    @work(exclusive=True, group="scientific")
    async def fold_protein(self, fasta_path: Path) -> None:
        await self.post_notice(
            f"Started AlphaFold 2 for `{fasta_path}`. This may take a long time; the run is preserved if PyMOL rendering fails."
        )
        try:
            result = await self.biotech_workflow.predict_and_render(fasta_path)
        except ToolIntegrationError as exc:
            await self.post_notice(str(exc), kind="error")
        except asyncio.CancelledError:
            await self.post_notice("The scientific job was cancelled.", kind="error")
            raise
        else:
            lines = [
                f"**Protein prediction completed** — run `{result.run_id}`",
                "",
                f"- Structure: `{result.structure_path}`",
                f"- Provenance: `{result.manifest_path}`",
            ]
            if result.preview_path:
                lines.append(f"- PyMOL preview: `{result.preview_path}`")
            if result.session_path:
                lines.append(f"- PyMOL session: `{result.session_path}`")
            lines.extend(f"- Warning: {warning}" for warning in result.warnings)
            lines.append("")
            lines.append("Use `/pymol` to open the latest structure interactively.")
            await self.post_notice("\n".join(lines))
        finally:
            self._scientific_job_running = False

    @work(exclusive=True, group="pymol-open")
    async def open_pymol(self, structure_path: Path | None) -> None:
        try:
            opened = await self.biotech_workflow.open_in_pymol(structure_path)
        except ToolIntegrationError as exc:
            await self.post_notice(str(exc), kind="error")
        else:
            await self.post_notice(f"Opened `{opened}` in PyMOL.")

    async def send_message(self, text: str) -> None:
        if self._waiting_for is not None:
            await self.post_notice("Still working on your last question — one at a time for now.", kind="error")
            return
        if not self.session_id:
            await self.post_notice("Still connecting to the AI system — try again in a moment.", kind="error")
            return

        await self.mount_message(UserMessage(text))
        self._agents.clear()
        self.query_one("#agent-panel", AgentPanel).clear()

        # Pull in anything relevant that was compressed earlier with
        # /digest, so using the store doesn't require remembering it
        # exists and pasting a search result in by hand — see
        # bedrock.context.auto_inject for what this can and can't do.
        # The bubble above shows exactly what the user typed; only the
        # text actually sent to the model gets the extra context, and the
        # notice below says so out loud rather than doing it silently.
        injected = find_relevant_context(text, self.context_store)
        outgoing_text = with_injected_context(text, injected)
        if injected is not None:
            await self.post_notice(injected.summary)

        try:
            task_id = await self.client.send_message(self.session_id, outgoing_text)
        except InferenceError as exc:
            await self.post_notice(str(exc), kind="error")
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

    # ---- Stored context ---------------------------------------------------------

    @work(group="digest")
    async def command_digest(self, args: str) -> None:
        """``/digest`` in agent-pack mode.

        No ``backend`` is passed, so digests here are the computed ones only.
        That is the right call rather than an omission: this app has no direct
        model connection - it talks to the agent system over REST - and the
        summarising a model would do belongs on that backend, next to the
        agents, not in the client. The computed digest is the same either way.
        """
        await run_digest_command(args, store=self.context_store, notify=self.post_notice)

    @property
    def context_store(self) -> ContextStore:
        if self._context_store is None:
            self._context_store = ContextStore(self.settings.context_dir)
        return self._context_store

    # ---- Shared workspaces ------------------------------------------------------

    @work(group="workspace")
    async def command_workspace(self, args: str) -> None:
        await self.workspace_session.run(args, self.post_notice)

    @property
    def workspace_session(self) -> WorkspaceCommandSession:
        if self._workspace_session is None:
            store = WorkspaceStore.for_user(
                self.settings.workspace_dir,
                self.settings.user_id,
                self.settings.display_name,
            )
            self._workspace_session = WorkspaceCommandSession(
                store,
                self.context_store,
                selection_changed=self._set_workspace_name,
            )
        return self._workspace_session

    def _set_workspace_name(self, name: str | None) -> None:
        self.query_one(StatusBar).workspace_name = name

    # ---- Live events ------------------------------------------------------------

    async def handle_event(self, event: AgentEvent) -> None:
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
        payload = event.payload
        await self.mount_message(
            ApprovalPrompt(
                task_id=payload["task_id"],
                reason=payload.get("reason", "This needs a human check before it's finished."),
                preview=payload.get("preview", ""),
            )
        )

    @on(ApprovalPrompt.Decided)
    async def on_approval_decided(self, message: ApprovalPrompt.Decided) -> None:
        decision = "approve" if message.approved else "reject"
        await message.prompt.remove()
        try:
            await self.client.approve(message.task_id, decision, approved_by=self.settings.user_id)
        except InferenceError as exc:
            await self.post_notice(str(exc), kind="error")
            return
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
        if self._waiting_for is None or self._answer_widget is None:
            return
        if self._draft_stream is not None:
            # Defensive only: call_model always emits a done=True token
            # before a task can complete/fail, so this shouldn't normally
            # still be open — but never leave a stream dangling under a
            # widget we're about to overwrite outright.
            await self._answer_widget.end_stream()
            self._draft_stream = None
        payload = event.payload
        if failed:
            self._answer_widget.update(friendly_error(payload.get("error", "")))
        else:
            try:
                content = await self.client.get_result(payload["result_ref"])
            except InferenceError as exc:
                content = friendly_error(str(exc))
            self._answer_widget.update(content or "_(no answer produced)_")
        self._waiting_for = None
        self._answer_widget = None
        self._plan_seen = False
        self._draft_role = None
        self._draft_text = ""
        self._set_input_busy(False)
        self.scroll_chat_to_end()

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
        await self.query_one("#chat", VerticalScroll).remove_children()
        self._waiting_for = None
        self._answer_widget = None
        self._agents.clear()
        self.query_one("#agent-panel", AgentPanel).clear()
        self._set_input_busy(False)
        self._plan_seen = False
        self._draft_role = None
        self._draft_stream = None
        self._draft_text = ""

    async def action_quit(self) -> None:
        if self._workspace_session is not None:
            self._workspace_session.close()
        await self.client.close()
        self.exit()


def _path_argument(value: str) -> Path:
    """Accept shell-style surrounding quotes without ever invoking a shell."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    return Path(value).expanduser()


def main() -> None:
    AgentPackApp().run()


if __name__ == "__main__":
    main()
