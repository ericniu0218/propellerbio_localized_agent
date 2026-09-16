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
from bedrock.connectors import ConnectorError, SourceHub, SourceRef, parse_source_ref
from bedrock.context import ContextStore
from bedrock.inference.base import InferenceError
from bedrock.ui.agent_panel import AgentPanel
from bedrock.ui.approval import ApprovalPrompt
from bedrock.ui.commands import Command, parse_command, parse_scientific_request
from bedrock.ui.digest_command import run_digest_command
from bedrock.ui.source_command import run_sources_command
from bedrock.ui.widgets import AssistantMessage, StatusBar, SystemNotice, UserMessage

CONNECTION_CHECK_SECONDS = 10

AGENTPACK_HELP_TEXT = """\
**Commands**

| Command | What it does |
|---|---|
| `/fold <fasta-path or source://id>` | Predict one monomer with AlphaFold 2 and render it in PyMOL |
| `/pymol [structure-path]` | Open a structure, or the latest prediction, in PyMOL |
| `/tools` | Check AlphaFold 2 and PyMOL configuration |
| `/digest <path>` | Compress a file so a model can read it cheaply |
| `/digest` | List and search what you have compressed |
| `/sources` | List the company data sources you can pull from |
| `/clear` | Clear the conversation |
| `/help` | Show this help |
| `/exit` | Quit (also `Ctrl+C`) |

You can also ask naturally, for example: “Run AlphaFold 2 on `candidate.fasta`”,
“Run AlphaFold 2 on `registry://SEQ-00123`”, or “Open the latest structure
interactively in PyMOL.”
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
        sources: SourceHub | None = None,
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
        self._scientific_job_running = False
        # Tests pass a hub with fake connectors; real runs build one from settings.
        self._sources = sources

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
                self.exit()
            case "digest":
                self.command_digest(command.args)
            case "sources":
                self.command_sources(command.args)
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
                    source = parse_source_ref(_unquote(command.args))
                    if source is not None:
                        self.fold_protein(source=source)
                    else:
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
    async def fold_protein(
        self, fasta_path: Path | None = None, source: SourceRef | None = None
    ) -> None:
        """Run AlphaFold 2 on a local FASTA, or on a record pulled from a company source.

        A source record is fetched and staged with its provenance first, and
        that provenance goes into the run manifest, so the prediction names the
        registry record and version it came from. The fetch sits inside the
        same ``try`` as the prediction so a refused or failed fetch still frees
        the one-job-at-a-time slot.
        """
        provenance = None
        try:
            if source is not None:
                await self.post_notice(f"Fetching `{source}`…")
                record, fasta_path = await self.sources.fetch_to_file(source)
                provenance = record.provenance()
            await self.post_notice(
                f"Started AlphaFold 2 for `{source or fasta_path}`. This may take a long time; the run is preserved if PyMOL rendering fails."
            )
            result = await self.biotech_workflow.predict_and_render(
                fasta_path, source_provenance=provenance
            )
        except (ToolIntegrationError, ConnectorError) as exc:
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
            if provenance:
                version = f", version {provenance['version']}" if provenance.get("version") else ""
                lines.insert(
                    2,
                    f"- Source: `{provenance['ref']}`{version} (sha256 `{provenance['sha256'][:12]}`)",
                )
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

        try:
            task_id = await self.client.send_message(self.session_id, text)
        except InferenceError as exc:
            await self.post_notice(str(exc), kind="error")
            return

        self._waiting_for = task_id
        self._answer_widget = AssistantMessage("AI system")
        await self.mount_message(self._answer_widget)
        self._answer_widget.update("_Working on it…_")
        self._set_input_busy(True)

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
        await run_digest_command(
            args, store=self.context_store, notify=self.post_notice, sources=self.sources
        )

    @property
    def context_store(self) -> ContextStore:
        if self._context_store is None:
            self._context_store = ContextStore(self.settings.context_dir)
        return self._context_store

    @work(group="sources")
    async def command_sources(self, args: str) -> None:
        """``/sources``: list, search and fetch from company systems of record."""
        await run_sources_command(args, hub=self.sources, notify=self.post_notice)

    @property
    def sources(self) -> SourceHub:
        if self._sources is None:
            self._sources = SourceHub.from_settings(self.settings)
        return self._sources

    # ---- Live events ------------------------------------------------------------

    async def handle_event(self, event: AgentEvent) -> None:
        if event.type in ("agent.started", "agent.status"):
            self._update_agent(event)
        elif event.type == "approval.required":
            await self._show_approval(event)
        elif event.type == "task.completed":
            await self._finish(event, failed=False)
        elif event.type == "task.failed":
            await self._finish(event, failed=True)

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

    async def _finish(self, event: AgentEvent, failed: bool) -> None:
        # A multi-step request's *final* step carries a different task_id
        # than the one `send_message` returned (see
        # app/api/routes_session.py) — we treat whatever completes/fails
        # next as the answer to this turn rather than matching task ids,
        # since that's what a person watching the chat would expect too.
        if self._waiting_for is None or self._answer_widget is None:
            return
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

    async def action_quit(self) -> None:
        await self.client.close()
        self.exit()


def _unquote(value: str) -> str:
    """Accept shell-style surrounding quotes without ever invoking a shell."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    return value


def _path_argument(value: str) -> Path:
    return Path(_unquote(value)).expanduser()


def main() -> None:
    AgentPackApp().run()


if __name__ == "__main__":
    main()
