"""The Textual application: wires the UI to the inference backend.

Responsibilities of this file, and nothing more:

* build the screen (status bar, scrolling conversation, input box)
* turn what the user types into either a slash command or a chat message
* run the streaming request in a background worker and feed the text to the
  widget as it arrives
* keep the connection indicator up to date

It talks to the model *only* through the ``InferenceBackend`` interface.
"""

from __future__ import annotations

import asyncio

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input

from bedrock.config import Settings, load_settings
from bedrock.inference import InferenceBackend, InferenceError, create_backend
from bedrock.models import Conversation
from bedrock.ui.commands import HELP_TEXT, Command, parse_command
from bedrock.ui.widgets import AssistantMessage, StatusBar, SystemNotice, UserMessage

CONNECTION_CHECK_SECONDS = 10


class BedrockApp(App[None]):
    TITLE = "Bedrock Private AI"
    CSS_PATH = "ui/styles.tcss"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("escape", "cancel_stream", "Stop", show=False),
    ]

    def __init__(self, settings: Settings | None = None, backend: InferenceBackend | None = None) -> None:
        super().__init__()
        self.theme = "textual-dark"  # always a dark, black-background terminal look
        self.settings = settings or load_settings()
        # Tests pass a fake backend; real runs build one from settings.
        self.backend: InferenceBackend = backend or create_backend(self.settings)
        self.conversation = Conversation(system_prompt=self.settings.system_prompt)
        self.model_name: str | None = self.settings.model
        self._streaming = False

    # ---- Screen layout ----------------------------------------------------

    def compose(self) -> ComposeResult:
        yield StatusBar()
        yield VerticalScroll(id="chat")
        yield Input(placeholder="Type a message, or /help for commands", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        self._refresh_status()
        self.set_interval(CONNECTION_CHECK_SECONDS, self.check_connection)
        self.check_connection()
        self.pick_default_model()

    # ---- Connection + model bookkeeping -----------------------------------

    def _refresh_status(self) -> None:
        self.query_one(StatusBar).model_name = self.model_name or "no model"

    @work(exclusive=True, group="connection")
    async def check_connection(self) -> None:
        connected = await self.backend.check_connection()
        self.query_one(StatusBar).connected = connected

    @work(group="models")
    async def pick_default_model(self) -> None:
        """If no model was configured, use the first one the server offers."""
        if self.model_name:
            return
        try:
            models = await self.backend.list_models()
        except InferenceError:
            return  # Connection indicator already tells the story; try again on /model.
        if models:
            self.model_name = models[0].id
            self._refresh_status()

    # ---- Input handling ---------------------------------------------------

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
            await self.send_message(text)

    async def run_command(self, command: Command) -> None:
        match command.name:
            case "help":
                await self.post_notice(HELP_TEXT)
            case "clear":
                await self.action_clear_chat()
            case "exit" | "quit":
                self.exit()
            case "model":
                await self.command_model(command.args)
            case _:
                await self.post_notice(f"Unknown command `/{command.name}`. Type `/help` for a list.", kind="error")

    async def command_model(self, name: str) -> None:
        if name:
            self.model_name = name
            self._refresh_status()
            await self.post_notice(f"Switched to model `{name}`.")
            return
        try:
            models = await self.backend.list_models()
        except InferenceError as exc:
            await self.post_notice(str(exc), kind="error")
            return
        if not models:
            await self.post_notice("The server reports no models. Pull one first, e.g. `ollama pull llama3.2`.", kind="error")
            return
        lines = ["**Available models** (use `/model <name>` to switch)", ""]
        for model in models:
            marker = "→" if model.id == self.model_name else " "
            lines.append(f"- {marker} `{model.id}`")
        await self.post_notice("\n".join(lines))

    # ---- Chat -------------------------------------------------------------

    async def send_message(self, text: str) -> None:
        if self._streaming:
            await self.post_notice("Still responding - press Esc to stop it first.", kind="error")
            return
        if not self.model_name:
            await self.post_notice("No model selected. Use `/model` to see what is available.", kind="error")
            return

        self.conversation.add_user(text)
        await self.mount_message(UserMessage(text))

        reply_widget = AssistantMessage(self.model_name)
        await self.mount_message(reply_widget)
        self.stream_reply(reply_widget)

    @work(exclusive=True, group="chat")
    async def stream_reply(self, widget: AssistantMessage) -> None:
        """Background worker: pull fragments from the backend into the widget."""
        prompt_box = self.query_one("#prompt", Input)
        self._streaming = True
        prompt_box.placeholder = "Responding… (Esc to stop)"
        # Serialise the history *before* adding the empty reply placeholder,
        # otherwise the model would see a blank assistant turn at the end.
        api_messages = self.conversation.to_api()
        reply = self.conversation.add_assistant()
        stream = widget.begin_stream()
        try:
            async for fragment in self.backend.stream_chat(self.model_name, api_messages):
                reply.content += fragment
                await stream.write(fragment)
                self.scroll_chat_to_end()
        except InferenceError as exc:
            await self.post_notice(str(exc), kind="error")
        except asyncio.CancelledError:
            reply.content += "\n\n*(stopped)*"
            await stream.write("\n\n*(stopped)*")
            raise
        finally:
            await widget.end_stream()
            self._streaming = False
            prompt_box.placeholder = "Type a message, or /help for commands"
            self.scroll_chat_to_end()

    async def mount_message(self, widget) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(widget)
        self.scroll_chat_to_end()

    async def post_notice(self, text: str, kind: str = "info") -> None:
        await self.mount_message(SystemNotice(text, kind=kind))

    def scroll_chat_to_end(self) -> None:
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    # ---- Actions (key bindings) -------------------------------------------

    async def action_clear_chat(self) -> None:
        self.workers.cancel_group(self, "chat")
        self.conversation.clear()
        await self.query_one("#chat", VerticalScroll).remove_children()
        self._streaming = False

    def action_cancel_stream(self) -> None:
        self.workers.cancel_group(self, "chat")

    async def action_quit(self) -> None:
        await self.backend.close()
        self.exit()


def main() -> None:
    """Entry point for the ``bedrock`` command.

    Dispatches on ``settings.backend`` (env var ``BEDROCK_BACKEND``, or the
    config file — see ``bedrock.config.settings``): "openai" (default) runs
    this single-model-server app unchanged; "agentpack" runs
    ``AgentPackApp`` instead, which talks to the full biotech agent-pack
    system. One command either way — IT/the employee never choose a
    different program, just a setting.
    """
    settings = load_settings()
    if settings.backend == "agentpack":
        from bedrock.agentpack_app import AgentPackApp

        AgentPackApp(settings=settings).run()
    else:
        BedrockApp(settings=settings).run()


if __name__ == "__main__":
    main()
