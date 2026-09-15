"""Custom widgets for the chat screen.

None of these know anything about networking. They receive text and display it.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Label, Markdown, Static
from textual.widgets._markdown import MarkdownStream


class StatusBar(Horizontal):
    """Top bar: app name on the left, current local context on the right."""

    model_name: reactive[str] = reactive("no model")
    workspace_name: reactive[str | None] = reactive(None)
    connected: reactive[bool | None] = reactive(None)  # None = not checked yet

    def compose(self) -> ComposeResult:
        yield Label("Bedrock Private AI", id="status-title")
        yield Label("", id="status-workspace")
        yield Label("", id="status-model")
        yield Label("", id="status-connection")

    def watch_model_name(self, value: str) -> None:
        self.query_one("#status-model", Label).update(f"model: {value}")

    def watch_workspace_name(self, value: str | None) -> None:
        self.query_one("#status-workspace", Label).update(
            f"space: {value}" if value else ""
        )

    def watch_connected(self, value: bool | None) -> None:
        label = self.query_one("#status-connection", Label)
        label.remove_class("ok", "bad", "unknown")
        if value is None:
            label.update("● connecting…")
            label.add_class("unknown")
        elif value:
            label.update("● connected")
            label.add_class("ok")
        else:
            label.update("● disconnected")
            label.add_class("bad")


class UserMessage(Static):
    """A message typed by the employee. Rendered as plain text, not Markdown,
    so nothing the user types is accidentally reinterpreted."""

    def __init__(self, text: str) -> None:
        super().__init__(text, markup=False)


class AssistantMessage(Markdown):
    """A model reply. Supports token-by-token streaming via ``begin_stream``."""

    def __init__(self, model_name: str) -> None:
        super().__init__("")
        self._stream: MarkdownStream | None = None

    def begin_stream(self) -> MarkdownStream:
        self._stream = self.get_stream(self)
        return self._stream

    async def end_stream(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None


class SystemNotice(Markdown):
    """Help text, errors and other messages from the app itself (not the model)."""

    def __init__(self, text: str, kind: str = "info") -> None:
        super().__init__(text)
        self.add_class(kind)  # "info" or "error"
