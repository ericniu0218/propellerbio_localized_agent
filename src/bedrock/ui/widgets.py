"""Custom widgets for the chat screen.

None of these know anything about networking. They receive text and display it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import unicodedata

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.content import Content
from textual.reactive import reactive
from textual.widgets import Label, Markdown, OptionList, Static
from textual.widgets._markdown import MarkdownStream
from textual.widgets.option_list import Option, OptionDoesNotExist

from bedrock.models import UNTITLED

if TYPE_CHECKING:  # only for type hints; widgets never touch the disk
    from bedrock.storage import ConversationSummary

NEW_CONVERSATION = "+new"
"""Option id for the "New conversation" row. Real ids can never contain "+"."""


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
    """A model reply. Supports token-by-token streaming via ``begin_stream``.

    ``content`` is empty for a live reply, which then streams in. A reply read
    back from a saved conversation passes its text here instead, so restoring a
    conversation and receiving one produce the same widget.
    """

    def __init__(self, model_name: str, content: str = "") -> None:
        super().__init__(content)
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


class ConversationList(OptionList):
    """The sidebar: saved conversations, most recently used first.

    Clicking a row (or moving to it and pressing Enter) posts Textual's usual
    ``OptionList.OptionSelected`` message with the conversation id; the app
    decides what opening a conversation means. This widget only displays what
    it is handed, so it knows nothing about files or models.
    """

    def show(self, summaries: list[ConversationSummary], open_id: str | None) -> None:
        """Replace the rows, and highlight the conversation that is on screen."""
        previous_id = None
        previous_scroll = self.scroll_offset.y
        same_chat = getattr(self, "_open_id", None) == open_id
        if self.highlighted is not None:
            try:
                previous_id = self.get_option_at_index(self.highlighted).id
            except (OptionDoesNotExist, IndexError):
                pass
        rows: list[Option | None] = [Option("+ New conversation", id=NEW_CONVERSATION), None]
        seen: set[str] = set()
        labels = {}
        for summary in summaries:
            if summary.id in seen:
                continue
            seen.add(summary.id)
            if summary.generated_title:
                # Content(...) is plain text: a title the model wrote can never
                # be mistaken for Textual markup such as "[bold]".
                label = Content("".join(c for c in summary.title if unicodedata.category(c)[0] != "C"))
            else:
                # Dimmed, so it reads as "no name yet" rather than as a name.
                label = Content.from_markup(f"[dim italic]{UNTITLED}[/]")
            rows.append(Option(label, id=summary.id))
            labels[summary.id] = label
        previous_labels = getattr(self, "_labels", {})
        same_rows = list(previous_labels) == list(labels)
        if same_rows:
            for identifier, label in labels.items():
                if label != previous_labels[identifier]:
                    self.replace_option_prompt(identifier, label)
        else:
            self.set_options(rows)
        self._labels = labels
        self._open_id = open_id
        try:
            selected = previous_id if same_chat and previous_id is not None else open_id
            if not (same_chat and same_rows):
                self.highlighted = self.get_option_index(selected) if selected else None
        except OptionDoesNotExist:
            self.highlighted = None  # a new conversation that has not been saved yet
        if same_chat:
            self.call_after_refresh(self.scroll_to, y=previous_scroll, animate=False, immediate=True)
