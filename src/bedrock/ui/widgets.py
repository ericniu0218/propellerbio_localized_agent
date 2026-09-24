"""Custom widgets for the chat screen.

None of these know anything about networking. They receive text and display it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import unicodedata

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.content import Content
from textual.reactive import reactive
from textual.widgets import Input, Label, Markdown, OptionList, Static
from textual.widgets._markdown import MarkdownStream
from textual.widgets.option_list import Option, OptionDoesNotExist

from bedrock.attachments import dropped_references
from bedrock.models import UNTITLED

if TYPE_CHECKING:  # only for type hints; widgets never touch the disk
    from bedrock.storage import ConversationSummary

NEW_CONVERSATION = "+new"
"""Option id for the "New conversation" row. Real ids can never contain "+"."""


class StatusBar(Horizontal):
    """Top bar: app name on the left, current local context on the right.

    The ``context`` counter shows how much of the model's context window the
    last chat prompt filled, as the server counted it: ``context 3.1k/8k``.
    It is worked out from numbers the server already sends back with each
    reply, and it only ever appears here - nothing is logged, saved or sent.
    It stays blank until a reply has reported its size.
    """

    DEFAULT_CSS = """
    StatusBar #status-context {
        width: auto;
        color: $text-muted;
        margin-right: 3;
    }
    StatusBar #status-context.full {
        color: $warning;
    }
    """

    FULL_SHARE = 0.9
    """At or above this share of the window the counter is highlighted: the
    next turns will start leaving out the oldest messages."""

    model_name: reactive[str] = reactive("no model")
    context_used: reactive[int | None] = reactive(None)
    """Prompt tokens of the last chat reply; None until a server reports it."""
    context_window: reactive[int | None] = reactive(None)
    workspace_name: reactive[str | None] = reactive(None)
    connected: reactive[bool | None] = reactive(None)  # None = not checked yet
    # Set true while AgentPackClient is between a dropped live-event
    # connection and a successful reconnect — distinct from `connected`
    # (the periodic /health poll) because the two can disagree: health
    # checks can keep passing while the *event stream* specifically has
    # dropped and is being retried with backoff.
    reconnecting: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        yield Label("Bedrock Private AI", id="status-title")
        yield Label("", id="status-workspace")
        yield Label("", id="status-model")
        yield Label("", id="status-context")
        yield Label("", id="status-connection")

    def watch_model_name(self, value: str) -> None:
        self.query_one("#status-model", Label).update(f"model: {value}")

    def watch_context_used(self, _value: int | None) -> None:
        self._render_context_label()

    def watch_context_window(self, _value: int | None) -> None:
        self._render_context_label()

    def _render_context_label(self) -> None:
        label = self.query_one("#status-context", Label)
        used, window = self.context_used, self.context_window
        if used is None or not window:
            label.update("")
            label.remove_class("full")
            return
        label.update(f"context {_tokens(used)}/{_tokens(window)}")
        label.set_class(used >= window * self.FULL_SHARE, "full")

    def watch_workspace_name(self, value: str | None) -> None:
        self.query_one("#status-workspace", Label).update(
            f"space: {value}" if value else ""
        )

    def watch_connected(self, _value: bool | None) -> None:
        self._render_connection_label()

    def watch_reconnecting(self, _value: bool) -> None:
        self._render_connection_label()

    def _render_connection_label(self) -> None:
        label = self.query_one("#status-connection", Label)
        label.remove_class("ok", "bad", "unknown")
        if self.reconnecting:
            label.update("● reconnecting…")
            label.add_class("unknown")
        elif self.connected is None:
            label.update("● connecting…")
            label.add_class("unknown")
        elif self.connected:
            label.update("● connected")
            label.add_class("ok")
        else:
            label.update("● disconnected")
            label.add_class("bad")


def _tokens(count: int) -> str:
    """``3174`` as ``3.1k`` and ``8192`` as ``8k``: windows are powers of two, so k is 1024."""
    if count < 1024:
        return str(count)
    return f"{count / 1024:.1f}".removesuffix(".0") + "k"


class PromptInput(Input):
    """The message box. Dropping files onto the terminal attaches them.

    A terminal reports a dropped file as a paste of its path. When a paste is
    nothing but paths to existing files, they are written as ``@`` references
    instead, so the employee sees what will be attached and can type the
    question around them. Any other paste behaves exactly like ``Input``'s.
    See ``bedrock/attachments.py``; checking that the paths exist is the only
    disk access here.
    """

    def _on_paste(self, event: events.Paste) -> None:
        references = dropped_references(event.text) if event.text else None
        if references is None:
            return  # Textual then runs Input's own paste handler as usual.
        event.prevent_default()  # otherwise Input's handler pastes the raw path too
        event.stop()
        start, end = sorted(self.selection)
        if start == end and start > 0 and self.value[start - 1] == "@":
            start -= 1  # "type @, then drop the file": that @ is already in the reference
        if start > 0 and not self.value[start - 1].isspace():
            references = " " + references
        if end == len(self.value) or not self.value[end].isspace():
            references += " "
        self.replace(references, start, end)


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
