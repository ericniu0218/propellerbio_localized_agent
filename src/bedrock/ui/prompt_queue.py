"""Display-only prompt queue, mounted above the app's input.

Integration: call ``show_queue(snapshot)`` on mount and on QueueChanged.
Snapshots expose ``pending``, ``running`` (an item or None), and ``paused``;
items expose ``id``, ``text``, and ``model``. An optional ``conversation_id``
(or ``session_id``) resets selection when switching conversations. Structural
protocols let the execution session supply its own immutable snapshot types.

Handle PromptQueuePanel.RemoveRequested (``item_id``), PauseRequested,
ResumeRequested, and ClearRequested in the app. These requests never mutate
the displayed queue; only a new snapshot does. Load ui/styles.tcss in the host.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Protocol

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalScroll
from textual.events import DescendantFocus
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option


class QueueItemView(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def model(self) -> str | None: ...


class QueueSnapshotView(Protocol):
    @property
    def pending(self) -> Sequence[QueueItemView]: ...

    @property
    def running(self) -> QueueItemView | None: ...

    @property
    def paused(self) -> bool: ...


def _preview(text: str, limit: int = 160) -> str:
    # Bound retained previews and remove terminal controls. Preserve literal
    # markup-looking text: all widgets below render it as plain text.
    text = " ".join(text[:2048].split())
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C"))
    return text[:limit - 1] + "…" if len(text) > limit else text


class PromptQueuePanel(Widget):
    """A small queue view with selection, buttons and a Delete shortcut."""

    can_focus = True
    BINDINGS = [Binding("delete", "remove_selected", "Remove queued prompt", show=False)]

    class RemoveRequested(Message):
        def __init__(self, item_id: str) -> None:
            super().__init__()
            self.item_id = item_id

    class PauseRequested(Message):
        pass

    class ResumeRequested(Message):
        pass

    class ClearRequested(Message):
        pass

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rows: tuple[tuple[str, str], ...] = ()
        self._queue_running_label: str | None = None
        self._queue_paused = False
        self._conversation: str | None = None
        self._rendered_conversation: str | None = None
        self._rendered_ids: tuple[str, ...] = ()
        self._render_version = 0

    def compose(self) -> ComposeResult:
        yield Static("Queue · 0 waiting · Idle", classes="queue-heading", markup=False)
        yield Static("", classes="queue-running", markup=False)
        yield Static("No prompts waiting.", classes="queue-empty", markup=False)
        yield OptionList(id="queue-pending", markup=False, compact=True)
        with HorizontalScroll(classes="queue-actions"):
            yield Button("Remove selected", id="queue-remove", disabled=True)
            yield Button("Pause", id="queue-toggle", disabled=True)
            yield Button("Clear queue", id="queue-clear", disabled=True)
        yield Static(
            "Clear queue removes waiting prompts only. Pending prompts are kept only while the app is open.",
            classes="queue-hint", markup=False,
        )

    def on_mount(self) -> None:
        self._render_queue()

    def show_queue(self, snapshot: QueueSnapshotView) -> None:
        """Accept a snapshot before or after mounting, without retaining its bodies."""
        previous = (self._rows, self._queue_running_label, self._queue_paused, self._conversation)
        self._rows = tuple(
            (item.id, f"{number}. [{_preview(item.model or 'default model', 40)}] {_preview(item.text)}")
            for number, item in enumerate(snapshot.pending, 1)
        )
        running = snapshot.running
        self._queue_running_label = (
            f"Running · [{_preview(running.model or 'default model', 40)}] {_preview(running.text)}"
            if running is not None else None
        )
        self._queue_paused = snapshot.paused
        self._conversation = getattr(snapshot, "conversation_id", getattr(snapshot, "session_id", None))
        current = (self._rows, self._queue_running_label, self._queue_paused, self._conversation)
        if self.is_mounted and current != previous:
            self._render_queue()

    def _render_queue(self) -> None:
        pending = self.query_one("#queue-pending", OptionList)
        previous_scroll = pending.scroll_offset.y
        previous_index = pending.highlighted
        previous_id = (
            pending.get_option_at_index(previous_index).id
            if previous_index is not None and previous_index < pending.option_count else None
        )
        identifiers = tuple(identifier for identifier, _ in self._rows)
        same_conversation = self._conversation == self._rendered_conversation
        # The minimal shared snapshot has no conversation ID. Globally unique
        # item IDs still let us recognize a wholly replaced list and reset it.
        if self._conversation is None and self._rendered_ids and identifiers:
            same_conversation = bool(set(self._rendered_ids).intersection(identifiers))
        same_rows = same_conversation and identifiers == self._rendered_ids
        if same_rows:
            for index, (_, label) in enumerate(self._rows):
                if pending.get_option_at_index(index).prompt != label:
                    pending.replace_option_prompt_at_index(index, label)
        else:
            pending.set_options(Option(label, id=identifier) for identifier, label in self._rows)
            if identifiers:
                pending.highlighted = (
                    identifiers.index(previous_id) if same_conversation and previous_id in identifiers
                    else min(previous_index or 0, len(identifiers) - 1) if same_conversation
                    else 0
                )
            else:
                pending.highlighted = None

        state = "Paused" if self._queue_paused else "Running" if self._queue_running_label else "Ready" if identifiers else "Idle"
        heading = self.query_one(".queue-heading", Static)
        heading_text = f"Queue · {len(identifiers)} waiting · {state}"
        if heading.content != heading_text:
            heading.update(heading_text)
        running = self.query_one(".queue-running", Static)
        running_text = self._queue_running_label or ""
        if running.content != running_text:
            running.update(running_text)
        running.display = self._queue_running_label is not None
        self.query_one(".queue-empty").display = not identifiers
        pending.display = bool(identifiers)
        self.query_one("#queue-remove", Button).disabled = not identifiers
        toggle = self.query_one("#queue-toggle", Button)
        toggle.label = "Resume" if self._queue_paused else "Pause"
        toggle.disabled = not (identifiers or self._queue_running_label or self._queue_paused)
        self.query_one("#queue-clear", Button).disabled = not identifiers
        # Keep focus within the panel if a snapshot hides/disables its control.
        focused = self.app.focused
        if focused is not None and focused in self.walk_children() and (focused.disabled or not focused.display):
            self.focus()

        self._render_version += 1
        version = self._render_version
        self._rendered_ids = identifiers
        self._rendered_conversation = self._conversation
        restored_index = pending.highlighted

        def restore_scroll() -> None:
            if version == self._render_version and pending.highlighted == restored_index:
                keep_scroll = same_conversation and (same_rows or previous_id in identifiers)
                pending.scroll_to(y=previous_scroll if keep_scroll else 0, animate=False, immediate=True)

        self.call_after_refresh(restore_scroll)

    def action_remove_selected(self) -> None:
        pending = self.query_one("#queue-pending", OptionList)
        index = pending.highlighted
        if index is not None and index < len(self._rows):
            self.post_message(self.RemoveRequested(self._rows[index][0]))

    @on(DescendantFocus)
    def reveal_focused_control(self, event: DescendantFocus) -> None:
        if isinstance(event.widget, Button):
            # Narrow terminals scroll the action row horizontally. Reveal the
            # entire focused button immediately when navigating with Tab.
            event.widget.scroll_visible(animate=False, force=True, immediate=True)

    @on(Button.Pressed)
    def handle_queue_button(self, event: Button.Pressed) -> None:
        if event.button.id not in {"queue-remove", "queue-toggle", "queue-clear"}:
            return
        event.stop()
        if event.button.id == "queue-remove":
            self.action_remove_selected()
        elif event.button.id == "queue-toggle":
            self.post_message(self.ResumeRequested() if self._queue_paused else self.PauseRequested())
        elif self._rows:
            self.post_message(self.ClearRequested())
