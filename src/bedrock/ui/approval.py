"""'Waiting for you' prompt: shown inline in the chat when a high-stakes
answer needs a human decision before it's finished (docs/UX.md section 2).
Mounted like any other chat message, removed once decided.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Static


class ApprovalPrompt(Static):
    """A pending approval. Posts `ApprovalPrompt.Decided` when the user
    clicks a button; the app is responsible for calling the backend and
    removing this widget afterwards."""

    class Decided(Message):
        def __init__(self, prompt: "ApprovalPrompt", approved: bool) -> None:
            self.prompt = prompt
            self.task_id = prompt.task_id
            self.approved = approved
            super().__init__()

    def __init__(self, task_id: str, reason: str, preview: str) -> None:
        super().__init__()
        self.task_id = task_id
        self._reason = reason
        self._preview = preview

    def compose(self) -> ComposeResult:
        preview = self._preview.strip()
        if len(preview) > 300:
            preview = preview[:300].rstrip() + "…"
        yield Static(f"**Waiting for you** — {self._reason}\n\n{preview}", markup=True)
        with Horizontal(id="approval-buttons"):
            yield Button("Approve", id="approve", variant="success")
            yield Button("Needs changes", id="reject", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Decided(self, approved=(event.button.id == "approve")))
