"""'Waiting for you' prompt: shown inline in the chat when a high-stakes
answer needs a human decision before it's finished (docs/UX.md section 2).

The widget owns one thing the app cannot do for it: refusing a second submit.
``on_button_pressed`` runs synchronously on the event loop, so disabling both
buttons there - before anything is awaited - is what makes a double click
send one decision instead of two. By the time the app's asynchronous handler
starts, the window in which a second click could get through is already shut.

A prompt is removed only when the backend has confirmed the decision. Three
other endings are possible and none of them may look like success:

* **Not sent.** Local validation or the history write failed before the
  request left. Nothing happened, so the buttons come back.
* **Unknown.** The request was dispatched and then timed out, was cancelled,
  or came back as an ambiguous error. It may have been recorded. The prompt
  stays visible and unresolved with submission disabled, because re-sending
  could double-approve and this backend has no way to ask what it recorded.
* **Recorded with a history gap.** The decision went through but the journal
  write failed. The decision stands and is never repeated.
"""

from __future__ import annotations

import hashlib

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Static

from bedrock.ui.rendering import safe_block, safe_line

PREVIEW_LIMIT = 300


def approval_fingerprint(session_id: str, task_id: str, reason: str, preview: str) -> str:
    """Identify a pending request so a re-delivered event is not a new prompt.

    A task ID alone is not enough: the same task can legitimately come back
    with a revised answer, and that revision is a genuinely new decision for
    the employee to make. Including the reason and preview distinguishes "you
    already saw this" from "this changed". Held in memory only - it says
    nothing that belongs in durable history.
    """
    digest = hashlib.sha256()
    for part in (session_id, task_id, reason, preview):
        digest.update(part.encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


class ApprovalPrompt(Static):
    """A pending approval. Posts `ApprovalPrompt.Decided` when the user
    clicks a button; the app is responsible for calling the backend, writing
    history, and deciding whether this prompt may be removed."""

    class Decided(Message):
        def __init__(self, prompt: "ApprovalPrompt", approved: bool) -> None:
            self.prompt = prompt
            self.task_id = prompt.task_id
            self.approved = approved
            super().__init__()

    def __init__(
        self,
        task_id: str,
        reason: str,
        preview: str,
        *,
        session_id: str = "",
        fingerprint: str | None = None,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.session_id = session_id
        self._reason = reason
        self._preview = preview
        self.fingerprint = fingerprint or approval_fingerprint(
            session_id, task_id, reason, preview
        )
        # "pending" -> the employee may decide; "submitting" -> a decision is
        # in flight; "unresolved" -> dispatched with an unknown result.
        self.state = "pending"

    def compose(self) -> ComposeResult:
        preview = safe_block(self._preview, limit=PREVIEW_LIMIT)
        yield Static(
            f"Waiting for you — {safe_line(self._reason, limit=300)}\n\n{preview}",
            markup=False,
            id="approval-text",
        )
        with Horizontal(id="approval-buttons"):
            yield Button("Approve", id="approve", variant="success")
            yield Button("Needs changes", id="reject", variant="error")
        yield Static("", id="approval-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.state != "pending":
            return  # A second click on an in-flight or unresolved prompt does nothing.
        self.state = "submitting"
        self._set_buttons_enabled(False)
        self._set_status("Sending your decision…")
        self.post_message(self.Decided(self, approved=(event.button.id == "approve")))

    # ---- Endings the app chooses between ---------------------------------

    def release(self, message: str) -> None:
        """Known not-sent: nothing reached the backend, so decide again."""
        self.state = "pending"
        self._set_buttons_enabled(True)
        self._set_status(message)

    def mark_unresolved(self, message: str) -> None:
        """Dispatched, result unknown. Visibly unfinished, submission closed."""
        self.state = "unresolved"
        self._set_buttons_enabled(False)
        self._set_status(message)
        self.add_class("unresolved")

    @property
    def status_text(self) -> str:
        status = self.query("#approval-status")
        return str(status.first(Static).content) if status else ""

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for button in self.query(Button):
            button.disabled = not enabled

    def _set_status(self, message: str) -> None:
        status = self.query("#approval-status")
        if status:
            status.first(Static).update(safe_line(message, limit=400))
