"""Approving a high-stakes answer: once, or honestly not at all.

This is the only place in the client where a click causes something outside
this machine to happen, which makes it the only place where "we are not sure
what happened" is a real state the interface has to be able to show. The
backend offers no way to ask what it recorded and no idempotency key, so
after a request has been dispatched there is nothing safe to retry - and
these tests exist mostly to keep a future well-meaning retry out.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest
from textual.app import App
from textual.widgets import Button

from bedrock.agentpack.models import AgentEvent
from bedrock.agentpack_app import AgentPackApp
from bedrock.config import Settings
from bedrock.inference.base import InferenceError
from bedrock.ui.approval import ApprovalPrompt
from bedrock.ui.widgets import SystemNotice

from tests.conftest import FakeHistoryStore, make_services

PENDING = {
    "task_id": "task_7",
    "reason": "High-stakes output requires human approval before completion.",
    "preview": "Draft regulatory section.",
}


class ApprovalClient:
    """An agent-pack client whose ``approve`` can stall or fail on demand."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.approvals: list[tuple[str, str, str]] = []
        self.started = 0
        self._queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def check_connection(self) -> bool:
        return True

    async def create_session(self, user_id: str) -> str:
        return "sess_fake"

    async def send_message(self, session_id: str, text: str) -> str:
        return "task_7"

    async def get_result(self, result_ref: str) -> str:
        return ""

    async def approve(self, task_id, decision, approved_by, comment=None) -> None:
        self.started += 1
        if self.error is not None:
            raise self.error
        self.approvals.append((task_id, decision, approved_by))

    async def events(self, session_id: str) -> AsyncIterator[AgentEvent]:
        while True:
            yield await self._queue.get()

    def push(self, type: str, payload: dict) -> None:
        self._queue.put_nowait(AgentEvent(type=type, session_id="sess_fake", payload=payload))

    async def close(self) -> None:
        return None


def make_app(client: ApprovalClient, history: FakeHistoryStore | None = None) -> AgentPackApp:
    return AgentPackApp(
        settings=Settings(user_id="alice"),
        client=client,
        services=make_services(history=history or FakeHistoryStore()),
    )


def notices(app) -> str:
    return "\n".join(str(widget.source) for widget in app.query(SystemNotice))


def buttons_disabled(prompt: ApprovalPrompt) -> bool:
    return all(button.disabled for button in prompt.query(Button))


async def show_prompt(app, pilot, client, payload=None) -> ApprovalPrompt:
    client.push("approval.required", payload or PENDING)
    for _ in range(50):
        await pilot.pause()
        if app.query(ApprovalPrompt):
            break
    return app.query_one(ApprovalPrompt)


async def test_the_prompt_disables_both_buttons_while_handling_the_press(tmp_path):
    """The guard has to be synchronous. Textual delivers the button press to
    this widget before the app's asynchronous handler runs, so disabling here
    is what closes the window a second press could get through."""

    class Harness(App[None]):
        def compose(self):
            yield ApprovalPrompt(task_id="task_7", reason="why", preview="draft")

    app = Harness()
    async with app.run_test() as pilot:
        prompt = app.query_one(ApprovalPrompt)
        prompt.query_one("#approve", Button).press()
        await pilot.pause()

        assert prompt.state == "submitting"
        assert buttons_disabled(prompt)

        # A second press on an already-submitting prompt changes nothing.
        prompt.query_one("#reject", Button).press()
        await pilot.pause()
        assert prompt.state == "submitting"


async def test_two_presses_send_one_decision(tmp_path):
    """Both presses are queued before either is handled - the real race - and
    only one request reaches the AI system."""
    client = ApprovalClient()
    app = make_app(client)

    async with app.run_test() as pilot:
        prompt = await show_prompt(app, pilot, client)
        approve = prompt.query_one("#approve", Button)
        approve.press()
        approve.press()
        await pilot.pause()
        await pilot.pause()

        assert client.started == 1
        assert client.approvals == [("task_7", "approve", "alice")]
        assert not app.query(ApprovalPrompt)
        assert notices(app).count("Approved.") == 1


async def test_a_failed_intent_write_means_not_sent_and_decidable_again(tmp_path):
    """Before dispatch, a failure is knowledge: nothing left this machine, so
    the employee keeps their buttons."""
    client = ApprovalClient()
    history = FakeHistoryStore(fail_states={"intent"})
    app = make_app(client, history)

    async with app.run_test() as pilot:
        prompt = await show_prompt(app, pilot, client)
        prompt.query_one("#approve", Button).press()
        await pilot.pause()

        assert client.started == 0
        assert app.query(ApprovalPrompt)
        assert not buttons_disabled(prompt)
        assert prompt.state == "pending"
        assert "Not sent" in prompt.status_text
        assert "Approved." not in notices(app)


async def test_an_ambiguous_failure_after_dispatch_is_unknown_never_rejected(tmp_path):
    """``approve()`` collapses HTTP errors and network failures into one
    exception type. Reading its text to decide the answer was "rejected"
    would be inventing a decision; the honest state is unknown."""
    client = ApprovalClient(error=InferenceError("Could not record your decision: 403 Forbidden"))
    history = FakeHistoryStore()
    app = make_app(client, history)

    async with app.run_test() as pilot:
        prompt = await show_prompt(app, pilot, client)
        prompt.query_one("#approve", Button).press()
        await pilot.pause()

        assert client.started == 1
        assert app.query(ApprovalPrompt)          # stays visibly unfinished
        assert prompt.state == "unresolved"
        assert buttons_disabled(prompt)           # and cannot be re-sent
        assert "Unknown" in prompt.status_text
        assert "Approved." not in notices(app)
        assert "rejected" not in notices(app).lower()
        assert [row["state"] for row in history.rows] == ["intent", "unknown"]

        # Clicking again does not smuggle in a retry.
        prompt.query_one("#approve", Button).press()
        await pilot.pause()
        assert client.started == 1


async def test_a_confirmed_decision_survives_a_failed_journal_write(tmp_path):
    """The AI system has accepted it. Sending it again to make the journal
    tidy could double-approve, so the gap is reported instead."""
    client = ApprovalClient()
    history = FakeHistoryStore(fail_states={"succeeded"})
    app = make_app(client, history)

    async with app.run_test() as pilot:
        prompt = await show_prompt(app, pilot, client)
        prompt.query_one("#approve", Button).press()
        await pilot.pause()

        assert client.approvals == [("task_7", "approve", "alice")]
        assert client.started == 1
        assert not app.query(ApprovalPrompt)
        text = notices(app)
        assert "Approved." in text
        assert "Local history has a gap" in text
        assert "NOT been repeated" in text


async def test_a_re_delivered_event_reuses_the_pending_prompt(tmp_path):
    client = ApprovalClient()
    app = make_app(client)

    async with app.run_test() as pilot:
        await show_prompt(app, pilot, client)
        client.push("approval.required", PENDING)
        await pilot.pause()
        await pilot.pause()

        assert len(app.query(ApprovalPrompt)) == 1


async def test_a_revised_request_is_a_new_decision(tmp_path):
    """Same task, different answer. The employee has not seen this one."""
    client = ApprovalClient()
    app = make_app(client)

    async with app.run_test() as pilot:
        await show_prompt(app, pilot, client)
        client.push("approval.required", {**PENDING, "preview": "Revised regulatory section."})
        for _ in range(50):
            await pilot.pause()
            if len(app.query(ApprovalPrompt)) == 2:
                break

        assert len(app.query(ApprovalPrompt)) == 2


async def test_an_unresolved_prompt_is_never_auto_submitted_when_redelivered(tmp_path):
    client = ApprovalClient(error=InferenceError("timed out"))
    app = make_app(client)

    async with app.run_test() as pilot:
        prompt = await show_prompt(app, pilot, client)
        prompt.query_one("#approve", Button).press()
        await pilot.pause()
        assert prompt.state == "unresolved"

        client.push("approval.required", PENDING)
        await pilot.pause()
        await pilot.pause()

        assert len(app.query(ApprovalPrompt)) == 1
        assert client.started == 1


async def test_rejecting_reports_needs_changes(tmp_path):
    client = ApprovalClient()
    app = make_app(client)

    async with app.run_test() as pilot:
        prompt = await show_prompt(app, pilot, client)
        prompt.query_one("#reject", Button).press()
        await pilot.pause()

        assert client.approvals == [("task_7", "reject", "alice")]
        assert "Sent back for changes." in notices(app)


async def test_a_preview_cannot_repaint_the_terminal(tmp_path):
    client = ApprovalClient()
    app = make_app(client)

    async with app.run_test() as pilot:
        prompt = await show_prompt(
            app, pilot, client, {**PENDING, "preview": "Draft\x1b[2J removed"}
        )
        rendered = str(prompt.query_one("#approval-text").content)

        assert "\x1b" not in rendered
        assert "removed" in rendered
