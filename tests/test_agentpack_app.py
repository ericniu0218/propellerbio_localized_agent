"""Drive the real Textual agent-pack app headlessly with a fake client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest

from bedrock.agentpack.models import AgentEvent
from bedrock.agentpack_app import AgentPackApp
from bedrock.biotech import ProteinWorkflowResult, ToolStatus
from bedrock.config import Settings
from bedrock.ui.agent_panel import AgentPanel
from bedrock.ui.approval import ApprovalPrompt
from bedrock.ui.widgets import AssistantMessage, StatusBar, SystemNotice, UserMessage


class FakeAgentPackClient:
    def __init__(self) -> None:
        self.connected = True
        self.closed = False
        self.sent_messages: list[tuple[str, str]] = []
        self.approvals: list[tuple[str, str, str]] = []
        self.results: dict[str, str] = {}
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    async def check_connection(self) -> bool:
        return self.connected

    async def create_session(self, user_id: str) -> str:
        return "sess_fake"

    async def send_message(self, session_id: str, text: str) -> str:
        self.sent_messages.append((session_id, text))
        return "task_fake"

    async def get_result(self, result_ref: str) -> str:
        return self.results.get(result_ref, "")

    async def approve(self, task_id: str, decision: str, approved_by: str, comment: str | None = None) -> None:
        self.approvals.append((task_id, decision, approved_by))

    async def events(self, session_id: str) -> AsyncIterator[AgentEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    def push(self, type: str, payload: dict) -> None:
        self._queue.put_nowait(AgentEvent(type=type, session_id="sess_fake", payload=payload))

    async def close(self) -> None:
        self.closed = True


class FakeBiotechWorkflow:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.folded: list[Path] = []
        self.opened: list[Path | None] = []

    def status(self):
        return (
            ToolStatus("AlphaFold 2", True, ()),
            ToolStatus("open-source PyMOL", True, ()),
        )

    async def predict_and_render(self, fasta_path: Path) -> ProteinWorkflowResult:
        self.folded.append(fasta_path)
        return ProteinWorkflowResult(
            run_id="run_test",
            run_dir=self.root,
            structure_path=self.root / "ranked_0.pdb",
            preview_path=self.root / "preview.png",
            session_path=self.root / "structure.pse",
            manifest_path=self.root / "manifest.json",
        )

    async def open_in_pymol(self, structure_path: Path | None = None) -> Path:
        self.opened.append(structure_path)
        return structure_path or self.root / "ranked_0.pdb"


@pytest.fixture
def fake_client() -> FakeAgentPackClient:
    return FakeAgentPackClient()


def make_app(
    client: FakeAgentPackClient,
    biotech_workflow: FakeBiotechWorkflow | None = None,
) -> AgentPackApp:
    return AgentPackApp(
        settings=Settings(user_id="alice"),
        client=client,
        biotech_workflow=biotech_workflow,
    )


async def test_send_message_shows_final_answer_from_result_ref(fake_client: FakeAgentPackClient):
    fake_client.results["res_1"] = "The boiling point of water is 100C at sea level."
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"boiling point?", "enter")
        await pilot.pause()

        assert fake_client.sent_messages == [("sess_fake", "boiling point?")]
        assert str(app.query_one(UserMessage).content) == "boiling point?"
        assert "Working on it" in app.query_one(AssistantMessage).source

        fake_client.push("task.completed", {"task_id": "task_fake", "result_ref": "res_1", "citations": []})
        await pilot.pause()

        assert "boiling point of water" in app.query_one(AssistantMessage).source


async def test_agent_panel_shows_friendly_status_and_clears_on_done(fake_client: FakeAgentPackClient):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"hi", "enter")
        await pilot.pause()

        fake_client.push("agent.status", {
            "agent_id": "a1", "role": "specialist", "status": "running_tool", "task_title": "Literature scan",
        })
        await pilot.pause()
        panel = app.query_one(AgentPanel)
        assert any("Researcher" in line for line in panel.lines)

        fake_client.push("agent.status", {
            "agent_id": "a1", "role": "specialist", "status": "done", "task_title": "Literature scan",
        })
        await pilot.pause()
        assert panel.lines == ()  # done agents don't linger


async def test_failed_task_shows_plain_language_error(fake_client: FakeAgentPackClient):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"hi", "enter")
        await pilot.pause()

        fake_client.push("task.failed", {"task_id": "task_fake", "error": "BudgetExceededError: boom"})
        await pilot.pause()

        reply = app.query_one(AssistantMessage).source
        assert "BudgetExceededError" not in reply
        assert "thinking" in reply.lower() or "try again" in reply.lower()


async def test_approval_prompt_approve_calls_backend_and_clears(fake_client: FakeAgentPackClient):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"draft the regulatory section", "enter")
        await pilot.pause()

        fake_client.push("approval.required", {
            "task_id": "task_fake",
            "reason": "High-stakes output requires human approval before completion.",
            "preview": "Draft text here.",
        })
        await pilot.pause()
        assert app.query_one(ApprovalPrompt)

        await pilot.click("#approve")
        await pilot.pause()

        assert fake_client.approvals == [("task_fake", "approve", "alice")]
        assert not app.query(ApprovalPrompt)
        assert any("Approved" in n.source for n in app.query(SystemNotice))


async def test_one_task_at_a_time(fake_client: FakeAgentPackClient):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"first question", "enter")
        await pilot.pause()
        await pilot.press(*"second question", "enter")
        await pilot.pause()

        assert fake_client.sent_messages == [("sess_fake", "first question")]
        assert any("one at a time" in n.source for n in app.query(SystemNotice))


async def test_digest_works_in_agentpack_mode_without_a_model(fake_client, tmp_path):
    """`/digest` is available in both modes.

    This app has no direct model connection, so digests here are the computed
    ones only - which for structured data is the better summary anyway. Note
    the deliberate absence of ``workers.wait_for_complete()``: this app's event
    listener runs for the life of the session and would never finish.
    """
    from textual.widgets import Input

    path = tmp_path / "assay.csv"
    path.write_text("sample,ph\n" + "\n".join(f"S{i:04d},{6.0 + i % 10 * 0.1:.1f}" for i in range(50)))

    app = AgentPackApp(
        settings=Settings(user_id="alice", context_dir=str(tmp_path / "context")),
        client=fake_client,
    )
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = f"/digest {path}"
        await pilot.press("enter")

        for _ in range(100):
            await pilot.pause()
            if app.context_store.list():
                break

        stored = app.context_store.list()
        assert len(stored) == 1
        assert stored[0].source_kind == "tabular"
        assert stored[0].digest.generated_by == "rules"

        notices = "\n".join(str(w.source) for w in app.query(SystemNotice))
        assert "Compressed `assay.csv`" in notices


async def test_fold_command_runs_alphafold_then_reports_pymol_artifacts(
    fake_client: FakeAgentPackClient,
    tmp_path: Path,
):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(fake_client, workflow)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/fold /tmp/protein.fasta", "enter")
        await pilot.pause()

        assert workflow.folded == [Path("/tmp/protein.fasta")]
        notices = [notice.source for notice in app.query(SystemNotice)]
        assert any("Protein prediction completed" in notice for notice in notices)
        assert any("preview.png" in notice for notice in notices)


async def test_pymol_command_opens_latest_structure(
    fake_client: FakeAgentPackClient,
    tmp_path: Path,
):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(fake_client, workflow)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/pymol", "enter")
        await pilot.pause()

        assert workflow.opened == [None]
        assert any("Opened" in notice.source for notice in app.query(SystemNotice))


async def test_natural_language_pymol_request_opens_latest_structure(
    fake_client: FakeAgentPackClient,
    tmp_path: Path,
):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(fake_client, workflow)
    request = "Please open the latest structure interactively in PyMOL"
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*request, "enter")
        await pilot.pause()

        assert workflow.opened == [None]
        assert any(request in str(message.content) for message in app.query(UserMessage))


# ---- SSE reconnection (NEXT_STEPS.md §5) -----------------------------------


async def test_reconnecting_shows_status_and_a_single_notice_not_one_per_attempt(
    fake_client: FakeAgentPackClient,
):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()

        fake_client.push("client.reconnecting", {"attempt": 1, "delay_s": 1.0})
        await pilot.pause()
        fake_client.push("client.reconnecting", {"attempt": 2, "delay_s": 2.0})
        await pilot.pause()

        assert app.query_one(StatusBar).reconnecting is True
        reconnect_notices = [n for n in app.query(SystemNotice) if "reconnecting" in n.source.lower()]
        assert len(reconnect_notices) == 1  # not one per retry attempt


async def test_reconnected_clears_status_and_announces_recovery(fake_client: FakeAgentPackClient):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()

        fake_client.push("client.reconnecting", {"attempt": 1, "delay_s": 1.0})
        await pilot.pause()
        fake_client.push("client.reconnected", {})
        await pilot.pause()

        assert app.query_one(StatusBar).reconnecting is False
        assert any("reconnected" in n.source.lower() for n in app.query(SystemNotice))


async def test_reconnecting_does_not_disturb_an_in_progress_answer(fake_client: FakeAgentPackClient):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"boiling point?", "enter")
        await pilot.pause()

        fake_client.push("client.reconnecting", {"attempt": 1, "delay_s": 1.0})
        await pilot.pause()

        # The job keeps running on the backend regardless of our connection
        # state — the pending bubble must not be touched by a reconnect.
        assert "Working on it" in app.query_one(AssistantMessage).source


async def test_resync_replays_buffered_agent_status_and_resolves_via_terminal(
    fake_client: FakeAgentPackClient,
):
    fake_client.results["res_1"] = "100C at sea level."
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"boiling point?", "enter")
        await pilot.pause()

        fake_client.push(
            "session.resync",
            {
                "reason": "gap",
                "events": [
                    {
                        "type": "agent.status",
                        "session_id": "sess_fake",
                        "payload": {
                            "agent_id": "a1",
                            "role": "specialist",
                            "status": "running_tool",
                            "task_title": "Research",
                        },
                    }
                ],
                "terminal": {
                    "type": "task.completed",
                    "session_id": "sess_fake",
                    "payload": {"task_id": "task_fake", "result_ref": "res_1", "citations": []},
                },
            },
        )
        await pilot.pause()

        # The buffered agent.status was replayed (rebuilds the panel from
        # it, exercising that path), then immediately superseded by the
        # terminal event resolving the turn — the actually-important
        # outcome: a pending answer must resolve from resync's terminal
        # snapshot even when the completing event itself was evicted from
        # the backend's buffer before we reconnected.
        assert "100C" in app.query_one(AssistantMessage).source


async def test_resync_without_a_terminal_leaves_the_pending_answer_waiting(
    fake_client: FakeAgentPackClient,
):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"boiling point?", "enter")
        await pilot.pause()

        fake_client.push("session.resync", {"reason": "gap", "events": [], "terminal": None})
        await pilot.pause()

        # Nothing resolved the turn — still legitimately in progress.
        assert "Working on it" in app.query_one(AssistantMessage).source


# ---- Token streaming into the answer bubble (NEXT_STEPS.md §4) ------------


async def test_normal_tier_streams_specialist_tokens_marked_pending(fake_client: FakeAgentPackClient):
    """No task.planned event before the specialist's tokens -> normal tier
    -> the Specialist's draft *is* the answer, so it streams live."""
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"hi", "enter")
        await pilot.pause()

        fake_client.push("agent.token", {"agent_id": "a1", "role": "specialist", "task_id": "t1", "text": "The", "done": False})
        await pilot.pause()
        fake_client.push("agent.token", {"agent_id": "a1", "role": "specialist", "task_id": "t1", "text": " answer", "done": False})
        await pilot.pause()

        reply = app.query_one(AssistantMessage).source
        assert "not yet compliance-reviewed" in reply
        assert "The answer" in reply


async def test_hard_tier_ignores_research_step_and_streams_only_director_synthesis(
    fake_client: FakeAgentPackClient,
):
    """task.planned seen -> hard tier -> the Specialist's tokens are the
    upstream research step, not the answer; only the Director's synthesis
    tokens (which arrive later, same turn) are the visible draft."""
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"hard question", "enter")
        await pilot.pause()

        fake_client.push("task.planned", {"plan": {}})
        await pilot.pause()

        fake_client.push(
            "agent.token",
            {"agent_id": "s1", "role": "specialist", "task_id": "research", "text": "internal research notes", "done": False},
        )
        await pilot.pause()
        assert "internal research notes" not in app.query_one(AssistantMessage).source
        assert "Working on it" in app.query_one(AssistantMessage).source

        fake_client.push(
            "agent.token",
            {"agent_id": "d1", "role": "director", "task_id": "synth", "text": "The synthesized answer", "done": False},
        )
        await pilot.pause()
        reply = app.query_one(AssistantMessage).source
        assert "The synthesized answer" in reply
        assert "internal research notes" not in reply


async def test_runner_reformatting_tokens_are_never_shown_as_the_draft(fake_client: FakeAgentPackClient):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"hi", "enter")
        await pilot.pause()

        fake_client.push("agent.token", {"agent_id": "s1", "role": "specialist", "task_id": "t1", "text": "Draft text", "done": False})
        await pilot.pause()
        fake_client.push("agent.token", {"agent_id": "s1", "role": "specialist", "task_id": "t1", "text": "", "done": True})
        await pilot.pause()

        # Runner's later reformatting pass over the same turn must not
        # appear in the bubble -- only the confirmed get_result() swap may.
        fake_client.push("agent.token", {"agent_id": "r1", "role": "runner", "task_id": "t1", "text": "reformatted nonsense", "done": False})
        await pilot.pause()

        reply = app.query_one(AssistantMessage).source
        assert "reformatted nonsense" not in reply
        assert "Reviewing before this is shown as final" in reply


async def test_draft_finishing_shows_reviewing_state_before_resolution(fake_client: FakeAgentPackClient):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"hi", "enter")
        await pilot.pause()

        fake_client.push("agent.token", {"agent_id": "s1", "role": "specialist", "task_id": "t1", "text": "Draft", "done": False})
        await pilot.pause()
        fake_client.push("agent.token", {"agent_id": "s1", "role": "specialist", "task_id": "t1", "text": "", "done": True})
        await pilot.pause()

        reply = app.query_one(AssistantMessage).source
        assert "Reviewing before this is shown as final" in reply
        assert "Draft" in reply
        assert "not yet compliance-reviewed" not in reply  # superseded by the reviewing marker


async def test_a_streamed_draft_is_fully_retracted_when_review_fails_it(fake_client: FakeAgentPackClient):
    """The 'retracting streamed text' case from the spec: a citation-
    required domain with no citation is rejected by the Director and
    escalated (docs/ARCHITECTURE.md §8) -- the streamed pending draft must
    not linger or be mistaken for a real (if unreviewed) answer once
    that happens."""
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"hi", "enter")
        await pilot.pause()

        fake_client.push("agent.token", {"agent_id": "s1", "role": "specialist", "task_id": "t1", "text": "An unreviewed claim", "done": False})
        await pilot.pause()
        fake_client.push("agent.token", {"agent_id": "s1", "role": "specialist", "task_id": "t1", "text": "", "done": True})
        await pilot.pause()
        assert "An unreviewed claim" in app.query_one(AssistantMessage).source

        fake_client.push("task.failed", {"task_id": "task_fake", "error": "No citation attached to a claim in a citation-required domain."})
        await pilot.pause()

        reply = app.query_one(AssistantMessage).source
        assert "An unreviewed claim" not in reply
        assert "not yet compliance-reviewed" not in reply
        assert "Reviewing before this is shown as final" not in reply


async def test_final_swap_replaces_the_draft_with_the_confirmed_reviewed_text(
    fake_client: FakeAgentPackClient,
):
    fake_client.results["res_1"] = "Confirmed, reviewed answer with a citation [doc#1]."
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"hi", "enter")
        await pilot.pause()

        fake_client.push("agent.token", {"agent_id": "s1", "role": "specialist", "task_id": "t1", "text": "rough draft wording", "done": False})
        await pilot.pause()
        fake_client.push("agent.token", {"agent_id": "s1", "role": "specialist", "task_id": "t1", "text": "", "done": True})
        await pilot.pause()

        fake_client.push("task.completed", {"task_id": "task_fake", "result_ref": "res_1", "citations": ["doc#1"]})
        await pilot.pause()

        reply = app.query_one(AssistantMessage).source
        assert "Confirmed, reviewed answer" in reply
        assert "rough draft wording" not in reply
        assert "not yet compliance-reviewed" not in reply
        assert "Reviewing before this is shown as final" not in reply


async def test_a_new_turn_after_streaming_starts_clean(fake_client: FakeAgentPackClient):
    """Streaming state from one turn must never leak into the next."""
    fake_client.results["res_1"] = "First answer."
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"first", "enter")
        await pilot.pause()
        fake_client.push("agent.token", {"agent_id": "s1", "role": "specialist", "task_id": "t1", "text": "draft one", "done": False})
        await pilot.pause()
        fake_client.push("task.completed", {"task_id": "task_fake", "result_ref": "res_1", "citations": []})
        await pilot.pause()
        assert app._draft_role is None
        assert app._plan_seen is False

        await pilot.press(*"second", "enter")
        await pilot.pause()
        assert "Working on it" in app.query(AssistantMessage)[-1].source
