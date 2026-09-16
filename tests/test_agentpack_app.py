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
from bedrock.ui.widgets import AssistantMessage, SystemNotice, UserMessage
from textual.widgets import Input

from tests.conftest import (
    FakeRunCatalog,
    make_confidence_report,
    make_services,
)


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
        self.folded: list[tuple[Path, str | None]] = []
        self.opened: list[Path | None] = []

    def status(self):
        return (
            ToolStatus("AlphaFold 2", True, ()),
            ToolStatus("open-source PyMOL", True, ()),
        )

    async def predict_and_render(
        self, fasta_path: Path, *, project_id: str | None = None
    ) -> ProteinWorkflowResult:
        self.folded.append((fasta_path, project_id))
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
    services=None,
) -> AgentPackApp:
    return AgentPackApp(
        settings=Settings(user_id="alice"),
        client=client,
        biotech_workflow=biotech_workflow,
        services=services or make_services(),
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

        assert workflow.folded == [(Path("/tmp/protein.fasta"), None)]
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


# ---- Folding with a project label, and what a finished fold says -----------


async def test_fold_accepts_a_quoted_path_and_a_project_label(
    fake_client: FakeAgentPackClient, tmp_path: Path
):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(fake_client, workflow)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = '/fold "/approved/my candidate.fasta" --project X'
        await pilot.press("enter")
        await pilot.pause()

        assert workflow.folded == [(Path("/approved/my candidate.fasta"), "X")]


async def test_fold_completion_summarises_confidence_and_points_at_the_full_page(
    fake_client: FakeAgentPackClient, tmp_path: Path
):
    workflow = FakeBiotechWorkflow(tmp_path)
    services = make_services(
        runs=FakeRunCatalog(
            reports={"run_test": make_confidence_report("run_test", model_id="model_3")}
        )
    )
    app = make_app(fake_client, workflow, services)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/fold /tmp/protein.fasta", "enter")
        await pilot.pause()

        text = "\n".join(notice.source for notice in app.query(SystemNotice))
        assert "pLDDT 90 or above" in text
        assert "model_3" in text
        assert "/confidence run_test" in text
        assert "ranked_0.pdb" in text  # the structure is still the headline


async def test_a_structure_stays_usable_when_its_confidence_cannot_be_read(
    fake_client: FakeAgentPackClient, tmp_path: Path
):
    """An expensive prediction is not invalidated by a missing JSON file."""
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(fake_client, workflow, make_services(runs=FakeRunCatalog(reports={})))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/fold /tmp/protein.fasta", "enter")
        await pilot.pause()

        text = "\n".join(notice.source for notice in app.query(SystemNotice))
        assert "Protein prediction completed" in text
        assert "ranked_0.pdb" in text
        assert "Confidence is unavailable" in text


async def test_a_fold_typo_never_starts_a_job(fake_client: FakeAgentPackClient, tmp_path: Path):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(fake_client, workflow)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "/fold candidate.fasta --project"
        await pilot.press("enter")
        await pilot.pause()

        assert workflow.folded == []
        assert app.is_running
        text = "\n".join(notice.source for notice in app.query(SystemNotice))
        assert "expected one argument" in text
