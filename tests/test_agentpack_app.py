"""Drive the real Textual agent-pack app headlessly with a fake client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest

from bedrock.agentpack.models import AgentEvent
from bedrock.agentpack_app import AgentPackApp
from bedrock.biotech import ProteinWorkflowResult
from bedrock.config import Settings
from bedrock.inference import InferenceError
from bedrock.ui.agent_panel import AgentPanel
from bedrock.ui.approval import ApprovalPrompt
from bedrock.ui.widgets import AssistantMessage, StatusBar, SystemNotice, UserMessage
from textual.widgets import Input

from tests.conftest import (
    FakeBiotechWorkflow,
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

    def push(self, type: str, payload: dict, event_id: str = "") -> None:
        """``event_id`` imitates the id the real stream puts on every event.

        Pass the same id twice to imitate a reconnecting stream redelivering
        an event the app has already acted on.
        """
        self._queue.put_nowait(AgentEvent(type=type, session_id="sess_fake", payload=payload, event_id=event_id))

    async def close(self) -> None:
        self.closed = True


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


async def test_relevant_stored_context_is_added_to_the_outgoing_message_not_the_bubble(
    fake_client: FakeAgentPackClient, tmp_path
):
    """Closes the gap /digest had: compressing a file used to do nothing for
    the next question you asked unless you remembered to search it yourself."""
    app = AgentPackApp(settings=Settings(user_id="alice", context_dir=str(tmp_path / "ctx")), client=fake_client)

    assay_file = tmp_path / "assay.csv"
    assay_file.write_text("\n".join(["sample,ph,passed"] + [f"s{i},{6.0 + i * 0.1:.1f},true" for i in range(10)]))
    from bedrock.context import ingest

    ingest(assay_file, store=app.context_store)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"what was the ph in the assay results?", "enter")
        await pilot.pause()

        assert len(fake_client.sent_messages) == 1
        _, outgoing = fake_client.sent_messages[0]
        assert "Relevant stored context" in outgoing
        assert "assay.csv" in outgoing
        assert outgoing.endswith("what was the ph in the assay results?")

        # The user's own bubble stays exactly what they typed - the
        # injected context is only in what got sent, never shown as if
        # the user had typed it.
        assert str(app.query_one(UserMessage).content) == "what was the ph in the assay results?"
        notices = [n.source for n in app.query(SystemNotice)]
        assert any("Added stored context from" in n and "assay.csv" in n for n in notices)


async def test_unrelated_message_is_sent_unchanged_when_nothing_matches(
    fake_client: FakeAgentPackClient, tmp_path
):
    app = AgentPackApp(settings=Settings(user_id="alice", context_dir=str(tmp_path / "ctx")), client=fake_client)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"good morning", "enter")
        await pilot.pause()

        assert fake_client.sent_messages == [("sess_fake", "good morning")]


# ---- Switching automatic context off -----------------------------------------
#
# Automatic context matches by keyword, so it can attach a passage that only
# shares a word with the question. These tests pin down the way to stop that:
# a setting for every window, and `/digest auto` for the one in front of you.


ASSAY_QUESTION = "what was the ph in the assay results?"


def make_app_with_assay(client: FakeAgentPackClient, tmp_path, **settings) -> AgentPackApp:
    """An app whose context store holds a CSV that matches ASSAY_QUESTION."""
    app = AgentPackApp(
        settings=Settings(user_id="alice", context_dir=str(tmp_path / "ctx"), **settings),
        client=client,
        services=make_services(),
    )
    assay_file = tmp_path / "assay.csv"
    assay_file.write_text("\n".join(["sample,ph,passed"] + [f"s{i},{6.0 + i * 0.1:.1f},true" for i in range(10)]))
    from bedrock.context import ingest

    ingest(assay_file, store=app.context_store)
    return app


def notice_texts(app: AgentPackApp) -> list[str]:
    return [str(n.source) for n in app.query(SystemNotice)]


async def type_line(pilot, text: str) -> None:
    await pilot.press(*text, "enter")
    await pilot.pause()


async def test_injection_notice_says_how_to_stop_it(fake_client: FakeAgentPackClient, tmp_path):
    """The way out is on screen at the moment a bad match could appear."""
    app = make_app_with_assay(fake_client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_line(pilot, ASSAY_QUESTION)

        added = [n for n in notice_texts(app) if "Added stored context from" in n]
        assert len(added) == 1
        assert "`/digest auto off`" in added[0]


async def test_auto_context_setting_off_sends_messages_exactly_as_typed(
    fake_client: FakeAgentPackClient, tmp_path, monkeypatch
):
    app = make_app_with_assay(fake_client, tmp_path, auto_context=False)
    searched = []
    monkeypatch.setattr(app.context_store, "search", lambda *a, **k: searched.append(a) or [])
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_line(pilot, ASSAY_QUESTION)

        assert fake_client.sent_messages == [("sess_fake", ASSAY_QUESTION)]
        assert not any("Added stored context" in n for n in notice_texts(app))
        # Off means the store is not consulted at all, not searched and ignored.
        assert searched == []


async def test_digest_auto_off_then_on_changes_what_this_window_sends(
    fake_client: FakeAgentPackClient, tmp_path
):
    app = make_app_with_assay(fake_client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_line(pilot, "/digest auto off")
        assert app.auto_context.enabled is False
        assert "now **off**" in notice_texts(app)[-1]
        # The notice says where the lasting setting lives, since this one is not saved.
        assert "BEDROCK_AUTO_CONTEXT=0" in notice_texts(app)[-1]

        await type_line(pilot, ASSAY_QUESTION)
        assert fake_client.sent_messages == [("sess_fake", ASSAY_QUESTION)]
        assert not any("Added stored context" in n for n in notice_texts(app))

        fake_client.push("task.completed", {"task_id": "task_fake", "result_ref": "res_1", "citations": []})
        await pilot.pause()
        await pilot.pause()

        await type_line(pilot, "/digest auto on")
        assert app.auto_context.enabled is True
        assert "now **on**" in notice_texts(app)[-1]

        await type_line(pilot, ASSAY_QUESTION)
        assert len(fake_client.sent_messages) == 2
        _, outgoing = fake_client.sent_messages[1]
        assert "Relevant stored context" in outgoing
        assert outgoing.endswith(ASSAY_QUESTION)


async def test_digest_auto_off_typed_while_a_prompt_waits_applies_to_that_prompt(
    fake_client: FakeAgentPackClient, tmp_path
):
    """The switch is read when a message is sent, not when it was typed."""
    app = make_app_with_assay(fake_client, tmp_path)
    second = "and what was the ph of sample s3?"
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_line(pilot, ASSAY_QUESTION)
        await type_line(pilot, second)
        # The first went out with context; the second is waiting its turn.
        assert len(fake_client.sent_messages) == 1
        assert "Relevant stored context" in fake_client.sent_messages[0][1]
        assert [item.text for item in app.queue_snapshot().pending] == [second]

        # Commands run straight away, even while prompts wait.
        await type_line(pilot, "/digest auto off")
        assert app.auto_context.enabled is False

        fake_client.push("task.completed", {"task_id": "task_fake", "result_ref": "res_1", "citations": []})
        await pilot.pause()
        await pilot.pause()

        assert fake_client.sent_messages[-1] == ("sess_fake", second)


async def test_digest_auto_reports_the_setting_and_refuses_other_words(
    fake_client: FakeAgentPackClient, tmp_path
):
    app = make_app_with_assay(fake_client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_line(pilot, "/digest auto")
        assert "is **on**" in notice_texts(app)[-1]

        await type_line(pilot, "/digest auto off")
        await type_line(pilot, "/digest auto off")
        assert "already **off**" in notice_texts(app)[-1]

        await type_line(pilot, "/digest auto")
        assert "is **off**" in notice_texts(app)[-1]

        await type_line(pilot, "/digest auto maybe")
        refusal = app.query(SystemNotice)[-1]
        assert refusal.has_class("error")
        assert "takes `on` or `off`" in str(refusal.source)
        # A word it does not understand changes nothing.
        assert app.auto_context.enabled is False
        # And nothing was sent to the AI system by any of this.
        assert fake_client.sent_messages == []


async def test_help_texts_offer_the_switch_in_agent_pack_mode(fake_client: FakeAgentPackClient, tmp_path):
    app = make_app_with_assay(fake_client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_line(pilot, "/help")
        assert "/digest auto off" in notice_texts(app)[-1]

        # `/digest help` runs in a worker, so wait for its notice.
        await type_line(pilot, "/digest help")
        joined = await wait_for_notice(app, pilot, "compress data so a model can read it cheaply")
        assert "`/digest auto off` stops it for this window" in joined


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
        await pilot.pause()

        assert fake_client.approvals == [("task_fake", "approve", "alice")]
        assert not app.query(ApprovalPrompt)
        assert any("Approved" in n.source for n in app.query(SystemNotice))


async def test_one_task_at_a_time_and_the_next_one_waits(fake_client: FakeAgentPackClient):
    """A second question queues instead of being refused, and runs after the first."""
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"first question", "enter")
        await pilot.pause()
        await pilot.press(*"second question", "enter")
        await pilot.pause()

        assert fake_client.sent_messages == [("sess_fake", "first question")]
        snapshot = app.queue_snapshot()
        assert snapshot.running.text == "first question"
        assert [item.text for item in snapshot.pending] == ["second question"]

        fake_client.push("task.completed", {"task_id": "task_fake", "result_ref": "res_1", "citations": []})
        await pilot.pause()
        await pilot.pause()

        assert fake_client.sent_messages == [("sess_fake", "first question"), ("sess_fake", "second question")]
        assert app.queue_snapshot().pending == ()


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


# ---- Folding from a company source (data-source connectors) ----------------


def source_app(fake_client, tmp_path: Path, workflow: FakeBiotechWorkflow) -> AgentPackApp:
    exports = tmp_path / "registry-exports"
    exports.mkdir()
    (exports / "SEQ-00123.fasta").write_text(">SEQ-00123 kinase\nACDEFGHIK\n")
    return AgentPackApp(
        settings=Settings(
            user_id="alice",
            biotech_work_dir=tmp_path / "work",
            connectors={"registry": {"type": "share", "root": str(exports)}},
        ),
        client=fake_client,
        biotech_workflow=workflow,
    )


async def wait_for_notice(app: AgentPackApp, pilot, text: str) -> str:
    """Poll for a notice. This app's event listener never finishes, so
    ``workers.wait_for_complete()`` is not an option here."""
    for _ in range(200):
        await pilot.pause()
        joined = "\n".join(str(widget.source) for widget in app.query(SystemNotice))
        if text in joined:
            return joined
    raise AssertionError(f"no notice containing {text!r}")


async def test_fold_pulls_the_sequence_from_a_company_source_with_provenance(fake_client, tmp_path):
    from textual.widgets import Input

    workflow = FakeBiotechWorkflow(tmp_path)
    app = source_app(fake_client, tmp_path, workflow)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "Run AlphaFold 2 on registry://SEQ-00123.fasta"
        await pilot.press("enter")
        notices = await wait_for_notice(app, pilot, "Protein prediction completed")

        staged, project = workflow.folded[0]
        assert project is None  # no --project was given
        assert staged.parent == tmp_path / "work" / "sources"
        assert staged.read_text().startswith(">SEQ-00123 kinase")
        provenance = workflow.provenance[0]
        assert provenance["ref"] == "registry://SEQ-00123.fasta"
        assert provenance["fetched_by"] == "alice"
        assert "Source: `registry://SEQ-00123.fasta`" in notices


async def test_pymol_opens_a_structure_pulled_from_a_company_source(fake_client, tmp_path):
    from textual.widgets import Input

    workflow = FakeBiotechWorkflow(tmp_path)
    app = source_app(fake_client, tmp_path, workflow)
    (tmp_path / "registry-exports" / "STRUCT-9.pdb").write_text("ATOM      1  N   MET A   1\n")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "/pymol registry://STRUCT-9.pdb"
        await pilot.press("enter")
        await wait_for_notice(app, pilot, "Opened `registry://STRUCT-9.pdb` in PyMOL")

        (opened,) = workflow.opened
        # Named explicitly, so it is styled neutrally rather than as a prediction.
        assert opened is not None and opened.parent == tmp_path / "work" / "sources"
        assert (opened.parent / (opened.name + ".provenance.json")).is_file()


async def test_quitting_while_a_fold_runs_closes_cleanly(fake_client, tmp_path):
    """The screen is gone by the time the "cancelled" notice would be shown;
    that used to fail the worker and end the app with a traceback."""
    started = asyncio.Event()

    class BlockingWorkflow(FakeBiotechWorkflow):
        async def predict_and_render(self, fasta_path, **kwargs):
            started.set()
            await asyncio.Event().wait()

    app = make_app(fake_client, BlockingWorkflow(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/fold /tmp/protein.fasta", "enter")
        await started.wait()
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert app.return_code == 0
    assert app._exception is None


async def test_a_refused_source_fetch_reports_and_frees_the_job_slot(fake_client, tmp_path):
    from textual.widgets import Input

    workflow = FakeBiotechWorkflow(tmp_path)
    app = source_app(fake_client, tmp_path, workflow)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "/fold registry://../outside.fasta"
        await pilot.press("enter")
        await wait_for_notice(app, pilot, "points outside the `registry` share")

        assert workflow.folded == []
        assert app._scientific_job_running is False


async def test_a_company_source_can_also_carry_a_project_label(fake_client, tmp_path):
    """The two features are orthogonal: where a sequence came from and which
    project it belongs to are different questions, and a run answers both."""
    from textual.widgets import Input

    workflow = FakeBiotechWorkflow(tmp_path)
    app = source_app(fake_client, tmp_path, workflow)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "/fold registry://SEQ-00123.fasta --project X"
        await pilot.press("enter")
        await wait_for_notice(app, pilot, "Protein prediction completed")

        staged, project = workflow.folded[0]
        assert project == "X"
        assert staged.read_text().startswith(">SEQ-00123 kinase")
        assert workflow.provenance[0]["ref"] == "registry://SEQ-00123.fasta"
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

# ---- Prompt queue in agent-pack mode ----------------------------------------


async def test_waiting_for_approval_is_not_completion(fake_client: FakeAgentPackClient):
    """A human check pauses the work; the queue must not treat it as finished."""
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"first question", "enter")
        await pilot.pause()
        await pilot.press(*"second question", "enter")
        await pilot.pause()

        fake_client.push("approval.required", {"task_id": "task_fake", "reason": "check this", "preview": ""})
        await pilot.pause()
        assert app.query_one(ApprovalPrompt)
        assert fake_client.sent_messages == [("sess_fake", "first question")]
        assert [item.text for item in app.queue_snapshot().pending] == ["second question"]

        await pilot.click("#approve")
        await pilot.pause()
        # Approving is still not completion: only task.completed releases it.
        assert fake_client.sent_messages == [("sess_fake", "first question")]

        fake_client.push("task.completed", {"task_id": "task_fake", "result_ref": "res_1", "citations": []})
        await pilot.pause()
        await pilot.pause()
        assert fake_client.sent_messages[-1] == ("sess_fake", "second question")


async def test_a_completion_with_nothing_running_advances_nothing(fake_client: FakeAgentPackClient):
    """Duplicate, delayed and unrelated events are ignored once a run has ended."""
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"first question", "enter")
        await pilot.pause()

        fake_client.push("task.completed", {"task_id": "task_fake", "result_ref": "res_1", "citations": []})
        await pilot.pause()
        await pilot.pause()
        assert app.queue_snapshot().is_idle

        for payload in (
            {"task_id": "task_fake", "result_ref": "res_1", "citations": []},  # duplicate
            {"task_id": "task_late", "result_ref": "res_2", "citations": []},  # unrelated
        ):
            fake_client.push("task.completed", payload)
            await pilot.pause()

        await pilot.press(*"second question", "enter")
        await pilot.pause()
        assert fake_client.sent_messages == [("sess_fake", "first question"), ("sess_fake", "second question")]


async def test_a_duplicate_completion_cannot_finish_the_next_request(fake_client):
    """An old terminal event must not advance a new request."""
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        for text in ("first question", "second question", "third question"):
            await pilot.press(*text, "enter")
            await pilot.pause()

        for _ in range(2):  # the same completion event, delivered twice
            fake_client.push("task.completed", {"task_id": "task_fake", "result_ref": "res_1", "citations": []})
            await pilot.pause()
        await pilot.pause()

        sent = [text for _, text in fake_client.sent_messages]
        assert sent == ["first question", "second question"]
        assert [item.text for item in app.queue_snapshot().pending] == ["third question"]
        fake_client.push("task.completed", {"task_id": "next-final-step", "result_ref": "res_2"})
        await pilot.pause()
        await pilot.pause()
        assert [text for _, text in fake_client.sent_messages] == ["first question", "second question", "third question"]


async def test_a_redelivered_event_is_ignored(fake_client):
    """A reconnecting stream can send the same event again; the copy does nothing."""
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        for text in ("first question", "second question", "third question"):
            await pilot.press(*text, "enter")
            await pilot.pause()

        payload = {"task_id": "task_fake", "result_ref": "res_1", "citations": []}
        for _ in range(2):  # the very same event, delivered twice
            fake_client.push("task.completed", payload, event_id="evt_7")
            await pilot.pause()
            await pilot.pause()

        assert [text for _, text in fake_client.sent_messages] == ["first question", "second question"]
        assert app.queue_snapshot().running.text == "second question"
        assert [item.text for item in app.queue_snapshot().pending] == ["third question"]


async def test_two_requests_failing_identically_both_finish(fake_client):
    """Same words, different events: the second failure must still end its request.

    Completions are told apart by the stream's event id. Judging them by their
    contents alone would treat the second failure as a replay of the first and
    leave the queue busy for ever.
    """
    app = make_app(fake_client)
    same = {"task_id": "task_fake", "error": "BudgetExceededError: boom"}
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.queue_prompt("first")
        await pilot.pause()
        fake_client.push("task.failed", dict(same), event_id="evt_1")
        await pilot.pause()
        await pilot.pause()
        assert not app.prompt_queue.busy

        await app.queue_prompt("second")
        await pilot.pause()
        fake_client.push("task.failed", dict(same), event_id="evt_2")
        await pilot.pause()
        await pilot.pause()

        assert not app.prompt_queue.busy, "the queue is stuck on a request that already failed"
        await app.queue_prompt("third")
        await pilot.pause()
        assert [text for _, text in fake_client.sent_messages] == ["first", "second", "third"]


async def test_result_download_failure_pauses_pending_prompts(fake_client):
    async def fail_result(reference):
        raise InferenceError("Could not fetch the answer")

    fake_client.get_result = fail_result
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.queue_prompt("first")
        await app.queue_prompt("second")
        fake_client.push("task.completed", {"task_id": "final-step", "result_ref": "res_1"})
        await pilot.pause()
        assert app.queue_snapshot().paused
        assert app.queue_snapshot().running is None
        assert [item.text for item in app.queue_snapshot().pending] == ["second"]
        assert [text for _, text in fake_client.sent_messages] == ["first"]
        app.resume_queue()
        await pilot.pause()
        assert [text for _, text in fake_client.sent_messages] == ["first", "second"]


async def test_foreign_session_completion_does_not_advance_queue(fake_client):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.queue_prompt("first")
        await app.queue_prompt("second")
        await app._finish(AgentEvent("task.completed", "another-session", {"task_id": "foreign", "result_ref": "res_1"}), failed=False)
        assert app.queue_snapshot().running.text == "first"
        assert [text for _, text in fake_client.sent_messages] == ["first"]


async def test_clear_during_result_download_does_not_finish_new_prompt(fake_client):
    fetching, release = asyncio.Event(), asyncio.Event()

    async def slow_result(reference):
        fetching.set()
        await release.wait()
        return "Old answer"

    fake_client.get_result = slow_result
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.queue_prompt("first")
        fake_client.push("task.completed", {"task_id": "old-final", "result_ref": "res_1"})
        await asyncio.wait_for(fetching.wait(), 2)
        await app.action_clear_chat()
        await app.queue_prompt("second")
        await app.queue_prompt("third")
        release.set()
        await pilot.pause()
        assert app.queue_snapshot().running.text == "second"
        assert [item.text for item in app.queue_snapshot().pending] == ["third"]


async def test_a_failed_task_pauses_the_queue(fake_client: FakeAgentPackClient):
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"first question", "enter")
        await pilot.pause()
        await pilot.press(*"second question", "enter")
        await pilot.pause()

        fake_client.push("task.failed", {"task_id": "task_fake", "error": "BudgetExceededError: boom"})
        await pilot.pause()
        await pilot.pause()

        snapshot = app.queue_snapshot()
        assert snapshot.paused is True
        assert [item.text for item in snapshot.pending] == ["second question"]
        assert fake_client.sent_messages == [("sess_fake", "first question")]

        app.resume_queue()
        await pilot.pause()
        await pilot.pause()
        assert fake_client.sent_messages[-1] == ("sess_fake", "second question")
        fake_client.push("task.failed", {"task_id": "task_fake", "error": "BudgetExceededError: boom"})
        await pilot.pause()
        assert app.queue_snapshot().running.text == "second question"


async def test_a_scientific_job_holds_the_queue_until_it_finishes(fake_client, tmp_path):
    workflow = FakeBiotechWorkflow(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_fold(fasta_path: Path, *, project_id: str | None = None, source_provenance: dict | None = None):
        workflow.folded.append((fasta_path, project_id))
        started.set()
        await release.wait()
        return ProteinWorkflowResult(
            run_id="run_test",
            run_dir=tmp_path,
            structure_path=tmp_path / "ranked_0.pdb",
            preview_path=None,
            session_path=None,
            manifest_path=tmp_path / "manifest.json",
        )

    workflow.predict_and_render = slow_fold
    app = make_app(fake_client, workflow)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/fold /tmp/protein.fasta", "enter")
        await started.wait()
        await pilot.pause()
        await pilot.press(*"what does that structure show?", "enter")
        await pilot.pause()

        # The fold is the running request; the question waits for it.
        assert app.queue_snapshot().running.text == "/fold /tmp/protein.fasta"
        assert [item.text for item in app.queue_snapshot().pending] == ["what does that structure show?"]
        assert fake_client.sent_messages == []

        release.set()
        for _ in range(4):
            await pilot.pause()
        assert fake_client.sent_messages == [("sess_fake", "what does that structure show?")]


# ---------------------------------------------------------------------------
# /reviews in agent-pack mode
# ---------------------------------------------------------------------------


async def test_reviews_here_does_not_send_you_to_a_command_this_mode_refuses(
    fake_client: FakeAgentPackClient, tmp_path, monkeypatch
):
    """With no reviews yet, `/reviews` used to say "Start one with `/review`" -
    in the one mode that refuses `/review`. Every `/reviews` verb works here;
    only *starting* a review does not."""
    monkeypatch.setenv("BEDROCK_LITREVIEW_DIR", str(tmp_path / "litreview"))
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/reviews", "enter")
        for _ in range(6):
            await pilot.pause()

        notice = "\n".join(str(n.source) for n in app.query(SystemNotice))
        assert "No literature reviews on this machine yet" in notice, notice
        assert "Start one with `/review" not in notice
        assert "bedrock-litreview ask" in notice


async def test_the_retraction_settings_are_discoverable_here(fake_client: FakeAgentPackClient):
    """The reported gap: `/reviews settings` and `/reviews reinstate` work in
    agent-pack mode but were in none of its help tables."""
    app = make_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/help", "enter")
        await pilot.pause()

        helped = "\n".join(str(n.source) for n in app.query(SystemNotice))
        assert "/reviews settings" in helped, helped
        assert "/reviews reinstate" in helped
