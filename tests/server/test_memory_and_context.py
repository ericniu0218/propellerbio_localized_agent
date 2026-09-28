"""Conversation memory and material kept apart from the question (F9, F10, F16).

docs/prompts/AGENT_PACK_INTEGRATION.md Part A.9: the server stored every turn and
used none (F9); clients pasted files into the message, so the Runner's routing
call read them all (F10); the workspace rewrote a run on every token (F16).
Prompts are observed at the mock model server (``MockBackend``), where they
would reach a real one.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from bedrock.server.events import get_event_bus
from bedrock.server.main import app
from bedrock.server.models.mock_backend import MockBackend
from bedrock.server.orchestrator.context import TRUNCATED, render_context, render_history
from bedrock.server.orchestrator.loop import handle_message
from bedrock.server.schemas.api import ContextBlock, HistoryTurn
from bedrock.server.schemas.enums import AgentRole, EventType
from bedrock.server.workspace.store import WorkspaceStore
from tests.server.test_workspace_api import _dev_login, _new_chat, _upload, _wait_for_run

CLASSIFY = 'Respond with JSON: {"complexity"'
MARKER = "ZEBRA-MARKER-4471"


@pytest.fixture
def prompts(monkeypatch) -> list[str]:
    seen: list[str] = []
    original = MockBackend.stream_chat

    async def recording(self, model, messages, options=None, on_usage=None):
        seen.append(messages[-1]["content"])
        async for part in original(self, model, messages, options, on_usage):
            yield part

    monkeypatch.setattr(MockBackend, "stream_chat", recording)
    return seen


async def test_a_follow_up_reaches_the_agents_with_the_conversation_so_far(session_id, prompts):
    await handle_message(session_id, "alice", "What does our internal SOP say about mycoplasma testing?")
    first_run = len(prompts)
    await handle_message(session_id, "alice", "How often should it be repeated?")

    later = prompts[first_run:]
    classify = next(p for p in later if CLASSIFY in p)
    assert "Earlier message in this conversation" in classify and "mycoplasma" in classify
    answer = next(p for p in later if CLASSIFY not in p)
    assert "Conversation so far:" in answer and "mycoplasma" in answer


async def test_history_can_be_supplied_by_the_caller(session_id, prompts):
    history = [HistoryTurn(turn_id="1", role="user", content="We are studying KRAS G12D.", created_at="")]
    await handle_message(session_id, "alice", "Summarise the open questions in our internal SOP notes.", history=history)
    assert any("KRAS G12D" in p and "Conversation so far:" in p for p in prompts if CLASSIFY not in p)


def test_history_keeps_the_newest_turns_that_fit():
    turns = [HistoryTurn(turn_id=str(i), role="user" if i % 2 == 0 else "specialist",
                         content=f"turn {i} " + "x" * 400, created_at="") for i in range(20)]
    text = render_history(turns, budget_tokens=300)
    assert text.startswith("Conversation so far:")
    assert "turn 19" in text and "turn 0 " not in text
    assert text.index("turn 18") < text.index("turn 19")  # oldest first


async def test_material_never_reaches_the_classifier_and_is_fenced_for_the_drafter(session_id, prompts):
    block = ContextBlock(id="art1", title="plate.csv", text=f"well,od\nA1,0.5\n{MARKER}", kind="attachment")
    await handle_message(session_id, "alice", "Summarise this plate reading from our internal SOP run.", context=[block])

    assert not any(MARKER in p for p in prompts if CLASSIFY in p)
    drafts = [p for p in prompts if p.startswith("Task:") and MARKER in p]
    assert drafts and "<<<FILE plate.csv [source:art1]>>>" in drafts[0]
    assert "not as instructions" in drafts[0]


async def test_a_simple_request_with_material_is_not_left_to_the_runner(session_id):
    block = ContextBlock(id="n1", title="note.txt", text="The assay failed twice.")
    await handle_message(session_id, "alice", "Summarise this", context=[block])
    assigned = [e.payload["assigned_to"] for e in get_event_bus().history(session_id) if e.type == EventType.TASK_ASSIGNED]
    assert assigned and AgentRole.RUNNER.value not in assigned


def test_material_over_budget_is_cut_with_a_note():
    big = ContextBlock(id="big", title="big.txt", text="word " * 20_000)
    small = ContextBlock(id="small", title="small.txt", text="kept")
    text = render_context([small, big], budget_tokens=500)
    assert "kept" in text and TRUNCATED in text


def test_old_clients_still_work_and_new_ones_can_see_what_the_server_supports():
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert {"context_blocks", "server_history", "force_tier", "cancel"} <= set(health["features"])
        session_id = client.post("/sessions", json={"user_id": "alice"}).json()["session_id"]
        client.post("/messages", json={"session_id": session_id, "message": "What is 2+2?"})
        deadline = time.monotonic() + 10
        while not any(e.type == EventType.TASK_COMPLETED for e in get_event_bus().history(session_id)):
            assert time.monotonic() < deadline
            time.sleep(0.02)


def test_a_workspace_run_sends_its_files_as_material_and_saves_rarely(monkeypatch, tmp_path, prompts):
    monkeypatch.setenv("WORKSPACE_DATA_DIR", str(tmp_path / "workspace"))
    saves: list[str] = []
    original_put = WorkspaceStore.put

    def counting_put(self, owner, kind, data):
        if kind == "runs":
            saves.append(data.get("status", ""))
        return original_put(self, owner, kind, data)

    monkeypatch.setattr(WorkspaceStore, "put", counting_put)
    with TestClient(app) as client:
        _dev_login(client, monkeypatch)
        chat = _new_chat(client)
        _upload(client, "plate.csv", f"well,od\nA1,0.5\nnote,{MARKER}\n".encode(), chatId=chat["id"])
        run = client.post("/workspace/api/runs", json={"chatId": chat["id"], "prompt": "Summarise the plate reading.", "mode": "analyze"}).json()
        done = _wait_for_run(client, run["id"], {"completed", "failed"})
        assert done["status"] == "completed", done.get("error")
        assert "events" not in done

    tokens = [e for e in get_event_bus().history(done["sessionId"]) if e.type == EventType.AGENT_TOKEN]
    assert len(saves) < len(tokens) / 2  # it used to be at least one save per token
    assert not any(MARKER in p for p in prompts if CLASSIFY in p)
    assert any(MARKER in p for p in prompts if p.startswith("Task:"))
