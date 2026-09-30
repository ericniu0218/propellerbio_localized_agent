"""Every chat request ends in a state the person can read.

Reported: after a report was saved to a project, asking "were there any harmful
side effects?" showed three dots for ever - in project chat, from the research
page's "Ask about it in a chat", and in an ordinary chat. The recorded runs said
why: both safety questions were `waiting` for human approval (the agents hold
clinical and safety answers for review), and the chat drew the same dots for
waiting as for writing, with the approval buttons only in the side panel. A
failed run posted nothing at all, and a run cut off by a restart stayed
"running" for ever.

These tests drive the real workspace API with a scripted agent system in place
of `handle_message`: the events are the real event bus's, the run lifecycle
and the chat are the real ones. No model is called.
"""

from __future__ import annotations

import asyncio

import pytest

from bedrock.inference import InferenceError
from bedrock.server.events import get_event_bus
from bedrock.server.schemas.enums import AgentRole, EventType, TaskDomain
from bedrock.server.schemas.events import (
    AgentTokenPayload,
    ApprovalRequiredPayload,
    TaskCompletedPayload,
    TaskFailedPayload,
)
from bedrock.server.task_store.store import get_task_store
from bedrock.server.workspace import routes as workspace_routes
from bedrock.server.workspace.chat_lifecycle import explain_failure, report_context
from tests.server.test_workspace_api import _dev_login, _new_chat, _wait_for_run, client, workspace_dir  # noqa: F401

API = "/workspace/api"
SEEN: list[dict] = []


def script(*steps, hold: asyncio.Event | None = None):
    """A stand-in for the orchestrator's `handle_message` that publishes `steps`."""

    async def handle_message(session_id, user_id, message, root_task_id=None, force_tier=None, *, context=None,
                             history=None):
        SEEN.append({"prompt": message, "context": context or [], "history": history or []})
        bus = get_event_bus()
        for step in steps:
            if step == "hold":
                await (hold or asyncio.Event()).wait()
                continue
            if isinstance(step, Exception):
                raise step
            kind, payload = step
            if kind == "token":
                await bus.publish_typed(session_id, EventType.AGENT_TOKEN, AgentTokenPayload(
                    agent_id="a1", role=AgentRole.SPECIALIST, text=payload))
            elif kind == "done":
                ref = get_task_store().store_result(session_id, payload)
                await bus.publish_typed(session_id, EventType.TASK_COMPLETED, TaskCompletedPayload(
                    task_id="t1", title="answer", result_ref=ref))
            elif kind == "fail":
                await bus.publish_typed(session_id, EventType.TASK_FAILED, TaskFailedPayload(
                    task_id="t1", title="answer", error=payload))
            elif kind == "approval":
                ref = get_task_store().store_result(session_id, payload)
                await bus.publish_typed(session_id, EventType.APPROVAL_REQUIRED, ApprovalRequiredPayload(
                    task_id="t1", title="answer", reason="High-stakes output requires human approval before completion.",
                    domain=TaskDomain.CLINICAL, preview=payload[:20], result_ref=ref))
        return "t1"

    return handle_message


@pytest.fixture
def agents(monkeypatch):
    SEEN.clear()

    def use(*steps, hold=None):
        monkeypatch.setattr("bedrock.server.orchestrator.loop.handle_message", script(*steps, hold=hold))

    return use


def ask(client, chat, text="Were there any harmful side effects?"):
    r = client.post(f"{API}/runs", json={"chatId": chat["id"], "prompt": text, "mode": "ask"})
    assert r.status_code == 200, r.text
    return r.json()


def messages(client, chat):
    return next(c for c in client.get(f"{API}/state").json()["chats"] if c["id"] == chat["id"])["messages"]


# --- ordinary chat ------------------------------------------------------------------------------------


def test_an_answer_is_posted_and_the_run_completes(client, monkeypatch, agents):
    _dev_login(client, monkeypatch)
    agents(("token", "Nausea was "), ("done", "Nausea was the most common adverse event [1]."))
    chat = _new_chat(client)
    run = _wait_for_run(client, ask(client, chat)["id"], {"completed", "failed"})
    assert run["status"] == "completed"
    assert [m["role"] for m in messages(client, chat)] == ["user", "assistant"]


@pytest.mark.parametrize("steps, kind, words", [
    ([InferenceError("Request failed: connection refused. Is the server running?", unavailable=True)],
     "unavailable", "model server is unavailable"),
    ([("fail", "404: model 'qwen3:30b' not found, try pulling it first")], "missing_model", "not available on the model server"),
    ([("fail", "This model's maximum context length is 12288 tokens")], "context", "context window"),
    ([("done", "")], "empty", "empty answer"),
    ([("fail", "Rejected by human approver.")], "rejected", "sent this answer back"),
])
def test_a_failure_ends_in_a_reason_the_person_can_act_on(client, monkeypatch, agents, steps, kind, words):
    _dev_login(client, monkeypatch)
    agents(*steps)
    chat = _new_chat(client)
    run = _wait_for_run(client, ask(client, chat)["id"], {"completed", "failed"})
    assert run["status"] == "failed" and run["errorKind"] == kind and words in run["error"]
    # The question is kept, and nothing is posted as if it were an answer.
    assert [m["role"] for m in messages(client, chat)] == ["user"]


def test_a_stream_that_breaks_keeps_its_partial_answer_marked_incomplete(client, monkeypatch, agents):
    _dev_login(client, monkeypatch)
    agents(("token", "Gastrointestinal events were "), ("token", "the most common"),
           InferenceError("Request failed: connection reset", unavailable=True))
    chat = _new_chat(client)
    run = _wait_for_run(client, ask(client, chat)["id"], {"completed", "failed"})
    assert run["errorKind"] == "unavailable"
    answer = messages(client, chat)[-1]
    assert answer["incomplete"] is True and answer["text"] == "Gastrointestinal events were the most common"


def test_a_request_that_hears_nothing_is_stopped_as_stalled(client, monkeypatch, agents):
    _dev_login(client, monkeypatch)
    monkeypatch.setenv("BEDROCK_CHAT_STALL_SECONDS", "0.5")
    agents("hold")
    chat = _new_chat(client)
    run = _wait_for_run(client, ask(client, chat)["id"], {"completed", "failed"})
    assert run["errorKind"] == "stalled" and "Retry" in run["error"]


def test_waiting_for_a_person_is_not_a_stall_and_approval_posts_the_answer(client, monkeypatch, agents):
    """The reported case: a safety question held for review."""
    from bedrock.server.compliance import approvals

    _dev_login(client, monkeypatch)
    monkeypatch.setenv("BEDROCK_CHAT_STALL_SECONDS", "0.3")
    decided = asyncio.Event()

    async def gate_wait(*_):
        await decided.wait()

    draft = "Gastrointestinal adverse events were most frequent [8]. No study read reported hepatotoxicity rates."
    agents(("approval", draft), "hold", ("done", draft), hold=decided)
    monkeypatch.setattr(approvals, "resolve_approval", lambda *a, **k: _set(decided))
    chat = _new_chat(client)
    run = _wait_for_run(client, ask(client, chat)["id"], {"waiting"})
    asyncio.run(asyncio.sleep(0.8))  # longer than the stall limit: a person may take their time
    run = _wait_for_run(client, run["id"], {"waiting"})
    assert run["draft"] == draft and run["currentStep"] == "Waiting for your review"
    assert client.post(f"{API}/runs/{run['id']}/decision", json={"decision": "approve"}).status_code == 200
    end = _wait_for_run(client, run["id"], {"completed", "failed"})
    assert end["status"] == "completed", (end.get("errorKind"), end.get("error"))
    assert messages(client, chat)[-1]["text"] == draft


async def _set(event):
    event.set()


# --- cancel, retry, restart -----------------------------------------------------------------------------------


def test_stop_then_retry_asks_again_without_repeating_the_question(client, monkeypatch, agents):
    _dev_login(client, monkeypatch)
    agents("hold")
    chat = _new_chat(client)
    first = ask(client, chat, "Is it safe in pregnancy?")
    _wait_for_run(client, first["id"], {"queued", "running"})
    assert client.post(f"{API}/runs/{first['id']}/cancel").status_code == 200
    stopped = _wait_for_run(client, first["id"], {"cancelled"})
    assert "Retry" in stopped["error"]

    agents(("done", "No study read enrolled pregnant participants [2]."))
    again = client.post(f"{API}/runs/{first['id']}/retry").json()
    assert _wait_for_run(client, again["id"], {"completed", "failed"})["status"] == "completed"
    assert [m["text"] for m in messages(client, chat) if m["role"] == "user"] == ["Is it safe in pregnancy?"]
    assert SEEN[-1]["prompt"] == "Is it safe in pregnancy?"
    assert all(t.content != "Is it safe in pregnancy?" for t in SEEN[-1]["history"])  # not twice
    # A second click does not start a second run.
    assert client.post(f"{API}/runs/{first['id']}/retry").status_code == 409


def test_a_run_cut_off_by_a_restart_says_so_instead_of_running_for_ever(client, monkeypatch):
    _dev_login(client, monkeypatch)
    chat = _new_chat(client)
    s = workspace_routes.store()
    owner = client.get(f"{API}/state").json()["user"]["id"]
    orphan = s.put(owner, "runs", {"chatId": chat["id"], "prompt": "q", "mode": "ask", "status": "running",
                                   "createdAt": "2026-09-29T00:00:00+00:00"})
    run = next(r for r in client.get(f"{API}/state").json()["runs"] if r["id"] == orphan["id"])
    assert run["status"] == "stopped" and run["errorKind"] == "restarted"


def test_a_brand_new_run_is_never_mistaken_for_an_orphan(client, monkeypatch):
    _dev_login(client, monkeypatch)
    chat = _new_chat(client)
    s = workspace_routes.store()
    owner = client.get(f"{API}/state").json()["user"]["id"]
    fresh = s.put(owner, "runs", {"chatId": chat["id"], "prompt": "q", "mode": "ask", "status": "queued"})
    run = next(r for r in client.get(f"{API}/state").json()["runs"] if r["id"] == fresh["id"])
    assert run["status"] == "queued"


# --- a saved report, from all three places ----------------------------------------------------------------------

REPORT = """# Does tirzepatide reduce liver fat in adults with obesity?

> Based on 16 papers read in detail.

## Answer

Tirzepatide reduced liver fat content by -8.09% vs -3.38% with insulin degludec at week 52 [10].

## Evidence at a glance

| Ref | What the study was |
|---|---|
| [10] | substudy of a randomised controlled trial |

## Clinical efficacy

""" + ("Long clinical detail. " * 400) + """

## Safety

Gastrointestinal adverse events were the most frequently reported events across treatment groups [8].

## Limitations, gaps and open questions

- 9 of 16 papers read were available as abstract only.

## Abbreviations

- **ALT** — alanine aminotransferase

## References

1. Example reference.

---

# Appendix

## Evidence quotes

- **E1** long appendix text
"""


def test_a_safety_question_gets_the_safety_section_and_how_to_read_absence():
    text, shown, left_out = report_context(REPORT, "Were there any harmful side effects?", budget_chars=4000)
    assert "safety" in shown and "answer" in shown and "limitations, gaps and open questions" in shown
    assert "Gastrointestinal adverse events" in text
    assert "not evidence of safety" in text
    assert "clinical efficacy" in left_out and "appendix" in left_out  # too long for the budget: named, not cut
    assert "Long clinical detail" not in text


def _save_report(client, **where):
    """A report saved to the workspace as the research page does it."""
    owner = client.get(f"{API}/state").json()["user"]["id"]
    s = workspace_routes.store()
    return s.put(owner, "artifacts", {"name": "Literature review - tirzepatide.md", "text": REPORT, "type": ".md",
                                      "scanStatus": "no obvious matches", "researchId": "r1", **where})


@pytest.mark.parametrize("where", ["project", "chat"])
def test_project_chat_and_ask_in_chat_both_read_the_right_sections(client, monkeypatch, agents, where):
    _dev_login(client, monkeypatch)
    agents(("done", "Gastrointestinal events were most frequent [8]."))
    if where == "project":
        project = client.post(f"{API}/records/projects", json={"data": {"name": "Obesity"}}).json()
        chat = _new_chat(client, projectId=project["id"])
        _save_report(client, projectId=project["id"])
    else:  # the research page's "Ask about it in a chat" attaches the report to one chat
        chat = _new_chat(client)
        artifact = _save_report(client, chatId=chat["id"])
        chat_record = next(c for c in client.get(f"{API}/state").json()["chats"] if c["id"] == chat["id"])
        chat_record["artifactIds"] = [artifact["id"]]
        workspace_routes.store().put(chat_record["ownerId"], "chats", chat_record)
    run = _wait_for_run(client, ask(client, chat)["id"], {"completed", "failed"})
    assert run["status"] == "completed"
    block = next(b for b in SEEN[-1]["context"] if "Literature review" in b.title)
    assert "safety" in block.title and "Gastrointestinal adverse events" in block.text


def test_a_deleted_attachment_is_said_not_a_404(client, monkeypatch, agents):
    _dev_login(client, monkeypatch)
    agents(("done", "An answer [1]."))
    chat = _new_chat(client, artifactIds=["gone-artifact-id"])
    run = ask(client, chat)
    assert "no longer in the workspace" in run["contextNote"]
    assert _wait_for_run(client, run["id"], {"completed", "failed"})["status"] == "completed"


def test_explanations_never_invent_an_answer():
    for error in ("", "weird", "Request failed: ConnectError"):
        kind, message = explain_failure(error)
        assert "Retry" in message and kind in ("failed", "unavailable")
