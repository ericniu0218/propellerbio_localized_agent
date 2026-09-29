"""Compliance defects fixed when the agent pack moved into bedrock-private-ai.

(docs/prompts/AGENT_PACK_INTEGRATION.md, Part A.9) F4: one audit row per token;
F5: PHI written to the audit log unredacted; F6: web approvals left no signature
record; F7: a model rewrote answers after a human approved them; F11: the
normal tier skipped the citation guard.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from fastapi.testclient import TestClient

from bedrock.server.compliance.approval_gates import get_approval_gate
from bedrock.server.compliance.audit import AGENT_OUTPUT, audit_event_sink
from bedrock.server.db import get_database
from bedrock.server.events import get_event_bus
from bedrock.server.main import app
from bedrock.server.orchestrator.loop import handle_message
from bedrock.server.schemas.enums import ApprovalDecision, EventType
from bedrock.server.task_store.store import get_task_store
from bedrock.server.tools.base import ToolResult
from bedrock.server.tools.literature_search import LiteratureSearchTool
from tests.server.test_workspace_api import REGULATORY, _dev_login, _new_chat, _wait_for_run


async def _approve_when_asked(session_id: str, seen: dict) -> None:
    queue = get_event_bus().subscribe(session_id)
    try:
        while True:
            event = await asyncio.wait_for(queue.get(), 10)
            if event.type == EventType.APPROVAL_REQUIRED:
                seen["required"] = event.payload
                get_approval_gate().resolve(event.payload["task_id"], ApprovalDecision.APPROVE)
                return
    finally:
        get_event_bus().unsubscribe(session_id, queue)


# --- F4: one row per model reply ---------------------------------------------------------


async def test_a_streamed_reply_is_one_audit_row_with_its_hash(session_id):
    get_event_bus().register_audit_sink(audit_event_sink)
    db = get_database()
    await handle_message(session_id, db.get_session_user(session_id), "What is the boiling point of water?")

    entries = db.get_audit_log(session_id)
    assert EventType.AGENT_TOKEN.value not in {e.action for e in entries}
    outputs = [e for e in entries if e.action == AGENT_OUTPUT]
    # simple tier: the classification and the answer - and no formatting call any more
    assert len(outputs) == 2
    for row in outputs:
        text = row.detail["text"]
        assert row.detail["sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert row.detail["chars"] == len(text) and "incomplete" not in row.detail
    assert all("event_seq" in e.detail for e in entries if e.action != AGENT_OUTPUT)


# --- F5 and F7: redacted audit, and the approved text is the delivered text -------------------


async def test_phi_is_redacted_in_the_audit_log_and_the_approved_answer_is_delivered_unchanged(session_id):
    get_event_bus().register_audit_sink(audit_event_sink)
    db = get_database()
    message = REGULATORY + " Subject SSN 123-45-6789 had an adverse event."
    seen: dict = {}
    approver = asyncio.create_task(_approve_when_asked(session_id, seen))
    task_id = await handle_message(session_id, db.get_session_user(session_id), message)
    await approver

    audit_text = json.dumps([e.detail for e in db.get_audit_log(session_id)])
    assert "123-45-6789" not in audit_text
    assert "[REDACTED:SSN]" in audit_text
    assert any("SSN" in e.detail.get("redacted", []) for e in db.get_audit_log(session_id))
    assert any("123-45-6789" in turn.content for turn in db.get_history(session_id))  # history keeps it

    completed = next(e for e in get_event_bus().history(session_id) if e.type == EventType.TASK_COMPLETED)
    assert completed.payload["task_id"] == task_id
    assert completed.payload["result_ref"] == seen["required"]["result_ref"]
    delivered = get_task_store().get_result(completed.payload["result_ref"])
    assert seen["required"]["preview"] == delivered[:500]


# --- F6: a workspace approval leaves the signature record ------------------------------------


def test_a_workspace_approval_writes_the_signature_record(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE_DATA_DIR", str(tmp_path / "workspace"))
    with TestClient(app) as client:
        _dev_login(client, monkeypatch, "Alice")
        chat = _new_chat(client)
        run = client.post("/workspace/api/runs", json={"chatId": chat["id"], "prompt": REGULATORY, "mode": "ask"}).json()
        waiting = _wait_for_run(client, run["id"], {"waiting", "failed"})
        client.post(f"/workspace/api/runs/{run['id']}/decision", json={"decision": "approve"})
        _wait_for_run(client, run["id"], {"completed", "failed"})

    records = [e for e in get_database().get_audit_log(waiting["sessionId"]) if e.action == EventType.APPROVAL_RESOLVED.value]
    assert len(records) == 1
    assert records[0].detail["approved_by"] == "dev-alice"
    assert records[0].detail["decision"] == "approve"
    assert records[0].detail["signer_verified"] is False  # a development sign-in, not Keycloak


# --- F11: the normal tier checks citations too ---------------------------------------------------


async def test_a_normal_tier_literature_answer_without_citations_is_blocked(session_id, monkeypatch):
    async def nothing_found(self, args):
        return ToolResult(summary="No matching entries.", content="", citations=[])

    monkeypatch.setattr(LiteratureSearchTool, "run", nothing_found)
    await handle_message(session_id, "alice", "What does the published literature say about zqx-17?")

    history = get_event_bus().history(session_id)
    assert [e.type for e in history].count(EventType.REVIEW_REJECTED) == 3  # first try and two retries
    failed = [e.payload for e in history if e.type == EventType.TASK_FAILED]
    assert len(failed) == 1 and failed[0]["blocked_on_human"] is True
    assert EventType.TASK_COMPLETED not in [e.type for e in history]


async def test_a_normal_tier_literature_answer_with_citations_completes_reviewed(session_id):
    await handle_message(session_id, "alice", "What does the published literature say about cell culture?")
    history = get_event_bus().history(session_id)
    assert EventType.REVIEW_APPROVED in [e.type for e in history]
    completed = next(e.payload for e in history if e.type == EventType.TASK_COMPLETED)
    assert completed["citations"]
    assert "**Sources**" in get_task_store().get_result(completed["result_ref"])
