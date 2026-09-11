"""Exercise AgentPackClient against a fake agent-pack server."""

import json

import httpx
import pytest

from bedrock.agentpack.client import AgentPackClient
from bedrock.inference import InferenceError


def make_client(handler) -> AgentPackClient:
    return AgentPackClient("http://fake", transport=httpx.MockTransport(handler))


async def test_create_session_and_send_message():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            assert json.loads(request.content) == {"user_id": "alice"}
            return httpx.Response(200, json={"session_id": "sess_1", "user_id": "alice", "created_at": "now"})
        if request.url.path == "/messages":
            assert json.loads(request.content) == {"session_id": "sess_1", "message": "hi"}
            return httpx.Response(200, json={"session_id": "sess_1", "task_id": "task_1", "accepted": True})
        raise AssertionError(f"unexpected path {request.url.path}")

    client = make_client(handler)
    session_id = await client.create_session("alice")
    assert session_id == "sess_1"
    task_id = await client.send_message(session_id, "hi")
    assert task_id == "task_1"


async def test_get_result():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/results/res_1"
        return httpx.Response(200, json={"result_ref": "res_1", "content": "the answer", "content_type": "text/plain"})

    client = make_client(handler)
    assert await client.get_result("res_1") == "the answer"


async def test_approve_posts_decision():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"task_id": "task_1", "decision": "approve"})

    client = make_client(handler)
    await client.approve("task_1", "approve", approved_by="alice")
    assert seen["body"] == {"task_id": "task_1", "decision": "approve", "approved_by": "alice", "comment": None}


async def test_connection_refused_is_reported_cleanly():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = make_client(handler)
    assert await client.check_connection() is False
    with pytest.raises(InferenceError, match="Is the AI system running"):
        await client.create_session("alice")


def sse_envelope(event_type: str, session_id: str, payload: dict) -> str:
    body = {"event_id": "e1", "type": event_type, "session_id": session_id, "timestamp": "now", "payload": payload}
    return f"event: {event_type}\ndata: {json.dumps(body)}\n\n"


async def test_events_parses_sse_stream():
    text = sse_envelope("agent.status", "sess_1", {"agent_id": "a1", "role": "runner", "status": "thinking"}) + sse_envelope(
        "task.completed", "sess_1", {"task_id": "task_1", "result_ref": "res_1"}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sessions/sess_1/events"
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})

    client = make_client(handler)
    events = [e async for e in client.events("sess_1")]
    assert [e.type for e in events] == ["agent.status", "task.completed"]
    assert events[0].payload == {"agent_id": "a1", "role": "runner", "status": "thinking"}
    assert events[1].payload["result_ref"] == "res_1"
    assert all(e.session_id == "sess_1" for e in events)
