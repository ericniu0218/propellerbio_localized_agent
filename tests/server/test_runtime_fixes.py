"""Runtime defects fixed when the agent pack moved into bedrock-private-ai.

Each test names the defect it pins (docs/prompts/AGENT_PACK_INTEGRATION.md,
Part A.9): F1 a failing request left clients waiting for ever; F2 cancel did not
cancel; F6 the workspace resolved approvals from a worker thread; F8 Keycloak
sign-in never reached the workspace; F12 real deployments defaulted to demo
data; F18 an identity could pick its own folder.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from bedrock.inference import InferenceError
from bedrock.server.agents.base import BudgetExceededError
from bedrock.server.api import routes_auth, routes_session
from bedrock.server.auth.oidc import OIDCUser
from bedrock.server.compliance.approval_gates import ApprovalGate, get_approval_gate
from bedrock.server.events import get_event_bus
from bedrock.server.main import app
from bedrock.server.orchestrator.loop import handle_message
from bedrock.server.orchestrator.runs import CANCELLED, MODEL_DOWN, OVER_BUDGET, get_message_runs
from bedrock.server.paths import safe_component
from bedrock.server.schemas.enums import AgentRole, EventType
from bedrock.server.workspace.store import demo_mode
from tests.server.test_workspace_api import REGULATORY, _dev_login, _new_chat, _wait_for_run


def _failures(session_id: str) -> list[dict]:
    return [e.payload for e in get_event_bus().history(session_id) if e.type == EventType.TASK_FAILED]


# --- F1: a failure is reported, once, in plain words ------------------------------


async def test_a_request_that_raises_ends_in_exactly_one_plain_failure(session_id):
    async def over_budget():
        raise BudgetExceededError(AgentRole.RUNNER, "task_boom", 9_000, 8_000)

    await get_message_runs().start(session_id=session_id, root_task_id="task_boom", coro=over_budget())

    assert _failures(session_id) == [
        {"task_id": "task_boom", "title": "Your request", "error": OVER_BUDGET, "blocked_on_human": False}
    ]
    assert not get_message_runs().running("task_boom")


def test_a_message_whose_model_server_is_down_fails_visibly(monkeypatch):
    async def model_down(*_args, **_kwargs):
        raise InferenceError("connection refused", unavailable=True)

    monkeypatch.setattr(routes_session, "handle_message", model_down)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={"user_id": "alice"}).json()["session_id"]
        task_id = client.post("/messages", json={"session_id": session_id, "message": "hi"}).json()["task_id"]
        deadline = time.monotonic() + 5
        while not _failures(session_id) and time.monotonic() < deadline:
            time.sleep(0.02)
    assert [(f["task_id"], f["error"]) for f in _failures(session_id)] == [(task_id, MODEL_DOWN)]


# --- F2: cancel stops the work, even at the approval gate ----------------------------


async def test_cancel_stops_a_run_waiting_for_a_human(session_id):
    bus = get_event_bus()
    queue = bus.subscribe(session_id)
    runs = get_message_runs()
    runs.start(session_id=session_id, root_task_id="task_gate",
               coro=handle_message(session_id, "alice", REGULATORY, "task_gate"))
    try:
        while True:
            event = await asyncio.wait_for(queue.get(), 10)
            if event.type == EventType.APPROVAL_REQUIRED:
                gated = event.payload["task_id"]
                break
    finally:
        bus.unsubscribe(session_id, queue)

    assert await runs.cancel(gated)  # a plan step's id finds its run

    types = [e.type for e in bus.history(session_id)]
    assert EventType.TASK_COMPLETED not in types
    assert [f["error"] for f in _failures(session_id)] == [CANCELLED]
    assert not get_approval_gate().is_pending(gated)
    assert not runs.running("task_gate")


def test_cancelling_a_finished_task_does_not_contradict_it(monkeypatch):
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={"user_id": "alice"}).json()["session_id"]
        task_id = client.post("/messages", json={"session_id": session_id, "message": "What is 2+2?"}).json()["task_id"]
        deadline = time.monotonic() + 10
        while not any(e.type == EventType.TASK_COMPLETED for e in get_event_bus().history(session_id)):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert client.post("/tasks/cancel", json={"task_id": task_id}).json()["status"] == "completed"
    assert _failures(session_id) == []


# --- F6: workspace approvals are resolved on the event loop ---------------------------


def test_a_workspace_approval_is_resolved_on_the_event_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE_DATA_DIR", str(tmp_path / "workspace"))
    where: list[str] = []
    original = ApprovalGate.resolve

    def recording(self, task_id, decision):
        try:
            asyncio.get_running_loop()
            where.append("event loop")
        except RuntimeError:
            where.append("worker thread")
        return original(self, task_id, decision)

    monkeypatch.setattr(ApprovalGate, "resolve", recording)
    with TestClient(app) as client:
        _dev_login(client, monkeypatch)
        chat = _new_chat(client)
        run = client.post("/workspace/api/runs", json={"chatId": chat["id"], "prompt": REGULATORY, "mode": "ask"}).json()
        assert _wait_for_run(client, run["id"], {"waiting", "failed"})["status"] == "waiting"
        client.post(f"/workspace/api/runs/{run['id']}/decision", json={"decision": "approve"})
        assert _wait_for_run(client, run["id"], {"completed", "failed"})["status"] == "completed"
    assert where == ["event loop"]


# --- F8: Keycloak sign-in lands in the workspace ------------------------------------------


class _FakeKeycloak:
    async def authorize_url(self, state: str) -> str:
        return f"https://keycloak.example/auth?state={state}"

    async def exchange_code(self, code: str) -> dict:
        return {"id_token": "id", "access_token": "access"}

    async def verify_id_token(self, id_token: str, access_token: str) -> OIDCUser:
        return OIDCUser(sub="kc-1", email="alice@example.com", name="Alice Smith", preferred_username="alice")


def _sign_in(client, query: str = "") -> str:
    login = client.get(f"/auth/login{query}", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    callback = client.get(f"/auth/callback?code=abc&state={state}", follow_redirects=False)
    assert callback.status_code == 302
    return callback.headers["location"]


def test_signing_in_with_keycloak_opens_the_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE_DATA_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(routes_auth, "_require_oidc_client", lambda: _FakeKeycloak())
    with TestClient(app) as client:
        assert urlparse(_sign_in(client)).path == "/"
        state = client.get("/workspace/api/state")
        assert state.status_code == 200
        assert state.json()["user"] == {"id": "alice", "name": "Alice Smith"}


def test_the_legacy_page_still_gets_its_session_and_other_destinations_are_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE_DATA_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(routes_auth, "_require_oidc_client", lambda: _FakeKeycloak())
    with TestClient(app) as client:
        legacy = urlparse(_sign_in(client, "?next=/legacy"))
        assert legacy.path == "/legacy"
        assert parse_qs(legacy.query)["user_id"] == ["alice"]
        assert parse_qs(legacy.query)["session_id"][0].startswith("sess_")
        assert urlparse(_sign_in(client, "?next=https://elsewhere.example/")).path == "/"


def test_a_sign_in_state_works_once(monkeypatch):
    monkeypatch.setattr(routes_auth, "_require_oidc_client", lambda: _FakeKeycloak())
    with TestClient(app) as client:
        login = client.get("/auth/login", follow_redirects=False)
        state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
        assert client.get(f"/auth/callback?code=a&state={state}", follow_redirects=False).status_code == 302
        assert client.get(f"/auth/callback?code=a&state={state}", follow_redirects=False).status_code == 400


# --- F12: demo data follows the server's mode ------------------------------------------------


def test_demo_mode_follows_the_server_mode_unless_told(monkeypatch):
    import bedrock.server.config as config

    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(mode="real"))
    assert demo_mode() is False
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(mode="mock"))
    assert demo_mode() is True
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(mode="real"))
    assert demo_mode() is True


# --- F18: an identity never picks its own folder ---------------------------------------------


def test_identities_become_safe_folder_names(tmp_path):
    assert safe_component("alice") == "alice"
    assert safe_component("alice.smith@example.com") == "alice.smith@example.com"
    for hostile in ("..", ".", "../../etc", "a/b", "a\\b", "", "x" * 65, "name\0"):
        name = safe_component(hostile)
        assert name.startswith("u-") and "/" not in name and "\\" not in name
        assert (tmp_path / name).resolve().parent == tmp_path.resolve()
    assert safe_component("a/b") == safe_component("a/b")  # stable: the same person, the same folder
