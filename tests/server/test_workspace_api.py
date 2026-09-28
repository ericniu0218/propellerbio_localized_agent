"""The workspace API, as it behaves today.

These are characterisation tests: they were written against the unmodified
code, before the agent pack moved into bedrock-private-ai, so that the move and
every later change to `app/workspace/` has something to be held to. Nothing in
`app/workspace/` had a test before this file. Where a test pins behaviour a
later change is meant to alter (the demo default, the approval record), that
change updates the test in the same commit and says why.

Every test gets its own workspace database through `WORKSPACE_DATA_DIR`, which
the store reads on every request. `DEMO_MODE` is read per call too, so each test
sets the mode it needs.
"""

from __future__ import annotations

import asyncio
import base64
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

_REAL_SLEEP = asyncio.sleep
COOKIE = "bedrock_workspace"
REGULATORY = "This is a final report for regulatory submission on our safety findings."


@pytest.fixture(autouse=True)
def workspace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DATA_DIR", str(tmp_path / "workspace"))
    monkeypatch.delenv("DEMO_MODE", raising=False)
    return tmp_path / "workspace"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def instant(monkeypatch):
    """Demo providers pause between steps; tests do not need to watch them."""

    async def _instant(*_args, **_kwargs):
        await _REAL_SLEEP(0)

    monkeypatch.setattr(asyncio, "sleep", _instant)


def _dev_login(client, monkeypatch, name="Alice") -> str:
    monkeypatch.setenv("DEMO_MODE", "false")
    client.cookies.clear()
    assert client.post("/workspace/api/dev-login", json={"name": name}).status_code == 200
    return client.cookies.get(COOKIE)


def _as(client, token: str) -> None:
    client.cookies.clear()
    client.cookies.set(COOKIE, token)


def _new_chat(client, **fields) -> dict:
    response = client.post("/workspace/api/records/chats", json={"data": {"title": "New chat", "messages": [], **fields}})
    assert response.status_code == 200, response.text
    return response.json()


def _upload(client, name: str, raw: bytes, **fields):
    return client.post("/workspace/api/uploads", json={"name": name, "base64": base64.b64encode(raw).decode(), **fields})


def _wait_for_run(client, run_id: str, statuses: set[str], deadline_s: float = 15.0) -> dict:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        run = next(r for r in client.get("/workspace/api/state").json()["runs"] if r["id"] == run_id)
        if run["status"] in statuses:
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached {statuses}; last status {run['status']}")


# --- configuration and sign-in -------------------------------------------------


def test_config_defaults_to_demo_mode_without_a_dev_login(client):
    config = client.get("/workspace/api/config").json()
    assert config["demo"] is True
    assert config["devLogin"] is False
    assert config["sso"] is False


def test_state_requires_a_signed_in_session(client):
    response = client.get("/workspace/api/state")
    assert response.status_code == 401


def test_demo_login_seeds_the_fictional_workspace(client):
    assert client.post("/workspace/api/demo-login").status_code == 200
    state = client.get("/workspace/api/state").json()
    assert state["user"] == {"id": "demo-maya", "name": "Maya Chen"}
    assert {p["name"] for p in state["projects"]} == {"KRAS G12D Program", "EGFR Resistance Screen"}
    assert len(state["chats"]) == 2
    assert all("text" not in a for a in state["artifacts"])  # metadata only in /state


def test_dev_login_is_refused_in_demo_mode_and_seeds_nothing_otherwise(client, monkeypatch):
    assert client.post("/workspace/api/dev-login", json={"name": "Alice"}).status_code == 403
    _dev_login(client, monkeypatch, "Alice Smith")
    state = client.get("/workspace/api/state").json()
    assert state["user"] == {"id": "dev-alice-smith", "name": "Alice Smith"}
    assert state["projects"] == [] and state["chats"] == []


def test_demo_login_is_refused_outside_demo_mode(client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    assert client.post("/workspace/api/demo-login").status_code == 403


def test_logout_ends_the_session(client, monkeypatch):
    _dev_login(client, monkeypatch)
    assert client.post("/workspace/api/logout").status_code == 200
    assert client.get("/workspace/api/state").status_code == 401


# --- records ---------------------------------------------------------------------


def test_another_user_cannot_see_change_or_delete_a_chat(client, monkeypatch):
    alice = _dev_login(client, monkeypatch, "Alice")
    chat = _new_chat(client, title="Alice's chat")
    bob = _dev_login(client, monkeypatch, "Bob")

    _as(client, bob)
    assert client.get("/workspace/api/state").json()["chats"] == []
    assert client.patch(f"/workspace/api/records/chats/{chat['id']}", json={"data": {"title": "x"}}).status_code == 404
    assert client.delete(f"/workspace/api/records/chats/{chat['id']}").status_code == 404

    _as(client, alice)
    assert [c["title"] for c in client.get("/workspace/api/state").json()["chats"]] == ["Alice's chat"]


def test_record_validation(client, monkeypatch):
    _dev_login(client, monkeypatch)
    assert client.post("/workspace/api/records/widgets", json={"data": {}}).status_code == 400
    sensitive = {"data": {"title": "t", "content": "patient: John Doe", "scope": "Personal"}}
    assert client.post("/workspace/api/records/memory", json=sensitive).status_code == 422
    too_big = {"data": {"title": "x" * 300_001}}
    assert client.post("/workspace/api/records/chats", json=too_big).status_code == 413


def test_update_cannot_change_ownership_fields(client, monkeypatch):
    _dev_login(client, monkeypatch)
    chat = _new_chat(client)
    patched = client.patch(
        f"/workspace/api/records/chats/{chat['id']}",
        json={"data": {"title": "Renamed", "ownerId": "someone-else", "id": "other"}},
    ).json()
    assert patched["title"] == "Renamed"
    assert patched["ownerId"] == chat["ownerId"] and patched["id"] == chat["id"]


# --- uploads and artifacts -------------------------------------------------------


def test_csv_upload_is_parsed_profiled_and_attached_to_its_chat(client, monkeypatch):
    _dev_login(client, monkeypatch)
    chat = _new_chat(client)
    raw = b"compound,ic50_uM\nA,0.5\nB,1.5\n"
    artifact = _upload(client, "results.csv", raw, chatId=chat["id"]).json()
    assert artifact["parseStatus"] == "parsed"
    assert artifact["scanStatus"] == "no obvious matches"
    assert artifact["profile"].startswith("2 rows · 2 columns")
    assert "text" not in artifact
    state = client.get("/workspace/api/state").json()
    assert next(c for c in state["chats"] if c["id"] == chat["id"])["artifactIds"] == [artifact["id"]]
    download = client.get(f"/workspace/api/artifacts/{artifact['id']}/download")
    assert download.content == raw


def test_upload_rejects_bad_base64_and_oversized_files(client, monkeypatch):
    _dev_login(client, monkeypatch)
    bad = client.post("/workspace/api/uploads", json={"name": "a.txt", "base64": "not base64!!"})
    assert bad.status_code == 422
    assert _upload(client, "big.txt", b"\0" * 20_000_001).status_code == 413


def test_sensitive_upload_is_quarantined_from_chat_context_until_redacted(client, monkeypatch):
    from app.workspace.routes import build_context, store

    _dev_login(client, monkeypatch)
    chat = _new_chat(client)
    artifact = _upload(client, "notes.txt", b"Contact jane.doe@example.com about the assay.", chatId=chat["id"]).json()
    assert artifact["scanStatus"] == "review"
    owner = client.get("/workspace/api/state").json()["user"]["id"]
    chat = store().get(owner, chat["id"], "chats")
    assert build_context(store(), owner, chat)["artifacts"] == []

    assert client.post(f"/workspace/api/artifacts/{artifact['id']}/redact").status_code == 200
    redacted = client.get(f"/workspace/api/artifacts/{artifact['id']}").json()
    assert "jane.doe@example.com" not in redacted["text"] and "[REDACTED]" in redacted["text"]
    assert redacted["scanStatus"].startswith("redacted by rule")
    assert [a["id"] for a in build_context(store(), owner, chat)["artifacts"]] == [artifact["id"]]


def test_a_transcript_upload_marked_as_a_meeting_creates_a_meeting(client, monkeypatch):
    _dev_login(client, monkeypatch)
    transcript = b"[00:00:05] Maya: Decision: repeat the assay.\n[00:00:09] Alex: I will prepare samples by Friday.\n"
    artifact = _upload(client, "weekly.txt", transcript, meeting=True).json()
    meetings = client.get("/workspace/api/state").json()["meetings"]
    assert [m["artifactId"] for m in meetings] == [artifact["id"]]
    assert meetings[0]["title"] == "weekly"
    assert meetings[0]["decisions"] == ["[00:00:05] Maya: Decision: repeat the assay."]


# --- runs ------------------------------------------------------------------------


def test_a_demo_run_completes_and_answers_in_its_chat(client, instant):
    client.post("/workspace/api/demo-login")
    chat = next(c for c in client.get("/workspace/api/state").json()["chats"] if c["title"] == "General lab questions")
    run = client.post("/workspace/api/runs", json={"chatId": chat["id"], "prompt": "What is a Western blot?", "mode": "ask"}).json()
    assert run["status"] == "queued"
    done = _wait_for_run(client, run["id"], {"completed"})
    assert done["answer"].startswith("DEMO ANALYSIS")
    messages = next(c for c in client.get("/workspace/api/state").json()["chats"] if c["id"] == chat["id"])["messages"]
    assert [m["role"] for m in messages[-2:]] == ["user", "assistant"]
    assert messages[-1]["runId"] == run["id"]

    followup = client.post(f"/workspace/api/runs/{run['id']}/followups", json={"prompt": "And a Northern blot?"}).json()
    assert followup["status"] == "running"
    end = time.monotonic() + 10
    while time.monotonic() < end:
        run_now = next(r for r in client.get("/workspace/api/state").json()["runs"] if r["id"] == run["id"])
        if run_now["followups"][0]["status"] == "completed":
            break
        time.sleep(0.05)
    assert run_now["followups"][0]["answer"].startswith("DEMO FOLLOW-UP")


def test_a_running_chat_refuses_a_second_run_in_the_same_lane_and_can_be_cancelled(client):
    client.post("/workspace/api/demo-login")
    chat = next(c for c in client.get("/workspace/api/state").json()["chats"] if c["title"] == "General lab questions")
    run = client.post("/workspace/api/runs", json={"chatId": chat["id"], "prompt": "first", "mode": "ask"}).json()

    again = client.post("/workspace/api/runs", json={"chatId": chat["id"], "prompt": "second", "mode": "auto"})
    assert again.status_code == 409
    assert client.post(f"/workspace/api/runs/{run['id']}/followups", json={"prompt": "too soon"}).status_code == 409
    assert client.post(f"/workspace/api/runs/{run['id']}/decision", json={"decision": "approve"}).status_code == 409
    assert client.delete(f"/workspace/api/records/chats/{chat['id']}").status_code == 409

    assert client.post(f"/workspace/api/runs/{run['id']}/cancel").json() == {"ok": True}
    assert _wait_for_run(client, run["id"], {"cancelled"})["currentStep"] == "Stopped"


def test_a_sensitive_prompt_is_refused(client, monkeypatch):
    _dev_login(client, monkeypatch)
    chat = _new_chat(client)
    response = client.post("/workspace/api/runs", json={"chatId": chat["id"], "prompt": "MRN: 12345 results?", "mode": "ask"})
    assert response.status_code == 422


def test_a_high_stakes_run_waits_for_a_human_and_completes_when_approved(client, monkeypatch):
    """The real orchestrator path: DEMO_MODE off, mock model, a regulatory request."""
    _dev_login(client, monkeypatch)
    chat = _new_chat(client)
    run = client.post("/workspace/api/runs", json={"chatId": chat["id"], "prompt": REGULATORY, "mode": "ask"}).json()
    waiting = _wait_for_run(client, run["id"], {"waiting", "failed", "completed"})
    assert waiting["status"] == "waiting", waiting.get("error")
    assert waiting["approvalTaskId"]

    assert client.post(f"/workspace/api/runs/{run['id']}/decision", json={"decision": "approve"}).json() == {"ok": True}
    done = _wait_for_run(client, run["id"], {"completed", "failed"})
    assert done["status"] == "completed"
    assert done["answer"]
    actions = [row["action"] for row in client.get("/workspace/api/state").json()["audit"]]
    assert "human approve" in actions
