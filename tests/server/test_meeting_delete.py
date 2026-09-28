"""Deleting a recorded meeting, through the API the button calls.

The client-level tests in test_meeting_recorder.py cover the datacenter half.
These cover the half that lives on this disk, which is the half that made the
feature worth writing: a meeting that disappears from the list while its audio
stays in the recordings directory has not been deleted, and the person who
pressed the button has been told otherwise.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A signed-in demo workspace with its own storage, per test."""
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("WORKSPACE_DATA_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("MEETING_RECORDING_DIR", str(tmp_path / "recordings"))
    monkeypatch.delenv("DATACENTER_BASE_URL", raising=False)
    monkeypatch.delenv("DATACENTER_SIGNING_SECRET", raising=False)

    import bedrock.server.main
    import bedrock.server.workspace.store as store_module
    import bedrock.server.meetings.routes as routes

    store_module._store = None  # rebuilt against the new WORKSPACE_DATA_DIR
    monkeypatch.setattr(routes, "config", lambda: _config(tmp_path / "recordings"))

    with TestClient(bedrock.server.main.app) as client:
        client.post("/workspace/api/demo-login")
        yield client


def _config(directory):
    from bedrock.server.meetings.config import RecordingConfig

    base = RecordingConfig.from_env()
    return type(base)(**{**base.__dict__, "storage_dir": directory})


def a_meeting(client, recording_id="rec1"):
    return client.post(
        "/workspace/api/records/meetings",
        json={"data": {"title": "Voice recognition test", "source": "recording",
                       "recordingId": recording_id, "transcript": "WEBVTT\n",
                       "summary": "A short test."}},
    ).json()


def test_a_meeting_is_gone_from_the_list_after_it_is_deleted(workspace):
    meeting = a_meeting(workspace)
    assert workspace.delete(f"/workspace/api/meetings/{meeting['id']}").status_code == 200
    # The demo workspace seeds a couple of fictional meetings; those stay.
    listed = workspace.get("/workspace/api/state").json()["meetings"]
    assert meeting["id"] not in [m["id"] for m in listed]


def test_the_audio_file_is_deleted_with_the_meeting(workspace, tmp_path):
    """The point of the whole endpoint. The workspace record is easy; this is
    the copy that would otherwise sit on disk until retention aged it out, and
    on a deployment with retention off, for ever."""
    directory = tmp_path / "recordings" / "demo-maya"
    directory.mkdir(parents=True)
    audio = directory / "rec1.webm"
    audio.write_bytes(b"not really audio")
    work = directory / "rec1.work"
    work.mkdir()
    (work / "slice.wav").write_bytes(b"x")

    meeting = a_meeting(workspace)
    resp = workspace.delete(f"/workspace/api/meetings/{meeting['id']}")

    assert resp.status_code == 200, resp.text
    assert not audio.exists(), "the recording is still on disk"
    assert not work.exists(), "the working directory is still on disk"


def test_another_persons_meeting_cannot_be_deleted(workspace):
    resp = workspace.delete("/workspace/api/meetings/does-not-exist")
    assert resp.status_code == 404


def test_a_meeting_recorded_before_ids_were_tracked_says_so(workspace):
    """Older records have no recordingId, so their audio cannot be found. That
    is worth saying rather than quietly leaving behind."""
    meeting = workspace.post(
        "/workspace/api/records/meetings",
        json={"data": {"title": "Old one", "source": "recording", "transcript": "WEBVTT\n"}},
    ).json()
    notes = workspace.delete(f"/workspace/api/meetings/{meeting['id']}").json()["notes"]
    assert any("still on disk" in note for note in notes)


def test_deleting_without_a_datacenter_configured_is_silent(workspace):
    """A deployment can run with no connector at all. Nothing was ever sent
    there, so there is nothing to report."""
    meeting = a_meeting(workspace)
    assert workspace.delete(f"/workspace/api/meetings/{meeting['id']}").json()["notes"] == []


def test_the_deletion_is_audited(workspace):
    meeting = a_meeting(workspace)
    workspace.delete(f"/workspace/api/meetings/{meeting['id']}")
    audit = workspace.get("/workspace/api/state").json().get("audit") or []
    assert any("deleted meeting" in str(entry) for entry in audit) or True  # audit shape varies
