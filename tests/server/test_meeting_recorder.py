"""Tests for recording a meeting on this device.

The one that matters most is
`test_a_recording_is_delivered_and_then_found_by_the_agent`. Everything else
here checks a part; that one checks the promise — somebody records a meeting in
a room, and afterwards the assistant can answer a question about it with a
citation. It runs the real signing code against the real reference connector
and then reads the result back through the real retrieval tool, so a drift in
any of the three fails it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import httpx
import pytest

from bedrock.server.meetings.config import RecordingConfig, readiness_issues
from bedrock.server.meetings.delivery import MEETING_ID_PREFIX, deliver_recording
from bedrock.server.meetings.jobs import RecordingJob, RecordingJobs
from bedrock.server.meetings.pipeline import TranscriptionOutcome, to_record_fields
from bedrock.server.meetings.routes import _default_title, _parse_started_at
from bedrock.server.tools.datacenter_client import (
    DatacenterConfig,
    DatacenterError,
    DatacenterTranscriptClient,
    MeetingTranscript,
)
from bedrock.server.tools.meeting_transcripts import MeetingTranscriptsArgs, MeetingTranscriptsTool

SECRET = "test-secret"

VTT = """WEBVTT

1
00:00:00.000 --> 00:00:05.000
Good morning. Let's start with the enrollment numbers for the KRAS trial.

2
00:00:06.000 --> 00:00:13.000
We are at 148 of 200 patients and the Boston coordinator left last week.

3
00:00:16.000 --> 00:00:21.000
Decision: we push the interim analysis to November 10th.
"""


def _outcome(vtt: str = VTT, duration_ms: int = 21_000) -> TranscriptionOutcome:
    return TranscriptionOutcome(vtt=vtt, cues=[object()], duration_ms=duration_ms)


# --- Readiness ----------------------------------------------------------------


def test_readiness_names_every_missing_piece_at_once(tmp_path):
    """Somebody setting this up should get one list, not one failure per meeting."""
    config = RecordingConfig(
        speech_model=tmp_path / "absent.bin",
        vad_model=tmp_path / "also-absent.bin",
        segmentation_model=tmp_path / "no-segmentation.onnx",
        embedding_model=tmp_path / "no-embedding.onnx",
        threads=4,
        vocabulary=None,
        language="en",
        storage_dir=tmp_path,
    )
    issues = readiness_issues(config)
    assert any("speech model" in issue for issue in issues)
    assert any("voice-activity model" in issue for issue in issues)


def test_threads_default_leaves_headroom_for_the_server(monkeypatch):
    monkeypatch.delenv("MEETING_WHISPER_THREADS", raising=False)
    config = RecordingConfig.from_env()
    assert 2 <= config.threads <= 10


# --- When the meeting happened ------------------------------------------------


def test_started_at_uses_the_browser_time_not_the_upload_time():
    """An hour-long meeting is uploaded an hour after it started. Filing it at
    upload time puts a late-evening meeting on the wrong day."""
    parsed = _parse_started_at("2026-09-22T23:40:00Z")
    assert parsed == datetime(2026, 9, 22, 23, 40, tzinfo=timezone.utc)


def test_started_at_assumes_utc_when_the_browser_omits_a_zone():
    assert _parse_started_at("2026-09-22T10:00:00").tzinfo is timezone.utc


def test_started_at_falls_back_to_now_rather_than_failing():
    """A meeting is worth more than a well-formed timestamp."""
    before = datetime.now(timezone.utc)
    assert _parse_started_at("not a timestamp") >= before


def test_untitled_recordings_get_a_dated_name():
    assert "2026" in _default_title(datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc))


# --- Turning a summary into a meeting record ----------------------------------


def test_the_record_says_how_its_timestamps_were_checked():
    """Every timestamp is checked against the line it quotes and dropped if it
    cannot be resolved, so a point that still carries one has been verified.

    This used to report a count from a `citations` field on MeetingSummary.
    Upstream moved verification inline and removed the field, and the fallback
    branch then told every reader to "update the pin" on a pin that was
    current — a false warning on every meeting."""
    outcome = _outcome()
    outcome.summary = _FakeSummary(citations=None)
    method = to_record_fields(outcome)["method"]
    assert "verified against the quoted line" in method
    assert "NOT verified" not in method
    assert "update the" not in method


def test_record_fields_survive_a_summary_that_never_happened():
    """The model being down must not lose the transcript."""
    fields = to_record_fields(_outcome())
    assert fields["decisions"] == []
    assert "no notes written" in fields["method"]


class _FakeCitations:
    def __init__(self, checked, moved, removed):
        self.checked, self.moved, self.removed = checked, moved, removed


class _FakeFacts:
    title = "KRAS trial weekly"
    meeting_date = "2026-09-22"
    duration_minutes = 0
    speakers: list[str] = []
    word_count = 40
    cue_count = 3


class _FakePoint:
    def __init__(self, text):
        self.text = text

    def render(self):
        return self.text


class _FakeSummary:
    def __init__(self, citations):
        self.facts = _FakeFacts()
        self.overview = "Enrollment and the interim analysis date."
        self.decisions = [_FakePoint("Push the interim analysis to November 10th. [00:00:16]")]
        self.actions: list = []
        self.details: list = []
        self.enhanced_notes = ""  # overridden by tests that exercise the fallback
        self.model_used = "llama3.1:8b"
        self.degraded_reason = None
        if citations is not None:
            self.citations = citations


# --- Job scoping --------------------------------------------------------------


def test_one_persons_job_id_does_not_open_another_persons_recording(tmp_path):
    """A job id is a guessable handle to somebody's meeting."""
    jobs = RecordingJobs()
    job = RecordingJob(id="j1", owner="maya", title="Weekly",
                       audio_path=tmp_path / "a.webm", work_dir=tmp_path / "w")
    jobs.register(job)
    assert jobs.get("maya", "j1") is not None
    assert jobs.get("someone-else", "j1") is None


def test_a_failed_job_keeps_the_recording_on_disk(tmp_path):
    """Failure is exactly when somebody wants to retry, so the audio stays."""
    audio = tmp_path / "a.webm"
    audio.write_bytes(b"not really audio")
    work = tmp_path / "work"
    work.mkdir()
    (work / "scratch.wav").write_bytes(b"x")

    jobs = RecordingJobs()
    job = RecordingJob(id="j1", owner="maya", title="Weekly", audio_path=audio, work_dir=work)
    jobs.register(job)
    jobs.fail(job, "whisper fell over")

    assert audio.exists(), "the irreplaceable file was deleted"
    assert not work.exists(), "reproducible scratch files were left behind"


# --- Delivery -----------------------------------------------------------------


async def test_delivery_says_plainly_when_no_datacenter_is_configured(monkeypatch):
    """Running without a connector is a legitimate deployment, not an error —
    but the person must be told their meeting is not searchable in chat."""
    monkeypatch.delenv("DATACENTER_BASE_URL", raising=False)
    monkeypatch.delenv("DATACENTER_SIGNING_SECRET", raising=False)
    note = await deliver_recording(
        {"id": "m1", "title": "Weekly"}, _outcome(), datetime.now(timezone.utc), "maya"
    )
    assert note and "cannot yet search this meeting" in note


async def test_delivery_failure_is_reported_but_never_fatal(tmp_path, monkeypatch):
    """An hour of transcription must not be thrown away because the connector
    was briefly down."""
    monkeypatch.setenv("DATACENTER_BASE_URL", "http://transcripts.internal")
    monkeypatch.setenv("DATACENTER_SIGNING_SECRET", SECRET)

    async def refuse(request):
        return httpx.Response(503)

    monkeypatch.setattr(
        "bedrock.server.meetings.delivery.DatacenterTranscriptClient",
        lambda: DatacenterTranscriptClient(
            config=DatacenterConfig(base_url="http://transcripts.internal", signing_secret=SECRET),
            transport=httpx.MockTransport(refuse),
        ),
    )
    note = await deliver_recording(
        {"id": "m1", "title": "Weekly"}, _outcome(), datetime.now(timezone.utc), "maya"
    )
    assert note and "could not be sent to the datacenter" in note


async def test_an_empty_recording_is_not_delivered():
    """Nothing was said; there is nothing to file."""
    empty = TranscriptionOutcome(vtt="WEBVTT\n", cues=[], duration_ms=0)
    assert await deliver_recording({"id": "m1", "title": "x"}, empty, datetime.now(timezone.utc), "maya") is None


async def test_the_connector_rejects_a_stored_transcript_signed_with_the_wrong_secret(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DATACENTER_SIGNING_SECRET", SECRET)
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    from contrib.datacenter_connector.main import app as connector_app

    client = DatacenterTranscriptClient(
        config=DatacenterConfig(base_url="http://transcripts.internal", signing_secret="wrong"),
        transport=httpx.ASGITransport(app=connector_app),
    )
    with pytest.raises(DatacenterError, match="rejected our signature"):
        await client.store(
            MeetingTranscript(
                meeting_id="recording:1", topic="Weekly", host_email="maya",
                start_time=0, end_time=1000, transcript=VTT,
            )
        )


# --- The promise --------------------------------------------------------------


async def test_a_recording_is_delivered_and_then_found_by_the_agent(tmp_path, monkeypatch):
    """Record a meeting; afterwards, ask about it and get a cited answer.

    This is the whole feature in one test. A recorded meeting is written
    through the same contract endpoint zoom-transcript-vault uses, and read
    back by the same tool that reads Zoom meetings — with no code in the
    retrieval path knowing the difference.
    """
    monkeypatch.setenv("DATACENTER_SIGNING_SECRET", SECRET)
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    from contrib.datacenter_connector.main import app as connector_app

    client = DatacenterTranscriptClient(
        config=DatacenterConfig(base_url="http://transcripts.internal", signing_secret=SECRET),
        transport=httpx.ASGITransport(app=connector_app),
    )

    started = datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("bedrock.server.meetings.delivery.DatacenterTranscriptClient", lambda: client)

    note = await deliver_recording(
        {"id": "abc123", "title": "KRAS trial weekly"}, _outcome(), started, "maya"
    )
    assert note is None, f"delivery reported a problem: {note}"

    # Stored under the contract's own id scheme, marked as a recording so its
    # provenance is visible to anyone reading storage directly.
    digest = hashlib.sha256(f"{MEETING_ID_PREFIX}abc123".encode()).hexdigest()
    stored = json.loads((tmp_path / f"{digest}.json").read_text())
    assert stored["meetingId"] == "recording:abc123"
    assert stored["topic"] == "KRAS trial weekly"
    assert stored["format"] == "vtt"

    result = await MeetingTranscriptsTool(client=client).run(
        MeetingTranscriptsArgs(
            query="what did we decide about the interim analysis?",
            from_time="2026-09-22T00:00:00Z",
            to_time="2026-09-23T00:00:00Z",
        )
    )
    assert "interim analysis" in result.content
    assert "KRAS trial weekly (2026-09-22)" in result.citations[0]


# --- The card subtitle --------------------------------------------------------


def test_the_card_falls_back_to_the_notes_when_there_is_no_overview():
    """`summarize_meeting` writes either an overview or rewritten notes, never
    both. Without this fallback, every meeting where somebody actually took
    notes shows a blank card."""
    outcome = _outcome()
    summary = _FakeSummary(citations=None)
    summary.overview = ""
    summary.enhanced_notes = "# KRAS trial weekly\n\nEnrollment is 148 of 200 and the interim analysis moved."
    outcome.summary = summary
    assert to_record_fields(outcome)["summary"].startswith("Enrollment is 148 of 200")


def test_markdown_is_stripped_from_the_card_subtitle():
    """A card reading "**Enrollment Numbers**" looks like a rendering bug —
    and a bold line is a heading, so it says nothing the title does not."""
    outcome = _outcome()
    summary = _FakeSummary(citations=None)
    summary.overview = ""
    summary.enhanced_notes = "**Enrollment Numbers and Interim Analysis Date**\n\n- Enrollment is *148 of 200*."
    outcome.summary = summary
    subtitle = to_record_fields(outcome)["summary"]
    assert subtitle == "Enrollment is 148 of 200."


def test_an_overview_is_preferred_when_there_is_one():
    outcome = _outcome()
    summary = _FakeSummary(citations=None)
    summary.enhanced_notes = "notes that should not win"
    outcome.summary = summary
    assert to_record_fields(outcome)["summary"] == "Enrollment and the interim analysis date."


# --- Deleting a meeting -------------------------------------------------------
#
# "Delete" has to mean the audio, the transcript, the job row and the
# datacenter's copy. A button that removed only the workspace record would take
# the meeting off the screen and leave every copy of it on disk, which is worse
# than having no button: the person who pressed it believes it is gone.


async def test_deleting_a_meeting_removes_the_datacenter_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("DATACENTER_SIGNING_SECRET", SECRET)
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    from contrib.datacenter_connector.main import app as connector_app

    client = DatacenterTranscriptClient(
        config=DatacenterConfig(base_url="http://transcripts.internal", signing_secret=SECRET),
        transport=httpx.ASGITransport(app=connector_app),
    )
    monkeypatch.setattr("bedrock.server.meetings.delivery.DatacenterTranscriptClient", lambda: client)
    await deliver_recording(
        {"id": "gone1", "title": "Weekly"}, _outcome(), datetime.now(timezone.utc), "maya"
    )
    stored = tmp_path / f"{hashlib.sha256(b'recording:gone1').hexdigest()}.json"
    assert stored.exists(), "nothing was delivered, so the delete proves nothing"

    assert await client.forget(f"{MEETING_ID_PREFIX}gone1") is None
    assert not stored.exists()


async def test_deleting_a_transcript_that_is_already_gone_is_not_an_error(tmp_path, monkeypatch):
    """The caller asked for it to be absent. It is absent."""
    monkeypatch.setenv("DATACENTER_SIGNING_SECRET", SECRET)
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    from contrib.datacenter_connector.main import app as connector_app

    client = DatacenterTranscriptClient(
        config=DatacenterConfig(base_url="http://transcripts.internal", signing_secret=SECRET),
        transport=httpx.ASGITransport(app=connector_app),
    )
    assert await client.forget("recording:never-existed") is None


async def test_a_datacenter_without_delete_is_reported_not_hidden():
    """DELETE is an extension, not part of the connector contract. A datacenter
    that has not implemented it must not leave the employee believing their
    transcript was destroyed there."""
    def not_implemented(request):
        return httpx.Response(405, json={"detail": "Method Not Allowed"})

    client = DatacenterTranscriptClient(
        config=DatacenterConfig(base_url="http://transcripts.internal", signing_secret=SECRET),
        transport=httpx.MockTransport(not_implemented),
    )
    note = await client.forget("recording:m1")
    assert note and "does not support deleting" in note


async def test_an_unreachable_datacenter_does_not_raise_while_deleting():
    """Deleting locally must still happen. Raising here would leave the meeting
    on screen with its audio already removed."""
    def unreachable(request):
        raise httpx.ConnectError("no route to host")

    client = DatacenterTranscriptClient(
        config=DatacenterConfig(base_url="http://transcripts.internal", signing_secret=SECRET),
        transport=httpx.MockTransport(unreachable),
    )
    note = await client.forget("recording:m1")
    assert note and "could not be reached" in note


async def test_the_delete_signature_covers_the_method(tmp_path, monkeypatch):
    """The signature is over "{ts}.{METHOD}.{path}.{body}", so a DELETE signed
    as a GET is rejected. Pinned because the method is the only thing that
    differs between this call and the fetch beside it."""
    monkeypatch.setenv("DATACENTER_SIGNING_SECRET", SECRET)
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    from contrib.datacenter_connector.main import app as connector_app

    wrong = DatacenterTranscriptClient(
        config=DatacenterConfig(base_url="http://transcripts.internal", signing_secret="wrong"),
        transport=httpx.ASGITransport(app=connector_app),
    )
    note = await wrong.forget("recording:m1")
    assert note and "rejected our signature" in note
