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


# --- Which speech model, and which languages ----------------------------------


def _config(tmp_path, **overrides) -> RecordingConfig:
    base = dict(
        speech_model=tmp_path / "ggml-large-v3-turbo.bin",
        vad_model=tmp_path / "vad.bin",
        segmentation_model=tmp_path / "seg.onnx",
        embedding_model=tmp_path / "emb.onnx",
        threads=4,
        vocabulary=None,
        language="detect",
        storage_dir=tmp_path,
    )
    base.update(overrides)
    return RecordingConfig(**base)


def test_the_multilingual_model_is_preferred_and_the_old_one_still_accepted(tmp_path, monkeypatch):
    """An installation from before the model change has only small.en. It keeps
    working; a fresh one is pointed at the multilingual model by name."""
    monkeypatch.setenv("MEETING_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("MEETING_SPEECH_MODEL", raising=False)
    assert RecordingConfig.from_env().speech_model.name == "ggml-large-v3-turbo.bin"

    (tmp_path / "ggml-small.en.bin").write_bytes(b"x")
    assert RecordingConfig.from_env().speech_model.name == "ggml-small.en.bin"
    assert not RecordingConfig.from_env().multilingual

    (tmp_path / "ggml-large-v3-turbo.bin").write_bytes(b"x")
    assert RecordingConfig.from_env().speech_model.name == "ggml-large-v3-turbo.bin"
    assert RecordingConfig.from_env().multilingual


def test_an_english_only_model_is_refused_for_another_language(tmp_path):
    (tmp_path / "ggml-small.en.bin").write_bytes(b"x")
    (tmp_path / "vad.bin").write_bytes(b"x")
    english_only = _config(tmp_path, speech_model=tmp_path / "ggml-small.en.bin")

    assert not [i for i in readiness_issues(english_only) if "model" in i.lower() and "voice" not in i]
    issues = readiness_issues(RecordingConfig(**{**english_only.__dict__, "language": "zh"}))
    assert any("English-only" in i and "ggml-large-v3-turbo.bin" in i for i in issues)


def test_readiness_tells_the_dialog_which_languages_it_can_offer(tmp_path, monkeypatch):
    from bedrock.server.meetings import routes

    (tmp_path / "ggml-small.en.bin").write_bytes(b"x")
    (tmp_path / "ggml-large-v3-turbo.bin").write_bytes(b"x")
    (tmp_path / "vad.bin").write_bytes(b"x")

    monkeypatch.setattr(routes, "config", lambda: _config(tmp_path, speech_model=tmp_path / "ggml-small.en.bin"))
    answer = routes.readiness(user={"id": "u"})
    assert answer["multilingual"] is False
    assert [l["code"] for l in answer["languages"]] == ["en"]
    assert answer["recommendedModel"] == "ggml-large-v3-turbo.bin"

    monkeypatch.setattr(routes, "config", lambda: _config(tmp_path))
    answer = routes.readiness(user={"id": "u"})
    assert answer["multilingual"] is True
    codes = [l["code"] for l in answer["languages"]]
    assert "zh" in codes and "auto" in codes and codes[:2] == ["detect", "en"]
    assert answer["defaultLanguage"] == "detect"
    assert answer["consentRequired"] is False


def test_a_language_the_server_cannot_do_is_refused_before_the_upload_is_kept(tmp_path):
    from fastapi import HTTPException

    from bedrock.server.meetings.routes import _resolve_language

    multilingual = _config(tmp_path)
    assert _resolve_language(None, multilingual) is multilingual
    assert _resolve_language("", multilingual) is multilingual
    assert _resolve_language("ZH", multilingual).language == "zh"
    assert _resolve_language("auto", multilingual).language == "auto"

    with pytest.raises(HTTPException) as refused:
        _resolve_language("klingon", multilingual)
    assert refused.value.status_code == 422

    english_only = _config(tmp_path, speech_model=tmp_path / "ggml-small.en.bin")
    assert _resolve_language("en", english_only).language == "en"
    # Detection on an English-only model simply means English; not an error.
    assert _resolve_language("detect", english_only).language == "detect"
    with pytest.raises(HTTPException) as refused:
        _resolve_language("zh", english_only)
    assert refused.value.status_code == 422
    assert "English-only" in refused.value.detail


def test_the_language_survives_a_restart(tmp_path):
    """A job resumed after a restart is transcribed in the language it was
    recorded in, not the server's default."""
    from bedrock.server.meetings.job_store import PersistedJob, RecordingJobStore

    store = RecordingJobStore(tmp_path / "jobs.sqlite")
    store.save(PersistedJob(
        id="j1", owner="u", title="t", audio_path=str(tmp_path / "a.webm"),
        work_dir=str(tmp_path / "w"), status="queued", stage="Queued",
        started_at="2026-09-28T00:00:00+00:00", language="zh",
    ))
    (job,) = store.unfinished()
    assert job.language == "zh"

    # Rows written before the field existed carry no language and mean "default".
    store.save(PersistedJob(
        id="j0", owner="u", title="t", audio_path=str(tmp_path / "b.webm"),
        work_dir=str(tmp_path / "w0"), status="queued", stage="Queued",
        started_at="2026-09-28T00:00:00+00:00",
    ))
    assert {j.id: j.language for j in store.unfinished()} == {"j1": "zh", "j0": None}


def test_the_vocabulary_is_the_whole_prompt_and_the_language_passes_through(tmp_path):
    """The Mandarin script hint is the transcriber's job, added once it knows
    the recording is Mandarin — chosen or detected — so the server's prompt is
    only the vocabulary."""
    from bedrock.server.meetings.pipeline import _whisper_config

    assert _whisper_config(_config(tmp_path)).initial_prompt is None
    assert _whisper_config(_config(tmp_path, vocabulary="KRAS, Priya")).initial_prompt == "KRAS, Priya"
    zh = _whisper_config(_config(tmp_path, language="zh", vocabulary="KRAS, Priya"))
    assert zh.initial_prompt == "KRAS, Priya"
    assert zh.language == "zh"
    assert _whisper_config(_config(tmp_path, language="detect")).language == "detect"


def test_room_sensitivity_reaches_whisper(tmp_path, monkeypatch):
    from bedrock.server.meetings.pipeline import _whisper_config

    monkeypatch.setenv("MEETING_MODEL_DIR", str(tmp_path))
    for name in ("MEETING_VAD_THRESHOLD", "MEETING_VAD_PAD_MS", "MEETING_NORMALIZE_AUDIO"):
        monkeypatch.delenv(name, raising=False)
    whisper = _whisper_config(RecordingConfig.from_env())
    assert whisper.normalize_audio is False  # measured to lower speech-to-noise on a real room
    assert whisper.vad_threshold == 0.4
    assert whisper.vad_speech_pad_ms == 60

    monkeypatch.setenv("MEETING_VAD_THRESHOLD", "0.3")
    monkeypatch.setenv("MEETING_VAD_PAD_MS", "100")
    monkeypatch.setenv("MEETING_NORMALIZE_AUDIO", "on")
    whisper = _whisper_config(RecordingConfig.from_env())
    assert (whisper.vad_threshold, whisper.vad_speech_pad_ms, whisper.normalize_audio) == (0.3, 100, True)


def test_a_meeting_nobody_named_is_marked_for_naming_from_its_notes(tmp_path):
    from bedrock.server.meetings.job_store import PersistedJob, RecordingJobStore

    store = RecordingJobStore(tmp_path / "jobs.sqlite")
    store.save(PersistedJob(
        id="j1", owner="u", title="Meeting — 28 Sep 2026, 21:00", audio_path=str(tmp_path / "a.webm"),
        work_dir=str(tmp_path / "w"), status="queued", stage="Queued",
        started_at="2026-09-28T00:00:00+00:00", auto_title=True,
    ))
    (job,) = store.unfinished()
    assert job.auto_title is True


def test_the_meeting_page_is_told_what_was_heard():
    from bedrock.meetings.transcribe import LanguageDecision
    from bedrock.server.meetings.pipeline import describe_language

    one = describe_language(LanguageDecision(language="zh", heard={"zh": 0.93, "en": 0.07}, sampled=8))
    assert one == "Transcribed as Mandarin (detected: Mandarin 93%, English 7%)."
    mixed = describe_language(LanguageDecision(language="auto", heard={"zh": 0.6, "en": 0.4}, sampled=8))
    assert mixed.startswith("More than one language was heard (Mandarin 60%, English 40%)")
    assert "each speaker turn" in mixed


# --- Speaker labels after diarisation ----------------------------------------


def _cue(start, end, text, speaker=None):
    from bedrock.meetings.transcript import Cue

    return Cue(start_ms=start, end_ms=end, text=text, speaker=speaker)


def test_speaker_labels_are_renumbered_in_order_of_appearance():
    from bedrock.server.meetings.pipeline import tidy_speakers

    # Speaker 5's one line is long enough to be a person, not a phantom.
    cues = [_cue(0, 1000, "a", "Speaker 2"), _cue(2000, 6000, "b", "Speaker 5"), _cue(7000, 8000, "c", "Speaker 2"),
            _cue(9000, 10000, "d", "Priya")]
    tidied, count = tidy_speakers(cues)
    assert [c.speaker for c in tidied] == ["Speaker 1", "Speaker 2", "Speaker 1", "Priya"]
    assert count == 3


def test_an_unlabelled_line_goes_to_its_nearest_neighbour_within_two_seconds():
    from bedrock.server.meetings.pipeline import tidy_speakers

    cues = [
        _cue(0, 1000, "a", "Speaker 1"),
        _cue(1500, 2000, "what?"),               # 500 ms after Speaker 1
        _cue(2200, 3000, "b", "Speaker 2"),
        _cue(20000, 21000, "far away"),          # nobody within two seconds
        _cue(30000, 31000, "lead-in"),           # only a following neighbour, 400 ms later
        _cue(31400, 32000, "c", "Speaker 2"),
    ]
    tidied, count = tidy_speakers(cues)
    assert [c.speaker for c in tidied] == ["Speaker 1", "Speaker 1", "Speaker 2", None, "Speaker 2", "Speaker 2"]
    assert count == 2


def test_tidying_nothing_is_nothing():
    from bedrock.server.meetings.pipeline import tidy_speakers

    assert tidy_speakers([]) == ([], 0)
    cues = [_cue(0, 1000, "a"), _cue(9000, 10000, "b")]
    assert tidy_speakers(cues) == (cues, 0)


def test_a_one_line_phantom_speaker_is_folded_into_the_previous_speaker():
    from bedrock.server.meetings.pipeline import tidy_speakers

    cues = [
        _cue(0, 3000, "a", "Speaker 1"), _cue(3500, 6000, "b", "Speaker 2"),
        _cue(6200, 6800, "Ha!", "Speaker 5"),                 # one short line: folded into Speaker 2
        _cue(7000, 9000, "c", "Speaker 1"),
        _cue(9500, 14000, "a long remark from somebody else", "Speaker 7"),  # long: kept
        _cue(15000, 16000, "d", "Speaker 2"),
    ]
    tidied, count = tidy_speakers(cues)
    assert [c.speaker for c in tidied] == ["Speaker 1", "Speaker 2", "Speaker 2", "Speaker 1", "Speaker 3", "Speaker 2"]
    assert count == 3


def test_phantoms_are_not_folded_when_everyone_would_be_one():
    from bedrock.server.meetings.pipeline import tidy_speakers

    cues = [_cue(0, 1000, "a", "Speaker 3"), _cue(2000, 3000, "b", "Speaker 8")]
    tidied, count = tidy_speakers(cues)
    assert [c.speaker for c in tidied] == ["Speaker 1", "Speaker 2"] and count == 2


# --- Vocabulary packs and the decoder's prompt budget --------------------------


def test_the_biotech_pack_loads_and_goes_after_configured_terms(monkeypatch):
    from bedrock.server.meetings.vocabulary_source import from_packs, suggest

    monkeypatch.setenv("MEETING_VOCABULARY_PACK", "biotech")
    monkeypatch.setenv("MEETING_VOCABULARY", "Priya, Ravi")
    monkeypatch.delenv("MEETING_VOCABULARY_CORPUS", raising=False)
    pack = from_packs()
    assert len(pack) > 400 and "dupilumab" in pack and "GLP-1" in pack and "TSLP" in pack
    assert not any(t.startswith("#") for t in pack)
    terms = suggest([])
    assert terms[:2] == ["Priya", "Ravi"] and terms[2] == pack[0]


def test_an_unknown_pack_is_skipped_not_fatal(monkeypatch, tmp_path):
    from bedrock.server.meetings.vocabulary_source import from_packs

    own = tmp_path / "mine.txt"
    own.write_text("# mine\nKRAS\n\nG12D # comment\n")
    assert from_packs(f"no-such-pack, {own}") == ["KRAS", "G12D"]


def test_the_decoder_hint_is_cut_to_whispers_budget_from_the_front():
    from bedrock.inference import estimate_tokens
    from bedrock.server.meetings.pipeline import HINT_TOKEN_BUDGET, hint_within_budget

    assert hint_within_budget(None) is None
    assert hint_within_budget("KRAS, Priya") == "KRAS, Priya"
    long = ", ".join(f"term{i}" for i in range(600))
    hint = hint_within_budget(long)
    assert hint.startswith("term0, term1, term2")
    assert estimate_tokens(hint) <= HINT_TOKEN_BUDGET
    assert "term599" not in hint


def test_documents_folder_comes_from_the_environment(monkeypatch, tmp_path):
    from bedrock.server.meetings.vocabulary_source import suggest

    (tmp_path / "paper.txt").write_text("Bimagrumab and Bimagrumab again. Bimagrumab.")
    monkeypatch.setenv("MEETING_VOCABULARY_CORPUS", str(tmp_path))
    monkeypatch.delenv("MEETING_VOCABULARY_PACK", raising=False)
    monkeypatch.delenv("MEETING_VOCABULARY", raising=False)
    assert "Bimagrumab" in suggest([])


# --- Export, translation and the Mandarin model --------------------------------


MEETING = {
    "id": "m1", "title": "KRAS weekly: dose response go-ahead", "recordedAt": "2026-09-29T20:18:26+00:00",
    "durationMs": 125_000, "speakerCount": 2,
    "caveats": ["Transcribed as Mandarin (detected: Mandarin 93%, English 7%)."],
    "summary": "The team agreed to start the dose response study on Monday.",
    "decisions": ["Start the dose response study Monday — Speaker 1 [00:00:22]"],
    "actions": ["Order the EGFR antibody before Thursday — Ravi [00:00:27]"],
    "findings": [],
    "transcript": "WEBVTT\n\n1\n00:00:00.000 --> 00:00:04.000\nSpeaker 1: 大家早上好，我们开始今天的周会。\n\n2\n00:00:05.000 --> 00:00:09.000\nSpeaker 2: The controls came back clean.\n",
    "transcriptEnglish": "WEBVTT\n\n1\n00:00:00.000 --> 00:00:04.000\nSpeaker 1: Good morning everyone, let's start today's weekly meeting.\n\n2\n00:00:05.000 --> 00:00:09.000\nSpeaker 2: The controls came back clean.\n",
    "translatedLines": 1,
}


def test_the_text_export_has_notes_then_every_line_and_english_only_when_asked():
    from bedrock.server.meetings import export

    plain = export.as_text(MEETING)
    assert plain.startswith("KRAS weekly: dose response go-ahead\n")
    assert "2 min · 2 speakers" in plain
    assert "SUMMARY\nThe team agreed" in plain and "- Order the EGFR antibody" in plain
    assert "[00:00:00] Speaker 1: 大家早上好，我们开始今天的周会。" in plain
    assert "EN:" not in plain
    bilingual = export.as_text(MEETING, english=True)
    assert "    EN: Good morning everyone, let's start today's weekly meeting." in bilingual
    assert bilingual.count("EN:") == 1  # the English line is not repeated under itself


def test_every_format_renders_and_names_its_file():
    from bedrock.server.meetings import export

    for fmt in export.FORMATS:
        body, media = export.render(MEETING, fmt, english=True)
        assert body and media == export.FORMATS[fmt]
    assert export.filename(MEETING, "docx") == "KRAS-weekly-dose-response-go-ahead.docx"
    assert export.as_vtt(MEETING, english=True) == MEETING["transcriptEnglish"]
    assert export.as_vtt(MEETING) == MEETING["transcript"]
    assert "## Transcript" in export.as_markdown(MEETING)


def test_the_word_export_carries_the_transcript_and_the_english():
    from docx import Document
    import io

    from bedrock.server.meetings import export

    doc = Document(io.BytesIO(export.as_docx(MEETING, english=True)))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "KRAS weekly: dose response go-ahead" in text
    assert "Speaker 1: 大家早上好，我们开始今天的周会。" in text
    assert "Good morning everyone" in text
    assert "Order the EGFR antibody before Thursday" in text


def test_the_export_route_refuses_an_unknown_format(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from bedrock.server.meetings import routes

    with pytest.raises(HTTPException) as refused:
        routes.download_transcript("m1", format="pdf", user={"id": "u"})
    assert refused.value.status_code == 422


def test_translations_become_a_parallel_english_transcript(tmp_path):
    from bedrock.meetings.transcript import Cue
    from bedrock.server.meetings.pipeline import TranscriptionOutcome, english_vtt

    cues = [Cue(0, 1000, "Speaker 1", "大家早上好"), Cue(1000, 2000, "Speaker 2", "Hello")]
    outcome = TranscriptionOutcome(vtt="", cues=cues, duration_ms=2000)
    assert english_vtt(outcome) is None
    outcome.translations = {0: "Good morning everyone"}
    vtt = english_vtt(outcome)
    assert "Speaker 1: Good morning everyone" in vtt and "Speaker 2: Hello" in vtt
    assert "大家早上好" not in vtt


def test_mandarin_gets_its_own_model_when_one_is_installed(tmp_path, monkeypatch):
    from bedrock.server.meetings.pipeline import _whisper_config

    monkeypatch.setenv("MEETING_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("MEETING_SPEECH_MODEL_ZH", raising=False)
    assert RecordingConfig.from_env().speech_model_zh is None
    assert _whisper_config(RecordingConfig.from_env()).models_by_language == {}

    (tmp_path / "ggml-large-v3.bin").write_bytes(b"x")
    cfg = RecordingConfig.from_env()
    assert cfg.speech_model_zh == tmp_path / "ggml-large-v3.bin"
    assert _whisper_config(cfg).models_by_language == {"zh": tmp_path / "ggml-large-v3.bin"}

    monkeypatch.setenv("MEETING_SPEECH_MODEL_ZH", str(tmp_path / "other.bin"))
    assert RecordingConfig.from_env().speech_model_zh == tmp_path / "other.bin"


def test_the_export_header_reads_naturally_for_short_and_single_speaker_meetings():
    from bedrock.server.meetings import export

    short = {**MEETING, "durationMs": 27_000, "speakerCount": 1}
    assert "27 s · 1 speaker" in export.as_text(short).splitlines()[1]
    assert "2 min · 2 speakers" in export.as_text(MEETING).splitlines()[1]


def test_one_speaker_is_reported_in_the_singular():
    from bedrock.server.meetings.pipeline import _speakers

    assert _speakers(1) == "1 speaker was" and _speakers(3) == "3 speakers were"
