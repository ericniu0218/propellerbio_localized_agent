"""The recording API the workspace's Meetings tab calls.

Three endpoints, and the shape of them follows from how long transcription
takes (see `jobs.py`):

    GET  /workspace/api/meetings/readiness      can this server transcribe at all?
    POST /workspace/api/meetings/recordings     here is the audio; returns a job
    GET  /workspace/api/meetings/recordings/:id what is that job doing now?

`readiness` exists so the browser can grey out the Record button and say what
is missing *before* somebody records an hour of audio that this server has no
whisper model to process. Finding that out afterwards means the meeting is gone.

The upload is multipart rather than the base64 JSON the rest of the workspace
uses. Base64 inflates by a third, and an hour of Opus is already ~14 MB; the
existing `/uploads` cap of 28 MB of base64 would refuse a long meeting.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.meetings.config import (
    MAX_RECORDING_BYTES,
    MAX_RECORDING_SECONDS,
    RecordingConfig,
    readiness_issues,
)
from app.meetings.delivery import deliver_recording
from app.meetings.enrollment import (
    ENROLLMENT_AVAILABLE,
    ENROLLMENT_MISSING,
    VoicePrintStore,
    enroll_from_wav,
)
from app.meetings.governance import capture_consent, consent_required
from app.meetings.job_store import PersistedJob, RecordingJobStore
from app.meetings.jobs import RecordingJob, RecordingJobs, process
from app.meetings.pipeline import PIPELINE_AVAILABLE, PIPELINE_MISSING
from app.meetings.retention import audio_retention_days, sweep, transcript_retention_days
from app.meetings.vocabulary_source import auto_enabled, suggest
from bedrock.meetings.transcribe import WhisperConfig, WhisperTranscriber
from app.workspace.routes import identity, require_object, store
from app.workspace.store import uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspace/api/meetings", tags=["meetings"])

# One registry for the process, backed by a store so an interrupted job can be
# picked up after a restart rather than silently lost.
_job_store = RecordingJobStore(RecordingConfig.from_env().storage_dir / "jobs.sqlite")
_jobs = RecordingJobs(_job_store)

# Browsers hand us whatever MediaRecorder produced. ffmpeg reads all of these;
# the list exists to reject an obviously wrong upload early rather than to be a
# security boundary — ffmpeg is what actually decides if the bytes are audio.
ALLOWED_SUFFIXES = {".webm", ".ogg", ".oga", ".opus", ".m4a", ".mp4", ".wav", ".mp3", ".aac"}


def config() -> RecordingConfig:
    return RecordingConfig.from_env()


@router.get("/readiness")
def readiness(user=Depends(identity)):
    """What would stop this server transcribing a recording right now."""
    if not PIPELINE_AVAILABLE:
        return {"ready": False, "issues": [PIPELINE_MISSING]}
    issues = readiness_issues(config())
    return {"ready": not issues, "issues": issues}


@router.post("/recordings")
async def create_recording(
    audio: UploadFile = File(...),
    title: str = Form("Untitled meeting"),
    projectId: str | None = Form(None),
    notes: str | None = Form(None),
    startedAt: str | None = Form(None),
    durationSeconds: float | None = Form(None),
    expectedSpeakers: int | None = Form(None),
    consentConfirmed: bool = Form(False),
    participants: str | None = Form(None),
    user=Depends(identity),
):
    """Accept a recording and start turning it into a meeting."""
    if not PIPELINE_AVAILABLE:
        raise HTTPException(503, PIPELINE_MISSING)

    recording_config = config()
    if auto_enabled():
        # Harvested terms go after the configured ones: whisper weights the
        # start of its prompt most heavily, and somebody typed those on purpose.
        harvested = suggest(store().list(user["id"], "meetings"))
        if harvested:
            recording_config = replace(recording_config, vocabulary=", ".join(harvested))
    issues = readiness_issues(recording_config)
    if issues:
        raise HTTPException(503, "This server cannot transcribe recordings: " + "; ".join(issues))

    workspace = store()
    if projectId:
        require_object(workspace, user["id"], projectId, "projects")

    # A deployment can require that somebody states the room was told. Off by
    # default: a hard block that surprises somebody mid-meeting is worse than a
    # warning, and that is a policy decision, not a default.
    if consent_required() and not consentConfirmed:
        raise HTTPException(
            422,
            "This deployment requires confirming that everyone present knows "
            "they are being recorded before a recording can be saved.",
        )
    consent = capture_consent(
        user, consentConfirmed, (participants or "").split(",") if participants else []
    ).to_record()

    if durationSeconds is not None and durationSeconds > MAX_RECORDING_SECONDS:
        raise HTTPException(
            413,
            f"This recording is {durationSeconds / 3600:.1f} hours long. The limit "
            f"is {MAX_RECORDING_SECONDS // 3600} hours — split it and upload each part.",
        )

    suffix = Path(audio.filename or "recording.webm").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"{suffix or 'That file type'} is not an audio format we can read.")

    # Streamed to disk in chunks rather than read into memory, because the
    # whole point of the size cap is that these files are large.
    job_id = uid()
    recordings_dir = recording_config.storage_dir / user["id"]
    recordings_dir.mkdir(parents=True, exist_ok=True)
    audio_path = recordings_dir / f"{job_id}{suffix}"

    written = 0
    try:
        with audio_path.open("wb") as handle:
            while chunk := await audio.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_RECORDING_BYTES:
                    raise HTTPException(
                        413,
                        f"This recording is larger than {MAX_RECORDING_BYTES // (1024 * 1024)} MB.",
                    )
                handle.write(chunk)
    except HTTPException:
        audio_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(500, f"The recording could not be saved: {exc}") from exc

    if written == 0:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(422, "The recording was empty. Nothing was captured.")

    started_at = _parse_started_at(startedAt)
    clean_title = (title or "").strip() or _default_title(started_at)

    job = RecordingJob(
        id=job_id,
        owner=user["id"],
        title=clean_title[:200],
        audio_path=audio_path,
        work_dir=recordings_dir / f"{job_id}.work",
    )
    _jobs.register(job)

    _job_store.save(
        PersistedJob(
            id=job_id, owner=user["id"], title=job.title,
            audio_path=str(audio_path), work_dir=str(job.work_dir),
            status="queued", stage="Queued", started_at=started_at.isoformat(),
            project_id=projectId, typed_notes=notes,
            expected_speakers=expectedSpeakers, consent=consent,
            access_groups=None,
        )
    )
    _jobs.start(job, _runner_for(job, recording_config, workspace, projectId, notes,
                                 started_at, expectedSpeakers, consent, user["id"]))
    workspace.audit(user["id"], "recorded meeting", job_id)
    return job.public()


def voice_store() -> VoicePrintStore:
    return VoicePrintStore(config().storage_dir / "voice-prints.sqlite")


@router.get("/voices")
def list_voices(user=Depends(identity)):
    """Everyone this workspace can recognise by voice."""
    if not ENROLLMENT_AVAILABLE:
        return {"available": False, "reason": ENROLLMENT_MISSING, "voices": []}
    return {"available": True, "voices": [v.public() for v in voice_store().list(user["id"])]}


@router.post("/voices")
async def enroll_voice(
    audio: UploadFile = File(...),
    name: str = Form(...),
    user=Depends(identity),
):
    """Save a voice print from a short sample, and keep only the vector.

    The audio is written to a temporary file because the embedding model reads
    a file, and deleted as soon as the vector exists. A voice print cannot be
    played back — it is 192 numbers — which is the only reason it is reasonable
    to keep one at all.
    """
    if not ENROLLMENT_AVAILABLE:
        raise HTTPException(503, ENROLLMENT_MISSING)
    person = (name or "").strip()
    if not person:
        raise HTTPException(422, "Give the person a name.")

    recording_config = config()
    if not recording_config.embedding_model.exists():
        raise HTTPException(503, f"No speaker-embedding model at {recording_config.embedding_model}.")

    suffix = Path(audio.filename or "sample.webm").suffix.lower() or ".webm"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"{suffix} is not an audio format we can read.")

    scratch = recording_config.storage_dir / user["id"] / "enroll"
    scratch.mkdir(parents=True, exist_ok=True)
    person_id = uid()
    raw = scratch / f"{person_id}{suffix}"
    wav = scratch / f"{person_id}.wav"
    try:
        written = 0
        with raw.open("wb") as handle:
            while chunk := await audio.read(1024 * 1024):
                written += len(chunk)
                if written > 50 * 1024 * 1024:
                    raise HTTPException(413, "That sample is too large.")
                handle.write(chunk)
        if not written:
            raise HTTPException(422, "The sample was empty. Nothing was captured.")

        transcriber_config = RecordingConfig.from_env()
        await WhisperTranscriber(
            WhisperConfig(model=transcriber_config.speech_model, vad_model=transcriber_config.vad_model)
        ).to_wav(raw, scratch)
        converted = scratch / f"{raw.stem}.16k.wav"
        try:
            vector, seconds = enroll_from_wav(converted, recording_config.embedding_model)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        saved = voice_store().save(user["id"], person_id, person, vector, seconds)
    finally:
        # The sample itself is never kept. Only the vector is.
        for path in (raw, wav, scratch / f"{raw.stem}.16k.wav"):
            path.unlink(missing_ok=True)

    store().audit(user["id"], f"saved a voice print for {person}", person_id)
    return saved.public()


@router.delete("/voices/{person_id}")
def delete_voice(person_id: str, user=Depends(identity)):
    if not voice_store().delete(user["id"], person_id):
        raise HTTPException(404, "That voice print is unavailable.")
    store().audit(user["id"], "deleted a voice print", person_id)
    return {"ok": True}


class SpeakerNames(BaseModel):
    names: dict[str, str]


@router.post("/{meeting_id}/speakers")
def rename_speakers(meeting_id: str, body: SpeakerNames, user=Depends(identity)):
    """Replace "Speaker 2" with a real name everywhere in one meeting.

    Diarisation can tell voices apart but has no idea whose they are, so the
    labels it produces are placeholders and somebody has to say who is who.
    The rename happens on the server, in one pass over every field, because the
    label appears in the transcript, in each decision and action item, and in
    the people list — renaming in only some of those leaves a meeting that
    contradicts itself.

    Whole-label matches only. Substituting "Speaker 1" inside a longer string
    would corrupt "Speaker 10" on a meeting with a dozen people in it.
    """
    workspace = store()
    meeting = require_object(workspace, user["id"], meeting_id, "meetings")

    renames = {
        old.strip(): new.strip()
        for old, new in body.names.items()
        if old.strip() and new.strip() and old.strip() != new.strip()
    }
    if not renames:
        return meeting

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(old) for old in sorted(renames, key=len, reverse=True)) + r")\b"
    )

    def swap(value):
        if isinstance(value, str):
            return pattern.sub(lambda m: renames[m.group(1)], value)
        if isinstance(value, list):
            return [swap(item) for item in value]
        return value

    for field in ("transcript", "summary", "enhancedNotes", "decisions", "actions",
                  "findings", "deadlines", "questions", "people"):
        if field in meeting:
            meeting[field] = swap(meeting[field])

    meeting["speakerNames"] = {**meeting.get("speakerNames", {}), **renames}
    saved = workspace.put(user["id"], "meetings", meeting)
    workspace.audit(user["id"], "renamed meeting speakers", meeting_id)
    return saved


@router.get("/vocabulary")
def vocabulary(user=Depends(identity)):
    """What whisper is being seeded with, and what else is worth adding.

    Suggestions are harvested from this workspace's own transcripts and
    documents, because the gene and compound names are already in the building
    and hand-maintaining the list means it is wrong within a month.
    """
    workspace = store()
    meetings = workspace.list(user["id"], "meetings")
    configured = [
        term.strip()
        for term in re.split(r"[,;\n]", os.environ.get("MEETING_VOCABULARY", ""))
        if term.strip()
    ]
    everything = suggest(meetings)
    return {
        "configured": configured,
        "suggested": [term for term in everything if term not in configured],
        "auto": auto_enabled(),
        "inUse": everything if auto_enabled() else configured,
    }


@router.get("/retention")
def retention_policy(user=Depends(identity)):
    """What this deployment keeps, and for how long."""
    audio, transcripts = audio_retention_days(), transcript_retention_days()
    return {
        "audioDays": audio,
        "transcriptDays": transcripts,
        "describes": (
            f"Recordings are deleted after {audio} days."
            if audio > 0
            else "Recordings are kept indefinitely."
        )
        + (
            f" Meetings are deleted after {transcripts} days."
            if transcripts > 0
            else " Meetings and their transcripts are kept indefinitely."
        ),
    }


def run_retention_sweep() -> dict:
    """One pass of the retention clocks. Called on startup; safe to call again.

    Audio only by default. Deleting a transcript is opt-in, because a
    transcript is the institutional memory this product exists to build.
    """
    report = sweep(config().storage_dir)
    return {
        "audioDeleted": report.audio_deleted,
        "bytesFreed": report.audio_bytes,
        "errors": report.errors,
    }


def _runner_for(job, recording_config, workspace, project_id, notes, started_at,
                expected_speakers, consent, owner):
    """One runner shape for both a fresh upload and a recovered job."""

    async def runner(running_job: RecordingJob) -> None:
        prints = voice_store().vectors(owner) if ENROLLMENT_AVAILABLE else []
        await process(
            running_job,
            _jobs,
            recording_config,
            store=workspace,
            project_id=project_id,
            typed_notes=notes,
            started_at=started_at,
            deliver=deliver_recording,
            expected_speakers=expected_speakers,
            consent=consent,
            voice_prints=prints,
        )

    return runner


def resume_interrupted() -> list[str]:
    """Pick up anything a restart interrupted. Called once, on startup.

    A job whose audio has gone is dropped rather than retried for ever: there
    is nothing left to transcribe, and leaving the row would make every startup
    try again.
    """
    if not PIPELINE_AVAILABLE:
        return []
    resumed = []
    workspace = store()
    for stale in _job_store.unfinished():
        audio = Path(stale.audio_path)
        if not audio.exists():
            logger.warning("Dropping recording job %s: its audio is gone", stale.id)
            _job_store.forget(stale.id)
            continue
        claimed = _job_store.claim(stale.id)
        if claimed is None:
            logger.warning("Abandoning recording job %s after repeated failures", stale.id)
            continue
        job = RecordingJob(
            id=claimed.id, owner=claimed.owner, title=claimed.title,
            audio_path=audio, work_dir=Path(claimed.work_dir),
        )
        _jobs.register(job)
        _jobs.advance(job, "Resuming after a restart")
        _jobs.start(job, _runner_for(
            job, config(), workspace, claimed.project_id, claimed.typed_notes,
            _parse_started_at(claimed.started_at), claimed.expected_speakers,
            claimed.consent, claimed.owner,
        ))
        resumed.append(claimed.id)
    if resumed:
        logger.info("Resumed %d interrupted recording job(s)", len(resumed))
    return resumed


@router.get("/recordings/{job_id}")
def recording_status(job_id: str, user=Depends(identity)):
    job = _jobs.get(user["id"], job_id)
    if job is None:
        raise HTTPException(404, "That recording is unavailable or has expired.")
    return job.public()


@router.get("/recordings")
def recording_list(user=Depends(identity)):
    """Recordings still in flight, so a page reload does not lose track of one."""
    return {"recordings": _jobs.for_owner(user["id"])}


def _parse_started_at(value: str | None) -> datetime:
    """When the meeting actually happened, which is not when it was uploaded.

    The browser sends the moment Record was pressed. It matters: this timestamp
    is what "what did we discuss on Tuesday?" searches against, and an
    hour-long meeting uploaded after it ends would otherwise be filed an hour
    late — on the wrong day, for anything recorded near midnight.
    """
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("Unparseable startedAt %r; using now", value)
    return datetime.now(timezone.utc)


def _default_title(started_at: datetime) -> str:
    return f"Meeting — {started_at.astimezone().strftime('%d %b %Y, %H:%M')}"
