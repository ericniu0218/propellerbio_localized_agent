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
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.meetings.config import (
    MAX_RECORDING_BYTES,
    MAX_RECORDING_SECONDS,
    RecordingConfig,
    readiness_issues,
)
from app.meetings.delivery import deliver_recording
from app.meetings.jobs import RecordingJob, RecordingJobs, process
from app.meetings.pipeline import PIPELINE_AVAILABLE, PIPELINE_MISSING
from app.workspace.routes import identity, require_object, store
from app.workspace.store import uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspace/api/meetings", tags=["meetings"])

# One registry for the process, like the workspace's own `jobs` dict.
_jobs = RecordingJobs()

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
    user=Depends(identity),
):
    """Accept a recording and start turning it into a meeting."""
    if not PIPELINE_AVAILABLE:
        raise HTTPException(503, PIPELINE_MISSING)

    recording_config = config()
    issues = readiness_issues(recording_config)
    if issues:
        raise HTTPException(503, "This server cannot transcribe recordings: " + "; ".join(issues))

    workspace = store()
    if projectId:
        require_object(workspace, user["id"], projectId, "projects")

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

    async def runner(running_job: RecordingJob) -> None:
        await process(
            running_job,
            _jobs,
            recording_config,
            store=workspace,
            project_id=projectId,
            typed_notes=notes,
            started_at=started_at,
            deliver=deliver_recording,
        )

    _jobs.start(job, runner)
    workspace.audit(user["id"], "recorded meeting", job_id)
    return job.public()


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
