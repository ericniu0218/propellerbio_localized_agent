"""Tracking a recording while it is being turned into notes.

Transcription is the only thing in this product whose cost is measured in
minutes of wall-clock time. On this hardware whisper's small.en model runs at
roughly real time, so an hour-long meeting is an hour of work after the meeting
ends. That rules out doing it inside the upload request: the browser would time
out, and a refresh would lose the recording.

So an upload returns immediately with a job id, the work happens in a
background task, and the browser polls. The states are deliberately specific —
"Transcribing speech" rather than "Working" — because somebody watching a
progress bar for forty minutes deserves to know which stage they are in and
that it has not hung.

Jobs live in memory. A server restart mid-transcription loses the job but not
the recording: the audio is on disk under `MEETING_RECORDING_DIR` and can be
re-submitted. Making jobs survive restarts means a real queue, which is the
right call when this moves off one box and is noted in docs/MEETINGS.md.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.meetings.config import RecordingConfig
from app.meetings.governance import groups_for, redact_record
from app.meetings.job_store import RecordingJobStore
from app.meetings.pipeline import (
    RecordingError,
    TranscriptionOutcome,
    summarise,
    to_record_fields,
    transcribe,
)

logger = logging.getLogger(__name__)

# A finished job is kept so the browser can collect the result after a reload,
# then dropped. Long enough to survive somebody closing the laptop lid and
# coming back; short enough that the dict cannot grow without bound.
COMPLETED_JOB_TTL_S = 6 * 60 * 60


@dataclass
class RecordingJob:
    id: str
    owner: str
    title: str
    audio_path: Path
    work_dir: Path
    status: str = "queued"
    stage: str = "Queued"
    error: str | None = None
    meeting_id: str | None = None
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    def public(self) -> dict:
        """What the browser is allowed to see. No filesystem paths."""
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "title": self.title,
            "error": self.error,
            "meetingId": self.meeting_id,
            "notes": self.notes,
        }


class RecordingJobs:
    """In-process registry of recordings being transcribed.

    Backed by `RecordingJobStore` when one is given, so an interrupted job can
    be picked up after a restart. Without one the behaviour is exactly what it
    was: fine for a test, and the reason the store is optional.
    """

    def __init__(self, store: RecordingJobStore | None = None) -> None:
        self._jobs: dict[str, RecordingJob] = {}
        self._tasks: set[asyncio.Task] = set()
        self.store = store

    def get(self, owner: str, job_id: str) -> RecordingJob | None:
        job = self._jobs.get(job_id)
        # Scoped by owner for the same reason every workspace read is: a job id
        # is a guessable handle to somebody's meeting.
        return job if job and job.owner == owner else None

    def for_owner(self, owner: str) -> list[dict]:
        return [job.public() for job in self._jobs.values() if job.owner == owner]

    def register(self, job: RecordingJob) -> None:
        self._sweep()
        self._jobs[job.id] = job

    def start(self, job: RecordingJob, runner) -> None:
        """Run `runner(job)` in the background, keeping a reference to the task.

        Without holding the reference the event loop is free to garbage-collect
        a running task, which drops the transcription silently partway through.
        """
        task = asyncio.create_task(self._guard(job, runner))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _guard(self, job: RecordingJob, runner) -> None:
        try:
            await runner(job)
        except RecordingError as exc:
            self.fail(job, str(exc))
        except Exception as exc:  # pragma: no cover - unexpected only
            logger.exception("Recording job %s crashed", job.id)
            self.fail(job, f"The recording could not be processed: {exc}")

    def advance(self, job: RecordingJob, stage: str) -> None:
        job.status = "running"
        job.stage = stage
        if self.store:
            self.store.update_status(job.id, "running", stage)

    def finish(self, job: RecordingJob, meeting_id: str, notes: list[str]) -> None:
        job.status = "done"
        job.stage = "Done"
        job.meeting_id = meeting_id
        job.notes = notes
        job.finished_at = time.monotonic()
        # The meeting is the record now; the job has nothing left to say.
        if self.store:
            self.store.forget(job.id)
        self._cleanup(job)

    def fail(self, job: RecordingJob, error: str) -> None:
        job.status = "failed"
        job.stage = "Failed"
        job.error = error
        job.finished_at = time.monotonic()
        # Kept, not forgotten: a failed job still holds the audio and the
        # parameters, which is exactly what a retry needs.
        if self.store:
            self.store.update_status(job.id, "failed", "Failed")
        self._cleanup(job)

    def _cleanup(self, job: RecordingJob) -> None:
        """Delete whisper's scratch files; keep the original recording.

        The slices and 16 kHz conversions are large and reproducible. The
        recording itself is the one artefact that cannot be recreated, so it
        stays on disk whatever happened — including after a failure, which is
        exactly when somebody will want to retry it.
        """
        shutil.rmtree(job.work_dir, ignore_errors=True)

    def _sweep(self) -> None:
        cutoff = time.monotonic() - COMPLETED_JOB_TTL_S
        for job_id, job in list(self._jobs.items()):
            if job.finished_at is not None and job.finished_at < cutoff:
                del self._jobs[job_id]


async def process(
    job: RecordingJob,
    jobs: RecordingJobs,
    config: RecordingConfig,
    *,
    store,
    project_id: str | None,
    typed_notes: str | None,
    started_at: datetime,
    deliver,
    expected_speakers: int | None = None,
    minimum_speakers: int | None = None,
    consent: dict | None = None,
    access_groups: list[str] | None = None,
    voice_prints: list | None = None,
) -> None:
    """The whole path from a file on disk to a meeting somebody can ask about."""

    def progress(stage: str) -> None:
        jobs.advance(job, stage)

    outcome: TranscriptionOutcome = await transcribe(
        job.audio_path,
        config,
        job.work_dir,
        on_progress=progress,
        expected_speakers=expected_speakers,
        minimum_speakers=minimum_speakers,
        voice_prints=voice_prints,
    )
    outcome = await summarise(
        outcome,
        config,
        title=job.title,
        meeting_date=started_at.date(),
        notes=typed_notes,
        on_progress=progress,
    )

    jobs.advance(job, "Saving")
    fields = to_record_fields(outcome)

    # Redaction happens before anything is stored or delivered, and covers the
    # transcript and everything derived from it together — a patient name taken
    # out of the transcript but left in an action item has not been taken out.
    fields["transcript"] = outcome.vtt[:100_000]
    fields, redaction = redact_record(fields)

    record = store.put(
        job.owner,
        "meetings",
        {
            "title": job.title,
            "projectId": project_id,
            "transcriptTruncated": len(outcome.vtt) > 100_000,
            "recordedAt": started_at.isoformat(),
            "durationMs": outcome.duration_ms,
            "source": "recording",
            "attributed": outcome.attributed,
            "speakerCount": outcome.speaker_count,
            "caveats": outcome.notes + ([f"PHI {redaction.describe()}."] if redaction.applied else []),
            "redaction": {"applied": redaction.applied, "findings": sorted(set(redaction.findings))},
            "consent": consent,
            "accessGroups": groups_for(project_id, access_groups),
            "demo": False,
            **fields,
        },
    )

    # Delivery is best-effort on purpose. The meeting is already saved and
    # readable in the workspace; failing the whole job because the datacenter
    # connector is unreachable would throw away an hour of transcription for a
    # problem that a retry fixes.
    jobs.advance(job, "Delivering to datacenter")
    delivery_note = await deliver(record, outcome, started_at, job.owner)

    notes = list(outcome.notes)
    if delivery_note:
        notes.append(delivery_note)
    jobs.finish(job, record["id"], notes)
