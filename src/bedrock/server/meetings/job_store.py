"""Recording jobs that survive a restart.

Transcription is the only thing in this product measured in tens of minutes,
and until now the jobs tracking it lived only in one process's memory. A
restart — a deploy, a crash, somebody's laptop sleeping — lost the job. The
audio was still on disk, so nothing was destroyed, but the person who recorded
saw a meeting that never appeared and had no way to ask for it again. On a
45-minute meeting that fails at minute 40, "re-record it" is not an answer.

So everything needed to run a job again is written down before the work
starts, and cleared when it finishes. On startup, anything still marked
unfinished is picked up and run.

What is *not* here is a queue. This is one row per job in the same SQLite the
rest of the workspace uses, which is right while this runs on one box and
wrong the moment it runs on two — two processes would both claim the same
interrupted job. `claim()` exists to make that boundary explicit rather than
implicit: it is the one place that would need a real queue's semantics.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Statuses that mean "this job still has work to do".
UNFINISHED = ("queued", "running")


@dataclass
class PersistedJob:
    """Everything needed to run one recording again from the beginning.

    Deliberately re-runs from the audio rather than resuming mid-whisper.
    Whisper has no checkpoint to resume from, and a partially written
    transcript is worse than one that takes another fifteen minutes: the first
    looks complete.
    """

    id: str
    owner: str
    title: str
    audio_path: str
    work_dir: str
    status: str
    stage: str
    started_at: str
    project_id: str | None = None
    typed_notes: str | None = None
    expected_speakers: int | None = None
    consent: dict | None = None
    access_groups: list[str] | None = None
    language: str | None = None
    """The language chosen in the Record dialog, kept so a job resumed after a
    restart is transcribed in the language it was recorded in rather than the
    server's default. None means the server default, as before."""
    auto_title: bool = False
    """Nobody named this meeting, so it is named from its notes."""
    attempts: int = 0


class RecordingJobStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS recording_jobs (
                       id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                       audio_path TEXT NOT NULL, work_dir TEXT NOT NULL,
                       status TEXT NOT NULL, stage TEXT NOT NULL,
                       started_at TEXT NOT NULL, payload TEXT NOT NULL,
                       attempts INTEGER NOT NULL DEFAULT 0,
                       updated TEXT NOT NULL)"""
            )

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    def save(self, job: PersistedJob) -> None:
        payload = json.dumps(
            {
                "project_id": job.project_id,
                "typed_notes": job.typed_notes,
                "expected_speakers": job.expected_speakers,
                "consent": job.consent,
                "access_groups": job.access_groups,
                "language": job.language,
                "auto_title": job.auto_title,
            }
        )
        with self._db() as db:
            db.execute(
                """INSERT OR REPLACE INTO recording_jobs
                   (id,owner,title,audio_path,work_dir,status,stage,started_at,payload,attempts,updated)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (job.id, job.owner, job.title, job.audio_path, job.work_dir, job.status,
                 job.stage, job.started_at, payload, job.attempts,
                 datetime.now(timezone.utc).isoformat()),
            )

    def update_status(self, job_id: str, status: str, stage: str) -> None:
        with self._db() as db:
            db.execute(
                "UPDATE recording_jobs SET status=?, stage=?, updated=? WHERE id=?",
                (status, stage, datetime.now(timezone.utc).isoformat(), job_id),
            )

    def forget(self, job_id: str) -> None:
        """A finished job needs nothing kept — the meeting is the record now."""
        with self._db() as db:
            db.execute("DELETE FROM recording_jobs WHERE id=?", (job_id,))

    def unfinished(self) -> list[PersistedJob]:
        with self._db() as db:
            rows = db.execute(
                f"SELECT * FROM recording_jobs WHERE status IN ({','.join('?' * len(UNFINISHED))})"
                " ORDER BY updated",
                UNFINISHED,
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def claim(self, job_id: str, max_attempts: int = 3) -> PersistedJob | None:
        """Take ownership of an interrupted job, once.

        Increments the attempt count in the same statement that checks it, so a
        job that crashes the process cannot be retried for ever — three goes at
        audio that reliably kills the server is enough to conclude the audio is
        the problem, and a crash loop on startup is worse than one lost job.
        """
        with self._db() as db:
            row = db.execute("SELECT * FROM recording_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["attempts"] >= max_attempts:
                if row is not None:
                    db.execute(
                        "UPDATE recording_jobs SET status='failed', stage=? WHERE id=?",
                        ("Abandoned after repeated failures", job_id),
                    )
                return None
            db.execute(
                "UPDATE recording_jobs SET attempts=attempts+1, updated=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), job_id),
            )
        job = self._from_row(row)
        job.attempts += 1
        return job

    @staticmethod
    def _from_row(row: sqlite3.Row) -> PersistedJob:
        payload = json.loads(row["payload"])
        return PersistedJob(
            id=row["id"], owner=row["owner"], title=row["title"],
            audio_path=row["audio_path"], work_dir=row["work_dir"],
            status=row["status"], stage=row["stage"], started_at=row["started_at"],
            attempts=row["attempts"], **payload,
        )
