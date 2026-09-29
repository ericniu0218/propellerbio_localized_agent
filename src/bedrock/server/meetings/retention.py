"""Deleting meetings that have outlived their purpose.

Audio and transcripts accumulated for ever, in two places at once: raw
recordings under `MEETING_RECORDING_DIR` and transcripts in the datacenter.
Nothing pruned either. For a company whose meetings discuss patients and
unpublished results, "we keep everything indefinitely" is a decision somebody
should make on purpose, and until now it was being made by omission.

Redaction made this sharper rather than softer. The stored transcript has
identifiers removed; the **audio does not**, because destroying the only
record of what was said in order to protect it is the worse mistake, and
re-transcribing is how a wrong redaction gets fixed. That means the audio is
now the most sensitive thing this feature holds, and the shortest clock here is
the one on the audio.

Two separate clocks, because they protect different things:

* `MEETING_AUDIO_RETENTION_DAYS` — how long the recording itself is kept.
  Default 30. Once it passes, the audio is deleted and the meeting, its
  transcript and its notes all remain. Almost nobody needs the audio of a
  meeting from last month; everybody needs to be able to search it.
* `MEETING_TRANSCRIPT_RETENTION_DAYS` — how long the meeting itself is kept.
  Default 0, meaning never delete. A transcript is the institutional memory
  this product exists to build, so deleting one is opt-in in a way deleting
  audio is not.

Nothing here runs on a timer. It is a function the deployment calls — from the
startup hook, a cron, or an admin endpoint — because a background thread that
silently deletes company records is not something to switch on by importing a
module.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_DAYS = 30
DEFAULT_TRANSCRIPT_DAYS = 0  # never

# Suffixes the recorder writes. Scratch directories (`*.work`) are already
# removed when a job finishes; this is the audio that is kept on purpose.
AUDIO_SUFFIXES = {".webm", ".ogg", ".oga", ".opus", ".m4a", ".mp4", ".wav", ".mp3", ".aac"}


def audio_retention_days() -> int:
    return int(os.environ.get("MEETING_AUDIO_RETENTION_DAYS", DEFAULT_AUDIO_DAYS))


def transcript_retention_days() -> int:
    return int(os.environ.get("MEETING_TRANSCRIPT_RETENTION_DAYS", DEFAULT_TRANSCRIPT_DAYS))


@dataclass
class SweepReport:
    audio_deleted: int = 0
    audio_bytes: int = 0
    meetings_deleted: int = 0
    kept: int = 0
    errors: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = []
        if self.audio_deleted:
            parts.append(f"{self.audio_deleted} recording(s), {self.audio_bytes // 1_000_000} MB")
        if self.meetings_deleted:
            parts.append(f"{self.meetings_deleted} meeting(s)")
        return "deleted " + ", ".join(parts) if parts else "nothing to delete"


def _age_days(moment: datetime) -> float:
    return (datetime.now(timezone.utc) - moment).total_seconds() / 86_400


def sweep_audio(recordings_dir: Path, days: int | None = None) -> SweepReport:
    """Delete recordings older than the audio retention window.

    Only files are removed; the meeting, its transcript and its notes stay. A
    meeting whose audio has been aged out is still searchable and still
    readable, which is the point of separating the two clocks.
    """
    report = SweepReport()
    days = audio_retention_days() if days is None else days
    if days <= 0:
        return report  # 0 means keep for ever, and is a real answer

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    root = Path(recordings_dir)
    if not root.exists():
        return report

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError as exc:  # pragma: no cover - racing with another sweep
            report.errors.append(f"{path.name}: {exc}")
            continue
        if modified > cutoff:
            report.kept += 1
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            report.audio_deleted += 1
            report.audio_bytes += size
        except OSError as exc:
            report.errors.append(f"{path.name}: {exc}")
    return report


def expired_meetings(meetings: list[dict], days: int | None = None) -> list[dict]:
    """Meetings past the transcript retention window.

    Returns them rather than deleting them, so the caller owns the deletion and
    can audit it. A meeting with no recorded date is never expired: guessing
    that an undated record is old enough to destroy is not a guess worth making.
    """
    days = transcript_retention_days() if days is None else days
    if days <= 0:
        return []
    expired = []
    for meeting in meetings:
        stamp = meeting.get("recordedAt")
        if not stamp:
            continue
        try:
            recorded = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        except ValueError:
            continue
        if recorded.tzinfo is None:
            recorded = recorded.replace(tzinfo=timezone.utc)
        if _age_days(recorded) > days:
            expired.append(meeting)
    return expired


def sweep(recordings_dir: Path, store=None, owners: list[str] | None = None) -> SweepReport:
    """One pass of both clocks.

    `store` and `owners` are optional: without them only the audio clock runs,
    which is the useful half and needs nothing but a directory.
    """
    report = sweep_audio(recordings_dir)

    if store is not None and transcript_retention_days() > 0:
        for owner in owners or []:
            for meeting in expired_meetings(store.list(owner, "meetings")):
                try:
                    store.delete(owner, meeting["id"])
                    store.audit(owner, "deleted an expired meeting", meeting["id"])
                    report.meetings_deleted += 1
                except Exception as exc:  # pragma: no cover
                    report.errors.append(f"{meeting.get('id')}: {exc}")

    if report.audio_deleted or report.meetings_deleted:
        logger.info("Meeting retention sweep %s", report.describe())
    return report
