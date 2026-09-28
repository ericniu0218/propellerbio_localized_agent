"""Jobs that survive a restart, and records that do not outlive their purpose.

The durability tests exist because of one failure: a 45-minute transcription
that dies at minute 40 used to be gone, with the audio still sitting on disk
and no way to ask for it again.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from bedrock.server.meetings.job_store import PersistedJob, RecordingJobStore
from bedrock.server.meetings.jobs import RecordingJob, RecordingJobs
from bedrock.server.meetings.retention import expired_meetings, sweep_audio


def a_job(job_id: str = "j1", status: str = "queued") -> PersistedJob:
    return PersistedJob(
        id=job_id, owner="maya", title="KRAS weekly",
        audio_path="/tmp/a.webm", work_dir="/tmp/a.work",
        status=status, stage="Queued",
        started_at="2026-09-24T14:00:00+00:00",
        project_id="proj-1", typed_notes="enrollment", expected_speakers=3,
        consent={"confirmed": True}, access_groups=["project:proj-1"],
    )


# --- Surviving a restart ------------------------------------------------------


def test_everything_needed_to_run_a_job_again_is_written_down(tmp_path):
    store = RecordingJobStore(tmp_path / "jobs.sqlite")
    store.save(a_job())
    (recovered,) = store.unfinished()
    assert recovered.title == "KRAS weekly"
    assert recovered.project_id == "proj-1"
    assert recovered.expected_speakers == 3
    assert recovered.consent == {"confirmed": True}
    assert recovered.access_groups == ["project:proj-1"]


def test_a_finished_job_is_forgotten(tmp_path):
    """The meeting is the record now; the job has nothing left to say."""
    store = RecordingJobStore(tmp_path / "jobs.sqlite")
    store.save(a_job())
    store.forget("j1")
    assert store.unfinished() == []


def test_a_failed_job_is_kept_because_a_retry_needs_it(tmp_path):
    store = RecordingJobStore(tmp_path / "jobs.sqlite")
    store.save(a_job())
    store.update_status("j1", "failed", "Failed")
    assert store.unfinished() == []  # not picked up automatically...
    assert store.claim("j1") is not None  # ...but still there to retry


def test_claiming_a_job_counts_the_attempt(tmp_path):
    store = RecordingJobStore(tmp_path / "jobs.sqlite")
    store.save(a_job())
    assert store.claim("j1", max_attempts=3).attempts == 1
    assert store.claim("j1", max_attempts=3).attempts == 2


def test_a_job_that_keeps_killing_the_server_is_abandoned(tmp_path):
    """Three goes at audio that reliably crashes is enough to conclude the
    audio is the problem. A crash loop on startup is worse than one lost job."""
    store = RecordingJobStore(tmp_path / "jobs.sqlite")
    store.save(a_job())
    for _ in range(3):
        store.claim("j1", max_attempts=3)
    assert store.claim("j1", max_attempts=3) is None
    assert store.unfinished() == []


def test_claiming_something_that_does_not_exist_is_not_an_error(tmp_path):
    assert RecordingJobStore(tmp_path / "jobs.sqlite").claim("nope") is None


def test_progress_is_persisted_as_it_happens(tmp_path):
    """So a restart can tell the difference between queued and nearly done."""
    store = RecordingJobStore(tmp_path / "jobs.sqlite")
    store.save(a_job())
    jobs = RecordingJobs(store)
    job = RecordingJob(id="j1", owner="maya", title="KRAS weekly",
                       audio_path=tmp_path / "a.webm", work_dir=tmp_path / "w")
    jobs.register(job)
    jobs.advance(job, "Transcribing speech")
    assert store.unfinished()[0].stage == "Transcribing speech"


def test_jobs_work_without_a_store_at_all(tmp_path):
    """The store is optional, which is what keeps the tests simple."""
    jobs = RecordingJobs()
    job = RecordingJob(id="j1", owner="maya", title="x",
                       audio_path=tmp_path / "a", work_dir=tmp_path / "w")
    jobs.register(job)
    jobs.advance(job, "Transcribing speech")
    assert jobs.get("maya", "j1").stage == "Transcribing speech"


# --- Retention ----------------------------------------------------------------


def test_old_audio_is_deleted_and_recent_audio_is_not(tmp_path):
    old, new = tmp_path / "old.webm", tmp_path / "new.webm"
    old.write_bytes(b"x" * 1000)
    new.write_bytes(b"y" * 1000)
    long_ago = time.time() - 60 * 86_400
    os.utime(old, (long_ago, long_ago))

    report = sweep_audio(tmp_path, days=30)
    assert report.audio_deleted == 1
    assert not old.exists()
    assert new.exists()


def test_zero_days_means_keep_for_ever(tmp_path):
    path = tmp_path / "old.webm"
    path.write_bytes(b"x")
    long_ago = time.time() - 9999 * 86_400
    os.utime(path, (long_ago, long_ago))
    assert sweep_audio(tmp_path, days=0).audio_deleted == 0
    assert path.exists()


def test_only_audio_is_swept(tmp_path):
    """The transcript lives in the workspace, not on this disk, and the two
    clocks are separate on purpose."""
    keep = tmp_path / "notes.json"
    keep.write_text("{}")
    long_ago = time.time() - 60 * 86_400
    os.utime(keep, (long_ago, long_ago))
    assert sweep_audio(tmp_path, days=30).audio_deleted == 0
    assert keep.exists()


def test_a_missing_directory_is_not_an_error(tmp_path):
    assert sweep_audio(tmp_path / "nothing-here", days=30).audio_deleted == 0


def test_an_expired_meeting_is_reported_not_deleted():
    """The caller owns the deletion, so it can be audited."""
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    meetings = [{"id": "m1", "recordedAt": old}, {"id": "m2", "recordedAt": datetime.now(timezone.utc).isoformat()}]
    assert [m["id"] for m in expired_meetings(meetings, days=365)] == ["m1"]


def test_a_meeting_with_no_date_is_never_expired():
    """Guessing that an undated record is old enough to destroy is not a guess
    worth making."""
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    assert expired_meetings([{"id": "m1"}, {"id": "m2", "recordedAt": old}], days=365) == [
        {"id": "m2", "recordedAt": old}
    ]


def test_transcripts_are_kept_for_ever_by_default():
    old = (datetime.now(timezone.utc) - timedelta(days=9999)).isoformat()
    assert expired_meetings([{"id": "m1", "recordedAt": old}], days=0) == []


def test_an_unparseable_date_is_left_alone():
    assert expired_meetings([{"id": "m1", "recordedAt": "whenever"}], days=1) == []


# --- One at a time ------------------------------------------------------------


async def test_a_second_recording_waits_rather_than_competing(tmp_path):
    """Transcription is CPU-bound and holds the whole recording in memory.
    Two forty-minute meetings running together on a laptop with its swap full
    got slower than either alone and stopped making visible progress. The
    second one now says it is waiting, which is both faster and honest."""
    import asyncio

    jobs = RecordingJobs()
    started, release = [], asyncio.Event()

    async def slow(job):
        started.append(job.id)
        await release.wait()

    made = []
    for name in ("a", "b"):
        job = RecordingJob(id=name, owner="maya", title=name,
                           audio_path=tmp_path / name, work_dir=tmp_path / f"{name}.w")
        jobs.register(job)
        jobs.start(job, slow)
        made.append(job)

    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert started == ["a"], "the second should not have begun"
    assert "Waiting" in made[1].stage

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert started == ["a", "b"], "the second should run once the first is done"
