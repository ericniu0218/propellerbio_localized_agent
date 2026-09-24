"""The preview that appears while a meeting is still happening.

The preview is explicitly not the transcript. Most of these tests are about
keeping that distinction true, because a preview presented as a record would be
the worst kind of wrong: readable, plausible, and not what the product stands
behind.
"""

from __future__ import annotations

import time

from app.meetings.live import IDLE_TIMEOUT_S, MAX_PREVIEW_CHARS, LiveSessions


def test_a_preview_accumulates_clip_by_clip():
    live = LiveSessions()
    session = live.open("s1", "maya")
    live.append(session, "Good morning everyone.")
    live.append(session, "Enrollment is 148 of 200.")
    assert session.text == "Good morning everyone. Enrollment is 148 of 200."
    assert session.clips == 2


def test_a_clip_that_transcribes_to_nothing_leaves_no_gap():
    """Silence between sentences must not become a hole in the middle of one."""
    live = LiveSessions()
    session = live.open("s1", "maya")
    live.append(session, "Good morning.")
    live.append(session, "   ")
    live.append(session, "Enrollment is 148 of 200.")
    assert session.text == "Good morning. Enrollment is 148 of 200."


def test_a_preview_always_says_it_is_provisional():
    """It has no speaker names, no vocabulary repair and no citation checking.
    Anything reading it has to be able to tell."""
    live = LiveSessions()
    payload = live.open("s1", "maya").public()
    assert payload["provisional"] is True
    assert "not corrected" in payload["note"]


def test_preview_text_is_bounded():
    live = LiveSessions()
    session = live.open("s1", "maya")
    for _ in range(400):
        live.append(session, "x" * 1000)
    assert len(session.text) <= MAX_PREVIEW_CHARS


def test_one_persons_preview_is_not_readable_by_another():
    live = LiveSessions()
    live.open("s1", "maya")
    assert live.get("maya", "s1") is not None
    assert live.get("someone-else", "s1") is None


def test_a_failed_clip_is_counted_not_hidden():
    """A meeting where half the preview clips failed is worth knowing about."""
    live = LiveSessions()
    session = live.open("s1", "maya")
    live.note_failure(session)
    assert session.public()["failures"] == 1


def test_closing_a_preview_forgets_it():
    live = LiveSessions()
    live.open("s1", "maya")
    live.close("maya", "s1")
    assert live.get("maya", "s1") is None


def test_closing_someone_elses_preview_does_nothing():
    live = LiveSessions()
    live.open("s1", "maya")
    live.close("intruder", "s1")
    assert live.get("maya", "s1") is not None


def test_an_abandoned_preview_is_swept():
    """A closed laptop should not leave partial transcript text in memory for
    the life of the process."""
    live = LiveSessions()
    stale = live.open("s1", "maya")
    stale.updated_at = time.monotonic() - IDLE_TIMEOUT_S - 1
    live.open("s2", "maya")  # opening sweeps
    assert live.get("maya", "s1") is None
    assert live.get("maya", "s2") is not None


def test_a_fresh_preview_survives_the_sweep():
    live = LiveSessions()
    live.open("s1", "maya")
    live.sweep()
    assert live.get("maya", "s1") is not None
