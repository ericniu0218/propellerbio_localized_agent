"""Finding a meeting by what it meant, not only by the words it used.

The first test is the whole point: "recruitment" against a meeting that only
ever says "enrollment" scored zero for this tool's entire life.
"""

from __future__ import annotations

import pytest

from app.meetings.semantic import MIN_SEMANTIC_MATCH, enabled, load, similarities
from app.tools.datacenter_client import MeetingTranscript
from app.tools.meeting_transcripts import rank_excerpts

TRANSCRIPT = """WEBVTT

1
00:00:01.000 --> 00:00:06.000
Alice: Let's start with the enrollment numbers. We are at 148 of 200 patients.

2
00:00:06.000 --> 00:00:12.000
Bob: The site coordinator in Boston left last week, so we are not opening a new site.

3
00:00:12.000 --> 00:00:18.000
Cara: I will revise the study document by Friday and send the tracker tonight.
"""

OTHER = """WEBVTT

1
00:00:01.000 --> 00:00:06.000
Dan: The assay plate needs another run before we can read the compound results.
"""


def meeting(transcript: str, topic: str, meeting_id: str) -> MeetingTranscript:
    return MeetingTranscript(
        meeting_id=meeting_id, topic=topic, host_email="maya",
        start_time=1_790_000_000_000, end_time=1_790_000_600_000, transcript=transcript,
    )


needs_model = pytest.mark.skipif(
    load() is None, reason="no text-embedding model available in this environment"
)


@needs_model
def test_a_synonym_finds_the_meeting():
    """The gap this closes. "Recruitment" and "enrollment" are the same thing
    to a clinical team, and share no words at all."""
    excerpts, matched = rank_excerpts(
        "what did we decide about recruitment?", [meeting(TRANSCRIPT, "Weekly", "m1")], top_k=3
    )
    assert matched is True
    assert "148 of 200" in excerpts[0].render()


@needs_model
def test_the_right_meeting_wins_when_two_are_offered():
    excerpts, matched = rank_excerpts(
        "how many people have we recruited?",
        [meeting(OTHER, "Chemistry", "m2"), meeting(TRANSCRIPT, "Weekly", "m1")],
        top_k=1,
    )
    assert matched is True
    assert excerpts[0].meeting.meeting_id == "m1"


@needs_model
def test_an_unrelated_question_still_reports_no_match():
    """The property that stops an agent presenting a travel-budget meeting as
    an answer. Semantic similarity is never zero, so this needs a floor."""
    _excerpts, matched = rank_excerpts(
        "quarterly travel budget", [meeting(TRANSCRIPT, "Weekly", "m1")], top_k=3
    )
    assert matched is False


@needs_model
def test_a_question_of_only_stopwords_matches_nothing():
    """It says nothing specific, but embeds to a perfectly ordinary vector."""
    _excerpts, matched = rank_excerpts(
        "what did we do about it", [meeting(TRANSCRIPT, "Weekly", "m1")], top_k=3
    )
    assert matched is False


@needs_model
def test_an_exact_term_still_wins_on_keywords():
    """Meaning must not blur G12C into G12D. Keyword overlap survives the
    hybrid, which is why the two scores are added rather than replaced."""
    excerpts, matched = rank_excerpts(
        "Boston site coordinator", [meeting(TRANSCRIPT, "Weekly", "m1")], top_k=3
    )
    assert matched is True
    assert "Boston" in excerpts[0].render()


@needs_model
def test_similarity_is_a_cosine_in_the_expected_range():
    model = load()
    (same,) = similarities(model, "enrollment numbers", ["enrollment numbers"])
    (different,) = similarities(model, "enrollment numbers", ["the assay plate run"])
    assert 0.9 <= same <= 1.01
    assert different < same


def test_ranking_still_works_with_no_model(monkeypatch):
    """Without an embedding model this falls back to keyword overlap — what it
    always did, which was never wrong, only narrow."""
    monkeypatch.setenv("MEETING_SEMANTIC", "false")
    assert enabled() is False
    excerpts, matched = rank_excerpts(
        "enrollment", [meeting(TRANSCRIPT, "Weekly", "m1")], top_k=3
    )
    assert matched is True
    assert "148 of 200" in excerpts[0].render()


def test_the_threshold_is_a_deliberate_choice_to_reject():
    """Documented rather than asserted-away: a miss is honest, a false positive
    hands somebody the wrong meeting and calls it an answer."""
    assert MIN_SEMANTIC_MATCH > 0.124  # an unrelated question measured this high
