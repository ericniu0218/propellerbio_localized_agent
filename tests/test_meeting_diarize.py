"""Telling voices apart on a single mixed microphone.

`flatten` is where the bug was, and it is the reason most of these exist. The
diariser reports simultaneous speech as overlapping turns; resolving those the
wrong way round collapsed a real two-person recording onto one speaker, because
the long turn covering the monologue swallowed every interjection inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

from bedrock.meetings.diarize import (
    DiarizationResult,
    SpeakerTurn,
    assign,
    consolidate,
    flatten,
    label_for,
    models_present,
    speaker_change_points,
)


@dataclass(frozen=True)
class FakeCue:
    start_ms: int
    end_ms: int
    speaker: str | None = None
    text: str = "words"


def result(*turns: SpeakerTurn) -> DiarizationResult:
    return DiarizationResult(
        turns=list(turns), speaker_count=len({t.speaker for t in turns}), estimated=False
    )


# --- Resolving overlapping turns ----------------------------------------------


def test_an_interjection_inside_a_long_turn_survives():
    """The exact shape that broke it: S0 talks for eight seconds, S1 cuts in for
    two. Resolving longest-first deletes S1 entirely."""
    timeline = flatten([SpeakerTurn(250, 8700, 0), SpeakerTurn(3560, 5250, 1)])
    owners = {(t.start_ms, t.end_ms): t.speaker for t in timeline}
    assert owners == {(250, 3560): 0, (3560, 5250): 1, (5250, 8700): 0}


def test_the_resolved_timeline_never_overlaps_itself():
    timeline = flatten(
        [SpeakerTurn(0, 10000, 0), SpeakerTurn(2000, 3000, 1), SpeakerTurn(2500, 4000, 2)]
    )
    for earlier, later in zip(timeline, timeline[1:]):
        assert earlier.end_ms <= later.start_ms


def test_turns_that_do_not_overlap_are_left_alone():
    turns = [SpeakerTurn(0, 1000, 0), SpeakerTurn(2000, 3000, 1)]
    assert flatten(turns) == turns


def test_flatten_of_nothing_is_nothing():
    assert flatten([]) == []


# --- Attaching speakers to cues -----------------------------------------------


def test_a_cue_gets_the_speaker_who_holds_most_of_it():
    cues = [FakeCue(0, 3000), FakeCue(3600, 5200)]
    labelled, counts = assign(cues, result(SpeakerTurn(250, 8700, 0), SpeakerTurn(3560, 5250, 1)))
    assert labelled[0].speaker == "Speaker 1"
    assert labelled[1].speaker == "Speaker 2"
    assert counts == {0: 1, 1: 1}


def test_the_question_and_its_answer_land_on_different_people():
    """"What's your name?" / "My name's Luke" — the case that was wrong."""
    cues = [FakeCue(69000, 71000), FakeCue(71160, 72190)]
    labelled, _ = assign(cues, result(SpeakerTurn(53660, 74030, 0), SpeakerTurn(71160, 72190, 1)))
    assert labelled[0].speaker != labelled[1].speaker


def test_a_cue_nobody_was_speaking_over_keeps_no_speaker():
    cues = [FakeCue(20000, 21000)]
    labelled, counts = assign(cues, result(SpeakerTurn(0, 5000, 0)))
    assert labelled[0].speaker is None
    assert counts == {}


def test_a_failed_diarisation_leaves_every_cue_untouched():
    """A meeting whose voices could not be told apart is still a meeting."""
    cues = [FakeCue(0, 1000)]
    labelled, counts = assign(cues, DiarizationResult([], 0, False, "models missing"))
    assert labelled == cues
    assert counts == {}


# --- Labels -------------------------------------------------------------------


def test_speaker_labels_are_one_based():
    """"Speaker 0" reads like a bug to everyone who is not a programmer."""
    assert label_for(0) == "Speaker 1"
    assert label_for(3) == "Speaker 4"


# --- Readiness ----------------------------------------------------------------


def test_missing_models_are_both_reported_at_once(tmp_path):
    missing = models_present(tmp_path / "seg.onnx", tmp_path / "emb.onnx")
    assert len(missing) == 2
    assert any("segmentation" in m for m in missing)
    assert any("embedding" in m for m in missing)


# --- Cutting only where it matters --------------------------------------------


def test_only_speaker_changes_become_cut_points():
    """Cutting at every turn boundary fragments the audio needlessly. Whisper
    pads each slice to its 30-second window, so a one-second slice costs what a
    thirty-second one does: on an 18-minute recording, cutting everywhere made
    440 slices and took 1011s, against 209s cutting only on voice changes."""
    turns = [SpeakerTurn(0, 5000, 0), SpeakerTurn(5000, 9000, 0), SpeakerTurn(9000, 12000, 1)]
    assert speaker_change_points(result(*turns)) == [9000]


def test_a_cut_lands_midway_through_the_gap():
    """So neither speaker is clipped, and the cut falls in silence if there is any."""
    assert speaker_change_points(result(SpeakerTurn(0, 4000, 0), SpeakerTurn(5000, 9000, 1))) == [4500]


def test_one_speaker_throughout_needs_no_cuts_at_all():
    turns = [SpeakerTurn(0, 5000, 0), SpeakerTurn(6000, 9000, 0)]
    assert speaker_change_points(result(*turns)) == []


def test_no_diarisation_means_no_cuts():
    assert speaker_change_points(DiarizationResult([], 0, False, "unavailable")) == []


# --- Working out the speaker count without being told -------------------------
#
# This is what makes "how many people are here?" a hint instead of a
# requirement. Raw clustering splintered a genuine three-person call into 17
# speakers; you do not always know who is coming to a club meeting.


def vectors_for(groups: dict[int, list[float]]) -> dict[int, list[float]]:
    return groups


def long_turns(*speakers: int) -> list[SpeakerTurn]:
    """One ten-second turn each, so nothing is dropped as noise."""
    return [SpeakerTurn(i * 10_000, i * 10_000 + 10_000, s) for i, s in enumerate(speakers)]


A = [1.0, 0.0, 0.0]
A_ISH = [0.97, 0.24, 0.0]      # same voice, different stretch of speech
B = [0.0, 1.0, 0.0]            # clearly someone else


def test_one_voice_split_in_two_is_put_back_together():
    result = DiarizationResult(long_turns(0, 1), 2, estimated=True)
    merged = consolidate(result, {0: A, 1: A_ISH})
    assert merged.speaker_count == 1


def test_two_different_voices_are_left_alone():
    result = DiarizationResult(long_turns(0, 1), 2, estimated=True)
    assert consolidate(result, {0: A, 1: B}).speaker_count == 2


def test_a_cluster_too_small_to_be_a_person_is_absorbed():
    """Half a second of a door closing is not a speaker."""
    turns = [SpeakerTurn(0, 10_000, 0), SpeakerTurn(10_000, 10_400, 1)]
    merged = consolidate(DiarizationResult(turns, 2, estimated=True), {0: A, 1: B})
    assert merged.speaker_count == 1


def test_a_minimum_stops_it_merging_below_what_you_know():
    """"At least five people" for a club meeting nobody counted."""
    result = DiarizationResult(long_turns(0, 1, 2), 3, estimated=True)
    vectors = {0: A, 1: A_ISH, 2: [0.99, 0.14, 0.0]}  # all one voice
    assert consolidate(result, vectors).speaker_count == 1
    assert consolidate(result, vectors, minimum_speakers=3).speaker_count == 3


def test_a_minimum_never_invents_speakers_that_are_not_there():
    """A floor stops merging; it cannot split one cluster into more."""
    result = DiarizationResult(long_turns(0), 1, estimated=True)
    assert consolidate(result, {0: A}, minimum_speakers=5).speaker_count == 1


def test_every_turn_survives_consolidation():
    """Merging relabels speech; it must never drop any."""
    result = DiarizationResult(long_turns(0, 1, 2), 3, estimated=True)
    merged = consolidate(result, {0: A, 1: A_ISH, 2: B})
    assert len(merged.turns) == len(result.turns)
    assert [t.start_ms for t in merged.turns] == [t.start_ms for t in result.turns]


def test_consolidation_without_vectors_changes_nothing():
    result = DiarizationResult(long_turns(0, 1), 2, estimated=True)
    assert consolidate(result, {}) is result


def test_consolidation_of_a_failed_diarisation_changes_nothing():
    empty = DiarizationResult([], 0, False, "models missing")
    assert consolidate(empty, {0: A}) is empty


def test_the_floor_on_being_a_speaker_scales_with_the_meeting():
    """Two seconds is a sensible floor in a two-minute recording and nothing at
    all in a forty-minute one. A real club meeting consolidated to 88
    "speakers" — one cluster holding two thirds of the speech and eighty-seven
    fragments around it — because one laptop microphone in a room gives the
    same person very different embeddings depending on where they sat. With a
    proportional floor the same recording gives 5."""
    # One speaker talking for ten minutes, plus a three-second fragment that is
    # comfortably over the absolute floor but a rounding error in the meeting.
    turns = [SpeakerTurn(0, 600_000, 0), SpeakerTurn(600_000, 603_000, 1)]
    result = DiarizationResult(turns, 2, estimated=True)
    assert consolidate(result, {0: A, 1: B}, min_share=0.01).speaker_count == 1


def test_a_short_meeting_is_judged_on_the_absolute_floor():
    """The proportional floor must not swallow a real speaker in a short
    recording: 1% of ninety seconds is under a second."""
    turns = [SpeakerTurn(0, 60_000, 0), SpeakerTurn(60_000, 90_000, 1)]
    result = DiarizationResult(turns, 2, estimated=True)
    assert consolidate(result, {0: A, 1: B}, min_share=0.01).speaker_count == 2
