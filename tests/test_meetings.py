"""Tests for meeting parsing and summarisation.

None of these need a model running. The backend is faked, which is the point of
the InferenceBackend protocol: the summariser never knows what is on the other
end. The prompts themselves are only exercised against a real model by hand -
see docs/MEETINGS.md.
"""

from __future__ import annotations

import datetime
import json

import pytest

from bedrock.inference.base import InferenceError
from bedrock.meetings.summarize import (
    Point,
    _dedupe,
    _repair_timestamp,
    chunk_cues,
    compute_facts,
    summarize_meeting,
)
from bedrock.meetings.transcript import Cue, parse_vtt, speakers, word_count

VTT = """WEBVTT

1
00:00:01.000 --> 00:00:06.000
Alice Chen: We're at 148 of 200. Three weeks behind.

2
00:00:06.500 --> 00:00:12.000
Bob Ruiz: Site four is the problem. Their coordinator left in August.

3
00:00:12.500 --> 00:00:19.000
Alice Chen: Then we're not opening a new site. Bob, send me the numbers by Thursday.

4
00:00:19.500 --> 00:00:25.000
Priya Nair: I'll have the redlined SAP ready by the end of next week.
"""


class FakeBackend:
    """Replays canned replies in order; records the prompts it was given."""

    def __init__(self, replies: list[str] | None = None, error: Exception | None = None):
        self._replies = list(replies or [])
        self._error = error
        self.prompts: list[str] = []

    async def stream_chat(self, model, messages):
        self.prompts.append(messages[0]["content"])
        if self._error:
            raise self._error
        reply = self._replies.pop(0) if self._replies else ""
        for piece in reply.split(" "):
            yield piece + " "

    async def check_connection(self):
        return True

    async def list_models(self):
        return []

    async def close(self):
        return None


def extraction(decisions=(), actions=()) -> str:
    return json.dumps({"decisions": list(decisions), "actions": list(actions)})


# --- parsing -------------------------------------------------------------


def test_parses_cues_speakers_and_timestamps():
    cues = parse_vtt(VTT)
    assert len(cues) == 4
    assert speakers(cues) == ["Alice Chen", "Bob Ruiz", "Priya Nair"]
    assert cues[0].timestamp == "00:00:01"
    assert word_count(cues) > 20


def test_crlf_and_junk_blocks_survive():
    cues = parse_vtt("WEBVTT\n\nNOTE a comment\n\ngarbage\n\n" + VTT.replace("\n", "\r\n"))
    assert len(cues) == 4
    assert "\r" not in cues[0].text


# --- facts, which never involve a model ----------------------------------


def test_facts_are_computed_without_a_model():
    facts = compute_facts(parse_vtt(VTT), title="Sync", meeting_date=datetime.date(2026, 9, 14))
    assert facts.speakers == ["Alice Chen", "Bob Ruiz", "Priya Nair"]
    assert facts.cue_count == 4
    assert facts.meeting_date == "2026-09-14"


def test_chunking_never_splits_an_utterance():
    cues = parse_vtt(VTT)
    chunks = chunk_cues(cues, words_per_chunk=8)
    assert len(chunks) > 1
    assert sum(len(chunk) for chunk in chunks) == len(cues)  # nothing lost
    assert all(chunk for chunk in chunks)  # nothing empty


# --- the hallucination guard ---------------------------------------------


def test_a_timestamp_that_was_never_said_is_not_cited():
    cues = parse_vtt(VTT)
    # 00:47:00 does not exist in a 25-second meeting.
    assert _repair_timestamp("00:47:00", "nothing resembling this transcript", cues) is None


def test_an_invented_timestamp_is_repaired_from_the_quoted_text():
    cues = parse_vtt(VTT)
    repaired = _repair_timestamp("00:47:00", "coordinator left in August", cues)
    assert repaired == "00:00:06"


def test_a_real_timestamp_is_kept_when_that_line_says_it():
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:12", "Not opening a new site", cues) == "00:00:12"


def test_a_real_timestamp_on_the_wrong_line_moves_to_the_line_that_says_it():
    """The failure seen in testing: an action item citing the cue after the one
    where it was agreed. 00:00:19 exists, so it used to pass as verified."""
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:19", "Send the enrollment numbers by Thursday", cues) == "00:00:12"


def test_a_real_timestamp_that_nothing_supports_is_not_cited():
    """This used to be kept, because the time exists. Existing is not enough."""
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:12", "anything", cues) is None


def test_numbers_are_evidence_at_any_length():
    """A fact is usually about a number, and "148" is shorter than four letters."""
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:01", "Enrollment is 148 of 200", cues) == "00:00:01"
    assert _repair_timestamp("00:00:19", "Enrollment is 148 of 200", cues) == "00:00:01"


def test_plurals_and_case_still_match():
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:12", "BOB SENDS THE NUMBER THURSDAY", cues) == "00:00:12"


def test_a_paraphrase_sharing_one_word_with_its_own_line_keeps_its_citation():
    """One shared word is enough to confirm the line the model chose, when no
    other line shares more - so a light paraphrase is not stripped."""
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:01", "Recruitment is behind schedule", cues) == "00:00:01"


def test_one_shared_word_is_not_enough_to_move_a_citation():
    """Moving picks a line out of the whole meeting, which needs two words."""
    cues = parse_vtt(VTT)
    # "behind" is only on 00:00:01, and 00:00:19 shares nothing with this.
    assert _repair_timestamp("00:00:19", "Recruitment is behind schedule", cues) is None


def test_filler_words_are_not_evidence():
    cues = parse_vtt(VTT)
    # "have", "that" and "will" appear in the meeting but prove nothing.
    assert _repair_timestamp("00:00:19", "that will have", cues) is None


def test_a_tie_keeps_the_line_the_model_chose():
    cues = [
        Cue(start_ms=1000, end_ms=2000, speaker="A", text="The budget review is Friday."),
        Cue(start_ms=5000, end_ms=6000, speaker="B", text="Friday works for the budget review."),
    ]
    assert _repair_timestamp("00:00:05", "Budget review on Friday", cues) == "00:00:05"
    assert _repair_timestamp("00:00:01", "Budget review on Friday", cues) == "00:00:01"


def test_two_cues_starting_in_the_same_second_are_both_considered():
    """Separate participant tracks can start cues in the same second."""
    cues = [
        Cue(start_ms=12000, end_ms=13000, speaker="A", text="Sounds good."),
        Cue(start_ms=12400, end_ms=15000, speaker="B", text="We close the Boston site in March."),
    ]
    assert _repair_timestamp("00:00:12", "Boston site closes in March", cues) == "00:00:12"


@pytest.mark.parametrize("written", ["0:00:12", "00:12", "00:00:12.500", " 00:00:12 "])
def test_loosely_formatted_timestamps_are_read(written):
    cues = parse_vtt(VTT)
    assert _repair_timestamp(written, "Not opening a new site", cues) == "00:00:12"


@pytest.mark.parametrize("written", ["soon", "99:99", "00:61:00", "", None, 12])
def test_something_that_is_not_a_time_is_rederived_from_the_words(written):
    cues = parse_vtt(VTT)
    assert _repair_timestamp(written, "Send the enrollment numbers by Thursday", cues) == "00:00:12"


# --- summarising ---------------------------------------------------------


async def test_structured_points_are_checked_and_counted():
    backend = FakeBackend([
        json.dumps({
            "decisions": [{"text": "Not opening a new site", "timestamp": "00:00:12"}],
            "actions": [
                # Cited on Priya's line, six seconds after Alice said it.
                {"text": "Send the enrollment numbers by Thursday", "owner": "Bob Ruiz", "timestamp": "00:00:19"},
                # No line of this meeting says anything like it.
                {"text": "Book the auditorium", "owner": "Priya Nair", "timestamp": "00:00:06"},
                # Left uncited by the model; the guard cites it from its words.
                {"text": "Redlined SAP ready by the end of next week", "owner": "Priya Nair", "timestamp": None},
            ],
        }),
        "A short overview of the meeting.",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")

    assert summary.decisions[0].timestamp == "00:00:12"
    assert [action.timestamp for action in summary.actions] == ["00:00:12", None, "00:00:19"]
    # Three timestamps were supplied; the uncited one is not a model citation.
    assert (summary.citations.checked, summary.citations.moved, summary.citations.removed) == (3, 1, 1)
    assert "[00:00:19]" not in summary.render().split("Send the enrollment numbers")[1].split("\n")[0]


async def test_extracts_decisions_and_actions():
    backend = FakeBackend([
        extraction(
            decisions=[{"text": "Not opening a new site", "timestamp": "00:00:12"}],
            actions=[{"text": "Send enrollment numbers", "owner": "Bob Ruiz", "timestamp": "00:00:12"}],
        ),
        "A short overview of the meeting.",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")

    assert summary.model_used == "fake"
    assert summary.degraded_reason is None
    assert summary.decisions[0].text == "Not opening a new site"
    assert summary.actions[0].owner == "Bob Ruiz"
    assert "## Decisions" in summary.render()


async def test_facts_survive_when_the_model_is_unreachable():
    """The whole point of computing facts separately: inference being down
    must not cost you the meeting."""
    backend = FakeBackend(error=InferenceError("connection refused. Is the inference server running?"))
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")

    assert summary.degraded_reason is not None
    assert summary.model_used is None
    assert summary.facts.speakers == ["Alice Chen", "Bob Ruiz", "Priya Nair"]
    assert "connection refused" in summary.render()


async def test_one_unparseable_chunk_does_not_lose_the_others():
    cues = parse_vtt(VTT)
    chunks = chunk_cues(cues, words_per_chunk=8)
    replies = ["not json at all"] + [
        extraction(decisions=[{"text": f"Decision {i}", "timestamp": "00:00:12"}])
        for i in range(1, len(chunks))
    ] + ["overview"]
    backend = FakeBackend(replies)

    summary = await summarize_meeting(
        cues, backend=backend, model="fake", title="Sync", words_per_chunk=8
    )
    assert summary.decisions  # the surviving chunks still contributed
    assert summary.degraded_reason is None


async def test_json_wrapped_in_a_code_fence_is_still_read():
    backend = FakeBackend([
        "```json\n" + extraction(decisions=[{"text": "Fenced", "timestamp": "00:00:12"}]) + "\n```",
        "overview",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.decisions[0].text == "Fenced"


async def test_notes_become_the_skeleton_of_the_output():
    backend = FakeBackend([extraction(), "## enrollment\n- filled in from the transcript"])
    summary = await summarize_meeting(
        parse_vtt(VTT), backend=backend, model="fake", title="Sync",
        notes="## enrollment\n- site 4?",
    )
    assert summary.enhanced_notes.startswith("## enrollment")
    # The user's own notes must reach the model, or this is just a summary.
    assert any("site 4?" in prompt for prompt in backend.prompts)
    assert "## Overview" not in summary.render()


async def test_citations_in_rewritten_notes_are_checked():
    """The notes mode was the one with no check at all: whatever the model
    wrote in brackets went straight into notes.md."""
    written = "\n".join([
        "## enrollment",
        "- At 148 of 200, three weeks behind [00:00:01]",
        "- Site four lost its coordinator in August [00:00:19]",
        "- Budget was approved [00:47:00].",
        "- [00:00:12] Not opening a new site",
    ])
    backend = FakeBackend([extraction(), written])
    summary = await summarize_meeting(
        parse_vtt(VTT), backend=backend, model="fake", title="Sync",
        notes="## enrollment\n- site 4?",
    )

    assert summary.enhanced_notes.split("\n") == [
        "## enrollment",
        "- At 148 of 200, three weeks behind [00:00:01]",  # right: untouched
        "- Site four lost its coordinator in August [00:00:06]",  # wrong line: moved
        "- Budget was approved.",  # said nowhere: removed, no stray space
        "- [00:00:12] Not opening a new site",  # leads its line: checked against what follows
    ]
    assert (summary.citations.checked, summary.citations.moved, summary.citations.removed) == (4, 1, 1)


async def test_overview_citations_are_checked_too():
    backend = FakeBackend([extraction(), "Enrollment is 148 of 200 (00:00:12). Bob owes numbers by Thursday [00:47:00]."])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")

    assert summary.overview == "Enrollment is 148 of 200 (00:00:01). Bob owes numbers by Thursday [00:00:12]."


async def test_an_empty_transcript_says_so_rather_than_calling_the_model():
    backend = FakeBackend([])
    summary = await summarize_meeting([], backend=backend, model="fake", title="Sync")
    assert summary.degraded_reason == "the transcript was empty"
    assert backend.prompts == []


def test_the_same_decision_from_two_chunks_is_reported_once():
    points = [
        Point(text="Interim analysis moves to March", timestamp="00:00:01"),
        Point(text="The interim analysis is moving to March", timestamp="00:00:12"),
        Point(text="Bob will call the site four PI", timestamp="00:00:19"),
    ]
    assert len(_dedupe(points)) == 2


def test_a_long_meeting_keeps_both_ends_not_just_the_start():
    """Trimming to the first N cues lost the wrap-up, which is where decisions
    and owners actually get stated."""
    from bedrock.meetings.summarize import _for_one_prompt

    cues = [
        Cue(start_ms=i * 1000, end_ms=i * 1000 + 900, speaker="A", text=f"line {i}")
        for i in range(500)
    ]
    rendered = _for_one_prompt(cues, limit=100)

    assert "line 0" in rendered
    assert "line 499" in rendered  # the close survives
    assert "line 250" not in rendered  # the middle is what gets dropped
    assert "omitted from the middle" in rendered


def test_a_short_meeting_is_sent_whole():
    from bedrock.meetings.summarize import _for_one_prompt

    cues = [Cue(start_ms=0, end_ms=900, speaker="A", text="only line")]
    assert "omitted" not in _for_one_prompt(cues, limit=100)


# --- the free-text guard, piece by piece ---------------------------------


def clean(text: str, notes: str = "") -> str:
    from bedrock.meetings.citations import CitationGuard

    return CitationGuard(parse_vtt(VTT)).clean_text(text, leave_alone=notes)


def test_times_the_person_typed_in_their_own_notes_are_left_alone():
    """"[14:00] standup" in their rough notes is a time of day, not a citation."""
    notes = "- [14:00] standup with site four\n- follow up"
    written = "- [14:00] standup with site four\n- Follow up with Bob [00:47:00]"
    assert clean(written, notes) == "- [14:00] standup with site four\n- Follow up with Bob"


def test_a_line_cited_only_in_round_brackets_is_still_checked():
    assert clean("Site four lost its coordinator (00:00:19)") == "Site four lost its coordinator (00:00:06)"
    assert clean("Budget was approved (00:47:00).") == "Budget was approved."


def test_brackets_that_are_not_times_are_left_alone():
    text = "- [x] done\n- [TBD] owner\n- see [the SAP](https://intranet/sap)\n- dated [2026-09-14]\n- at (10:30) daily"
    assert clean(text) == text


def test_each_citation_on_a_line_is_checked_against_its_own_sentence():
    text = "Enrollment is 148 of 200 [00:00:01]. Site four lost its coordinator [00:00:01]."
    assert clean(text) == "Enrollment is 148 of 200 [00:00:01]. Site four lost its coordinator [00:00:06]."


def test_a_citation_after_the_full_stop_belongs_to_the_sentence_before_it():
    text = "Enrollment is 148 of 200. Site four lost its coordinator. [00:00:01]"
    assert clean(text) == "Enrollment is 148 of 200. Site four lost its coordinator. [00:00:06]"


def test_a_semicolon_does_not_cut_a_note_in_half():
    """Found against a realistic meeting: splitting here left only "QA is
    assessing impact", which shares no word with the line, and a correct
    citation was removed."""
    cues = [
        Cue(start_ms=1000, end_ms=5000, speaker="M", text="Freezer three hit minus 62 on Saturday."),
        Cue(start_ms=8000, end_ms=12000, speaker="M", text="Samples were moved to freezer five. QA is reviewing them."),
    ]
    from bedrock.meetings.citations import CitationGuard

    text = "- Samples relocated to freezer five; QA assessing impact [00:00:08]"
    assert CitationGuard(cues).clean_text(text) == text


def test_several_times_in_one_bracket_are_each_checked():
    assert clean("Site four lost its coordinator [00:00:06, 00:47:00]") == "Site four lost its coordinator [00:00:06]"
    # Both endpoints of a range move to the one line that says it, once.
    assert clean("Site four lost its coordinator [00:00:01 - 00:00:19]") == "Site four lost its coordinator [00:00:06]"
    # A note drawing on two lines keeps both, as written, when each says its part.
    assert clean("Coordinator left, not opening a new site [00:00:06 - 00:00:12]") == (
        "Coordinator left, not opening a new site [00:00:06 - 00:00:12]"
    )


def test_a_line_that_says_part_of_a_note_keeps_its_citation():
    """00:00:12 shares two words with this note; 00:00:06 shares three. Both
    say part of it, so the model's choice is not second-guessed."""
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:12", "Coordinator left, not opening a new site", cues) == "00:00:12"


def test_a_correct_loosely_written_citation_keeps_its_spelling():
    assert clean("Not opening a new site [0:12]") == "Not opening a new site [0:12]"


def test_fenced_code_is_never_changed():
    text = "```\nBudget [00:47:00]\n```\nBudget was approved [00:47:00]"
    assert clean(text) == "```\nBudget [00:47:00]\n```\nBudget was approved"


def test_the_report_line_is_plain_and_absent_when_nothing_was_cited():
    from bedrock.meetings.citations import CitationReport

    assert CitationReport().describe() is None
    assert CitationReport(checked=1, moved=0, removed=1).describe() == (
        "Checked 1 citation against the transcript: 0 moved to the line that says it, "
        "1 removed because no line says it."
    )


# --- The model handing back its own JSON skeleton -----------------------------


def test_a_placeholder_owner_becomes_no_owner():
    """llama3.1:8b returned `"owner": "name or null"` verbatim. An action item
    assigned to "name or null" reads as a broken product; no owner is honest."""
    from bedrock.meetings.summarize import _points
    from bedrock.meetings.citations import CitationGuard

    points = _points(
        [{"text": "Send the enrollment tracker tonight.", "owner": "name or null"}],
        CitationGuard([]),
        with_owner=True,
    )
    assert len(points) == 1
    assert points[0].owner is None
    assert "name or null" not in points[0].render()


def test_a_point_that_is_only_a_placeholder_is_dropped():
    """The same run produced a "fact" whose entire text was the placeholder."""
    from bedrock.meetings.summarize import _points
    from bedrock.meetings.citations import CitationGuard

    points = _points(
        [
            {"text": "name or null"},
            {"text": "..."},
            {"text": "<the concrete detail>"},
            {"text": "148 of 200 patients"},
        ],
        CitationGuard([]),
        with_owner=False,
    )
    assert [p.text for p in points] == ["148 of 200 patients"]


def test_a_real_owner_still_survives():
    """The filter must not eat ordinary names."""
    from bedrock.meetings.summarize import _points
    from bedrock.meetings.citations import CitationGuard

    points = _points(
        [{"text": "Update the protocol document.", "owner": "Priya"}],
        CitationGuard([]),
        with_owner=True,
    )
    assert points[0].owner == "Priya"


def test_a_first_person_owner_becomes_no_owner():
    """"I will send the tracker" names nobody, and an in-room recording has no
    speaker labels to resolve the pronoun against."""
    from bedrock.meetings.summarize import _points
    from bedrock.meetings.citations import CitationGuard

    points = _points(
        [{"text": "Send the enrollment tracker tonight.", "owner": "I"}],
        CitationGuard([]),
        with_owner=True,
    )
    assert points[0].owner is None


# --- Cue hygiene and speaker-aware slicing ------------------------------------


def test_a_cue_of_pure_punctuation_is_not_speech():
    """Given room noise whisper emits a cue whose whole text is ",". Two of
    them appeared in one real recording, spanning eighteen and four seconds,
    and read like the recorder had broken."""
    from bedrock.meetings.transcribe import is_speech

    assert not is_speech(",")
    assert not is_speech(" . ")
    assert not is_speech("[BLANK_AUDIO]")
    assert is_speech("Good morning.")


def test_leading_punctuation_is_trimmed_from_a_cue():
    """Slicing on a speech boundary leaves whisper starting a cue with the
    punctuation that ended the previous one."""
    from bedrock.meetings.transcribe import clean_cue_text

    assert clean_cue_text(", and I'm also going to start the project") == (
        "and I'm also going to start the project"
    )
    assert clean_cue_text("  Good morning.  ") == "Good morning."


def test_words_inside_a_cue_are_never_touched():
    from bedrock.meetings.transcribe import clean_cue_text

    assert clean_cue_text("148 of 200, and rising") == "148 of 200, and rising"


def test_a_turn_is_cut_where_the_voice_changes():
    """Two people talking without a gap are one VAD region, so a caller that
    knows where the speaker changes has to be able to force the cut."""
    from bedrock.meetings.transcribe import _split_turns_at

    assert _split_turns_at([(0, 10000)], [4000]) == [(0, 4000), (4000, 10000)]


def test_a_cut_that_would_leave_a_fragment_is_skipped():
    """A boundary a tenth of a second inside a turn would hand whisper half a
    syllable, which transcribes worse than leaving both speakers together."""
    from bedrock.meetings.transcribe import _split_turns_at

    assert _split_turns_at([(0, 10000)], [100, 9950]) == [(0, 10000)]


def test_a_boundary_outside_a_turn_changes_nothing():
    from bedrock.meetings.transcribe import _split_turns_at

    assert _split_turns_at([(0, 5000)], [8000]) == [(0, 5000)]


def test_several_cuts_in_one_turn_all_apply():
    from bedrock.meetings.transcribe import _split_turns_at

    assert _split_turns_at([(0, 12000)], [4000, 8000]) == [(0, 4000), (4000, 8000), (8000, 12000)]


def test_a_cut_into_near_silence_is_refused():
    """Speaker boundaries and speech boundaries come from two different models
    and do not agree, so a boundary regularly lands mid-silence. Cutting there
    hands whisper a slice with no words in it, and what comes back is not
    silence but invention — on a twelve-turn recording this turned 12 cues into
    21, of which 8 were hallucinated filler."""
    from bedrock.meetings.transcribe import _split_turns_at

    # Speech on both sides of the cut: a legitimate speaker change, so cut.
    regions = [(0, 4000), (9000, 13000)]
    assert _split_turns_at([(0, 13000)], [6500], regions) == [(0, 6500), (6500, 13000)]

    # Now a turn whose first side holds almost no speech. Cutting there hands
    # whisper a slice of silence, and it fills it in rather than returning
    # nothing, so the cut is refused and both speakers stay in one cue.
    sparse = [(0, 200), (9000, 13000)]
    assert _split_turns_at([(0, 13000)], [6500], sparse) == [(0, 13000)]


def test_without_regions_the_speech_check_is_skipped():
    """Callers that have no VAD data still get the length guard."""
    from bedrock.meetings.transcribe import _split_turns_at

    assert _split_turns_at([(0, 10000)], [5000]) == [(0, 5000), (5000, 10000)]
