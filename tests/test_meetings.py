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
from bedrock.meetings.citations import resolve_timestamp
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
        self.options: list = []

    async def stream_chat(self, model, messages, options=None, on_usage=None):
        self.prompts.append(messages[0]["content"])
        self.options.append(options)
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


def notes_reply(markdown: str, citations=()) -> str:
    """The JSON envelope the notes prompt asks for."""
    return json.dumps({"markdown": markdown, "citations": list(citations)})


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


# --- the citation guard ---------------------------------------------------
#
# A timestamp is a claim that particular words were said at a particular
# moment. These tests hold the line that the claim is checked against the
# words, not merely against the list of timestamps that exist.


def test_a_real_timestamp_on_the_wrong_cue_is_repaired():
    """The bug this guard exists for.

    00:00:12 is a real moment in this meeting, and the loose check accepted it
    for that reason alone. But the quote comes from the cue six seconds
    earlier, so 00:00:06 is the honest citation.
    """
    cues = [
        Cue(6000, 11000, "Bob", "Their coordinator left in August."),
        Cue(12000, 18000, "Alice", "We are not opening a new site."),
    ]
    assert _repair_timestamp("00:00:12", "coordinator left in August", cues) == "00:00:06"


def test_a_real_timestamp_is_kept_when_its_cue_holds_the_quote():
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:12", "not opening a new site", cues) == "00:00:12"


def test_a_real_timestamp_alone_never_validates_a_citation():
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:12", "anything at all, unrelated", cues) is None


def test_a_timestamp_that_was_never_said_is_not_cited():
    cues = parse_vtt(VTT)
    # 00:47:00 does not exist in a 25-second meeting.
    assert _repair_timestamp("00:47:00", "nothing resembling this transcript", cues) is None


@pytest.mark.parametrize("suggested", ["00:47:00", None, "", "later on", 12, {"at": 1}])
def test_a_unique_quote_repairs_any_suggested_timestamp(suggested):
    cues = parse_vtt(VTT)
    assert resolve_timestamp("coordinator left in August", suggested, cues) == "00:00:06"


def test_word_overlap_alone_is_not_evidence():
    """"Site" and "coordinator" both appear in cue 2, but never as this
    sentence. Loose overlap used to accept exactly this."""
    cues = parse_vtt(VTT)
    assert resolve_timestamp("the site coordinator agreed", None, cues) is None


def test_contradictory_evidence_is_rejected():
    cues = parse_vtt(VTT)
    assert resolve_timestamp("coordinator never left in August", None, cues) is None


@pytest.mark.parametrize(
    "evidence",
    [None, 12, ["coordinator", "left"], "", "   ", "...", " - , . ", "left in", "August"],
)
def test_evidence_that_cannot_locate_anything_is_rejected(evidence):
    """Non-strings, blanks, punctuation, and quotes too short to identify one
    utterance. Two words appear in too many meetings to pin a moment down."""
    cues = parse_vtt(VTT)
    assert resolve_timestamp(evidence, "00:00:06", cues) is None


def test_three_tokens_is_enough():
    cues = parse_vtt(VTT)
    assert resolve_timestamp("left in August", None, cues) == "00:00:06"


@pytest.mark.parametrize(
    "evidence",
    [
        "THEIR COORDINATOR LEFT IN AUGUST",
        "their   coordinator\n left in August",
        "  coordinator left in August  ",
    ],
)
def test_case_and_whitespace_do_not_matter(evidence):
    cues = [Cue(6000, 11000, "Bob", "Their coordinator left in August.")]
    assert resolve_timestamp(evidence, None, cues) == "00:00:06"


def test_typographic_quotes_are_folded():
    cues = [Cue(6000, 11000, "Alice", "We’re not opening a new site.")]
    assert resolve_timestamp("We're not opening a new site", None, cues) == "00:00:06"
    assert resolve_timestamp("We’re not opening a new site", None, cues) == "00:00:06"


@pytest.mark.parametrize(
    "evidence",
    [
        "We're at 140 of 200",          # a changed number
        "We're not at 148 of 200",      # an inserted negation
        "at 200 of 148 we're",          # reordered words
    ],
)
def test_numbers_negation_and_word_order_stay_significant(evidence):
    cues = parse_vtt(VTT)
    assert resolve_timestamp(evidence, None, cues) is None


def test_a_quote_may_not_match_inside_a_larger_word():
    cues = [Cue(3000, 8000, None, "The offsite fourteen-day window closed.")]
    assert resolve_timestamp("site four", None, cues) is None
    assert resolve_timestamp("offsite fourteen-day window", None, cues) == "00:00:03"


def test_a_repeated_quote_is_pinned_by_a_supported_timestamp():
    cues = [
        Cue(5000, 9000, "Alice", "We are three weeks behind."),
        Cue(65000, 69000, "Bob", "We are three weeks behind."),
    ]
    assert resolve_timestamp("three weeks behind", "00:01:05", cues) == "00:01:05"
    assert resolve_timestamp("three weeks behind", "00:00:05", cues) == "00:00:05"


def test_a_repeated_quote_with_no_supported_timestamp_is_ambiguous():
    """Choosing the earliest or the nearest would be a guess printed as a
    fact, so the note keeps its words and loses its citation."""
    cues = [
        Cue(5000, 9000, "Alice", "We are three weeks behind."),
        Cue(65000, 69000, "Bob", "We are three weeks behind."),
    ]
    assert resolve_timestamp("three weeks behind", "00:47:00", cues) is None
    assert resolve_timestamp("three weeks behind", None, cues) is None


def test_two_cues_in_the_same_displayed_second_are_both_checked():
    """Overlapping speakers share a whole-second label. A lookup keyed by
    timestamp would drop one of them; every cue has to be tested."""
    cues = [
        Cue(12100, 15000, "Alice", "We are not opening a new site."),
        Cue(12800, 16000, "Bob", "Send me the numbers by Thursday."),
    ]
    assert resolve_timestamp("numbers by Thursday", None, cues) == "00:00:12"
    assert resolve_timestamp("not opening a new site", "00:00:12", cues) == "00:00:12"


def test_the_citation_is_the_cue_start_floored_to_a_second():
    """Cue-level precision, not word-level alignment: 6.9s displays as 06."""
    cues = [Cue(6900, 11000, "Bob", "Their coordinator left in August.")]
    assert resolve_timestamp("coordinator left in August", None, cues) == "00:00:06"


def test_evidence_split_across_two_cues_is_not_accepted():
    cues = [
        Cue(6000, 11000, "Bob", "Site four is the problem."),
        Cue(12000, 18000, "Alice", "Their coordinator left in August."),
    ]
    assert resolve_timestamp("the problem. Their coordinator", None, cues) is None


# --- summarising ---------------------------------------------------------


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
        # A realistic decision, not a marker word: the substance gate drops a
        # bare noun, and this test is about the code fence, not the content.
        "```json\n" + extraction(decisions=[{"text": "Move the interim analysis to March", "timestamp": "00:00:12"}]) + "\n```",
        "overview",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.decisions[0].text == "Move the interim analysis to March"


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


# --- citations through the real summariser --------------------------------


@pytest.mark.parametrize("kind", ["decisions", "actions"])
async def test_a_wrong_but_real_timestamp_is_repaired_end_to_end(kind):
    """The six-second mismatch, through the path the product actually runs.

    The model attaches cue 2's words to cue 3's timestamp. Both timestamps are
    real, so timestamp existence cannot tell them apart; the quote can.
    """
    item = {
        "text": "Site four lost its study coordinator",
        "quote": "Their coordinator left in August.",
        "timestamp": "00:00:12",
    }
    if kind == "actions":
        item["owner"] = "Bob Ruiz"
    backend = FakeBackend([extraction(**{kind: [item]}), "An overview."])

    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")

    points = getattr(summary, kind)
    assert points[0].timestamp == "00:00:06"
    assert points[0].text == "Site four lost its study coordinator"
    rendered = summary.render()
    assert "[00:00:06]" in rendered
    assert "00:00:12" not in rendered


@pytest.mark.parametrize("kind", ["decisions", "actions"])
async def test_an_uncheckable_quote_costs_the_citation_not_the_point(kind):
    item = {
        "text": "Site four lost its study coordinator",
        "quote": "the coordinator resigned over the summer",
        "timestamp": "00:00:12",
    }
    backend = FakeBackend([extraction(**{kind: [item]}), "An overview."])

    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")

    points = getattr(summary, kind)
    assert points[0].text == "Site four lost its study coordinator"
    assert points[0].timestamp is None
    assert "Site four lost its study coordinator" in summary.render()
    assert "00:00:12" not in summary.render()


async def test_a_malformed_quote_does_not_fall_back_to_the_text():
    """An empty or non-string quote is a broken claim, not a missing one, so it
    gets no second chance at the paraphrase."""
    backend = FakeBackend([
        extraction(decisions=[
            {"text": "Their coordinator left in August.", "quote": "", "timestamp": "00:00:06"},
            {"text": "We're at 148 of 200. Three weeks behind.", "quote": 12, "timestamp": "00:00:01"},
        ]),
        "An overview.",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert len(summary.decisions) == 2
    assert all(point.timestamp is None for point in summary.decisions)


async def test_an_older_reply_without_a_quote_is_judged_on_its_text():
    """Compatibility with the previous prompt: no "quote" field means the text
    is tried as evidence, against the same strict matcher. A verbatim line
    earns its citation; a paraphrase does not."""
    backend = FakeBackend([
        extraction(decisions=[
            {"text": "Their coordinator left in August.", "timestamp": "00:47:00"},
            {"text": "Site four lost its study coordinator", "timestamp": "00:00:12"},
        ]),
        "An overview.",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")

    verbatim, paraphrase = summary.decisions
    assert verbatim.timestamp == "00:00:06"
    assert paraphrase.timestamp is None
    assert paraphrase.text == "Site four lost its study coordinator"


async def test_evidence_outside_the_chunk_the_model_saw_is_not_used():
    """Each chunk is checked against its own cues. Rescuing a citation from a
    passage that model call never read would cite something nobody quoted."""
    cues = parse_vtt(VTT)
    chunks = chunk_cues(cues, words_per_chunk=8)
    assert len(chunks) > 2
    first = extraction(decisions=[{
        "text": "Priya is redlining the SAP",
        # Said in the last cue, which is not in the first chunk.
        "quote": "I'll have the redlined SAP ready",
        "timestamp": "00:00:19",
    }])
    backend = FakeBackend([first] + [extraction()] * (len(chunks) - 1) + ["An overview."])

    summary = await summarize_meeting(
        cues, backend=backend, model="fake", title="Sync", words_per_chunk=8
    )
    assert summary.decisions[0].text == "Priya is redlining the SAP"
    assert summary.decisions[0].timestamp is None


# --- citations in the prose halves ----------------------------------------


NOTES = "## Enrollment\n- site 4?\n\n## Statistics\n- SAP timing\n"


async def test_enhanced_notes_repair_the_same_six_second_mismatch():
    backend = FakeBackend([
        extraction(),
        notes_reply(
            "## Enrollment\n- The coordinator left in August. [[cite:c1]]",
            [{
                "id": "c1",
                "quote": "Their coordinator left in August.",
                "timestamp": "00:00:12",
            }],
        ),
    ])
    summary = await summarize_meeting(
        parse_vtt(VTT), backend=backend, model="fake", title="Sync", notes=NOTES
    )
    assert "[00:00:06]" in summary.enhanced_notes
    assert "00:00:12" not in summary.enhanced_notes
    assert "cite:" not in summary.enhanced_notes


async def test_unverifiable_note_citations_vanish_without_taking_the_note():
    markdown = (
        "## Enrollment\n"
        "- Checked out. [[cite:c1]]\n"
        "- Unknown id. [[cite:nope]]\n"
        "- Claimed twice. [[cite:dup]]\n"
        "- Evidence nobody said. [[cite:c9]]\n"
        "- Written straight into the prose. [00:00:19]\n"
    )
    backend = FakeBackend([
        extraction(),
        notes_reply(markdown, [
            {"id": "c1", "quote": "Their coordinator left in August.", "timestamp": "00:00:06"},
            {"id": "dup", "quote": "Their coordinator left in August.", "timestamp": "00:00:06"},
            {"id": "dup", "quote": "Site four is the problem.", "timestamp": "00:00:06"},
            {"id": "c9", "quote": "nothing like this was ever said", "timestamp": "00:00:01"},
        ]),
    ])
    summary = await summarize_meeting(
        parse_vtt(VTT), backend=backend, model="fake", title="Sync", notes=NOTES
    )

    notes = summary.enhanced_notes
    assert notes.count("[00:00:06]") == 1  # only the one that checked out
    assert "00:00:19" not in notes
    assert "cite:" not in notes
    for line in ("Unknown id.", "Claimed twice.", "Evidence nobody said.",
                 "Written straight into the prose."):
        assert line in notes
    assert notes.splitlines()[1] == "- Checked out. [00:00:06]"


async def test_a_timestamp_copied_out_of_the_users_own_notes_is_not_a_citation():
    """Someone typing "[00:14:00]" during a call has not verified anything, and
    once the model echoes it into the rewrite it looks exactly like a generated
    citation. Their own notes are never edited - only the generated output."""
    rough = "## Enrollment\n- decision at [00:14:00]?\n"
    backend = FakeBackend([
        extraction(),
        notes_reply("## Enrollment\n- They decided at [00:14:00].", []),
    ])
    summary = await summarize_meeting(
        parse_vtt(VTT), backend=backend, model="fake", title="Sync", notes=rough
    )
    assert summary.enhanced_notes == "## Enrollment\n- They decided at."
    assert "[00:14:00]" in rough  # the user's own text, untouched


async def test_plain_markdown_notes_stay_readable_and_lose_their_citations():
    backend = FakeBackend([
        extraction(),
        "## Enrollment\n- Site four lost its coordinator. [00:00:12]\n\n## Statistics\n- SAP next week.",
    ])
    summary = await summarize_meeting(
        parse_vtt(VTT), backend=backend, model="fake", title="Sync", notes=NOTES
    )
    assert summary.enhanced_notes == (
        "## Enrollment\n- Site four lost its coordinator.\n\n## Statistics\n- SAP next week."
    )


async def test_an_unusable_envelope_falls_back_to_the_rough_notes():
    """Never render a JSON blob where someone's notes should be."""
    backend = FakeBackend([extraction(), json.dumps({"citations": [], "markdown": None})])
    summary = await summarize_meeting(
        parse_vtt(VTT), backend=backend, model="fake", title="Sync", notes=NOTES
    )
    assert summary.enhanced_notes == NOTES.strip()
    assert "markdown" not in summary.enhanced_notes


async def test_headings_and_bullet_order_survive_verification():
    markdown = (
        "## Enrollment\n- 148 of 200, three weeks behind. [[cite:c1]]\n"
        "\n## Statistics\n- Redlined SAP next week. [[cite:c2]]\n"
    )
    backend = FakeBackend([
        extraction(),
        notes_reply(markdown, [
            {"id": "c1", "quote": "We're at 148 of 200.", "timestamp": "00:00:01"},
            {"id": "c2", "quote": "redlined SAP ready by the end of next week", "timestamp": "00:00:19"},
        ]),
    ])
    summary = await summarize_meeting(
        parse_vtt(VTT), backend=backend, model="fake", title="Sync", notes=NOTES
    )
    headings = [line for line in summary.enhanced_notes.splitlines() if line.startswith("## ")]
    assert headings == ["## Enrollment", "## Statistics"]
    assert "[00:00:01]" in summary.enhanced_notes
    assert "[00:00:19]" in summary.enhanced_notes


@pytest.mark.parametrize(
    "prose, expected",
    [
        ("Behind [00:00:01] today.", "Behind today."),
        ("Behind [[00:00:01]] today.", "Behind today."),
        ("Behind [12:34] today.", "Behind today."),
        # Not the citation syntax, so not ours to remove.
        ("We start at 10:30 tomorrow.", "We start at 10:30 tomorrow."),
        ("See (00:00:01) for that.", "See (00:00:01) for that."),
        ("Option [4] was chosen.", "Option [4] was chosen."),
        ("Budget is [TBD] still.", "Budget is [TBD] still."),
    ],
)
async def test_only_bracketed_clock_times_count_as_citation_syntax(prose, expected):
    """Ordinary times and numbers in prose survive; the citation form does not."""
    backend = FakeBackend([extraction(), prose])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == expected


async def test_the_overview_never_carries_an_unverified_citation():
    backend = FakeBackend([
        extraction(),
        "Enrollment is behind [00:00:01] and site four has no coordinator [00:00:06].",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == "Enrollment is behind and site four has no coordinator."
    assert "00:00:01" not in summary.render()


# --- Briefs that talk about the meeting, not about themselves -----------------
#
# Three of these came back from llama3.1:8b in one afternoon of real use. The
# first one sat in the meetings list looking like a broken feature.


async def test_the_model_asking_for_a_transcript_is_not_a_summary():
    """A one-second recording produced a polite request addressed to whoever
    was reading. It rendered where the summary goes, so to anyone skimming the
    list the product had simply broken."""
    backend = FakeBackend([extraction(), (
        "There is no provided transcript, only an opening phrase. Please provide "
        "the transcript for a specific discussion and I will be happy to create "
        "an overview of what was talked about."
    )])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == ""
    assert "Please provide" not in summary.render()
    # The computed half is untouched: who spoke, how long, how many words.
    assert summary.facts.word_count > 0


async def test_a_brief_that_only_describes_the_recording_is_dropped():
    backend = FakeBackend([extraction(), "This appears to be a conversation between two individuals."])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == ""


async def test_the_padding_goes_and_the_content_stays():
    """The common shape: one sentence of hedging about the format, then the
    part the reader actually wants."""
    backend = FakeBackend([extraction(), (
        "This appears to be a recording of a group conversation, likely a podcast. "
        "Site four has no coordinator and the assay repeats on Thursday."
    )])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == "Site four has no coordinator and the assay repeats on Thursday."


async def test_a_real_brief_is_left_exactly_as_written():
    """Including its bullets - the format the brief is now asked for."""
    brief = (
        "A two-person check on the recorder before a client demo.\n\n"
        "- Site four has no coordinator; the assay repeats on Thursday.\n"
        "- Delivery is a downloadable file, not an app store."
    )
    backend = FakeBackend([extraction(), brief])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == brief


async def test_a_one_line_brief_beginning_with_this_survives():
    """"This was a microphone test" is a good brief for a microphone test. Only
    hedging or naming the format as the subject counts as padding."""
    backend = FakeBackend([extraction(), "This was a microphone test before the club meeting."])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == "This was a microphone test before the club meeting."


async def test_the_prompts_own_example_is_never_reported_as_a_fact():
    """The overview prompt shows one good bullet so the model stops writing
    prose. On a recording with nothing in it, llama3.1:8b had nothing to write
    and wrote the example: a one-second clip came back claiming enrollment was
    148 of 200 and a readout had moved to March. An invented clinical figure in
    a meeting summary is the worst thing this module can produce."""
    backend = FakeBackend([extraction(), (
        "- Enrollment is 148 of 200.\n- The readout moves to March."
    )])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == ""
    assert "148" not in summary.render()


async def test_one_borrowed_figure_does_not_cost_the_rest_of_the_brief():
    """The commoner shape: a real brief with one line lifted from the example."""
    backend = FakeBackend([extraction(), (
        "A two-person check on the recorder.\n"
        "- Enrollment is 148 of 200.\n"
        "- Delivery is a downloadable file, not an app store."
    )])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert "148" not in summary.overview
    assert "downloadable file" in summary.overview
    assert "A two-person check on the recorder." in summary.overview


async def test_a_recording_too_short_to_summarise_is_not_summarised():
    """The gate that stops the invention at its source. Three runs against a
    one-second recording produced three different meetings, none of which
    happened. The model is not asked when there is nothing to ask about."""
    tiny = parse_vtt("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nSometimes.\n")
    backend = FakeBackend([extraction(), "A productive planning session about the Q3 roadmap."])
    summary = await summarize_meeting(tiny, backend=backend, model="fake", title="Ajawkd")

    assert summary.overview == ""
    assert "roadmap" not in summary.render()
    assert any("Too short to summarise" in n for n in summary.notices)
    # The computed half survives: this is a downgrade, not a failure.
    assert summary.degraded_reason is None
    assert summary.facts.word_count == 1


async def test_a_short_but_real_recording_is_still_summarised():
    """The floor has to sit below a genuine short meeting. A 1m32s test
    recording runs to roughly four hundred words."""
    lines = ["WEBVTT", ""]
    for i in range(6):
        lines += [str(i + 1),
                  f"00:00:{i * 5:02d}.000 --> 00:00:{i * 5 + 4:02d}.000",
                  "We agreed the assay repeats on Thursday and site four still has no coordinator.",
                  ""]
    backend = FakeBackend([extraction(), "The assay repeats on Thursday."])
    summary = await summarize_meeting(
        parse_vtt("\n".join(lines)), backend=backend, model="fake", title="Short"
    )
    assert summary.overview == "The assay repeats on Thursday."


async def test_a_line_that_announces_the_list_is_not_part_of_the_brief():
    """llama3.1:8b writes these constantly once a transcript is long enough
    that the format instruction stops landing."""
    backend = FakeBackend([extraction(), (
        "The team reviewed the assay.\n\n"
        "Here are some specific points that can be noted from the conversation:\n\n"
        "- Site four still has no coordinator."
    )])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert "Here are some" not in summary.overview
    assert summary.overview == (
        "The team reviewed the assay.\n\n- Site four still has no coordinator."
    )


async def test_the_models_guesses_about_garbled_words_are_removed():
    """A forty-minute recording of people talking over each other produced a
    bracketed guess per paragraph - "a 'sasu' (which is likely a mishearing of
    'sister-in-law')". The sentence around it is usually fine; the guess is
    invention wearing a helpful hat."""
    backend = FakeBackend([extraction(), (
        "- She was called a \u201csasu\u201d (which is likely a mishearing of "
        "\u201csister-in-law\u201d), then corrected.\n"
        "- He was the \u201cvet\u201d (short for veterinarian) in the game."
    )])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert "likely" not in summary.overview
    assert "short for" not in summary.overview
    assert "then corrected" in summary.overview
    assert "in the game" in summary.overview


async def test_an_ordinary_bracket_is_not_a_guess():
    """The filter is for speculation, not for parentheses."""
    backend = FakeBackend([extraction(), "- Delivery is a downloadable file (Mac and Windows)."])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == "- Delivery is a downloadable file (Mac and Windows)."


async def test_asterisk_bullets_become_ordinary_ones():
    """The prompt asks for "- ". Markdown renders both; a meeting that mixes
    them looks like two summaries stapled together."""
    backend = FakeBackend([extraction(), "* Site four has no coordinator.\n* The assay repeats."])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.overview == "- Site four has no coordinator.\n- The assay repeats."


# --- Who said it, as opposed to who will do it --------------------------------
#
# Taken from the transcript at the verified timestamp, never from the model.
# The cue already carries whichever name came from the Zoom participant track,
# the voice print or the meeting window, so a point cannot be attributed to
# somebody who never spoke.


async def test_a_decision_is_credited_to_whoever_said_it():
    backend = FakeBackend([
        extraction(decisions=[{
            "text": "No new site is being opened.",
            "quote": "Then we're not opening a new site",
            "timestamp": "00:00:12",
        }]),
        "A short brief.",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.decisions[0].speaker == "Alice Chen"
    assert "— Alice Chen" in summary.decisions[0].render()


async def test_the_owner_of_an_action_beats_the_speaker_who_assigned_it():
    """"Bob, send me the numbers" is spoken by Alice and owned by Bob. Showing
    Alice would credit the work to the person who asked for it."""
    backend = FakeBackend([
        extraction(actions=[{
            "text": "Send the enrollment numbers by Thursday.",
            "owner": "Bob",
            "quote": "Bob, send me the numbers by Thursday",
            "timestamp": "00:00:12",
        }]),
        "A short brief.",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    action = summary.actions[0]
    assert action.speaker == "Alice Chen" and action.owner == "Bob"
    assert "— Bob" in action.render() and "Alice" not in action.render()


async def test_a_point_with_no_verified_timestamp_is_credited_to_nobody():
    """The same honesty rule the citation follows: no timestamp, no claim."""
    backend = FakeBackend([
        extraction(decisions=[{"text": "Something was settled.", "quote": "nowhere in the transcript"}]),
        "A short brief.",
    ])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    assert summary.decisions[0].timestamp is None
    assert summary.decisions[0].speaker is None
    assert "—" not in summary.decisions[0].render()


async def test_the_participants_are_listed_when_they_have_real_names():
    backend = FakeBackend([extraction(), "A short brief."])
    summary = await summarize_meeting(parse_vtt(VTT), backend=backend, model="fake", title="Sync")
    rendered = summary.render()
    assert "## Participants" in rendered
    for name in ("Alice Chen", "Bob Ruiz", "Priya Nair"):
        assert f"- {name}" in rendered


async def test_a_list_of_speaker_1_and_speaker_2_is_not_a_participant_list():
    """Diarisation without names produces placeholders. Listing them under a
    heading that promises people is worse than leaving the heading out."""
    anonymous = parse_vtt(
        "WEBVTT\n\n1\n00:00:01.000 --> 00:00:06.000\n"
        "Speaker 1: We're at 148 of 200 and three weeks behind schedule.\n\n"
        "2\n00:00:06.500 --> 00:00:12.000\n"
        "Speaker 2: Site four is the problem since their coordinator left.\n"
    )
    backend = FakeBackend([extraction(), "A short brief."])
    summary = await summarize_meeting(anonymous, backend=backend, model="fake", title="Sync")
    assert "## Participants" not in summary.render()


# --- the whole command, end to end ----------------------------------------


async def test_the_written_notes_file_carries_only_verified_citations(tmp_path, monkeypatch):
    """Drive ``bedrock-meeting``'s real path and read the notes.md it writes.

    Whisper, ffmpeg, audio hardware, a model server and the network are all
    replaced at the seams the module already has, so this runs anywhere.
    """
    from bedrock.meetings import cli
    from bedrock.meetings.transcribe import TranscriptionResult

    folder = tmp_path / "2026-09-15 09.00.12 Phase II weekly sync"
    folder.mkdir()
    # A real Zoom folder to hand the command. Transcription itself is faked
    # below, so the bytes never matter - but the command checks that a
    # recording is there before it starts, and that check is worth keeping.
    (folder / "audio_only.m4a").write_bytes(b"")
    rough = tmp_path / "rough.md"
    rough.write_text("## Enrollment\n- site 4? decided around [00:14:00]\n")

    cues = parse_vtt(VTT)

    class FakeTranscriber:
        def __init__(self, config):
            self.config = config

        def readiness_issues(self):
            return ()

    async def fake_transcribe(folder, transcriber, work_dir=None):
        return TranscriptionResult(cues=cues, speakers=speakers(cues))

    backend = FakeBackend([
        extraction(actions=[{
            "text": "Send enrollment numbers",
            "owner": "Bob Ruiz",
            "quote": "Their coordinator left in August.",
            "timestamp": "00:00:12",          # real, but six seconds off
        }]),
        notes_reply(
            "## Enrollment\n"
            "- The coordinator left in August. [[cite:c1]]\n"
            "- Decided around [00:14:00], they said. [[cite:ghost]]\n",
            [{"id": "c1", "quote": "Their coordinator left in August.", "timestamp": "00:00:12"}],
        ),
    ])

    monkeypatch.setattr(cli, "WhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "transcribe_recording", fake_transcribe)
    monkeypatch.setattr(cli, "discover_participant_tracks", lambda folder: {})
    built = []
    monkeypatch.setattr(cli, "create_backend", lambda settings: built.append(settings) or backend)
    from bedrock.config import load_settings

    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(config_path=tmp_path / "none.toml", env={"BEDROCK_MODEL": "fake"}))

    args = cli.build_parser().parse_args([str(folder), "--notes", str(rough)])
    assert await cli._run(args) == 0

    written = (folder / "notes.md").read_text()
    assert "[00:00:06]" in written          # the repaired citation reached the file
    assert "00:00:12" not in written        # the wrong one did not
    assert "00:14:00" not in written        # nor the timestamp echoed from the notes
    assert "cite:" not in written
    assert "Send enrollment numbers" in written
    assert "Decided around, they said." in written
    # The user's own file is never rewritten.
    assert "[00:14:00]" in rough.read_text()
    # The backend came from the shared factory, with the long meeting timeout.
    assert built and built[0].request_timeout == 600.0


# --- The model handing back its own JSON skeleton -----------------------------


def test_a_placeholder_owner_becomes_no_owner():
    """llama3.1:8b returned `"owner": "name or null"` verbatim. An action item
    assigned to "name or null" reads as a broken product; no owner is honest."""
    from bedrock.meetings.summarize import _points

    points = _points(
        [{"text": "Send the enrollment tracker tonight.", "owner": "name or null"}],
        [],
        with_owner=True,
    )
    assert len(points) == 1
    assert points[0].owner is None
    assert "name or null" not in points[0].render()


def test_a_point_that_is_only_a_placeholder_is_dropped():
    """The same run produced a "fact" whose entire text was the placeholder."""
    from bedrock.meetings.summarize import _points

    points = _points(
        [
            {"text": "name or null"},
            {"text": "..."},
            {"text": "<the concrete detail>"},
            {"text": "148 of 200 patients"},
        ],
        [],
        with_owner=False,
    )
    assert [p.text for p in points] == ["148 of 200 patients"]


def test_a_real_owner_still_survives():
    """The filter must not eat ordinary names."""
    from bedrock.meetings.summarize import _points

    points = _points(
        [{"text": "Update the protocol document.", "owner": "Priya"}],
        [],
        with_owner=True,
    )
    assert points[0].owner == "Priya"


def test_a_first_person_owner_becomes_no_owner():
    """"I will send the tracker" names nobody, and an in-room recording has no
    speaker labels to resolve the pronoun against."""
    from bedrock.meetings.summarize import _points

    points = _points(
        [{"text": "Send the enrollment tracker tonight.", "owner": "I"}],
        [],
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


def test_only_the_result_lines_are_read_from_the_vad_log():
    """The VAD binary logs its own workings on the way to the answer:

        whisper_vad_segments_from_probs: VAD segment 654: start = 2334.47, ...
        Speech segment 0: start = 109.00, end = 966.00

    The first is a debug line in *seconds*, the second the result in
    centiseconds. A pattern matching any "start = …, end = …" read both, so
    every transcript was built from twice as many regions as existed, half at
    the wrong time scale. Short recordings survived it; a forty-minute meeting
    produced a turn ending before it started and ffmpeg aborted on it.
    """
    from bedrock.meetings.transcribe import _VAD_LINE

    assert _VAD_LINE.search("Speech segment 0: start = 109.00, end = 966.00")
    assert not _VAD_LINE.search(
        "whisper_vad_segments_from_probs: VAD segment 654: start = 2334.47, end = 2335.61"
    )


def test_a_region_that_ends_before_it_starts_is_discarded(tmp_path, monkeypatch):
    """Everything downstream assumes regions are sorted and forward-going, and
    nothing downstream checks. One bad pair reached ffmpeg as
    "-ss 20.005 -to 9.660" and took a whole meeting with it."""
    import asyncio

    from bedrock.meetings.transcribe import WhisperConfig, WhisperTranscriber

    log_text = (
        "Speech segment 0: start = 100.00, end = 200.00\n"
        "Speech segment 1: start = 900.00, end = 500.00\n"   # backwards
        "Speech segment 2: start = 300.00, end = 400.00\n"   # out of order
    )

    class Runner:
        async def run(self, args, log_path, env=None):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(log_text)

    transcriber = WhisperTranscriber(
        WhisperConfig(model=tmp_path / "m.bin", vad_model=tmp_path / "v.bin"), runner=Runner()
    )
    regions = asyncio.run(transcriber.detect_speech_regions(tmp_path / "a.wav", tmp_path))
    assert regions == [(1000, 2000), (3000, 4000)]
# --- Fitting the model's context window ---------------------------------------


def long_meeting(lines: int = 900, words: int = 18) -> list[Cue]:
    """About an hour of talk: far more than a laptop's 8k-token window."""
    return [
        Cue(i * 4000, i * 4000 + 3500, "Alice Chen" if i % 2 else "Bob Ruiz",
            f"line {i} " + " ".join(f"word{i}x{j}" for j in range(words)))
        for i in range(lines)
    ]


def scheduler_with(backend, **profile_changes):
    from dataclasses import replace

    from bedrock.config import PROFILES
    from bedrock.inference import Scheduler

    return Scheduler(backend, replace(PROFILES["laptop"], **profile_changes))


@pytest.mark.parametrize("window", [8192, 4096, 2048])
async def test_no_prompt_is_larger_than_the_window_can_hold(window):
    from bedrock.inference import estimate_tokens
    from bedrock.meetings.summarize import EXTRACT_REPLY_TOKENS, OVERVIEW_REPLY_TOKENS

    backend = FakeBackend()
    scheduler = scheduler_with(backend, context_window=window, reply_limit=min(1024, window // 4))
    summary = await summarize_meeting(long_meeting(), scheduler=scheduler, model="fake", title="Sync")

    assert summary.degraded_reason is None
    assert len(backend.prompts) > 2  # the transcript was split into several chunks
    for prompt, options in zip(backend.prompts, backend.options):
        assert options.max_tokens in (EXTRACT_REPLY_TOKENS, OVERVIEW_REPLY_TOKENS)
        assert options.reasoning is False
        assert estimate_tokens(prompt) + options.max_tokens <= window
    assert any("lines from the middle were left out" in notice for notice in summary.notices)


async def test_every_meeting_call_is_a_user_job_on_the_scheduler():
    from bedrock.inference import Priority

    backend = FakeBackend()
    scheduler = scheduler_with(backend)
    seen = []
    original = scheduler.stream

    def recording(model, messages, *, priority, options=None):
        seen.append(priority)
        return original(model, messages, priority=priority, options=options)

    scheduler.stream = recording
    await summarize_meeting(parse_vtt(VTT), scheduler=scheduler, model="fake", title="Sync")
    assert seen and set(seen) == {Priority.USER_JOB}


async def test_notes_citations_are_checked_only_against_the_lines_sent():
    """A long meeting sends the notes pass only its start and end. A quote from
    the unsent middle is real speech, but that model call never read it, so
    its citation cannot stand."""
    cues = long_meeting()
    middle = cues[len(cues) // 2]
    first = cues[0]
    from bedrock.inference import estimate_tokens
    from bedrock.meetings.summarize import _EXTRACT_PROMPT, EXTRACT_REPLY_TOKENS, prompt_budget

    room = prompt_budget(8192, EXTRACT_REPLY_TOKENS) - estimate_tokens(_EXTRACT_PROMPT.format(transcript=""))
    extraction_calls = len(chunk_cues(cues, max_tokens=room))
    backend = FakeBackend([extraction() for _ in range(extraction_calls)] + [
        notes_reply(
            "## Notes\n- From the start. [[cite:a]]\n- From the middle. [[cite:b]]\n",
            [
                {"id": "a", "quote": first.text, "timestamp": first.timestamp},
                {"id": "b", "quote": middle.text, "timestamp": middle.timestamp},
            ],
        )
    ])
    scheduler = scheduler_with(backend)

    summary = await summarize_meeting(cues, scheduler=scheduler, model="fake", title="Sync", notes="## Notes\n- stuff")

    notes_prompt = backend.prompts[-1]
    assert first.text in notes_prompt and middle.text not in notes_prompt
    assert f"[{first.timestamp}]" in summary.enhanced_notes
    assert f"[{middle.timestamp}]" not in summary.enhanced_notes
    assert "From the middle." in summary.enhanced_notes  # the words stay, the citation goes


async def test_notes_too_long_for_the_window_are_refused_plainly():
    backend = FakeBackend([extraction()])
    scheduler = scheduler_with(backend, context_window=2048, reply_limit=512)
    rough = "## My notes\n" + "- a very long note about everything " * 400

    summary = await summarize_meeting(parse_vtt(VTT), scheduler=scheduler, model="fake", title="Sync", notes=rough)

    assert summary.enhanced_notes == rough.strip()
    assert any("too long to rewrite" in notice for notice in summary.notices)
    assert len(backend.prompts) == 1  # only the extraction pass was sent
    assert "too long to rewrite" in summary.render()


def test_the_extraction_prompt_covers_everyday_task_language_and_writes_english():
    """The requests that make up most real action items are never announced as
    "action item". The prompt has to name them, and the notes have to come out
    in English whatever was spoken, with the quote left in the original."""
    from bedrock.meetings.summarize import _EXTRACT_PROMPT, _OVERVIEW_PROMPT

    prompt = _EXTRACT_PROMPT.format(transcript="x")
    for phrase in ("Can you send me", "Will do", "loop in", "I'll follow up", "Sending an email"):
        assert phrase in prompt
    assert "in English whatever language" in prompt
    assert "never translated" in prompt
    assert "in English whatever language" in _OVERVIEW_PROMPT.format(transcript="x")


def test_chinese_is_counted_by_character_not_by_space():
    """Splitting on whitespace made a Mandarin meeting one word per sentence,
    which put it under the floor below which no overview is written."""
    from bedrock.meetings.transcript import count_words

    assert count_words("the controls came back clean") == 5
    assert count_words("大家早上好，我们开始今天的周会。") == 14
    assert count_words("第一项是KRAS G12D对照实验的结果") == 11 + 2
    assert count_words("") == 0
    assert count_words("   ") == 0

    cues = parse_vtt("WEBVTT\n\n1\n00:00:00.000 --> 00:00:05.000\n大家早上好，我们开始今天的周会。\n")
    assert word_count(cues) == 14


def test_a_title_reply_is_cleaned_or_refused():
    from bedrock.meetings.summarize import clean_title

    assert clean_title("KRAS G12D control assay review") == "KRAS G12D control assay review"
    assert clean_title('"Dose response study kickoff."') == "Dose response study kickoff"
    assert clean_title("Title: Sequencing delay and antibody order\n\nThe meeting covered...") == "Sequencing delay and antibody order"
    assert clean_title("**Weekly KRAS sync**") == "Weekly KRAS sync"
    assert clean_title("Meeting about the assay") is None
    assert clean_title("This meeting was a weekly review of the KRAS program and its many open questions") is None
    assert clean_title("") is None
    assert clean_title("\n  \n") is None


async def test_a_blank_name_is_replaced_from_the_notes_but_only_when_there_are_some():
    from bedrock.meetings.summarize import MeetingSummary, Point, compute_facts, suggest_title

    facts = compute_facts(parse_vtt(VTT), title="Meeting — 28 Sep 2026, 21:00")

    class Scheduler:
        def __init__(self, reply): self.reply = reply; self.prompts = []
        async def complete(self, model, messages, priority=None, options=None):
            self.prompts.append(messages[0]["content"]); return self.reply

    summary = MeetingSummary(facts=facts, overview="The group reviewed the G12D control assay and agreed to start the dose response study on Monday.")
    summary.actions = [Point(text="Order the EGFR antibody before Thursday", owner="Ravi")]
    scheduler = Scheduler("Dose response study go-ahead")
    assert await suggest_title(scheduler, "m", summary) == "Dose response study go-ahead"
    assert "Order the EGFR antibody" in scheduler.prompts[0]

    thin = MeetingSummary(facts=facts, overview="Test.")
    scheduler = Scheduler("Anything")
    assert await suggest_title(scheduler, "m", thin) is None
    assert scheduler.prompts == []


def test_a_bullet_that_describes_the_recording_goes_and_so_does_an_empty_one():
    """Seen on a real recording: the brief was a list whose first bullet
    described the recording and whose second was a bare dash."""
    from bedrock.meetings.summarize import clean_overview

    brief = ("- This is a conversation between two students who are studying together in a library.\n"
             "\n-\n- They have a lab meeting on Friday from 3:00 to 6:00 p.m.\n"
             "- They made plans to check in later.")
    assert clean_overview(brief) == (
        "- They have a lab meeting on Friday from 3:00 to 6:00 p.m.\n- They made plans to check in later."
    )
    # A bullet about the content that happens to start with "this" stays.
    assert clean_overview("Weekly sync.\n\n- This assay needs a repeat.\n- Site four needs a coordinator.") == (
        "Weekly sync.\n\n- This assay needs a repeat.\n- Site four needs a coordinator."
    )
