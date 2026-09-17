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
    monkeypatch.setattr(cli, "OpenAICompatibleBackend", lambda *a, **kw: backend)

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
