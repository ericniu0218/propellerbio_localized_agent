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
from bedrock.meetings.transcript import parse_vtt, speakers, word_count

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


def test_a_real_timestamp_is_kept():
    cues = parse_vtt(VTT)
    assert _repair_timestamp("00:00:12", "anything", cues) == "00:00:12"


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
