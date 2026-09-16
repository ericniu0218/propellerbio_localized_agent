"""Turning a meeting transcript into notes someone will actually read.

Nobody wants to read a transcript. The product is the other thing: you type a
few scrappy lines during the call, and afterwards the assistant rewrites them
into real notes, using the transcript as the source of fact.

Three decisions shape this module.

*Rules first, model second.* The same split ``context/digest.py`` makes. Who
spoke, how long the meeting ran, how many words - all computed, never wrong,
never dependent on a GPU being up. The model is asked only for the part rules
cannot do: what was decided and who agreed to do what. If the model is down or
returns something unparseable, the computed facts still come back and the call
still succeeds. A summariser that fails completely when inference is unavailable
is worse than one that degrades.

*Map-reduce, not one giant prompt.* An hour of speech is roughly 11,000 tokens,
and a server's context window is not ours to assume - Ollama defaults to 4096.
So the transcript is cut into chunks that comfortably fit anywhere, each is read
on its own, and the findings are merged. This costs more calls and works on
every backend, including the small ones.

*Never cite a timestamp that was not said.* Models invent citations. Every
timestamp the model returns is checked against the real cue list; one that does
not exist is re-derived by finding the quoted text, and dropped if that fails
too. A note with no timestamp is honest. A note with a fabricated one is not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from bedrock.inference.base import InferenceBackend, InferenceError
from bedrock.meetings.transcript import Cue, speakers, word_count

# Chunks are sized for the smallest context window we expect to meet (Ollama's
# 4096-token default), leaving room for the instructions and the reply.
WORDS_PER_CHUNK = 1100
MAX_POINTS_PER_KIND = 25


@dataclass(frozen=True)
class Point:
    """One decision or action item, with the timestamp where it was said."""

    text: str
    timestamp: str | None = None
    owner: str | None = None

    def render(self) -> str:
        parts = [self.text]
        if self.owner:
            parts.append(f"— {self.owner}")
        if self.timestamp:
            parts.append(f"[{self.timestamp}]")
        return " ".join(parts)


@dataclass(frozen=True)
class MeetingFacts:
    """Computed from the transcript alone. Never involves a model."""

    title: str
    meeting_date: str | None
    duration_minutes: int
    speakers: list[str]
    word_count: int
    cue_count: int


@dataclass
class MeetingSummary:
    facts: MeetingFacts
    overview: str = ""
    decisions: list[Point] = field(default_factory=list)
    actions: list[Point] = field(default_factory=list)
    enhanced_notes: str = ""
    model_used: str | None = None
    degraded_reason: str | None = None
    """Set when the model could not be reached or could not be parsed. The
    facts above are still valid; everything the model produces is empty."""

    def render(self) -> str:
        lines = [f"# {self.facts.title}"]
        subtitle = []
        if self.facts.meeting_date:
            subtitle.append(self.facts.meeting_date)
        subtitle.append(f"{self.facts.duration_minutes} min")
        if self.facts.speakers:
            subtitle.append(", ".join(self.facts.speakers))
        lines.append("*" + " · ".join(subtitle) + "*")

        if self.degraded_reason:
            lines += ["", f"> Summary unavailable: {self.degraded_reason}"]
            return "\n".join(lines)

        if self.enhanced_notes:
            lines += ["", self.enhanced_notes]
        elif self.overview:
            lines += ["", "## Overview", "", self.overview]

        if self.decisions:
            lines += ["", "## Decisions", ""]
            lines += [f"- {point.render()}" for point in self.decisions]
        if self.actions:
            lines += ["", "## Action items", ""]
            lines += [f"- {point.render()}" for point in self.actions]
        return "\n".join(lines)


def compute_facts(cues: list[Cue], title: str, meeting_date: date | None = None) -> MeetingFacts:
    """Everything derivable without a model. This never fails."""
    duration_ms = max((cue.end_ms for cue in cues), default=0)
    return MeetingFacts(
        title=title,
        meeting_date=meeting_date.isoformat() if meeting_date else None,
        duration_minutes=round(duration_ms / 60_000),
        speakers=speakers(cues),
        word_count=word_count(cues),
        cue_count=len(cues),
    )


def chunk_cues(cues: list[Cue], words_per_chunk: int = WORDS_PER_CHUNK) -> list[list[Cue]]:
    """Split on cue boundaries so no utterance is ever cut in half."""
    chunks: list[list[Cue]] = []
    current: list[Cue] = []
    running = 0
    for cue in cues:
        length = len(cue.text.split())
        if current and running + length > words_per_chunk:
            chunks.append(current)
            current, running = [], 0
        current.append(cue)
        running += length
    if current:
        chunks.append(current)
    return chunks


_EXTRACT_PROMPT = """\
You are reading one part of a meeting transcript. Report only what is actually \
said in it.

Respond with JSON and nothing else:
{{"decisions": [{{"text": "...", "timestamp": "00:12:34"}}],
 "actions": [{{"text": "...", "owner": "name or null", "timestamp": "00:12:34"}}]}}

Rules:
- A decision is something the group settled on. Phrases like "then we", "let's",
  "agreed", "we're not going to", "we take" usually mark one. A topic they only
  discussed is not a decision.
- An action is something a specific person will do next.
- "owner" is the person who will DO the task, not the person who assigned it.
  In "Bob, send me the numbers", the owner is Bob, not the speaker.
- Write each item as a short standalone sentence. Do not start it with a name
  being addressed; put that name in "owner" instead.
- Use the timestamp shown at the start of the line where it was said.
- If this part contains none, return empty arrays. Never invent one.

Transcript part:
{transcript}
"""

_OVERVIEW_PROMPT = """\
Write a short overview of this meeting: what it was about and where it landed. \
Three sentences at most. Plain prose, no headings, no bullet points. Use only \
what the transcript says.

Transcript:
{transcript}
"""

_NOTES_PROMPT = """\
Someone typed these rough notes during the meeting. Rewrite them as clean notes \
they could send to a colleague.

Rules, in order of importance:
- Keep THEIR structure. Their headings and their bullets stay, in their order.
- Fill in detail underneath their points using the transcript.
- Add nothing that is not in the transcript. If a note is not discussed, leave \
it as they wrote it.
- Cite timestamps like [00:12:34] for facts you add.
- Return markdown. No preamble, no closing remark.

Their notes:
{notes}

Transcript:
{transcript}
"""


def _render(cues: list[Cue]) -> str:
    return "\n".join(cue.render() for cue in cues)


async def _complete(backend: InferenceBackend, model: str, prompt: str) -> str:
    """One non-streaming call, assembled from the streaming seam.

    The backend protocol is deliberately streaming-only; collecting the
    fragments here keeps that contract untouched.
    """
    parts: list[str] = []
    async for fragment in backend.stream_chat(model, [{"role": "user", "content": prompt}]):
        parts.append(fragment)
    return "".join(parts).strip()


def _extract_json(raw: str) -> dict:
    """Pull the JSON object out of a reply that may be wrapped in prose or fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    candidate = fenced.group(1) if fenced else raw[raw.find("{") : raw.rfind("}") + 1]
    if not candidate.strip():
        raise ValueError("no JSON object in reply")
    return json.loads(candidate)


def _valid_timestamps(cues: list[Cue]) -> set[str]:
    return {cue.timestamp for cue in cues}


def _repair_timestamp(value: object, text: str, cues: list[Cue]) -> str | None:
    """Keep a timestamp only if the meeting really contains it.

    A model that invents `00:47:00` for a 6-minute meeting is producing a
    citation that cannot be checked. Rather than pass that on, the quoted text
    is located in the transcript and that cue's real timestamp used; if the text
    cannot be found either, the point keeps its words and loses its citation.
    """
    if isinstance(value, str) and value in _valid_timestamps(cues):
        return value

    needle = {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 3}
    if needle:
        best, best_overlap = None, 0
        for cue in cues:
            overlap = len(needle & set(re.findall(r"[a-z0-9]+", cue.text.lower())))
            if overlap > best_overlap:
                best, best_overlap = cue, overlap
        if best is not None and best_overlap >= 2:
            return best.timestamp
    return None


def _points(raw_items: object, cues: list[Cue], with_owner: bool) -> list[Point]:
    if not isinstance(raw_items, list):
        return []
    points: list[Point] = []
    for item in raw_items[:MAX_POINTS_PER_KIND]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        owner = item.get("owner") if with_owner else None
        owner = str(owner).strip() if owner and str(owner).lower() != "null" else None
        points.append(
            Point(
                text=text,
                timestamp=_repair_timestamp(item.get("timestamp"), text, cues),
                owner=owner,
            )
        )
    return points


def _dedupe(points: list[Point]) -> list[Point]:
    """Chunks overlap in subject matter, so the same decision often comes back
    from two of them worded slightly differently."""
    kept: list[Point] = []
    seen: list[set[str]] = []
    for point in points:
        words = {w for w in re.findall(r"[a-z0-9]+", point.text.lower()) if len(w) > 3}
        # 0.5 Jaccard: "interim analysis moves to March" and "the interim
        # analysis is moving to March" share 3 of 5 content words (0.6) and must
        # collapse; genuinely different decisions score near zero.
        if any(words and len(words & other) / len(words | other) >= 0.5 for other in seen):
            continue
        kept.append(point)
        seen.append(words)
    return kept


async def summarize_meeting(
    cues: list[Cue],
    *,
    backend: InferenceBackend,
    model: str,
    title: str = "Meeting",
    meeting_date: date | None = None,
    notes: str | None = None,
    words_per_chunk: int = WORDS_PER_CHUNK,
) -> MeetingSummary:
    """Summarise a meeting, degrading to computed facts if the model is away.

    ``notes`` is the rough text the user typed during the call. When present it
    becomes the skeleton of the output, which is the whole point: the result
    reads like their notes, finished, rather than a generic summary that ignores
    what they thought was worth writing down.
    """
    facts = compute_facts(cues, title=title, meeting_date=meeting_date)
    summary = MeetingSummary(facts=facts)
    if not cues:
        summary.degraded_reason = "the transcript was empty"
        return summary

    chunks = chunk_cues(cues, words_per_chunk=words_per_chunk)
    decisions: list[Point] = []
    actions: list[Point] = []

    try:
        for chunk in chunks:
            reply = await _complete(
                backend, model, _EXTRACT_PROMPT.format(transcript=_render(chunk))
            )
            try:
                data = _extract_json(reply)
            except (ValueError, json.JSONDecodeError):
                # One unparseable chunk should not lose the other nine.
                continue
            decisions += _points(data.get("decisions"), chunk, with_owner=False)
            actions += _points(data.get("actions"), chunk, with_owner=True)

        if notes and notes.strip():
            summary.enhanced_notes = await _complete(
                backend,
                model,
                _NOTES_PROMPT.format(notes=notes.strip(), transcript=_render(cues[:400])),
            )
        else:
            summary.overview = await _complete(
                backend, model, _OVERVIEW_PROMPT.format(transcript=_render(cues[:400]))
            )
    except InferenceError as exc:
        summary.degraded_reason = str(exc)
        return summary

    summary.decisions = _dedupe(decisions)
    summary.actions = _dedupe(actions)
    summary.model_used = model
    return summary
