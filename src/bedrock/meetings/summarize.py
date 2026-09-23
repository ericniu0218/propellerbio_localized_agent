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

*Never cite a timestamp for something not said then.* Models invent citations,
and they also attach real timestamps to the wrong line. Every timestamp the
model returns - in a decision, an action, a fact, the overview or the rewritten
notes - is checked against the line it points at: kept if that line says it,
moved to the line that does, dropped if none does. See ``citations.py`` for how
"says it" is judged. A note with no timestamp is honest. A note with a wrong one
is not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from bedrock.inference.base import InferenceBackend, InferenceError
from bedrock.meetings.citations import CitationGuard, CitationReport
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
    details: list[Point] = field(default_factory=list)
    """Concrete specifics - figures, dates, statuses - extracted per chunk so
    they survive whatever the single-prompt overview chooses to compress."""
    enhanced_notes: str = ""
    model_used: str | None = None
    degraded_reason: str | None = None
    """Set when the model could not be reached or could not be parsed. The
    facts above are still valid; everything the model produces is empty."""
    citations: CitationReport = field(default_factory=CitationReport)
    """How many of the model's timestamps were checked, moved and removed."""

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
        if self.details:
            lines += ["", "## Details", ""]
            lines += [f"- {point.render()}" for point in self.details]
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

Respond with JSON and nothing else. Replace every <...> with real content from
the transcript; never copy the words inside the angle brackets:
{{"decisions": [{{"text": "<what the group settled on>", "timestamp": "00:12:34"}}],
 "actions": [{{"text": "<what will be done>", "owner": "<who will do it, or null>", "timestamp": "00:12:34"}}],
 "facts": [{{"text": "<the concrete detail>", "timestamp": "00:12:34"}}]}}

Rules:
- A decision is something the group settled on. Phrases like "then we", "let's",
  "agreed", "we're not going to", "we take" usually mark one. A topic they only
  discussed is not a decision.
- An action is something a specific person will do next.
- "owner" is the person who will DO the task, not the person who assigned it.
  In "Bob, send me the numbers", the owner is Bob, not the speaker.
- A fact is any concrete detail stated: a number, a count, a date, a deadline,
  a name, a status. Copy the actual value. "Enrollment is 148 of 200" is a
  fact; "they discussed enrollment" is not.
- Write each item as a short standalone sentence. Do not start it with a name
  being addressed; put that name in "owner" instead.
- Use the timestamp shown at the start of the line where it was said.
- If this part contains none, return empty arrays. Never invent one.

Transcript part:
{transcript}
"""

_OVERVIEW_PROMPT = """\
Write an overview of this meeting: what it was about and where it landed.

Keep every specific. Numbers, dates, deadlines, names and statuses go in \
exactly as said - write "enrollment is 148 of 200", never "they discussed \
enrollment numbers". A reader who was not there should learn the actual \
figures from this paragraph, not learn that figures exist.

Plain prose, no headings, no bullet points. Use only what the transcript says.

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
- Keep every specific: numbers, dates, deadlines and names go in exactly as \
said, never summarised into "the numbers" or "the deadline".
- Cite timestamps like [00:12:34] for facts you add.
- Return markdown. No preamble, no closing remark.

Their notes:
{notes}

Transcript:
{transcript}
"""


# The overview and notes passes get one prompt, not the chunked treatment the
# extraction pass gets, so a long meeting has to be trimmed to fit.
MAX_OVERVIEW_CUES = 400


def _render(cues: list[Cue]) -> str:
    return "\n".join(cue.render() for cue in cues)


def _for_one_prompt(cues: list[Cue], limit: int = MAX_OVERVIEW_CUES) -> str:
    """Trim a long meeting for a single-prompt pass, keeping both ends.

    Taking the first N cues loses the entire back half of a long meeting —
    including the wrap-up, which is where decisions and owners usually land.
    The opening states the agenda and the close states the outcome, so when a
    meeting is too long to send whole, those are the parts to keep.
    """
    if len(cues) <= limit:
        return _render(cues)
    head, tail = limit // 2, limit - limit // 2
    skipped = len(cues) - limit
    return (
        _render(cues[:head])
        + f"\n\n[... {skipped} lines omitted from the middle of the meeting ...]\n\n"
        + _render(cues[-tail:])
    )


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


def _repair_timestamp(value: object, text: str, cues: list[Cue]) -> str | None:
    """The timestamp a point saying ``text`` should cite, or None.

    A model that invents `00:47:00` for a 6-minute meeting, or cites the line
    after the one where something was agreed, is producing a citation that
    sends the reader to the wrong place. The cited line has to say the thing;
    otherwise the line that does is cited instead, and if no line does, the
    point keeps its words and loses its citation. See ``citations.py``.
    """
    return CitationGuard(cues).check(value, text)


# A small model sometimes hands the JSON skeleton back instead of filling it
# in: an action owned by "name or null", a fact whose text is "...". Those are
# the prompt's own placeholder words, not anything anybody said - but by the
# time they reach a reader they look exactly like content, and an action item
# assigned to "name or null" reads as a bug in the product rather than as the
# model having nothing to say. Seen from llama3.1:8b, twice in one 28-second
# meeting, which is why the prompt wording alone is not trusted to prevent it.
_SCHEMA_ECHOES = frozenset(
    {
        "",
        "...",
        "null",
        "none",
        "n/a",
        "na",
        "text",
        "owner",
        "timestamp",
        "name",
        "name or null",
        "who will do it, or null",
        "what the group settled on",
        "what will be done",
        "the concrete detail",
    }
)


# "I will send the tracker tonight" yields an owner of "I", which names nobody.
# On a transcript with speaker labels the pronoun could in principle be
# resolved to the speaker; on one recorded from a single microphone - which is
# every in-room recording - there are no labels to resolve it against, and
# "Send the tracker — I" is worse than no owner at all.
_PRONOUN_OWNERS = frozenset({"i", "me", "my", "we", "us", "our", "you", "they", "he", "she", "it"})


def _is_schema_echo(value: object) -> bool:
    """True when the model returned the placeholder instead of an answer."""
    cleaned = str(value or "").strip().strip("<>\"'").casefold()
    return cleaned in _SCHEMA_ECHOES


def _is_unusable_owner(value: object) -> bool:
    cleaned = str(value or "").strip().strip("<>\"'.,").casefold()
    return _is_schema_echo(value) or cleaned in _PRONOUN_OWNERS


def _points(raw_items: object, guard: CitationGuard, with_owner: bool) -> list[Point]:
    if not isinstance(raw_items, list):
        return []
    points: list[Point] = []
    for item in raw_items[:MAX_POINTS_PER_KIND]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text or _is_schema_echo(text):
            continue
        owner = item.get("owner") if with_owner else None
        # A dropped owner costs the reader a name. A placeholder owner tells
        # them the wrong thing, so an unusable one becomes no owner at all.
        owner = str(owner).strip() if owner and not _is_unusable_owner(owner) else None
        points.append(
            Point(
                text=text,
                timestamp=guard.check(item.get("timestamp"), text),
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
    details: list[Point] = []

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
            # Checked against this chunk only: it is all the model was shown,
            # so a line outside it cannot be where the model found anything.
            guard = CitationGuard(chunk, summary.citations)
            decisions += _points(data.get("decisions"), guard, with_owner=False)
            actions += _points(data.get("actions"), guard, with_owner=True)
            details += _points(data.get("facts"), guard, with_owner=False)

        # The notes and overview read the whole meeting (trimmed if long), so
        # their citations are checked against every line of it.
        whole_meeting = CitationGuard(cues, summary.citations)
        if notes and notes.strip():
            written = await _complete(
                backend,
                model,
                _NOTES_PROMPT.format(notes=notes.strip(), transcript=_for_one_prompt(cues)),
            )
            summary.enhanced_notes = whole_meeting.clean_text(written, leave_alone=notes)
        else:
            written = await _complete(
                backend, model, _OVERVIEW_PROMPT.format(transcript=_for_one_prompt(cues))
            )
            summary.overview = whole_meeting.clean_text(written)
    except InferenceError as exc:
        summary.degraded_reason = str(exc)
        return summary

    summary.decisions = _dedupe(decisions)
    summary.actions = _dedupe(actions)
    summary.details = _dedupe(details)
    summary.model_used = model
    return summary
