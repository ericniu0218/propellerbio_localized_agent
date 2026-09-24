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
far more than a laptop's context window (``MachineProfile.context_window``,
8192 tokens on the laptop profile). So the transcript is cut into chunks, each
read on its own, and the findings are merged. Every prompt is sized in tokens
against the window, leaving room for the reply: an extraction chunk is split
until it fits, and the single-prompt overview and notes passes keep as many
lines from the start and end of the meeting as fit. A prompt the window cannot
hold is never sent - a server would silently cut it (Ollama) or refuse it
(vLLM). If even the employee's own notes leave no room, the notes pass is
refused with a plain message instead.

*A job the employee asked for.* Every call goes through the ``Scheduler`` as
``Priority.USER_JOB`` with a reply limit and thinking switched off: a
reasoning model left to think would spend the whole limit before writing any
JSON.

*Never cite a timestamp that was not said.* Models invent citations. Every
timestamp the model returns is checked against the real cue list; one that does
not exist is re-derived by finding the quoted text, and dropped if that fails
too. A note with no timestamp is honest. A note with a fabricated one is not.
The check uses only the lines that model call was actually sent, which are
exactly the lines that fitted the window.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from bedrock.config.profile import DEFAULT_PROFILE, PROFILES
from bedrock.inference import Priority, RequestOptions, Scheduler, estimate_tokens
from bedrock.inference.base import InferenceBackend, InferenceError
from bedrock.json_reply import decode_json_reply
from bedrock.meetings.citations import (
    apply_citations,
    resolve_timestamp,
    strip_citation_markup,
)
from bedrock.meetings.transcript import Cue, speakers, word_count

# A chunk is at most this many words (about 1,500 tokens), and in any case no
# more than fits the window with the instructions and the reply.
WORDS_PER_CHUNK = 1100
MAX_POINTS_PER_KIND = 25

EXTRACT_REPLY_TOKENS = 768
"""Reply limit for one chunk's JSON: room for a dozen items with quotes."""

OVERVIEW_REPLY_TOKENS = 512
"""Reply limit for the overview: one full paragraph."""

PROMPT_SAFETY = 0.9
"""Token counts are estimated here, not measured, and the server wraps each
message in a few tokens of its own. Prompts are kept to 90% of the room left
after the reply, so an estimate that runs a little low still fits."""


def prompt_budget(context_window: int, reply_tokens: int) -> int:
    """How many tokens one prompt may use, leaving ``reply_tokens`` for the answer."""
    return max(0, int((context_window - reply_tokens) * PROMPT_SAFETY))


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
    notices: list[str] = field(default_factory=list)
    """Plain sentences about what could not be done in full - lines left out
    to fit the window, or notes too long to rewrite. Shown under the title."""
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
        for notice in self.notices:
            lines += ["", f"> {notice}"]

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


def chunk_cues(
    cues: list[Cue], words_per_chunk: int = WORDS_PER_CHUNK, max_tokens: int | None = None
) -> list[list[Cue]]:
    """Split on cue boundaries so no utterance is ever cut in half.

    A chunk ends before it would pass ``words_per_chunk`` words or, when given,
    ``max_tokens`` tokens of rendered transcript. A single line longer than
    ``max_tokens`` still becomes a chunk of its own; the caller refuses to
    send it.
    """
    chunks: list[list[Cue]] = []
    current: list[Cue] = []
    words = tokens = 0
    for cue in cues:
        length = len(cue.text.split())
        cost = _line_tokens(cue)
        too_long = words + length > words_per_chunk or (max_tokens is not None and tokens + cost > max_tokens)
        if current and too_long:
            chunks.append(current)
            current, words, tokens = [], 0, 0
        current.append(cue)
        words += length
        tokens += cost
    if current:
        chunks.append(current)
    return chunks


def _line_tokens(cue: Cue) -> int:
    """Tokens one cue takes in a prompt, including its line break."""
    return estimate_tokens(cue.render()) + 1


_EXTRACT_PROMPT = """\
You are reading one part of a meeting transcript. Report only what is actually \
said in it.

Respond with JSON and nothing else. Replace every <...> with real content from
the transcript; never copy the words inside the angle brackets:
{{"decisions": [{{"text": "<what the group settled on>", "quote": "<words copied exactly from that line>", "timestamp": "00:12:34"}}],
 "actions": [{{"text": "<what will be done>", "owner": "<who will do it, or null>", "quote": "<words copied exactly from that line>", "timestamp": "00:12:34"}}],
 "facts": [{{"text": "<the concrete detail>", "quote": "<words copied exactly from that line>", "timestamp": "00:12:34"}}]}}

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
- "quote" is a short run of words copied EXACTLY from the line above, letter
  for letter. Do not tidy it, shorten it below about five words, or join words
  from two different lines. It is what proves the timestamp, and an item whose
  quote cannot be found in the transcript keeps its text but loses its citation.
- Use the timestamp shown at the start of the line the quote came from.
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
- To cite a detail you added, put a marker like [[cite:c1]] at the end of that \
line and list c1 below. Never write a timestamp into the markdown yourself.

Respond with JSON and nothing else:
{{"markdown": "their notes, rewritten",
  "citations": [{{"id": "c1", "quote": "...", "timestamp": "00:12:34"}}]}}

Each citation's "quote" is a short run of words copied EXACTLY from one \
transcript line, letter for letter. Each id is used once. A marker whose quote \
cannot be found is removed, and the line it was on keeps its words.

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


@dataclass(frozen=True)
class _Excerpt:
    """The lines a single-prompt pass is given: the start and the end of the meeting."""

    head: list[Cue]
    tail: list[Cue]
    skipped: int

    @property
    def cues(self) -> list[Cue]:
        """Exactly the cues sent, in order - what the citation check must use."""
        return self.head + self.tail

    def render(self) -> str:
        if not self.skipped:
            return _render(self.cues)
        return (
            _render(self.head)
            + f"\n\n[... {self.skipped} lines omitted from the middle of the meeting ...]\n\n"
            + _render(self.tail)
        )


_OMISSION_TOKENS = 20
"""Room for the "[... N lines omitted ...]" marker."""


def _excerpt(cues: list[Cue], budget_tokens: int, limit: int = MAX_OVERVIEW_CUES) -> _Excerpt:
    """Trim a long meeting for a single-prompt pass, keeping both ends.

    Taking the first N cues loses the entire back half of a long meeting —
    including the wrap-up, which is where decisions and owners usually land.
    The opening states the agenda and the close states the outcome, so when a
    meeting is too long to send whole, those are the parts to keep. Lines are
    taken from the start and the end in turn until ``budget_tokens`` or
    ``limit`` lines are used up.

    Returned as cues rather than text because the citation check has to run
    against exactly this list. Checking a quote against the whole meeting would
    let a citation point at a passage that model call never read.
    """
    if len(cues) <= limit and sum(_line_tokens(cue) for cue in cues) <= budget_tokens:
        return _Excerpt(list(cues), [], 0)
    head: list[Cue] = []
    tail: list[Cue] = []
    used = _OMISSION_TOKENS
    first, last = 0, len(cues) - 1
    from_start = True
    while first <= last and len(head) + len(tail) < limit:
        cue = cues[first] if from_start else cues[last]
        cost = _line_tokens(cue)
        if used + cost > budget_tokens:
            break
        used += cost
        if from_start:
            head.append(cue)
            first += 1
        else:
            tail.insert(0, cue)
            last -= 1
        from_start = not from_start
    return _Excerpt(head, tail, len(cues) - len(head) - len(tail))


def _shown_cues(cues: list[Cue], limit: int = MAX_OVERVIEW_CUES, budget_tokens: int | None = None) -> list[Cue]:
    """The cues a single-prompt pass is actually given, in order."""
    budget = budget_tokens if budget_tokens is not None else 10**9
    return _excerpt(cues, budget, limit).cues


def _for_one_prompt(cues: list[Cue], limit: int = MAX_OVERVIEW_CUES, budget_tokens: int | None = None) -> str:
    """The text a single-prompt pass is given; see ``_excerpt``."""
    budget = budget_tokens if budget_tokens is not None else 10**9
    return _excerpt(cues, budget, limit).render()


async def _complete(scheduler: Scheduler, model: str, prompt: str, reply_tokens: int) -> str:
    """One call as a job the employee asked for, with its reply limit."""
    options = RequestOptions(max_tokens=reply_tokens, reasoning=False)
    reply = await scheduler.complete(
        model, [{"role": "user", "content": prompt}], priority=Priority.USER_JOB, options=options
    )
    return reply.strip()


def _repair_timestamp(value: object, evidence: object, cues: list[Cue]) -> str | None:
    """The timestamp a quoted line earns, checked against the words themselves.

    Kept as a name of its own because it reads at the call site, but the rule
    lives in ``citations.resolve_timestamp``: a timestamp survives only when the
    quote is found in a cue that really carries it. A timestamp that merely
    exists somewhere in the meeting proves nothing.
    """
    return resolve_timestamp(evidence, value, cues)


# The key is looked for, not its value. A model that sent ``"quote": ""`` made a
# broken claim rather than no claim, so it does not get a second attempt at the
# paraphrase - but a reply from the older prompt, with no ``quote`` key at all,
# still has its text tried as evidence against the same strict matcher.
_MISSING = object()


def _evidence_for(item: dict, text: str) -> object:
    supplied = item.get("quote", _MISSING)
    return text if supplied is _MISSING else supplied


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


def _points(raw_items: object, cues: list[Cue], with_owner: bool) -> list[Point]:
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
                timestamp=_repair_timestamp(
                    item.get("timestamp"), _evidence_for(item, text), cues
                ),
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


def _verified_notes(reply: str, cues: list[Cue], rough: str) -> str:
    """The rewritten notes, carrying only citations that checked out.

    Three shapes arrive here and all three have to end as readable notes:

    - The envelope that was asked for: its markers are resolved against its own
      citation records, and anything unsupported is erased.
    - Plain Markdown, because a small model ignored the JSON instruction. The
      prose is kept and every citation in it is removed unverified - it was
      never backed by a quote anybody could check.
    - An envelope with no usable markdown in it. Their own rough notes are
      returned instead, because rendering a JSON blob where somebody's notes
      should be is the one outcome worse than a plain rewrite.
    """
    try:
        envelope = decode_json_reply(reply, allow_surrounding_text=True)
    except (ValueError, json.JSONDecodeError):
        return strip_citation_markup(reply.strip())

    if not isinstance(envelope, dict):
        return strip_citation_markup(reply.strip())

    markdown = envelope.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        return rough.strip()
    return apply_citations(markdown.strip(), envelope.get("citations"), cues)


async def summarize_meeting(
    cues: list[Cue],
    *,
    model: str,
    scheduler: Scheduler | None = None,
    backend: InferenceBackend | None = None,
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

    Calls go through ``scheduler``, whose profile sets the window every prompt
    is sized against. A plain ``backend`` also works; it is given a scheduler
    of its own with the default profile.
    """
    if scheduler is None:
        if backend is None:
            raise ValueError("summarize_meeting needs a scheduler or a backend")
        scheduler = Scheduler(backend, PROFILES[DEFAULT_PROFILE])
    profile = scheduler.profile
    facts = compute_facts(cues, title=title, meeting_date=meeting_date)
    summary = MeetingSummary(facts=facts)
    if not cues:
        summary.degraded_reason = "the transcript was empty"
        return summary

    extract_room = prompt_budget(profile.context_window, EXTRACT_REPLY_TOKENS)
    extract_template = estimate_tokens(_EXTRACT_PROMPT.format(transcript=""))
    chunks = chunk_cues(cues, words_per_chunk=words_per_chunk, max_tokens=extract_room - extract_template)
    decisions: list[Point] = []
    actions: list[Point] = []
    details: list[Point] = []
    unsent_lines = 0

    try:
        for chunk in chunks:
            prompt = _EXTRACT_PROMPT.format(transcript=_render(chunk))
            if estimate_tokens(prompt) > extract_room:
                unsent_lines += len(chunk)  # one line longer than the whole window
                continue
            reply = await _complete(scheduler, model, prompt, EXTRACT_REPLY_TOKENS)
            try:
                data = decode_json_reply(reply, allow_surrounding_text=True)
            except (ValueError, json.JSONDecodeError):
                # One unparseable chunk should not lose the other nine.
                continue
            decisions += _points(data.get("decisions"), chunk, with_owner=False)
            actions += _points(data.get("actions"), chunk, with_owner=True)
            details += _points(data.get("facts"), chunk, with_owner=False)
        if unsent_lines:
            summary.notices.append(
                f"{unsent_lines} transcript line(s) were too long for the model's "
                f"{profile.context_window:,}-token window and were not read for decisions and actions."
            )

        # Both single-prompt passes are checked against the cues they were
        # shown, never the whole meeting.
        if notes and notes.strip():
            await _rewrite_notes(summary, scheduler, model, cues, notes)
        else:
            reply_tokens = OVERVIEW_REPLY_TOKENS
            room = prompt_budget(profile.context_window, reply_tokens) - estimate_tokens(
                _OVERVIEW_PROMPT.format(transcript="")
            )
            excerpt = _excerpt(cues, room)
            _note_trimmed(summary, excerpt, "overview")
            overview = await _complete(
                scheduler, model, _OVERVIEW_PROMPT.format(transcript=excerpt.render()), reply_tokens
            )
            # The overview prompt does not ask for citations. One that appears
            # anyway is the model's own invention, so it goes without checking.
            summary.overview = strip_citation_markup(overview)
    except InferenceError as exc:
        summary.degraded_reason = str(exc)
        return summary

    summary.decisions = _dedupe(decisions)
    summary.actions = _dedupe(actions)
    summary.details = _dedupe(details)
    summary.model_used = model
    return summary


MIN_NOTES_TRANSCRIPT_TOKENS = 200
"""Rewriting notes against less transcript than this would be guesswork."""


async def _rewrite_notes(summary: MeetingSummary, scheduler: Scheduler, model: str, cues: list[Cue], notes: str) -> None:
    """The notes pass: their notes, filled in from as much transcript as fits."""
    profile = scheduler.profile
    reply_tokens = profile.reply_limit
    room = prompt_budget(profile.context_window, reply_tokens) - estimate_tokens(
        _NOTES_PROMPT.format(notes=notes.strip(), transcript="")
    )
    if room < MIN_NOTES_TRANSCRIPT_TOKENS:
        summary.notices.append(
            f"Your notes are too long to rewrite within the model's {profile.context_window:,}-token "
            "window, so they are left as you wrote them. Shorten them, or split them and run again."
        )
        summary.enhanced_notes = notes.strip()
        return
    excerpt = _excerpt(cues, room)
    _note_trimmed(summary, excerpt, "notes")
    reply = await _complete(
        scheduler, model, _NOTES_PROMPT.format(notes=notes.strip(), transcript=excerpt.render()), reply_tokens
    )
    summary.enhanced_notes = _verified_notes(reply, excerpt.cues, notes)


def _note_trimmed(summary: MeetingSummary, excerpt: _Excerpt, what: str) -> None:
    if excerpt.skipped:
        summary.notices.append(
            f"The meeting is too long to send whole, so the {what} was written from its first "
            f"and last parts; {excerpt.skipped} lines from the middle were left out."
        )
