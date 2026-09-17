"""Checking that a meeting note's timestamp points at the moment it describes.

A citation can be wrong in two ways, and only one of them is obvious. A model
that writes ``[00:47:00]`` for a six-minute meeting has invented a moment; that
is easy to catch, because no line starts then. The quieter failure is a real
timestamp on the wrong line: an action item citing the cue six seconds after
the one where it was actually agreed. It looks verified - that moment exists -
and sends the reader to the wrong place. Both were seen in testing.

So a citation is checked against the line it points at, by the distinctive
words the note and that line share:

* **kept** when that line shares two or more, or shares one and no other line
  shares more. Two is enough on its own because one note often draws on
  several lines ("coordinator left, so no new site [00:00:06, 00:00:12]"), and
  each line it cites only has to say its part;
* **moved** otherwise, to the line that shares the most, if that is at least
  two - one shared word is too weak to pick a different line out of a meeting;
* **removed** when neither holds.

A note with no timestamp is honest; a note with a wrong one is not.

"Says the thing" is measured by shared words, not meaning. Numbers count at any
length, because "148 of 200" is exactly what a fact is about; other words count
from four letters, minus common function words, with regular plurals folded so
"numbers" meets "number". The limit is the same one keyword search has: a note
that paraphrases its line without sharing a single distinctive word with it
cannot be verified, so its citation is removed rather than guessed at.

The same check runs on two shapes of output. Structured points (decisions,
action items, facts) carry their timestamp in a field. Free-text notes and the
overview carry ``[hh:mm:ss]`` (or ``(hh:mm:ss)``) inline, and
:meth:`CitationGuard.clean_text` checks each one against the words of its own
sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bedrock.context.search import stem
from bedrock.meetings.transcript import Cue

# Shared words that count as a line saying something. Enough to keep the
# model's citation whatever other lines say, and needed before a citation is
# moved to, or added at, a line the model did not choose. One shared word is
# too weak to pick a line out of a whole meeting, though it confirms the line
# the model chose when no other line does better.
MIN_SHARED_WORDS = 2

# Common words of four letters or more that say nothing about which line a note
# came from. Kept short, like the search stopwords: every word added here is
# one more that can never count as evidence.
_FILLER = frozenset(
    """
    about after also back been before being could didn does doing done from going
    have here into just know like make more much need only other really should
    some that their them then there these they thing think this those very want
    well were what when where which while will with would yeah your
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+")

# One bracketed group on one line: ``[00:12:34]``, ``[12:34]``,
# ``[00:12:34, 00:13:02]``, ``[00:12:34 - 00:12:50]`` or ``(00:12:34)``.
# Whether its contents really are times is decided by _parse_times, so ``[x]``
# and ``[TBD]`` pass through. Round brackets only count with hours in every
# time, because "(10:30)" in a note is far more likely a time of day.
_BRACKET = re.compile(r"\[([^\[\]\n]{1,80})\]|\(([^()\n]{1,80})\)")
_TIME = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,]\d{1,3})?")
_SEPARATOR = re.compile(r"\s*(?:,|;|-|–|—|\bto\b)\s*")
# Where one sentence ends and the next begins, for finding what a citation
# cites. Not ";": "samples moved to freezer five; QA assessing impact [..]"
# cites the whole clause pair, and splitting there strips a correct citation.
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_FENCE = re.compile(r"^\s*(```|~~~)")


def content_words(text: str) -> set[str]:
    """The words that tell one line of a meeting apart from another."""
    words: set[str] = set()
    for token in _TOKEN.findall(text.lower()):
        if any(ch.isdigit() for ch in token):
            words.add(token)
        elif len(token) > 3 and token not in _FILLER:
            words.add(stem(token))
    return words


def normalise_timestamp(value: object) -> str | None:
    """``hh:mm:ss``, the form cues use, or None if this is not a time.

    Accepts ``0:12:34``, ``12:34`` (minutes and seconds) and a trailing
    fraction, because models copy the format loosely.
    """
    if not isinstance(value, str):
        return None
    match = _TIME.fullmatch(value.strip())
    if match is None:
        return None
    hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    if minutes > 59 or seconds > 59:
        return None
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class CitationReport:
    """What the guard did to the citations a model wrote.

    ``checked`` counts every timestamp the model supplied. Of those,
    ``moved`` now point at a different line and ``removed`` were dropped. A
    point the model left uncited, which the guard cited from its words, is not
    counted: there was no model citation to check.
    """

    checked: int = 0
    moved: int = 0
    removed: int = 0

    def record(self, supplied: bool, original: str | None, result: str | None) -> None:
        if not supplied:
            return
        self.checked += 1
        if result is None:
            self.removed += 1
        elif result != original:
            self.moved += 1

    def describe(self) -> str | None:
        """One line for the person running the tool, or None if nothing was cited."""
        if not self.checked:
            return None
        noun = "citation" if self.checked == 1 else "citations"
        return (
            f"Checked {self.checked} {noun} against the transcript: "
            f"{self.moved} moved to the line that says it, "
            f"{self.removed} removed because no line says it."
        )


class CitationGuard:
    """Checks citations against one set of cues.

    Built once per set of cues so each cue's words are worked out once, not
    once per citation. Several guards can share one :class:`CitationReport`,
    which is how a meeting read in chunks still reports a single total.
    """

    def __init__(self, cues: list[Cue], report: CitationReport | None = None) -> None:
        self._cues = [(cue, content_words(cue.text)) for cue in cues]
        self.report = report if report is not None else CitationReport()

    def check(self, value: object, text: str) -> str | None:
        """The timestamp ``text`` should cite, given the model's ``value``.

        ``value`` is whatever the model put in the timestamp field, including
        nothing. The result is the cited line's timestamp if that line says
        the thing, else the timestamp of the line that says it best, else None.
        """
        supplied = value is not None and str(value).strip().lower() not in {"", "null", "none"}
        original = normalise_timestamp(value)
        result = self._resolve(original, content_words(text))
        self.report.record(supplied, original, result)
        return result

    def _resolve(self, cited: str | None, needle: set[str]) -> str | None:
        if not needle:
            return None  # nothing distinctive to check it against
        best: Cue | None = None
        best_shared = 0
        cited_shared = -1  # stays -1 when no cue starts at the cited time
        for cue, words in self._cues:
            shared = len(needle & words)
            # Separate tracks can start two cues in the same second; the cited
            # time is as good as the better of them.
            if cited is not None and cue.timestamp == cited:
                cited_shared = max(cited_shared, shared)
            if shared > best_shared:  # strictly greater: the earliest line wins a tie
                best, best_shared = cue, shared
        if cited_shared >= MIN_SHARED_WORDS or (cited_shared >= 1 and cited_shared >= best_shared):
            return cited
        if best is not None and best_shared >= MIN_SHARED_WORDS:
            return best.timestamp
        return None

    def clean_text(self, markdown: str, *, leave_alone: str = "") -> str:
        """Check every ``[hh:mm:ss]`` and ``(hh:mm:ss)`` citation in free text.

        Each citation is checked against its own sentence - the words between
        the previous sentence break or citation and itself - or, for one that
        leads its line (``- [00:12:34] Budget approved``), against the words
        after it. Fenced code is left untouched.

        ``leave_alone`` is the person's own rough notes. A bracketed time they
        typed themselves (``[14:00] standup``) is theirs, not a citation, so an
        identical bracket in the output is never changed.
        """
        theirs = {match.group(0) for match in _BRACKET.finditer(leave_alone)}
        lines = markdown.split("\n")
        in_fence = False
        for index, line in enumerate(lines):
            if _FENCE.match(line):
                in_fence = not in_fence
                continue
            if not in_fence and ("[" in line or "(" in line):
                lines[index] = self._clean_line(line, theirs)
        return "\n".join(lines)

    def _clean_line(self, line: str, theirs: set[str]) -> str:
        groups = []
        for match in _BRACKET.finditer(line):
            square = match.group(1) is not None
            times = _parse_times(match.group(1) if square else match.group(2), hours_required=not square)
            if times and match.group(0) not in theirs:
                groups.append((match, times, square))
        if not groups:
            return line

        replacements: list[tuple[int, int, str]] = []
        previous_end = 0
        for position, (match, times, square) in enumerate(groups):
            next_start = groups[position + 1][0].start() if position + 1 < len(groups) else len(line)
            text = _cited_text(line[previous_end : match.start()], line[match.end() : next_start])
            previous_end = match.end()

            checked = [self.check(time, text) for time in times]
            if checked == [normalise_timestamp(time) for time in times]:
                continue  # every time stands as written, so leave the text alone
            kept = list(dict.fromkeys(result for result in checked if result is not None))
            opening, closing = ("[", "]") if square else ("(", ")")
            replacement = f"{opening}{', '.join(kept)}{closing}" if kept else ""
            replacements.append((match.start(), match.end(), replacement))

        for start, end, new in reversed(replacements):
            if not new:
                # Take one neighbouring space with the citation, so neither
                # "done [00:47:00]." nor "- [00:47:00] done" is left with a gap.
                if start > 0 and line[start - 1] == " ":
                    start -= 1
                elif end < len(line) and line[end] == " ":
                    end += 1
            line = line[:start] + new + line[end:]
        return line


def _parse_times(inside: str, *, hours_required: bool) -> list[str] | None:
    """The times in one bracket, or None unless it holds times and nothing else."""
    parts = [part for part in _SEPARATOR.split(inside.strip()) if part]
    if not parts or not all(normalise_timestamp(part) for part in parts):
        return None
    if hours_required and not all(part.count(":") == 2 for part in parts):
        return None
    return parts


def _cited_text(before: str, after: str) -> str:
    """The words a citation vouches for.

    Normally the sentence it ends: the last sentence before it that has any
    distinctive words, so "Enrollment is 148 of 200. Site four lost its
    coordinator [00:00:06]" is checked against the second sentence only. A
    citation with nothing before it (``- [00:00:06] Site four lost its
    coordinator``) vouches for what follows instead.
    """
    for sentence in reversed(_SENTENCE_END.split(before)):
        if content_words(sentence):
            return sentence
    return after
