"""Putting the company's own words back into a transcript.

Whisper spells what it does not know phonetically. `MEETING_VOCABULARY` seeds
the decoder with the real terms, which helps a great deal but is a bias, not a
guarantee: across test runs of the same 28 seconds of audio, "KRAS" came out
correct once and as "CRAS" the next time, and "Priya" as "PREA". For a company
whose meetings are *about* gene and drug names, a transcript that says CRAS is
not merely untidy — it is unsearchable, because retrieval matches on words.

So after transcription, each near-miss of a configured term is repaired. This
is deliberately conservative, because silently rewriting words in a record of
what somebody said is the kind of help that can do real damage:

* only single-word terms are considered. A multi-word entry like "interim
  analysis" is left to the decoder hint alone, because deciding where a phrase
  begins in a stream of words invites exactly the mistake this guards against;
* a token has to be *close* — one edit for a short term, two for a long one,
  and similar enough — and not already the term, so ordinary text is untouched.
  The edit budget is the guard that matters: on a four-letter name, two edits
  reaches ordinary English, and "Rain delayed the site visit" would become
  "Ravi delayed the site visit";
* an inflection is never "corrected". With "cohort" configured, the word
  "cohorts" is right as it stands, and replacing it would corrupt the sentence;
* every substitution is recorded and reported, so a reader can see that the
  transcript was edited and what it said before.

The last point is the important one. The rest of this product is careful to
distinguish what was said from what a model inferred — that is what
`citations.py` exists for — and a repair that hid itself would quietly undo
that.

Deliberately, this catches less than it could. "CRAS" is one edit from "KRAS"
and is repaired; "PREA" is two edits from "Priya" and is not, because allowing
two edits on a name that short also lets "Rain" become "Ravi". Whisper's own
vocabulary hint is what gets those right most of the time — this is the safety
net under it, and a net that grabs the wrong word is worse than no net.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Below this, two short words that merely rhyme start matching each other.
# "grass" is 0.67 similar to "KRAS" and must survive untouched.
MIN_SIMILARITY = 0.72
# A token already shaped like a proper noun gets a little more latitude on
# similarity, because the shape is itself evidence.
NAME_SIMILARITY = 0.6
MIN_TOKEN_LENGTH = 3

# How far a token may be from a term before it is a different word. This is
# scaled by term length, and it is the guard that matters most: on a short
# name two edits reaches ordinary vocabulary. "Rain" is two edits and 75%
# similar to "Ravi", which is more than enough to pass every similarity rule
# here — and rewriting "Rain delayed the site visit" into "Ravi delayed the
# site visit" would put words in somebody's mouth. One edit on a short term
# keeps that from being possible at all.
SHORT_TERM_LENGTH = 5


def max_edits_for(term: str) -> int:
    return 1 if len(term) <= SHORT_TERM_LENGTH else 2

# Word characters plus the internal hyphens and digits that gene and compound
# names are full of: "G12C", "BRD-1192", "PD-L1".
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]*")

# A cue's timing line, which must never be rewritten.
_TIMING = re.compile(r"-->")

_INFLECTIONS = ("s", "es", "ed", "ing", "'s")


@dataclass
class Correction:
    heard: str
    corrected: str
    count: int = 1


@dataclass
class VocabularyReport:
    corrections: list[Correction] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(c.count for c in self.corrections)

    def describe(self) -> str:
        """One line naming every repair, for the meeting's provenance."""
        if not self.corrections:
            return ""
        parts = ", ".join(
            f"{c.heard}→{c.corrected}" + (f" ×{c.count}" if c.count > 1 else "")
            for c in self.corrections
        )
        return f"vocabulary repaired {self.total} word(s): {parts}"


def parse_vocabulary(raw: str | None) -> list[str]:
    """The single-word terms from a `MEETING_VOCABULARY` setting."""
    if not raw:
        return []
    terms: list[str] = []
    for entry in re.split(r"[,;\n]", raw):
        entry = entry.strip()
        # Multi-word entries still reach whisper as a decoder hint; they are
        # just not candidates for after-the-fact repair.
        if entry and " " not in entry and len(entry) >= MIN_TOKEN_LENGTH:
            terms.append(entry)
    return terms


def _edit_distance(a: str, b: str, cap: int) -> int:
    """Levenshtein distance, giving up once it cannot come in under `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def _is_sentence_initial(text: str, start: int) -> bool:
    """Whether this token opens a sentence, where a capital means nothing."""
    before = text[:start].rstrip()
    return not before or before[-1] in ".!?"


def _is_inflection(token: str, term: str) -> bool:
    lowered, base = token.casefold(), term.casefold()
    return any(lowered == base + suffix for suffix in _INFLECTIONS)


# Words that are never a mis-hearing of anything, however close they look.
#
# This list is the difference between a useful repair and a corrupted record.
# "Luke" is one edit and 75% similar to "like", and on a real recording the
# repair turned three spoken "like"s into "Luke" — in a meeting that actually
# had somebody called Luke in it, so the result read as plausible and was
# wrong. Common speech is dense with short words that sit one edit from a
# name: like/Luke, well/Will, man/Max, rain/Ravi, the/Thea, sam/same.
#
# Frequency is the right test — a word this common in speech is far more
# likely to be itself than to be a fumbled proper noun — and a fixed list is
# enough, because the failure mode is confined to very common short words.
_COMMON_WORDS = frozenset(
    """
    the be to of and a in that have i it for not on with he as you do at this
    but his by from they we say her she or an will my one all would there their
    what so up out if about who get which go me when make can like time no just
    him know take people into year your good some could them see other than then
    now look only come its over think also back after use two how our work first
    well way even new want because any these give day most us is are was were
    been has had did does said says going got went come came thing things
    something anything nothing everyone someone yeah yes okay ok right sure
    maybe really very much many more less lest here there where why while
    before during again still every each both few own same such through
    between under above down off out over again once
    man men woman women boy girl kid guy name names call called
    rain sun cold warm class exam test paper note notes
    """.split()
)


def is_common_word(token: str) -> bool:
    return token.casefold().strip("'s") in _COMMON_WORDS


def looks_like_a_name(token: str, sentence_initial: bool) -> bool:
    """Whether the token is shaped like a proper noun rather than a word.

    This is what separates the two cases that string distance cannot. "PREA"
    and "grass" are both two edits and about 66% similar to a configured term
    ("Priya", "KRAS"), but one is whisper fumbling a name and the other is an
    ordinary English word used correctly. The shape tells them apart: whisper
    writes an unfamiliar proper noun as capitals or a capitalised oddity, and
    writes real words in lower case.

    A capital at the start of a sentence says nothing — every sentence has one —
    so it is not counted.
    """
    if any(character.isdigit() for character in token):
        return True  # G12C, PD-L1, BRD-1192
    if token.isupper() and len(token) >= MIN_TOKEN_LENGTH:
        return True
    return token[:1].isupper() and not sentence_initial


def _match(token: str, terms: list[str], sentence_initial: bool = True) -> str | None:
    """The term this token is a misspelling of, or None to leave it alone."""
    if len(token) < MIN_TOKEN_LENGTH:
        return None
    # An ordinary word spelled in lower case is itself. Capitals still qualify,
    # because whisper writes an unfamiliar name as capitals and "LUKE" in the
    # middle of a sentence is not the adverb.
    if not looks_like_a_name(token, sentence_initial) and is_common_word(token):
        return None
    lowered = token.casefold()
    # A name-shaped token is allowed a looser similarity, because the shape is
    # itself evidence. An ordinary lower-case word has to be a very close miss.
    threshold = NAME_SIMILARITY if looks_like_a_name(token, sentence_initial) else MIN_SIMILARITY
    best: tuple[float, str] | None = None
    for term in terms:
        if lowered == term.casefold() or _is_inflection(token, term):
            return None  # already right, or a legitimate variation of it
        budget = max_edits_for(term)
        if _edit_distance(lowered, term.casefold(), budget) > budget:
            continue
        ratio = SequenceMatcher(None, lowered, term.casefold()).ratio()
        if ratio >= threshold and (best is None or ratio > best[0]):
            best = (ratio, term)
    return best[1] if best else None


class _Repairer:
    """Applies the corrections and keeps one report across many lines."""

    def __init__(self, terms: list[str]) -> None:
        self.terms = terms
        self.report = VocabularyReport()
        self._seen: dict[tuple[str, str], Correction] = {}

    def line(self, text: str) -> str:
        return _TOKEN.sub(self._token, text)

    def _token(self, match: re.Match) -> str:
        token = match.group(0)
        term = _match(token, self.terms, _is_sentence_initial(match.string, match.start()))
        if term is None:
            return token
        key = (token, term)
        if key in self._seen:
            self._seen[key].count += 1
        else:
            correction = Correction(heard=token, corrected=term)
            self._seen[key] = correction
            self.report.corrections.append(correction)
        return term


def correct_texts(texts: list[str], vocabulary: str | None) -> tuple[list[str], VocabularyReport]:
    """Repair a list of spoken lines — one cue's text each.

    This is the form the pipeline uses, because the summariser reads cues, not
    the rendered WEBVTT. Correcting the rendered text alone would leave the
    notes still saying CRAS while the transcript underneath said KRAS.
    """
    terms = parse_vocabulary(vocabulary)
    if not terms:
        return list(texts), VocabularyReport()
    repairer = _Repairer(terms)
    return [repairer.line(text) for text in texts], repairer.report


def correct(vtt: str, vocabulary: str | None) -> tuple[str, VocabularyReport]:
    """Repair near-misses of the configured terms throughout a WEBVTT transcript."""
    terms = parse_vocabulary(vocabulary)
    if not terms or not vtt:
        return vtt, VocabularyReport()

    repairer = _Repairer(terms)
    lines = []
    for line in vtt.splitlines():
        # Header, cue numbers and timing lines carry no speech.
        if not line.strip() or _TIMING.search(line) or line.strip().isdigit() or line.strip() == "WEBVTT":
            lines.append(line)
            continue
        lines.append(repairer.line(line))
    return "\n".join(lines), repairer.report
