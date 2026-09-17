"""Checking that a citation really points at the words it claims to quote.

A meeting note that says *"we're not opening a new site [00:00:12]"* is making a
checkable claim: that those words were said at that moment. Before this module
existed the check was much weaker than it looked. A timestamp was accepted
whenever it appeared *anywhere* in the transcript, so a real timestamp attached
to the wrong sentence sailed through, and the fallback used loose word overlap,
which will happily match two common words shared by unrelated lines.

So the rule here is deliberately narrow, and it is worth being clear about what
it does and does not prove.

**What a validated timestamp means.** The model supplied a verbatim excerpt, and
that excerpt was found inside exactly one identifiable transcript cue, whose own
timestamp is the one printed. That is a claim about *where the quote is*.

**What it does not mean.** It is not a proof that the paraphrased decision
written next to the quote follows from it - a model can still summarise badly
while quoting accurately. It is also not word-level audio alignment:
``Cue.timestamp`` floors the cue's start to a whole second, so the citation
points at the start of the utterance, not at the instant the quoted word was
spoken.

**Why no model is involved.** A checker that asks a model whether the model was
right is not a check. Everything below is plain string work: deterministic, fast,
and it still runs with every server switched off.

The matcher is strict on purpose. Losing a citation costs a reader a convenience;
keeping a wrong one costs them the ability to trust any of them.
"""

from __future__ import annotations

import re
from collections import Counter

from bedrock.meetings.transcript import Cue

# Evidence shorter than this is not evidence. "Yes." appears in most meetings
# several times over, so quoting it locates nothing; three lexical tokens is the
# smallest span that usually identifies one utterance, and a shorter quote loses
# the citation rather than guessing at which "yes" was meant.
MIN_EVIDENCE_TOKENS = 3

# Word processors and models both produce curly quotes where the transcript has
# straight ones. Folding them together is safe: it cannot merge two different
# words, change a number, or flip a negation.
_TYPOGRAPHIC = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
    }
)

_LEXICAL = re.compile(r"[^\W_]+", re.UNICODE)

# The one bracketed-timestamp form this module recognises as a citation: square
# brackets - one pair or two, since models produce both - containing nothing but
# a clock time. "[00:12:34]" and "[12:34]" are citations; "we start at 10:30",
# "(00:12:34)" and "[4]" are ordinary prose and are left alone. Keeping the
# syntax this narrow is what lets the notes be scrubbed of unverified citations
# without touching anything the user meant to write.
_CLOCK = r"\s*\d{1,2}:\d{2}(?::\d{2})?\s*"
RAW_TIMESTAMP_CITATION = re.compile(
    rf"[ \t]*(?:\[\[{_CLOCK}\]\]|\[{_CLOCK}\])"
)


def normalize(text: str) -> str:
    """Case, whitespace and quote shape only.

    Everything that carries meaning survives: word order, numbers, negation and
    punctuation are all left exactly as they are, so "not opening" can never
    match "opening" and "148" can never match "140".
    """
    return " ".join(text.translate(_TYPOGRAPHIC).casefold().split())


def _is_word_char(char: str) -> bool:
    return bool(char) and (char.isalnum() or char == "_")


def _boundary_ok(neighbour: str, edge: str) -> bool:
    """A boundary is only required where two word characters would collide.

    This is what ``\\b`` means in a regular expression, and it is why "site
    four" does not match inside "offsite fourteen" while a quote that genuinely
    begins with punctuation still matches.
    """
    return not (_is_word_char(neighbour) and _is_word_char(edge))


def contains(haystack: str, needle: str) -> bool:
    """Is ``needle`` a contiguous, whole-word span of ``haystack``?

    Both arguments must already be normalized.
    """
    if not needle:
        return False
    start = haystack.find(needle)
    while start != -1:
        before = haystack[start - 1] if start else ""
        end = start + len(needle)
        after = haystack[end] if end < len(haystack) else ""
        if _boundary_ok(before, needle[0]) and _boundary_ok(after, needle[-1]):
            return True
        start = haystack.find(needle, start + 1)
    return False


def prepare_evidence(evidence: object) -> str | None:
    """Normalize a supplied quote, or reject it as unusable as evidence.

    Rejects anything that is not a string, is empty or whitespace-only, is made
    of punctuation alone, or is too short to identify one utterance.
    """
    if not isinstance(evidence, str):
        return None
    needle = normalize(evidence)
    if len(_LEXICAL.findall(needle)) < MIN_EVIDENCE_TOKENS:
        return None
    return needle


def resolve_timestamp(evidence: object, suggested: object, cues: list[Cue]) -> str | None:
    """The timestamp a quote earns, or ``None`` if it has not earned one.

    ``evidence`` is the excerpt the model says it copied from the transcript,
    ``suggested`` the timestamp it offered, and ``cues`` exactly the cues that
    model call was shown - never a wider search, or the citation would point at
    a passage nobody read.

    The four outcomes:

    - The quote is nowhere in these cues: no citation. An existing but unrelated
      timestamp does not rescue it.
    - The suggested timestamp belongs to a cue that really contains the quote:
      keep it. This is also how a quote repeated later in the meeting gets
      pinned to the occurrence the model meant.
    - Every cue containing the quote shows the same timestamp: repair to it,
      whichever wrong, missing or malformed value was suggested.
    - The quote appears at two different times and nothing chooses between them:
      no citation. Picking the earliest or the nearest would be a guess printed
      as a fact.
    """
    needle = prepare_evidence(evidence)
    if needle is None:
        return None

    # Every cue is tested, not one per displayed second: two speakers can both
    # start inside the same second, and a mapping keyed by timestamp would throw
    # one of them away.
    matches = [cue for cue in cues if contains(normalize(cue.text), needle)]
    if not matches:
        return None

    if isinstance(suggested, str):
        wanted = suggested.strip()
        if any(cue.timestamp == wanted for cue in matches):
            return wanted

    found = {cue.timestamp for cue in matches}
    return found.pop() if len(found) == 1 else None


def strip_raw_timestamp_citations(text: str) -> str:
    """Remove bracketed timestamps a model wrote itself.

    Nothing has verified these. They also arrive by a second route: the rough
    notes someone typed during the call may contain "[00:14:00]", and the model
    copies it into its rewrite, where it looks exactly like a generated
    citation. The user's own file is never touched - only the generated output.
    """
    return _tidy(RAW_TIMESTAMP_CITATION.sub("", text))


def _tidy(text: str) -> str:
    """Drop the trailing whitespace that removing a citation leaves behind."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


# The placeholder the notes prompt asks for. Narrow on purpose: an id is a short
# run of non-space, non-bracket characters, so nothing in ordinary prose can be
# mistaken for one. The single-bracket spelling is accepted too because small
# models drop a bracket level often enough to be worth tolerating.
_CITE_MARKER = re.compile(
    r"[ \t]*\[\[cite:\s*(?P<double>[^\]\s]{1,64})\s*\]\]"
    r"|[ \t]*\[cite:\s*(?P<single>[^\]\s]{1,64})\s*\]"
)


def _supported_ids(records: object, cues: list[Cue]) -> dict[str, str]:
    """Map each citation id that survives checking to its verified timestamp.

    An id used twice is dropped rather than resolved, because the two records
    can disagree and there is no way to tell which one a placeholder meant.
    """
    rows = [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []

    keys = [
        record["id"].strip()
        for record in rows
        if isinstance(record.get("id"), str) and record["id"].strip()
    ]
    used_once = {key for key, count in Counter(keys).items() if count == 1}

    supported: dict[str, str] = {}
    for record in rows:
        key = record.get("id")
        key = key.strip() if isinstance(key, str) else ""
        if key not in used_once:
            continue
        timestamp = resolve_timestamp(record.get("quote"), record.get("timestamp"), cues)
        if timestamp is not None:
            supported[key] = timestamp
    return supported


def apply_citations(markdown: str, records: object, cues: list[Cue]) -> str:
    """Turn verified citation placeholders into timestamps; erase the rest.

    Order matters. Any bracketed timestamp the model wrote directly is removed
    *first*, so an unverified citation cannot survive by being written in the
    output format instead of as a placeholder. Only then are the checked ones
    written in.

    A placeholder whose id is unknown, used twice, or backed by evidence that
    does not appear in the transcript is deleted along with the space in front
    of it. The sentence it was attached to stays: losing a citation must never
    cost the reader the note.
    """
    cleaned = RAW_TIMESTAMP_CITATION.sub("", markdown)
    supported = _supported_ids(records, cues)

    def replace(match: re.Match[str]) -> str:
        timestamp = supported.get(match.group("double") or match.group("single") or "")
        if timestamp is None:
            return ""
        before = cleaned[: match.start()]
        return f" [{timestamp}]" if before and not before.endswith("\n") else f"[{timestamp}]"

    return _tidy(_CITE_MARKER.sub(replace, cleaned))


def strip_citation_markup(text: str) -> str:
    """Remove every citation form without verifying any of it.

    Used where a response arrived in a shape we cannot check - plain Markdown
    where an envelope was asked for, or an overview that was told not to cite
    and did anyway. The prose survives; the unverifiable citations do not.
    """
    return _tidy(_CITE_MARKER.sub("", RAW_TIMESTAMP_CITATION.sub("", text)))
