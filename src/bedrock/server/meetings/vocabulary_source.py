"""Building the vocabulary from the company's own words, instead of by hand.

`MEETING_VOCABULARY` is the highest-value setting the recorder has: whisper
spells what it does not know phonetically, and seeding the decoder with the
real terms is what turns "CRAS" back into "KRAS". It was also hand-maintained,
which means in practice it is written once, is wrong within a month, and nobody
notices until a transcript is unsearchable.

The terms are already in the building. Gene symbols, compound codes and
colleagues' names are all over the documents this system already reads and the
meetings it has already transcribed. This harvests them.

What counts as a term is the whole question, and the answer is deliberately
narrow. A word is a candidate only if it is *shaped* like jargon — capitals,
internal digits, a hyphen joining letters to numbers — and appears more than
once, and is not ordinary English. That misses lower-case drug names and it is
the right trade: a vocabulary padded with common words biases the decoder
towards them, which makes transcripts worse, and the failure is silent.

Nothing here runs automatically. `suggest()` returns candidates for somebody to
look at; a deployment that trusts them sets `MEETING_VOCABULARY_AUTO=true` and
accepts that the decoder is being steered by a heuristic.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Shapes that mean "this is a name or a code, not a word":
#   KRAS, EGFR, PD-L1   — two or more capitals, optionally hyphen-digits
#   G12C, BRD-1192      — letters followed by digits
#   Sotorasib           — capitalised and long enough not to be a sentence start
_ALL_CAPS = re.compile(r"^[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)*$")
_ALPHANUMERIC = re.compile(r"^[A-Za-z]{1,6}[-]?\d{1,4}[A-Za-z]?$")
_CAPITALISED = re.compile(r"^[A-Z][a-z]{4,}$")

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]*")

MIN_OCCURRENCES = 2
MAX_TERMS = 120

# Capitalised words that are just sentences starting, or calendar noise. A
# vocabulary containing "Monday" steers the decoder for no benefit.
_NOT_JARGON = frozenset(
    """monday tuesday wednesday thursday friday saturday sunday january february
    march april may june july august september october november december
    okay right yeah hello thanks thank sorry please today tomorrow yesterday
    morning afternoon evening tonight meeting meetings agenda minutes action
    actions decision decisions update updates project projects team teams
    google zoom slack email report reports review reviews number numbers
    because before after during while there their about would could should
    something anything everything nothing someone everyone another""".split()
)


def _is_jargon(token: str) -> bool:
    if token.casefold() in _NOT_JARGON or len(token) < 3:
        return False
    if _ALL_CAPS.match(token) or _ALPHANUMERIC.match(token):
        return True
    # A capitalised word only counts when it is not merely opening a sentence,
    # which the caller establishes by position — see `harvest`.
    return bool(_CAPITALISED.match(token))


def harvest(texts: list[str], minimum: int = MIN_OCCURRENCES) -> list[tuple[str, int]]:
    """Candidate terms and how often each was seen, most frequent first.

    A capitalised word is only counted when it appears somewhere other than the
    start of a sentence at least once. "Boston" said mid-sentence is a place
    worth seeding; "Because" opening one is not a term, and without this check
    every common word that ever begins a sentence becomes vocabulary.
    """
    counts: Counter[str] = Counter()
    mid_sentence: set[str] = set()

    for text in texts:
        if not text:
            continue
        for sentence in re.split(r"[.!?\n]+", text):
            for index, match in enumerate(_TOKEN.finditer(sentence.strip())):
                token = match.group(0)
                if not _is_jargon(token):
                    continue
                counts[token] += 1
                if index > 0:
                    mid_sentence.add(token)

    survivors = [
        (token, count)
        for token, count in counts.items()
        if count >= minimum
        and (not _CAPITALISED.match(token) or token in mid_sentence)
    ]
    survivors.sort(key=lambda pair: (-pair[1], pair[0]))
    return survivors[:MAX_TERMS]


def from_meetings(meetings: list[dict]) -> list[str]:
    """Terms already appearing in this workspace's transcripts.

    Self-reinforcing on purpose: a term whisper got right once is more likely
    to be got right again, and one it mangled will not reach the threshold — so
    the list grows towards what the company actually says and away from noise.
    """
    texts = [m.get("transcript") or "" for m in meetings]
    texts += [m.get("summary") or "" for m in meetings]
    return [term for term, _count in harvest(texts)]


def from_documents(corpus_dir: Path, limit: int = 200) -> list[str]:
    """Terms from the documents this deployment has already ingested."""
    root = Path(corpus_dir)
    if not root.exists():
        return []
    texts = []
    for path in sorted(root.rglob("*"))[:limit]:
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".csv"}:
            try:
                texts.append(path.read_text(errors="replace")[:200_000])
            except OSError as exc:  # pragma: no cover
                logger.warning("Could not read %s: %s", path, exc)
    return [term for term, _count in harvest(texts)]


def suggest(meetings: list[dict], corpus_dir: Path | None = None) -> list[str]:
    """Everything worth considering, configured terms first.

    The configured list always wins and always comes first: somebody typed it
    deliberately, and whisper weights the beginning of its prompt most heavily.
    """
    configured = [
        term.strip()
        for term in re.split(r"[,;\n]", os.environ.get("MEETING_VOCABULARY", ""))
        if term.strip()
    ]
    seen = {term.casefold() for term in configured}
    suggested = list(configured)

    for term in from_meetings(meetings) + (from_documents(corpus_dir) if corpus_dir else []):
        if term.casefold() not in seen:
            seen.add(term.casefold())
            suggested.append(term)
    return suggested[:MAX_TERMS]


def auto_enabled() -> bool:
    """Whether suggestions are fed to the decoder without anyone approving them."""
    return os.environ.get("MEETING_VOCABULARY_AUTO", "false").lower() == "true"
