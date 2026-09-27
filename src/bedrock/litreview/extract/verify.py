"""Quote verification — the single most important file in this repository.

A language model predicts text that *sounds* right. Usually that overlaps with
what is true; when it does not, you get a confident, fluent, invented finding.
That is not a bug awaiting a patch, it is how the mechanism works.

So we do not ask the model to be honest. We make it copy the sentence it took
each finding from, and then we check, here, in code, that the sentence really
appears in the paper. If it does not, the finding is destroyed.

Everything else in this system rests on this check, which is why it is plain
string matching that anybody can read and nobody has to trust.

Why fuzzy matching is allowed at all: PDF extraction mangles text in four
predictable ways — ligatures, curly quotes, words hyphenated across a line
break, and collapsed whitespace. A strict comparison throws away perfectly
good findings because of a typographic artifact. So we normalise first, then
allow a high-threshold fuzzy match, and we *record which kind of match it was*.
A rising fuzzy rate is a signal that PDF parsing has degraded, not that the
model has.

Two rules keep a fuzzy match honest:

1. A fuzzy match is typography or a harmless word, never meaning. The words
   that differ between the model's quote and the paper are compared; if any
   is a number or a word listed in `knowledge/meaning_words.yaml` ("not",
   "reduced", "mice", "all"), the model has changed what the paper said and
   the quote fails. Without this, "did significantly reduce" verified against
   "did not significantly reduce".

2. What is stored and shown is the paper's text, not the model's. Verification
   returns *where* the quote is (`locate_quote`), widened to the paper's whole
   sentence so a qualifier the model left off ("deletion of", "but not in
   controls") is on the page. The model's version is kept for audit only.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import yaml

from bedrock.litreview.models import QuoteStatus

# Above this, a near-match is accepted as the same sentence. Set high on
# purpose: we are correcting for typography, not paraphrase.
FUZZY_THRESHOLD = 95

# A quote shorter than this cannot be verified meaningfully — a five-character
# string appears in almost any paper by chance, so matching proves nothing.
MIN_QUOTE_CHARS = 25

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}

# Curly quotes, dashes and the various unicode spaces PDFs are full of.
_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}

_HYPHEN_LINEBREAK = re.compile(r"-\s*\n\s*")
_WHITESPACE = re.compile(r"\s+")


@lru_cache(maxsize=16)
def normalize_with_map(text: str) -> tuple[str, list[int], list[int]]:
    """`normalize`, plus where each output character came from.

    Returns (normalised text, starts, ends): output character i was produced
    by `text[starts[i]:ends[i]]`. That is what lets a match found in
    normalised text be cut out of the paper's own text, byte for byte.

    Characters are normalised one at a time, each with any combining marks
    that follow it, so every output character has exactly one origin.
    """
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        piece = unicodedata.normalize("NFKC", text[i:j])
        for bad, good in _LIGATURES.items():
            piece = piece.replace(bad, good)
        for bad, good in _PUNCT.items():
            piece = piece.replace(bad, good)
        for ch in piece:
            chars.append(ch)
            starts.append(i)
            ends.append(j)
        i = j

    joined = "".join(chars)
    drop = [False] * len(joined)
    for m in _HYPHEN_LINEBREAK.finditer(joined):
        for k in range(m.start(), m.end()):
            drop[k] = True

    out: list[str] = []
    out_s: list[int] = []
    out_e: list[int] = []
    for k, ch in enumerate(joined):
        if drop[k]:
            continue
        if ch.isspace():
            if out and out[-1] == " ":
                out_e[-1] = ends[k]
                continue
            ch = " "
        out.append(ch)
        out_s.append(starts[k])
        out_e.append(ends[k])
    while out and out[0] == " ":
        out.pop(0), out_s.pop(0), out_e.pop(0)
    while out and out[-1] == " ":
        out.pop(), out_s.pop(), out_e.pop()
    # Lower-case last and whole, as `normalize` does: a Greek final sigma is
    # lower-cased by context. Only when that changes the length (a dotted
    # capital I) is it done character by character, to keep the map aligned.
    text_out = "".join(out)
    lowered = text_out.lower()
    if len(lowered) == len(text_out):
        return lowered, out_s, out_e
    chars, starts, ends = [], [], []
    for k, (ch, a, b) in enumerate(zip(text_out, out_s, out_e)):
        low_ch = ch.lower()
        if ch == "Σ" and k and text_out[k - 1].isalpha() and not text_out[k + 1 : k + 2].isalpha():
            low_ch = "ς"  # what str.lower() does for a final sigma
        for low in low_ch:
            chars.append(low)
            starts.append(a)
            ends.append(b)
    return "".join(chars), starts, ends


def normalize(text: str) -> str:
    """Strip away the differences that are typography rather than content.

    Applied identically to the quote and to the source before comparing, so a
    match means "the same words", not "the same bytes".

    Produces the same string as `normalize_with_map` (checked in tests and on
    every paper in evals/litreview/corpora*), about fifty times faster; use that one
    only when positions are needed.
    """
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    for bad, good in _PUNCT.items():
        text = text.replace(bad, good)
    # Rejoin "inflamma-\ntion" before collapsing whitespace, or the hyphen is
    # left stranded mid-word and nothing matches.
    text = _HYPHEN_LINEBREAK.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip().lower()


def _align(needle: str, haystack: str) -> tuple[float, int, int]:
    """Best similarity between `needle` and any window of `haystack`, 0-100,
    and where that window is.

    Uses rapidfuzz when it is installed, and falls back to a slower pure-Python
    equivalent otherwise. The fallback exists because this check must keep
    working on an air-gapped machine where somebody forgot a dependency —
    failing open here would silently disable the guarantee the whole system
    rests on.
    """
    if not needle or not haystack:
        return 0.0, 0, 0
    if len(needle) >= len(haystack):
        # The quote is the whole source or more: compare the two outright.
        return SequenceMatcher(None, needle, haystack, autojunk=False).ratio() * 100, 0, len(haystack)
    try:
        from rapidfuzz import fuzz

        a = fuzz.partial_ratio_alignment(needle, haystack)
        return float(a.score), a.dest_start, a.dest_end
    except ImportError:
        return _align_fallback(needle, haystack)


def _align_fallback(needle: str, haystack: str) -> tuple[float, int, int]:
    best, where = 0.0, 0
    window = len(needle)
    # Step in tenths of a window: fine enough to land near any real match,
    # coarse enough not to be quadratic on a full paper.
    step = max(1, window // 10)
    for start in range(0, len(haystack) - window + 1, step):
        chunk = haystack[start : start + window]
        score = SequenceMatcher(None, needle, chunk).ratio() * 100
        if score > best:
            best, where = score, start
        if best >= 99.5:
            break
    return best, where, where + window


# --- meaning ------------------------------------------------------------------

_WORDS_FILE = Path(__file__).resolve().parent.parent / "knowledge" / "meaning_words.yaml"
# Words, numbers and comparison signs. `\w` so that Greek letters are tokens:
# IFN-γ and IFN-α are different molecules.
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|\w+n't|\w+|[<>=±≤≥]")


@lru_cache(maxsize=1)
def meaning_words() -> dict:
    """The hand-written word lists: {'meaning': set of words, 'studied':
    {class: [terms]}, 'ambiguous_human': [terms]}."""
    data = yaml.safe_load(_WORDS_FILE.read_text(encoding="utf-8")) or {}
    studied = {k: [t.lower() for t in v] for k, v in (data.get("studied") or {}).items()}
    ambiguous = [t.lower() for t in data.get("ambiguous_human") or []]
    words = {w.lower() for group in (data.get("meaning") or {}).values() for w in group}
    # Who was studied is meaning too: "24 mice" is not "24 rats". Single words
    # only: splitting "in vitro" would make "in" a meaning word.
    words |= {t for terms in [*studied.values(), ambiguous] for t in terms if " " not in t}
    counted = [t.lower() for t in data.get("counted") or []]
    return {"meaning": words, "studied": studied, "ambiguous_human": ambiguous, "counted": counted}


_ANIMALS = ("mouse", "rat", "primate", "animal")


@lru_cache(maxsize=1)
def _studied_patterns() -> tuple[list[tuple[str, re.Pattern[str]]], re.Pattern[str] | None]:
    def pattern(terms: list[str]) -> re.Pattern[str]:
        return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)

    words = meaning_words()
    classes = [(name, pattern(terms)) for name, terms in words["studied"].items() if terms]
    ambiguous = pattern(words["ambiguous_human"]) if words["ambiguous_human"] else None
    return classes, ambiguous


def covered(claimed: str, said: set[str] | dict[str, str]) -> bool:
    """Does text naming the kinds of subject in `said` back a claim about
    `claimed`? Exact class, or "animal" backed by any particular animal."""
    return claimed in said or (claimed == "animal" and any(a in said for a in _ANIMALS))


def studied_in(text: str) -> dict[str, str]:
    """Which kinds of subject a text names: class -> the first word naming it.

    The one reading of "who was studied" in the codebase: the citation guard
    uses it on report sentences and the text they cite, provenance on the
    text a model's species came from. A life-stage word ("adult",
    "pregnant") names humans only when the text names no animal: "adult
    mice" is about mice.
    """
    classes, ambiguous = _studied_patterns()
    found: dict[str, str] = {}
    for name, pattern in classes:
        m = pattern.search(text)
        if m:
            found[name] = m.group(0)
    if ambiguous is not None and "human" not in found and not any(a in found for a in _ANIMALS):
        m = ambiguous.search(text)
        if m:
            found["human"] = m.group(0)
    return found


# Words that name no people where they stand: the adjective "human", and a
# word joined to the next by a hyphen ("adult-onset deletion", "maternal-fetal").
_NOT_PEOPLE = re.compile(r"\bhuman\b|\b[^\W\d_]+-(?=[^\W\d_])", re.I)


def people_named(text: str) -> str | None:
    """The first word in `text` that says people were studied ("patients",
    "participants", "children"), or None. `studied_in`'s human class, less
    words that name no people where they stand: "human bronchial epithelial
    cells" are cells, and "adult-onset Gpr75 deletion" was done in mice."""
    return studied_in(_NOT_PEOPLE.sub(" ", text)).get("human")


@lru_cache(maxsize=1)
def entity_words() -> frozenset[str]:
    """Every one-word name in the knowledge table ("tnf", "semaglutide",
    "psoriasis"), near-misses included. A quote that swaps one for another
    names a different molecule, drug or disease: "TNF" is not "TNFR", and
    semaglutide is not liraglutide, however close the spelling."""
    from bedrock.litreview.knowledge import load_knowledge

    words = set()
    for entity in load_knowledge().entities:
        for term in (entity.canonical, *entity.terms, *entity.near_miss, *entity.context_terms):
            tokens = _TOKEN.findall(normalize(term))
            if len(tokens) == 1 and tokens[0].isalpha():
                words.add(tokens[0])
    return frozenset(words)


def _numbered_name_part(tokens: list[str], k: int) -> bool:
    """Is tokens[k] the letter after a number, as the "a" in "il-17a"? The
    tokenizer splits "17a" into "17" and "a", so without this IL-17A and
    IL-17F compare as the same words."""
    tok = tokens[k]
    return k > 0 and tokens[k - 1][0].isdigit() and tok.isalpha() and len(tok) <= 2


def meaning_changes(quote_norm: str, source_norm: str) -> list[str]:
    """The meaning-bearing words that differ between two normalised texts.

    Empty means the texts differ only in typography or filler ("Moreover"
    for "Furthermore"). A number, a comparison sign, a listed word, a name
    from the knowledge table or the letter of a numbered name ("IL-17A") on
    either side of a difference means they do not say the same thing.
    """
    listed = meaning_words()["meaning"]
    names = entity_words()
    a, b = _TOKEN.findall(quote_norm), _TOKEN.findall(source_norm)
    ops = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    # Words the source has only at the very edges of the window are alignment
    # slack, not a change: the model quoted less than the window, exactly as
    # an exact partial quote does (and the whole sentence is shown anyway).
    while ops and ops[0][0] == "insert":
        ops = ops[1:]
    while ops and ops[-1][0] == "insert":
        ops = ops[:-1]
    changed: list[str] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        for tokens, k in [*((a, k) for k in range(i1, i2)), *((b, k) for k in range(j1, j2))]:
            tok = tokens[k]
            if (tok in listed or tok in names or tok.endswith("n't") or any(c.isdigit() for c in tok)
                    or not tok[0].isalnum() or _numbered_name_part(tokens, k)
                    or any(c.isalpha() and not c.isascii() for c in tok)):  # α, β, γ, κ …
                changed.append(tok)
    edge = (ops[0][3], ops[-1][4]) if ops else (0, len(b))
    if Counter(t for t in a if t[0].isdigit()) != Counter(t for t in b[edge[0]:edge[1]] if t[0].isdigit()):
        changed.append("numbers")
    return changed


# --- locating a quote -----------------------------------------------------------

# A sentence ends at . ! or ? followed by whitespace and a capital, digit or
# bracket, or at a blank line. Cautious on purpose: stopping early only means
# less of the sentence is shown, never a wrong one.
_SENTENCE_END = re.compile(r"[.!?](?=\s+[A-Z0-9(\[])|\n\s*\n")
_WIDEN_LIMIT = 400
# A full stop after one of these is not the end of a sentence: "Fig. 2A",
# "Smith et al. (2019)", "vs. 5 mg". Stopping there would cut off the very
# qualifier widening exists to show ("... (Fig. 2A) but not in knockouts").
_ABBREVIATION = re.compile(
    r"\b(?:figs?|al|vs|e\.g|i\.e|cf|ca|approx|ref|refs|no|nos|suppl?|eq|eqs|tab|resp|sp|spp|dr|st)$", re.I)


def _is_end(text: str, m: re.Match[str]) -> bool:
    if text[m.start()] != ".":
        return True
    return not _ABBREVIATION.search(text[max(0, m.start() - 8) : m.start()])


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with the same boundary rule widening uses."""
    out, last = [], 0
    for m in _SENTENCE_END.finditer(text or ""):
        if _is_end(text, m):
            piece = text[last : m.start() + 1 if text[m.start()] in ".!?" else m.start()].strip()
            if piece:
                out.append(piece)
            last = m.end()
    tail = (text or "")[last:].strip()
    if tail:
        out.append(tail)
    return out


def region_of(part: str, text: str) -> tuple[int, int] | None:
    """Where `part` (a chunk re-joined from `text`'s paragraphs) sits in
    `text`, in `text`'s offsets; None when it cannot be found."""
    s, starts, ends = normalize_with_map(text)
    p = normalize(part)
    k = s.find(p) if p else -1
    if k < 0:
        return None
    return starts[k], ends[k + len(p) - 1]


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen text[start:end] to the whole sentence (or sentences) it sits in,
    at most `_WIDEN_LIMIT` characters each way."""
    lo = max(0, start - _WIDEN_LIMIT)
    # With no boundary in reach, the start stays put — unless the text itself
    # begins in reach, which is then the sentence's start. No `endpos` on the
    # searches: the look-ahead must see past it, or the boundary just before a
    # quote that begins its sentence is missed and the previous sentence is
    # glued on.
    new_start = 0 if lo == 0 else start
    for m in _SENTENCE_END.finditer(text, lo):
        if m.start() >= start:
            break
        if m.end() <= start and _is_end(text, m):
            new_start = max(new_start, m.end())
    while new_start < start and text[new_start].isspace():
        new_start += 1
    m = next((m for m in _SENTENCE_END.finditer(text, max(end - 1, start)) if _is_end(text, m)), None)
    if m and m.start() > end + _WIDEN_LIMIT:
        m = None
    new_end = (m.start() + 1 if text[m.start()] in ".!?" else m.start()) if m else (
        len(text) if len(text) - end <= _WIDEN_LIMIT else end)
    new_end = max(new_end, end)
    while new_end > new_start and text[new_end - 1].isspace():
        new_end -= 1
    return new_start, new_end


@dataclass
class Located:
    """Where a quote is in its source, or why it could not be found there."""

    status: QuoteStatus
    start: int = 0
    end: int = 0
    text: str = ""
    """The source's own words at `start:end`, widened to whole sentences."""
    reason: str = ""


def _match(q: str, s: str) -> tuple[QuoteStatus, int, int, str]:
    """(status, start, end, reason) of normalised quote `q` in normalised source `s`."""
    if not q or not s:
        return "failed", 0, 0, "empty"
    k = s.find(q)
    if k >= 0:
        return "exact", k, k + len(q), ""
    score, a, b = _align(q, s)
    if score < FUZZY_THRESHOLD:
        return "failed", 0, 0, "not in source"
    # The window may start or end inside a word; take whole words.
    while a > 0 and s[a - 1].isalnum():
        a -= 1
    while b < len(s) and s[b].isalnum():
        b += 1
    changed = meaning_changes(q, s[a:b])
    if changed:
        return "failed", 0, 0, "meaning changed: " + ", ".join(dict.fromkeys(changed))
    return "fuzzy", a, b, ""


def locate_quote(quote: str, source_text: str, *, widen: bool = True) -> Located:
    """Find the model's quote in the source and return the source's own words.

    "exact": the normalised quote is a substring of the normalised source.
    "fuzzy": it is not, but a window of the source matches it closely and the
    words that differ carry no meaning. "failed": neither, or the differences
    change what was said; `reason` says which.
    """
    if not quote or not source_text:
        return Located("failed", reason="empty")
    if len(quote.strip()) < MIN_QUOTE_CHARS:
        return Located("failed", reason="too short")
    q = normalize(quote)
    s, starts, ends = normalize_with_map(source_text)
    status, a, b, reason = _match(q, s)
    if status == "failed":
        return Located("failed", reason=reason)
    start, end = starts[a], ends[b - 1]
    if widen:
        start, end = sentence_bounds(source_text, start, end)
    return Located(status, start, end, source_text[start:end])


def verify_quote(quote: str, source_text: str) -> QuoteStatus:
    """Did this sentence really appear in this paper?

    Returns "exact" when the normalised quote is a substring of the normalised
    source, "fuzzy" when it is close enough to be the same sentence through a
    parsing artifact, and "failed" otherwise.

    A "failed" result means the finding must be discarded. Never surface an
    unverified quote in a report — a citation that cannot be checked is worse
    than no citation, because it reads as though someone checked it.
    """
    if not quote or not source_text or len(quote.strip()) < MIN_QUOTE_CHARS:
        return "failed"
    # Same decision as `locate_quote`, without building the position map.
    return _match(normalize(quote), normalize(source_text))[0]


def verify_all(
    findings: list, source_text: str
) -> tuple[list[tuple[object, QuoteStatus]], dict[str, int]]:
    """Verify a batch and report the tally.

    Returns the surviving (finding, status) pairs plus counts per status. Log
    the counts on every run: the exact/fuzzy/failed ratio is a health metric
    for the parsing layer, and a fuzzy rate drifting above ~10% means PDF
    extraction has degraded and should be investigated before the prompts are.
    """
    kept: list[tuple[object, QuoteStatus]] = []
    tally = {"exact": 0, "fuzzy": 0, "failed": 0}

    for finding in findings:
        status = verify_quote(getattr(finding, "quote", ""), source_text)
        tally[status] += 1
        if status != "failed":
            kept.append((finding, status))

    return kept, tally


# --- results reported as the paper's own, when they are someone else's --------
#
# A verified quote is not necessarily this paper's own result. Europe PMC's
# full text strips reference numbers out of the running text but leaves the
# brackets that held them, so a sentence that cites earlier work through a
# numbered reference still carries "[]", "[12]", "[3, 4]" or "[5-7]" right
# where the reference used to be - almost always at the end of the sentence,
# next to the full stop. Author-year citations ("(Smith et al., 2019)") are
# left untouched altogether. A sentence that quotes one of these, or that
# opens by saying in words that it is summarising someone else's work
# ("Previous studies have shown ..."), is Introduction or Discussion prose
# about another paper, not a finding of this one - however cleanly it reads
# as a result and however well it verifies, because verification only checks
# that the paper wrote the sentence, never who the sentence is about.
#
# This is exactly the asthma S9 case
# (evals/litreview/hand_audit/qwen_asthma.md): a birth-cohort result quoted
# from the paper's own Introduction, its reference bracket stripped to "[]",
# reported as if this paper (a paper about MAIT cells and estrogen) had run
# that birth cohort itself.
#
# Each pattern below is deliberately narrow, because a false drop here throws
# away a real finding just as surely as a missed one lets a fabricated one
# through:
#
# - the bracket pattern only matches brackets that hold nothing but digits,
#   commas, dashes and spaces (or nothing at all) - a confidence interval
#   ("[95% CI 1.2-3.4]"), a concentration ("[Ca2+]i") or an isotope label
#   ("[3H]thymidine") always has a letter or a "%" inside, so none of those
#   match;
# - the author-year pattern requires a capitalised name followed by "et al.",
#   "and <Name>" or a comma, and then a plausible year - so it does not fire
#   on "(Fig. 2A)", "(Table 1)", "(n = 12)", "(p < 0.05)" or an abbreviation
#   like "(IL-17A)", none of which is a name-plus-year;
# - the trailing-superscript pattern only matches raised digits sitting right
#   at the end of the sentence (before at most one closing punctuation mark) -
#   a concentration written as "10^6 CFU/mL." has words after the superscript,
#   so it is never mistaken for a reference mark;
# - the lead-in phrases are matched only at the very start of the quote, since
#   that is how a paper actually opens a sentence summarising other work, not
#   a clause that can appear mid-sentence in a real result.
#
# The section check is here for the paper that behaves as its headings say:
# `parsing/sections.py` folds any heading matching "introduction" or
# "background" into the canonical section "other" (see its `_HEADING_MAP`),
# so there is no canonical "introduction" section to check for - checking for
# "other" is the real equivalent. But the S9 paper did not behave: its
# Introduction had no "Results" heading above it, so the parser's fallback
# rule (unlabelled body text becomes "results" when nothing else does) filed
# it as "results" (`parsing/sections.py` `_Collector.result`). The section
# check therefore does not catch S9 - only the sentence patterns above do -
# but it is kept for the well-behaved paper where it works.

# Digits, commas, dashes and spaces only (or nothing): what is left of a
# numbered reference list once Europe PMC strips the numbers themselves.
_CITATION_BRACKET = re.compile(r"\[[\d,\s\-‐‑‒–—―−]*\]")

# A run of raised digits immediately at the end of the sentence: a reference
# mark some extractions render this way instead of a bracket.
_TRAILING_SUPERSCRIPT = re.compile(r"[¹²³⁰⁴-⁹]+[.,;:)\]]?\s*$")

# "(Smith et al., 2019)", "(Smith and Jones 2019)", "(Smith, 2019a)": a
# capitalised name, an optional co-author, and a four-digit year.
_AUTHOR_YEAR = re.compile(
    r"\([A-Z][A-Za-z\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z\-]+))?,?\s*(?:19|20)\d{2}[a-z]?\)"
)

# How a paper's own Introduction or Discussion says "this is somebody else's
# result", checked only at the start of the quote.
#
# "human genetic studies have identified", "have been shown to" (the plural of
# "has been shown to" above), "genetic studies have shown" and "genome-wide
# association studies have identified" were added for the new obesity report's
# target section, which opens "Human genetic studies have identified GPR75
# loss-of-function variants to be strongly protective against obesity,
# establishing GPR75 as a compelling therapeutic target" (OPEN_ISSUES.md item
# 7; verbatim in evals/litreview/hand_audit/qwen_obesity_after.md and
# claude_obesity.md) - other people's human genetics, quoted in the background
# of a mouse paper, with no citation bracket for the marker check to catch.
# The other three are the same shape found while re-reading
# evals/litreview/corpora/psoriasis-prevention-targets/papers.jsonl: "Genetic
# studies have shown that polymorphisms in the IL-23 pathway are associated
# with an increased risk of developing psoriasis" and "Genome-wide association
# studies have identified SOCS1 as a susceptibility locus in psoriasis" both
# open a background sentence citing someone else's genetic-association work;
# "have been shown to" itself only ever occurs there mid-sentence ("KCs have
# been shown to be closely associated with psoriasis pathogenesis"), so it is
# kept narrow the same way "has been shown to" already is - see
# tests/litreview/test_verify.py's "only matters at the start of the
# sentence" test.
_OTHER_WORK_LEAD_IN = re.compile(
    r"^\s*(?:previous studies have shown|a recent study found|another study showed|"
    r"it has been reported|has been shown to|have been shown to|prior work|earlier studies|"
    r"human genetic studies have identified|genetic studies have shown|"
    r"genome-wide association studies have identified)\b",
    re.I,
)

# The canonical section names a paper's text can carry are exactly
# `parsing/sections.py` `CANONICAL`. Introduction and Background are not
# among them - both fold into "other" - so "other" is the real name to check.
# "unknown" belongs here for the same reason, not a different one: it is what
# `_Collector.result()` files a paper's leftover text under when not even
# Methods, Results or Discussion was recognised anywhere in the body - which
# means Introduction and Discussion are in that blob too, unseparated from
# anything else. A quote pulled from text the parser could not place is
# exactly as likely to be someone else's result, restated, as one pulled from
# "other" - not naming the section is not evidence that the section was
# Results, so it earns the same citation-marker scrutiny rather than a pass.
_INTRO_LIKE_SECTIONS = frozenset({"other", "unknown"})

# A narrow exemption from the citation-marker check, added after running it
# over 593 real evidence units from stored qwen3:8b reports turned up two of
# this shape: E257 and E270 (see their tests) are the paper's *own* finding,
# stated as such, with a citation bracket sitting inside a subordinate clause
# that only compares the result with earlier work ("...unlike the previously
# known...", "...are consistent with research demonstrating..."). The bracket
# there belongs to the comparison, not to the finding.
#
# Both conditions below must hold, and neither is enough alone:
#
# - the quote must *open* by claiming the result as this paper's own, in the
#   present tense of reporting it now ("We observed", "Our findings", "Here,
#   we", "In this study, we"). Past-tense framing of a *prior* paper's own
#   result reads the same on the surface but is not exempt - "we
#   demonstrated", "we have found" and "In our studies" are deliberately left
#   off this list, because the audit's counter-examples used exactly that
#   framing to introduce a citation to someone else's data anyway (see the
#   "must stay dropped" tests: AhR/AT2, estrogen, Hinks et al., "these
#   findings align with");
# - and every citation marker in the quote must come after some comparison
#   connective ("unlike", "consistent with", "in line with", "in agreement
#   with", "similar to", "as previously", "in contrast to", "as reported") -
#   a marker before any connective is not fenced off from the paper's own
#   claim, and is not exempted.
_OWN_WORK_OPENER = re.compile(
    r"^\s*(?:we\s+(?:also\s+)?(?:observed|found|show(?:ed)?|report)\b"
    r"|our\s+(?:findings|results|data)\b"
    r"|here,?\s+we\b"
    r"|in\s+this\s+study,?\s+we\b)",
    re.I,
)
_COMPARISON_CONNECTIVE = re.compile(
    r"\b(?:unlike|consistent with|in line with|in agreement with|similar to|"
    r"as previously|in contrast to|as reported)\b",
    re.I,
)


def _markers(quote: str) -> list[re.Match[str]]:
    return [*_CITATION_BRACKET.finditer(quote), *_AUTHOR_YEAR.finditer(quote), *_TRAILING_SUPERSCRIPT.finditer(quote)]


def _fenced_by_comparison(quote: str, markers: list[re.Match[str]]) -> bool:
    """Is every marker in `markers` preceded by a comparison connective, in a
    quote that opens by claiming the result as this paper's own?"""
    if not markers or not _OWN_WORK_OPENER.match(quote):
        return False
    connectives = [m.start() for m in _COMPARISON_CONNECTIVE.finditer(quote)]
    if not connectives:
        return False
    earliest = min(connectives)
    return all(m.start() > earliest for m in markers)


def other_work_reason(quote: str, section: str) -> str | None:
    """Why `quote` reports someone else's result rather than this paper's
    own, or None if nothing here says it does.

    Meant to run on the paper's own verified sentence (`Located.text`), right
    after quote verification and before every check downstream of it - a
    finding built on a borrowed result should never reach the entity check
    or provenance, both of which would happily pass a true sentence about
    the wrong paper.
    """
    if section in _INTRO_LIKE_SECTIONS:
        return "other_work:section"
    if _OTHER_WORK_LEAD_IN.match(quote):
        return "other_work:lead_in"
    markers = _markers(quote)
    if markers and not _fenced_by_comparison(quote, markers):
        return "other_work:citation_marker"
    return None


# --- sentences that report no result at all -----------------------------------
#
# A verified quote can be the paper's own words, about the paper's own work,
# and still not be a finding: it says what the study *set out* to do, what it
# *predicts*, what its results *might one day enable*, or what is still
# *unknown*. A hand audit of three benchmark reports found each of these cited
# as evidence (evals/litreview/hand_audit/):
#
#   asthma S32   "In this study, we investigated the role of the AhR in
#                regulating allergen-induced ROS generation ..." - an aim,
#                cited for "AhR regulates allergen-induced ROS generation".
#   obesity S56  "... our findings highlight a therapeutic potential for
#                obese/diabetic postmenopausal patients; cold exposure is
#                therefore predicted to metabolically benefit those
#                patients." - a prediction from mouse work, cited as
#                prevention evidence.
#   asthma S57   "Identifying key pathophysiological determinants in airway
#                epithelium enables novel therapeutics to prevent asthma
#                recurrence." - background, cited as a prevention finding.
#   migraine S23 the *report sentence* was the methods one: "The study used
#                a ... mouse model to investigate the effects of NF-kB
#                inhibition." It claims nothing.
#
# `RESULT_WORDS` alone cannot catch these: S32 contains "induced", S57
# "prevent", both result words used here to name a model or a hope.
#
# So the rule is scoped, the way the citation guard's `_null_only` scopes a
# negation. Each cue below marks a *stretch* of the sentence as aim, method,
# prediction or open question; that stretch is set aside, and the text is
# "not a result" only when a cue matched and nothing left outside the
# stretches reports a result. "To test this, we treated mice with X, which
# reduced Y by 40%" is a finding: the aim stretch ends at the comma, and
# "which reduced Y by 40%" is left over.
#
# How far a stretch reaches depends on the cue:
#
# - an aim or method cue ("we investigated", "to determine whether", "was
#   designed to") reaches *forward* from the cue to the next clause break
#   (`_CLAUSE_BREAK`: a comma, semicolon or colon, "which", "and", "but",
#   "to show" ...), so whatever the sentence says before the cue, and after
#   the break, is still read for a result;
# - a prediction, potential or open-question cue ("is predicted to", "may
#   offer", "remains unclear") covers its *whole clause*, back to the
#   previous break too, since the thing predicted or unknown is its subject
#   ("cold exposure is predicted to ..."). The breaks still bound it, so "X
#   reduced Y and may offer a new therapy" keeps "X reduced Y".
#
# Deliberately absent: plain "may", "suggest", "indicate" and "could". "These
# data suggest IL-23 drives psoriasis" follows a result as often as not, and
# "X may be mediated by Y" is how results sections hedge. Only the fixed
# phrases below, which state an aim or a hope rather than hedge a result,
# are cues. Measured before this was switched on, on the three stored
# qwen3:8b benchmark reports: 30 of 593 evidence units hit (every one an aim,
# a method, a prediction or an open question on reading; 3 of them are also
# someone else's work, which Stage 5 checks first), and 7 of 179 kept
# sentences would now be dropped by the citation guard. Over all 23,787
# abstract and results sentences of their papers, 276 hits contain a result
# word somewhere; a sample of 60, and every hit with a past-tense result verb,
# were read, and all but one used the word inside the aim ("to determine
# whether the increased ..."). The one ("... suggests a potential dual-hit
# mechanism") is why "potential" must be a noun below.
# tests/litreview/test_not_a_result.py has the real cases, kept and dropped.

# What a text that reports a result says. Verb and adjective forms only:
# "exacerbation" names an asthma event, "exacerbated" reports one. Shared by
# the citation guard (`report/guard.py`, check 6) and the rule below, so one
# vocabulary decides what "reports a result" means everywhere.
#
# Grouped by what kind of result each word or phrase reports, because a flat
# wall of words is hard to extend without breaking something else. A hand
# re-audit of the 179 sentences kept across the three stored benchmark
# reports (obesity, asthma, migraine) found real results this list missed,
# one per group below; each was added and checked against every one of those
# 179 sentences (and the 593 evidence units behind them) before being kept -
# see tests/litreview/test_result_words.py for the verbatim cases, kept and
# still dropped:
#
# - effect verbs: a treatment, a gene or a knockout did something to a
#   measurement ("led to weight loss", "results in glucose intolerance",
#   "potentiated", "normalized body weight", "reversed");
# - comparisons: two groups were found to differ, or not to. Expression
#   level is a comparison too ("highly expressed ... compared to",
#   "overexpressed in") - deliberately not bare "expressed", which only says
#   where a gene product exists, not what was found (PAR2 "is a G-protein
#   coupled receptor expressed in meningeal neurons ..." reports nothing).
#   "similar" is scoped to "similar between/among" (two arms or groups) for
#   the same reason: bare "is similar in <population> to" is how a trial
#   states its *aim* ("to confirm the hypothesis that the PK profile ... is
#   similar in healthy Chinese individuals to that of healthy non-Asian
#   individuals"; see test_not_a_result.py), not a result. "differences" is
#   scoped to "no difference(s)" for a sharper reason: E212 of the obesity
#   report (the MC4R "~0.6 kg/week" sentence, see test_qualifier.py's S6
#   cases) restates a real group-mean-difference number whose next sentence
#   says it "did not reach statistical significance" - a stored qualifier in
#   today's data model, not a word in the quote itself. Reading bare
#   "differences" as a result would move that sentence from check 6 (no
#   result cited) to check 8 (drops the qualifier), which only fires when
#   the qualifier was actually captured - not guaranteed for every unit, and
#   not for this one in the stored baseline report. Scoping to "no
#   difference(s)" keeps check 6 the one that catches it, as it does today;
# - null results: the paper looked and found nothing. Negative results are
#   first-class here, not a weaker case than a positive one - "no
#   differences ... indicating that X does not directly regulate Y" reports
#   a null result exactly as much as "X increased Y" reports a positive one.
#   "No signal of", "no evidence of" and "no cases of" state an absence just
#   as plainly ("No signal of drug-induced liver injury due to rimegepant
#   was identified" - migraine's safety finding, not a non-result);
# - safety/PK results: how a trial's safety or pharmacokinetic data came out
#   ("the most frequent AEs were ...", "elevations ... were observed").
#
# "block(ed|s|ing)" was added to the effect verbs after masking a
# model-naming adjective (`_NAMED_MODEL`, below) turned up a real result with
# no other result word left once the mask ran: "... fully blocked
# CGRP-induced cutaneous allodynia" (migraine's olcegepant/CGRP-R quote)
# reports what olcegepant did ("blocked") once "CGRP-induced" - the model,
# not the result - is masked out.
RESULT_WORDS = re.compile(
    r"\b(?:"
    # -- effect verbs ---------------------------------------------------------
    r"reduc(?:ed|es|ing|tion|tions)|decreas(?:ed|es|ing)|lower(?:ed|ing)?|attenuat(?:ed|es|ing)|"
    r"ameliorat(?:ed|es)|alleviat(?:ed|es)|diminish(?:ed|es)|abolish(?:ed|es)|abrogat(?:ed|es)|"
    r"mitigat(?:ed|es)|improv(?:ed|es|ing|ement)|prevent(?:ed|s|ing)|protect(?:ed|s|ive|ion)|"
    r"resolv(?:ed|es)|increas(?:ed|es|ing)|elevat(?:ed|es|ing|ion|ions)|worsen(?:ed|s|ing)|aggravat(?:ed|es)|"
    r"exacerbat(?:ed|es|ing)|promot(?:ed|es)|accelerat(?:ed|es)|augment(?:ed|s)|amplif(?:ied|ies)|"
    r"suppress(?:ed|es)|inhibit(?:ed|s)|induc(?:ed|es)|enhanc(?:ed|es)|restor(?:ed|es)|rescu(?:ed|es)|"
    r"block(?:ed|s|ing)?|"
    r"potentiat\w*|exaggerat\w*|normali[sz]\w*|revers\w*|led\s+to|result(?:s|ed|ing)?\s+in|"
    r"higher|greater|fewer|less|more|better|worse|significant(?:ly)?|associated|correlat\w*|"
    r"efficac\w*|effective|benefit\w*|remission|response|achieved|rose|fell|declined|lost|gained|"
    r"upregulat\w*|downregulat\w*|up-regulat\w*|down-regulat\w*|"
    # -- comparisons -----------------------------------------------------------
    r"no\s+differences?|"
    r"(?:highly|more|less|over|under)[- ]?express\w*|"
    r"express(?:ion|ed)\s+(?:was|were)\s+(?:increased|decreased|elevated|reduced)|"
    r"similar\s+(?:between|among)|"
    # -- null results ------------------------------------------------------------
    r"no effect|did not|failed|no\s+(?:signal|evidence|cases?)\s+of|"
    # -- safety / PK results --------------------------------------------------
    r"most\s+(?:frequent|common)|"
    # Base forms ("can reduce", "exacerbate"), change nouns ("no increase"),
    # and how safety results are written ("adverse events were infrequent").
    r"reduce|decrease|increase|lower|improve|prevent|protect|worsen|exacerbate|aggravate|attenuate|"
    r"alleviate|relieve[ds]?|suppress|inhibit|enhance|promote|restore|induce|change[ds]?|difference|"
    r"adverse|tolerat\w*|infrequent|mild|unchanged)\b|%|\bp\s*[<=>]|\bfold\b", re.I)


# A model or stimulus named after what provokes or receives it -
# "allergen-induced", "(NTG)-induced", "HDM-induced", "diet-induced",
# "imiquimod-treated", "obesity-associated" - names the experiment, not a
# result. Shared by `reports_result` and, through it, by the leftover text in
# `not_a_result_reason` below, so one rule decides this everywhere: masking
# only the compound adjective, never a real verb use ("X induced Y",
# "treated with X reduced Y"), because the lookbehind requires a hyphen stuck
# to the word before it. The hyphen itself is read the way `_PUNCT` above
# does: a paper's own text carries the same non-ASCII hyphens PDF extraction
# leaves behind ("drug‐induced", not "drug-induced"), and this runs on
# that raw text, never on `normalize`'s output.
_NAMED_MODEL = re.compile(
    r"(?<=[\w)\]])[-‐‑‒–—―−](?:induced|treated|challenged|exposed|mediated|driven|associated)\b", re.I)


def reports_result(text: str) -> bool:
    """Does this text use any word a reported result is written with?

    A wide net on purpose: it is used to decide that a text reports *no*
    result, and a false "no" throws a real finding away. A model or
    stimulus named after what provokes or receives it is masked first
    (`_NAMED_MODEL`), or "Diet-induced obesity (DIO) mice ... were infused
    with a NOX blocker" - a methods sentence - would count as reporting a
    result just because "induced" is also how a real result is written ("X
    induced Y"). A real result keeps working: "EN4 effectively suppressed
    CRE-induced c-Myc expression" still reports one, because "suppressed"
    sits outside the masked span."""
    return bool(RESULT_WORDS.search(_NAMED_MODEL.sub(" ", text)))


# Aim and method cues: what the study set out to do, or how. Each reaches
# forward to the next clause break.
_AIM_CUE = re.compile(
    r"\b(?:"
    r"we\s+(?:first\s+|further\s+|then\s+|also\s+|therefore\s+|next\s+)?"
    r"(?:investigated|aimed|sought\s+to|hypothesi[sz]ed|hypothesi[sz]e|examined|assessed|evaluated|explored)"
    r"|(?:the|our|this)\s+(?:main\s+|primary\s+)?(?:aim|objective|purpose|goal)s?\s+of\s+"
    r"(?:this|the\s+present|the\s+current|our)\s+(?:study|work|trial|experiment)s?"
    r"|(?:study|trial|experiment)s?\s+(?:was|were|is)\s+designed\s+to"
    r"|(?:was|were)\s+(?:investigated|examined|assessed|evaluated|explored)"
    r"|(?:the|this|our)\s+study\s+used"
    r"|to\s+(?:further\s+|better\s+)?(?:investigate|examine|determine|assess|evaluate|explore|clarify|test)\s+"
    r"(?:whether|if|how|the)"
    r")\b",
    re.I,
)

# Prediction, potential and open-question cues: a hope or a gap, not a
# result. Each covers its whole clause.
_HOPE_CUE = re.compile(
    r"\b(?:"
    r"(?:is|are)\s+(?:therefore\s+|thus\s+)?predicted\s+to"
    r"|may\s+offer"
    # "potential" as the thing highlighted ("a therapeutic potential for",
    # "the potential of", "the potential role of"), not as an adjective: "the
    # causal link ... suggests a potential dual-hit mechanism" reports a link.
    r"|(?:highlight|suggest|underscore)(?:s|ed|ing)?\s+(?:a|the)\s+(?:therapeutic\s+)?potential"
    r"\s+(?:for|of|in|to|role|use|value|benefit)"
    r"|enables?\s+novel\s+therapeutics"
    r"|warrants?\s+further\s+(?:investigation|study|studies|research)"
    r"|could\s+be\s+a\s+(?:potential|promising)\s+(?:therapeutic\s+)?(?:target|strategy)"
    r"|remain(?:s|ed)?\s+(?:largely\s+|still\s+|poorly\s+)?(?:unknown|unclear|elusive|understood)"
    r")\b",
    re.I,
)

# Where a cued stretch ends: a comma, semicolon or colon; "which"; a
# conjunction ("and", "but", "whereas", "thus"), since what follows it can be
# the result ("IL-6 was examined and was significantly elevated", "X reduced
# Y and may offer a new therapy"); or "to show"/"showing", which introduce
# what an experiment found ("The study used knockout mice to show that ...").
_CLAUSE_BREAK = re.compile(
    r"[,;:]|\s(?=(?:which|and|but|whereas|while|thus|hence|so)\b)"
    r"|\s(?=(?:to\s+(?:show|demonstrate|reveal|confirm)|showing|demonstrating|revealing)\b)",
    re.I,
)

# Hypothesis cues: a guess stated as a guess, not as what happened - a
# structured abstract's HYPOTHESIS part ("cold exposure might induce enhanced
# browning/beiging and improve glucose metabolism.", a migraine benchmark
# case; see tests/litreview/test_not_a_result.py). Each covers its whole
# clause, like a hope cue.
#
# "might <verb>" is read unrestricted: a results section essentially never
# hedges a genuine finding with "might" (that is what "may" and "could" are
# for), so any verb after it is read as a guess. "may <verb>" is not given
# the same latitude, because "may" hedges a real result too ("which may
# reflect a ceiling effect" must stay a finding) - it is scoped to the same
# result-shaped verbs RESULT_WORDS already lists, where reading the hedge as
# a guess costs nothing even on the rare sentence where that reading is
# wrong. "could induce", "we hypothesize(d) that" and "it is conceivable
# that" are literal phrases for the same reason.
_HYPOTHESIS_CUE = re.compile(
    r"\bwe\s+hypothesi[sz]ed?\s+that\b"
    r"|\bit\s+is\s+conceivable\s+that\b"
    r"|\bcould\s+induce\b"
    r"|\bmight\s+\w+\b"
    r"|\bmay\s+(?:induce|improve|benefit|enhance|promote|reduce|prevent|protect|"
    r"ameliorate|attenuate|exacerbate|alleviate|restore|suppress|inhibit)\b",
    re.I,
)

# Where a hypothesis clause ends: the same breaks as `_CLAUSE_BREAK`, minus a
# bare "and" - which a hypothesis's own coordinated verbs use, sharing one
# modal ("... might induce X and improve Y" is one guess about two things),
# not a new clause with a result of its own.
_HYPOTHESIS_BREAK = re.compile(
    r"[,;:]|\s(?=(?:which|but|whereas|while|thus|hence|so)\b)"
    r"|\s(?=(?:to\s+(?:show|demonstrate|reveal|confirm)|showing|demonstrating|revealing)\b)",
    re.I,
)


def _stance_spans(sentence: str) -> tuple[list[tuple[int, int]], str | None]:
    """The stretches of one sentence that state an aim, method, prediction,
    hypothesis or open question, and the kind of the first cue found ("aim",
    "hope" or "hypothesis")."""
    spans: list[tuple[int, int]] = []
    kind = None
    for m in _AIM_CUE.finditer(sentence):
        end = next((b.start() for b in _CLAUSE_BREAK.finditer(sentence, m.end())), len(sentence))
        spans.append((m.start(), end))
        kind = kind or "aim"
    for m in _HOPE_CUE.finditer(sentence):
        start = max((b.end() for b in _CLAUSE_BREAK.finditer(sentence, 0, m.start())), default=0)
        end = next((b.start() for b in _CLAUSE_BREAK.finditer(sentence, m.end())), len(sentence))
        spans.append((start, end))
        kind = kind or "hope"
    for m in _HYPOTHESIS_CUE.finditer(sentence):
        start = max((b.end() for b in _HYPOTHESIS_BREAK.finditer(sentence, 0, m.start())), default=0)
        end = next((b.start() for b in _HYPOTHESIS_BREAK.finditer(sentence, m.end())), len(sentence))
        spans.append((start, end))
        kind = kind or "hypothesis"
    return spans, kind


def not_a_result_reason(text: str) -> str | None:
    """Why `text` states no result - it only says what a study set out to do,
    how, what it predicts, hypothesises or hopes for, or what is unknown - or
    None.

    "not_a_result:aim", "not_a_result:hope" or "not_a_result:hypothesis",
    after the first cue that matched. None whenever no cue matches, and
    whenever anything outside the cued stretches, in any sentence of `text`,
    reports a result: a quote that opens with its aim and then gives the
    result is a finding. `reports_result` does its own masking of a
    model-naming adjective ("diet-induced"), so that masking happens here
    too, on exactly the same rule.

    Run on the paper's own verified sentence in Stage 5 (after
    `other_work_reason`), and by the citation guard on report sentences and
    on the quotes they cite.
    """
    kind = None
    left: list[str] = []
    for sentence in split_sentences(text):
        spans, found = _stance_spans(sentence)
        kind = kind or found
        keep, last = [], 0
        for start, end in sorted(spans):
            if start > last:
                keep.append(sentence[last:start])
            last = max(last, end)
        keep.append(sentence[last:])
        left.append(" ".join(keep))
    if kind is None:
        return None
    if reports_result(" ".join(left)):
        return None
    return f"not_a_result:{kind}"


# --- a conclusion sentence ------------------------------------------------------
#
# Authors' conclusions generalise, and the most common way they do is by who
# was studied. The asthma hand audit (S3, evals/litreview/hand_audit/
# qwen_asthma.md) found a report sentence copying an abstract's conclusion,
# "MUC1-CT reduces NLRP3 inflammasome-mediated pyroptosis ... thereby
# attenuating neutrophil airway inflammation in patients with asthma", when
# the mechanism was shown in a cell line and in mice. The sentence is the
# paper's own and verifies; what it overstates is its subjects.
#
# This block only says which sentences are conclusions. What is done about
# one that names people is `extract/provenance.py` `methods_subjects`: its
# people are checked against the paper's Methods. That catches the plain
# case - a conclusion naming people in a paper whose Methods studied none.
# It does not reach S3 itself: that paper also recruited patients (for
# induced sputum), and which of a paper's experiments backs which clause of
# its conclusion is beyond a word rule.
#
# A sentence is a conclusion when
#
# - it opens with a concluding phrase ("In conclusion", "Taken together",
#   "Collectively", "Overall,", "Our findings demonstrate", "This study
#   revealed", "These data suggest"). Not a bare "The results show": in a
#   Results section that opens a result, not a conclusion. And "Overall",
#   "Together" and "Altogether" only with a comma: "Overall survival was ..."
#   is a result;
# - in an abstract, it sits under a "Conclusion(s)" or "Interpretation"
#   label ("CONCLUSIONS:", or glued on as Europe PMC often leaves it:
#   "...p<0.001.ConclusionDPL combined with BUD ..."); or, in an abstract with
#   no labels, a concluding phrase opened an earlier sentence of its
#   paragraph. That is S3: "This study revealed that MUC1 downregulation ...
#   MUC1-CT reduces ...". In a labelled abstract the label decides, so a
#   "Our results show" under RESULTS does not make the rest of that part a
#   conclusion.

CONCLUSION_CUE = re.compile(
    r"^\W*(?:"
    r"in\s+(?:summary|conclusion|sum|short)\b"
    r"|to\s+(?:conclude|summari[sz]e)\b"
    r"|taken\s+together\b|collectively\b"
    r"|(?:overall|together|altogether|conclusively)\s*,"
    r"|(?:our|these|this|the\s+present|the\s+current|present)\s+(?:present\s+|current\s+|combined\s+)?"
    r"(?:findings|results|data|study|work|observations|analyses|experiments)\s+(?:\w+\s+){0,2}?"
    r"(?:show|suggest|indicat|demonstrat|reveal|support|highlight|provid|confirm|establish|impl|identif|"
    r"uncover|point)\w*"
    r")",
    re.I,
)

# Structured-abstract labels, in the three forms the corpora show: "LABEL:"
# (any case), glued to the previous sentence and the next word
# ("...unclear.MethodsKidney samples ...", Title case only, so ordinary
# prose never matches), and alone on a line.
_LABELS = (r"Background|Introduction|Objectives?|Aims?|Purpose|Materials\s+and\s+methods|Patients\s+and\s+methods|"
           r"Subjects\s+and\s+methods|Methods?|Design|Participants|Results|Findings|Conclusions?|Interpretation")
ABSTRACT_LABEL = re.compile(
    r"\b(?P<a>" + _LABELS + r")\s*:"
    r"|(?:(?<=[a-z0-9.%)\]\s])|^)(?-i:(?P<b>" + _LABELS + r"))(?-i:(?=[A-Z]))"
    r"|(?:^|(?<=\n))[ \t]*(?P<c>" + _LABELS + r")[ \t]*(?=\n|$)",
    re.I,
)
_CONCLUSION_LABEL = re.compile(r"^(?:conclusions?|interpretation)$", re.I)
_METHODS_LABEL = re.compile(r"^(?:(?:materials|patients|subjects)\s+and\s+methods|methods?|design|participants)$", re.I)


def abstract_labels(abstract: str) -> list[tuple[str, int, int]]:
    """(label, start, end) of each structured-abstract label, in order."""
    out = []
    for m in ABSTRACT_LABEL.finditer(abstract or ""):
        label = m.group("a") or m.group("b") or m.group("c")
        out.append((re.sub(r"\s+", " ", label), m.start(), m.end()))
    return out


# A structured-abstract label is sometimes not the very first word: Europe
# PMC's own text has "Key Results: Among 45 participants ..." and
# "Background and Purpose: Children with acute asthma ...". `ABSTRACT_LABEL`
# (built for `conclusion_reason`, which only needs to know *that* a label is
# there, not trim it) matches just "Results:" and "Purpose:" in those two -
# the bare word from `_LABELS` that sits right before the colon. What comes
# before it ("Key ", "Background and ") is plain words with no digits or
# punctuation, never real sentence content, so it is swept up with the label.
# A real sentence reaching a colon this early would almost always have a
# number or punctuation before it; this is what tells the two apart.
_LABEL_LEAD_IN = re.compile(r"^(?:[A-Za-z]+\s+){0,3}$")


def strip_leading_label(text: str, start: int) -> tuple[str, int]:
    """If `text` opens with a structured-abstract label ("Background and
    Purpose:", or the glued form "...p<0.001.ConclusionDPL combined..."),
    return (text with the label removed, start advanced by the same
    amount). Otherwise (text, start) unchanged.

    `start` is `text`'s offset in whatever section text it was found in
    (the paper's own text, at fixed character positions). Only `start` moves
    and only forward, by exactly the number of characters cut from the front
    of `text` - never `end`, which the caller keeps - so
    `section_text[start:end] == text` still holds after the strip, the same
    invariant `report/invariants.py`'s `_anchored()` checks on every run.
    """
    labels = abstract_labels(text)
    if not labels:
        return text, start
    _, lab_start, lab_end = labels[0]
    if lab_start > 0 and not _LABEL_LEAD_IN.match(text[:lab_start]):
        return text, start
    cut = lab_end
    while cut < len(text) and text[cut].isspace():
        cut += 1
    if cut == 0 or cut >= len(text):
        # Nothing to cut, or the span is nothing but a label: an empty quote
        # would be worse than a labelled one, so leave it for the checks
        # downstream to reject on their own terms.
        return text, start
    return text[cut:], start + cut


def abstract_methods(abstract: str) -> list[tuple[int, int]]:
    """Where a structured abstract's Methods part is: (start, end) of the
    text after each Methods-type label, up to the next label. Empty for an
    abstract with no such label - an unlabelled paragraph is not guessed at."""
    labels = abstract_labels(abstract)
    spans = []
    for i, (label, _, end) in enumerate(labels):
        if _METHODS_LABEL.match(label):
            stop = labels[i + 1][1] if i + 1 < len(labels) else len(abstract)
            spans.append((end, stop))
    return spans


def conclusion_reason(section_text: str, start: int, end: int, section: str) -> str | None:
    """Why the sentence at `section_text[start:end]` is the authors'
    conclusion - "conclusion:cue", "conclusion:label" or
    "conclusion:paragraph" - or None. See the block comment above."""
    quote = section_text[start:end]
    if any(CONCLUSION_CUE.search(s) for s in split_sentences(quote)):
        return "conclusion:cue"
    if section != "abstract":
        return None
    labels = [lab for lab in abstract_labels(section_text) if lab[2] <= start]
    if labels:
        return "conclusion:label" if _CONCLUSION_LABEL.match(labels[-1][0]) else None
    paragraph = section_text.rfind("\n\n", 0, start)
    head = section_text[paragraph + 2 if paragraph >= 0 else 0 : start]
    if any(CONCLUSION_CUE.search(s) for s in split_sentences(head)):
        return "conclusion:paragraph"
    return None


# --- a qualifier in the paper's next sentence ---------------------------------
#
# Widening shows the whole sentence a quote sits in, so a qualifier inside it
# ("but not in controls", "p = 0.099") is on the page. A qualifier in the
# *next* sentence was not: the obesity hand audit (S6,
# evals/litreview/hand_audit/qwen_obesity.md) found a report sentence
# restating
#
#     "In the MC4R mutation carriers, placebo subtracted group mean
#      differences in weight loss (dose 0.01 mg/kg/24 h) of ~0.6 kg/week were
#      observed ()."
#
# faithfully, while the paper's very next sentence reads "Although this result
# did not reach statistical significance (p = 0.09) when compared to placebo,
# weight loss from baseline was significantly reduced (p < 0.0001)." Neither
# the writer nor the guard ever saw it.
#
# So when the sentence right after a quote - same section, same paragraph -
# qualifies it, Stage 5 stores it beside the quote (`EvidenceUnit.
# qualifier_quote`), anchored like the context sentence. It is shown to the
# reader, given to the writer and citable, and the citation guard drops a
# sentence that states the quote's result while leaving out a "not
# significant" the qualifier says about it (`report/guard.py`, check 8).
#
# Everything that decides this is in this block, and it is deliberately
# narrow. What counts as a qualifier:
#
# - it opens with a word that concedes or contrasts: "Although", "However",
#   "But", "Nevertheless", "Nonetheless", "In contrast". Measured on the three
#   stored reports this also stores some contrasts about something else ("In
#   contrast, SCN5A was not significantly changed"). That costs only a line of
#   the paper's own text shown beside the quote; the guard's drop rule below
#   never fires on one, because it also needs the sentence to point back;
# - it says "this/these/the <difference|effect|result|trend|change> was/did
#   not ...";
# - it says the result was not significant AND points back at it in its first
#   clause: "Because the findings for the first secondary efficacy endpoint
#   were not statistically significant, ..." (migraine S9).
#
# A sentence that merely contains "not significant" is not a qualifier: "IL-6
# was not significantly different" after a TNF result is a separate finding.

_QUALIFIER_OPENER = re.compile(r"^(?:although|however|but|nevertheless|nonetheless|in\s+contrast)\b", re.I)
# What a sentence calls the result it points back at.
_RESULT_NOUN = (r"(?:result|difference|effect|finding|change|trend|increase|decrease|reduction|improvement|"
                r"association|correlation|observation|response|benefit)s?")
# Words that may open a pointing-back sentence before its subject; stripped
# before the sentence is split into clauses.
_LEAD_IN = re.compile(r"^(?:although|though|however|but|nevertheless|nonetheless|in\s+contrast|because|since|"
                      r"while)\b,?\s*", re.I)
# "Although this result ...", "Because the findings for the first secondary
# efficacy endpoint ...": the sentence's subject is the result just reported.
_POINTS_BACK = re.compile(
    r"^(?:(?:although|though|however|but|nevertheless|nonetheless|in\s+contrast|because|since|while),?\s+)?"
    r"(?:this|these|that|such|the)\s+(?:[\w-]+\s+)?" + _RESULT_NOUN + r"\b", re.I)
_THIS_WAS_NOT = re.compile(
    r"^(?:this|these|the)\s+(?:difference|effect|result|trend|change)s?\s+(?:was|were|did)\s+not\b", re.I)
# Saying a result was not statistically significant. "p = 0.05" and above, or
# "p > ...", count; "p < 0.05" does not.
NOT_SIGNIFICANT = re.compile(
    r"\bdid\s+not\s+reach\s+(?:statistical\s+)?significance|\bnot\s+(?:statistically\s+)?significant|"
    r"\bnon-?significant|\bn\.s\.|\bp\s*=\s*0?\.(?:0[5-9]|[1-9])|\bp\s*>\s*0?\.\d", re.I)
# What a report sentence says when it has kept that limit: the words above,
# or a hedge that only fits a non-significant result ("numerically favored",
# "a trend towards").
KEEPS_NOT_SIGNIFICANT = re.compile(
    NOT_SIGNIFICANT.pattern + r"|\binsignificant|\bdid\s+not\s+achieve|\bfailed\s+to\s+reach|\bnumerically|"
    r"\btrend(?:s|ed)?\b|\btended\b|\bmarginal(?:ly)?\b|\bno\s+(?:statistically\s+)?significant", re.I)
# Words that carry no subject of their own when comparing a qualifier's
# subject with the quote's words.
_FILLER = frozenset({"this", "these", "that", "those", "such", "were", "when", "than", "then", "from", "with",
                     "also", "into", "over", "after", "before", "their", "there", "which", "while", "both",
                     "each", "only", "compared", "versus", "group", "groups", "because", "since", "although",
                     "though", "however", "nevertheless", "nonetheless", "contrast"})
_CONTENT_WORD = re.compile(r"[^\W\d_][\w-]*")


def content_words(text: str) -> set[str]:
    """The words of `text` that say what it is about: four letters or more,
    or a name with a digit in it ("IL-6", "MC4R"); plural "s" dropped, filler
    and result words ("significantly", "reduced") left out."""
    words = (m.group(0).lower().strip("-") for m in _CONTENT_WORD.finditer(text))
    return {w[:-1] if w.endswith("s") else w for w in words
            if (len(w) >= 4 or any(c.isdigit() for c in w)) and w not in _FILLER and not RESULT_WORDS.fullmatch(w)}


def _first_clause(sentence: str) -> tuple[str, str]:
    """(first clause, rest): split at the first comma, semicolon, "but" or
    "whereas" outside brackets, after any leading "Although"/"However,".
    "Although this result did not reach statistical significance (p = 0.09)
    when compared to placebo" | "weight loss from baseline was significantly
    reduced (p < 0.0001)." """
    lead = _LEAD_IN.match(sentence)
    depth = 0
    for i in range(lead.end() if lead else 0, len(sentence)):
        c = sentence[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        elif depth == 0 and (c in ",;" or re.match(r"\s(?:but|whereas)\b", sentence[i:], re.I)):
            return sentence[:i], sentence[i:].lstrip(" ,;")
    return sentence, ""


def not_significant_clause(qualifier: str, quote: str) -> tuple[str, str] | None:
    """When `qualifier` says the result in `quote` was not significant: its
    (not-significant clause, the rest of the sentence). Otherwise None.

    Three things must hold. The sentence points back at the quote's result
    ("this result", "the findings for ..."); its first clause says "not
    significant"; and any subject it names in that clause is the quote's
    ("the findings for the first secondary efficacy endpoint" beside a quote
    about that endpoint), so "The reduction in IL-6 was not significant" is
    not read as a verdict on a TNF quote before it.
    """
    back = _POINTS_BACK.match(qualifier) or _THIS_WAS_NOT.match(qualifier)
    if back is None:
        return None
    first, rest = _first_clause(qualifier)
    verdict = NOT_SIGNIFICANT.search(first)
    if verdict is None:
        return None
    subject = content_words(qualifier[back.end():verdict.start()])
    if not subject <= content_words(quote):
        return None
    return first, rest


def is_qualifier(sentence: str, quote: str) -> bool:
    """Does `sentence`, the one right after `quote`, qualify it? See above."""
    return bool(_QUALIFIER_OPENER.match(sentence) or _THIS_WAS_NOT.match(sentence)
                or not_significant_clause(sentence, quote))


def qualifier_after(text: str, start: int, end: int) -> Located | None:
    """The sentence right after `text[start:end]`, when it qualifies it.

    Same section by construction (`text` is one section); same paragraph,
    because a blank line between them ends the search: a new paragraph is
    about something else. The sentence is found with the same boundaries
    widening uses and returned at its offsets in `text`, like a quote.
    """
    k = end
    while k < len(text) and text[k].isspace():
        k += 1
    if k >= len(text) or re.search(r"\n\s*\n", text[end:k]):
        return None
    a, b = sentence_bounds(text, k, k + 1)
    sentence = text[a:b]
    if a < end or not sentence or not is_qualifier(sentence, text[start:end]):
        return None
    return Located("exact", a, b, sentence)
