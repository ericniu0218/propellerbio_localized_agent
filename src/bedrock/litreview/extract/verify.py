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
_OTHER_WORK_LEAD_IN = re.compile(
    r"^\s*(?:previous studies have shown|a recent study found|another study showed|"
    r"it has been reported|has been shown to|prior work|earlier studies)\b",
    re.I,
)

# The canonical section names a paper's text can carry are exactly
# `parsing/sections.py` `CANONICAL`. Introduction and Background are not
# among them - both fold into "other" - so "other" is the real name to check.
_INTRO_LIKE_SECTIONS = frozenset({"other"})

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
