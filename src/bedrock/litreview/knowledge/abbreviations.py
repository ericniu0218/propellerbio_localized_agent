"""A paper's own abbreviations, and what a finding's target really names.

Two mistakes this module exists to stop, both seen in real reports:

- **An abbreviation read as somebody else's.** The asthma report filed "AHR"
  under the aryl hydrocarbon receptor (AhR). The paper had said, in its own
  Methods, "Airway hyper-responsiveness (AHR) was detected using ..." - it
  was a breathing readout, not a receptor. Letters alone cannot tell the two
  apart; the paper's definition can.
- **An outcome measure read as a target.** The migraine report gave "MMDs"
  (monthly migraine days) its own row of candidate targets. Nobody can
  target a count of headache days.

So, in code and from the paper's text only:

1. `find_abbreviations` reads every "long form (SHORT)" and "SHORT (long
   form)" in the paper - the Schwartz-Hearst heuristic (Schwartz & Hearst,
   2003): the short form's letters must be found, in order and from the
   right, in the words before the bracket, and its first letter must start
   a word. "Airway hyper-responsiveness (AHR)" passes; "treated mice (AHR)"
   does not. The result is a per-paper map, SHORT -> long form; the same
   paper always gives the same map.
2. `resolve_target` decides what an extracted target names. When it is an
   abbreviation the paper defines, the paper's long form is what gets
   judged and normalised - never the table's entry for the same letters.
   Otherwise the name itself is.
3. A name that is an outcome measure (`outcome_measures.yaml`: MMD, PASI,
   BMI, FEV1, ... - and AHR, but only where the paper defines it as airway
   hyperresponsiveness, because AHR is also the receptor's gene symbol) is
   flagged. The finding is kept - "GRI977143 reduced AHR" is a real result
   and belongs in the prevention section - but Stage 6 gives it no target
   row (`EvidenceUnit.outcome_measure`).

Why keep the finding rather than drop it: dropping would lose a verified
result. Why not blank its target: two different readouts from one paper
("AHR" and "eosinophil counts") would then look like the same finding to
`extract/dedupe.py`, and the evidence table would lose the name of what was
measured. Keeping the name and flagging it changes only the one thing that
was wrong - the row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from bedrock.litreview.knowledge import Knowledge, clean, fold, looks_like_target, term_pattern

MEASURES_TABLE = Path(__file__).parent / "outcome_measures.yaml"


# --- 1. the paper's own definitions ----------------------------------------------

# A bracket with no bracket inside it. Its content is cut at the first ";" or
# ",", as Schwartz and Hearst do: "(AHR; ref. 12)" defines AHR.
_BRACKET = re.compile(r"\(([^()]{1,200})\)")
# Where a sentence (or an earlier bracket) ends, the long form cannot reach back
# past it.
_BARRIER = re.compile(r"[.;:!?](?:\s|$)|[()\[\]\n]")


def _short_form_ok(short: str) -> bool:
    """Shaped like an abbreviation: 2-10 characters, starting with a letter or
    digit, holding at least one upper-case letter and no full stop, and no
    more than two words. The upper-case letter keeps "(mice)" and "(n = 6)"
    out; the full stop keeps "(Fig. C)" out."""
    return (
        2 <= len(short) <= 10
        and short[0].isalnum()
        and "." not in short
        and any(ch.isupper() for ch in short)
        and len(short.split()) <= 2
    )


def best_long_form(short: str, candidate: str) -> str | None:
    """The shortest tail of `candidate` that spells `short`, Schwartz-Hearst
    style, or None.

    Walks the short form from its last character to its first and finds each
    letter or digit in the candidate, also from the right. The first
    character must begin a word. "AHR" in "we measured airway
    hyperresponsiveness" finds R and H inside "hyperresponsiveness" and A at
    the start of "airway", so the long form is "airway hyperresponsiveness".
    """
    s = len(short) - 1
    i = len(candidate) - 1
    while s >= 0:
        ch = short[s].lower()
        if not ch.isalnum():
            s -= 1
            continue
        while i >= 0 and (candidate[i].lower() != ch or (s == 0 and i > 0 and candidate[i - 1].isalnum())):
            i -= 1
        if i < 0:
            return None
        i -= 1
        s -= 1
    start = max(candidate.rfind(" ", 0, i + 1), candidate.rfind("\n", 0, i + 1)) + 1
    long_form = candidate[start:].strip()
    return long_form or None


def _accept(short: str, long_form: str | None) -> str | None:
    """The checks Schwartz and Hearst put on a match: longer than the short
    form, not containing it, and not too many words for its letters."""
    if not long_form:
        return None
    words = long_form.split()
    letters = sum(1 for ch in short if ch.isalnum())
    if len(long_form) <= len(short) or len(words) > min(letters + 5, letters * 2):
        return None
    # The short form must not be one of the long form's own words: "Sirt6 KO
    # mice (Sirt6-/-)" names a genotype, it does not define "Sirt6".
    core = fold(short)
    if any(fold(w) == core for w in re.split(r"[^0-9A-Za-zα-ωΑ-Ω]+", long_form) if w):
        return None
    return long_form


def find_abbreviations(text: str) -> dict[str, str]:
    """Every abbreviation `text` defines, SHORT -> long form. The first
    definition wins when a paper defines one twice."""
    text = clean(text)
    found: dict[str, str] = {}
    for m in _BRACKET.finditer(text):
        inside = re.split(r"[;,]", m.group(1), maxsplit=1)[0].strip()
        before = text[: m.start()]
        cut = max((b.end() for b in _BARRIER.finditer(before, max(0, len(before) - 300))), default=0)
        before = before[max(cut, len(before) - 300):]
        if _short_form_ok(inside):
            # "long form (SHORT)": the long form is in the words before it.
            words = before.split()
            letters = sum(1 for ch in inside if ch.isalnum())
            window = " ".join(words[-min(letters + 5, letters * 2):])
            long_form = _accept(inside, best_long_form(inside, window))
            short = inside
        else:
            # "SHORT (long form)": the short form is the word just before it.
            last = re.search(r"(\S+)\s*$", before)
            short = last.group(1).strip(".,;:\"'") if last else ""
            if not short or not _short_form_ok(short) or len(inside.split()) < 2:
                continue
            long_form = _accept(short, best_long_form(short, inside))
            if long_form != inside:
                # The whole bracket must be the long form, or it is a remark
                # about the word before it, not its definition.
                continue
        if long_form and short not in found:
            found[short] = long_form
    return found


def paper_abbreviations(paper) -> dict[str, str]:
    """The abbreviation map of one paper, from all of its text (title,
    abstract, then every section)."""
    return find_abbreviations(paper.all_text())


def defined_as(name: str, abbreviations: dict[str, str]) -> str | None:
    """The paper's long form for `name`, when `name` is one of its
    abbreviations. "MMD" finds a paper's "MMDs" and the reverse.

    Exact spelling first. Failing that, a paper that defines exactly one
    abbreviation spelled like `name` in another case ("aryl hydrocarbon
    receptor (AhR)" when the model copied "AHR") still decides: the paper has
    said what those letters mean in it. Only the case may differ, never a
    character: "Sirt6-/-" is not "Sirt6"."""
    name = clean(name).strip()
    keys = [k for k in (name, name[:-1] if name.endswith("s") else "", name + "s") if len(k) >= 2]
    for key in keys:
        if key in abbreviations:
            return abbreviations[key]
    lowered = {k.lower() for k in keys}
    other_case = {long for short, long in abbreviations.items() if short.lower() in lowered}
    return other_case.pop() if len(other_case) == 1 else None


# --- 2. outcome measures ------------------------------------------------------------


@dataclass(frozen=True)
class MeasureTable:
    measures: tuple[str, ...]
    abbreviations: dict[str, str]
    ambiguous: dict[str, str]
    """Also the name of a target ("AHR" is the aryl hydrocarbon receptor's
    gene symbol): a measure only when the paper defines it as one, so these
    are never matched by name - only their long forms are."""
    patterns: tuple[re.Pattern[str], ...]


@lru_cache(maxsize=4)
def load_measures(path: str | None = None) -> MeasureTable:
    raw = yaml.safe_load(Path(path or MEASURES_TABLE).read_text(encoding="utf-8")) or {}
    measures = tuple(str(m) for m in raw.get("measures") or [])
    abbreviations = {str(k): str(v) for k, v in (raw.get("abbreviations") or {}).items()}
    ambiguous = {str(k): str(v) for k, v in (raw.get("ambiguous_abbreviations") or {}).items()}
    terms = list(measures)
    for short in abbreviations:
        terms.append(short)
        if short[-1].isupper():
            terms.append(short + "s")  # "MMDs", "AEs"
    # `term_pattern` makes short acronyms case-sensitive ("AHR" never matches
    # "AhR") and lets hyphens and spaces stand in for each other.
    patterns = tuple(re.compile(term_pattern(t)) for t in dict.fromkeys(terms))
    return MeasureTable(measures, abbreviations, ambiguous, patterns)


def is_outcome_measure(name: str, kb: Knowledge | None = None, table: MeasureTable | None = None) -> bool:
    """Does `name` name a readout rather than something one could act on?

    It must contain a listed measure, and nothing target-shaped may be left
    once the measure is taken out: "moderate to severe MMDs" is a measure,
    "AHR signaling" (a receptor's pathway) and "leptin-induced weight loss"
    (leptin is in the entity table) are not.
    """
    table = table or load_measures()
    text = clean(name).strip()
    if not text:
        return False
    spans = [m.span() for p in table.patterns for m in p.finditer(text)]
    if not spans:
        return False
    rest = list(text)
    for start, end in spans:
        rest[start:end] = " " * (end - start)
    rest_text = re.sub(r"\s+", " ", "".join(rest)).strip()
    # Piece by piece, so a long list ("AHR, elevated serum IgE, ...") is not
    # waved through by `looks_like_target`'s length limit.
    if any(looks_like_target(piece) for piece in re.split(r"[,;/]|\band\b|\bor\b", rest_text)):
        return False
    if kb is not None and kb.scan(rest_text, ("target", "drug")):
        return False
    return True


# --- 3. what a target names -----------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTarget:
    name: str
    """What goes in `EvidenceUnit.target`: the table's canonical name, or the
    paper's own name when the table does not know it (or knows the same
    letters as something else)."""
    outcome_measure: bool
    """A readout, not a target: kept as evidence, never given a target row."""
    long_form: str | None = None
    """The paper's own definition, when the target was one of its
    abbreviations."""


def resolve_target(raw: str, kb: Knowledge, abbreviations: dict[str, str] | None = None,
                   table: MeasureTable | None = None) -> ResolvedTarget:
    """Normalise an extracted target, reading the paper's own definitions first.

    - The paper defines the name ("Airway hyper-responsiveness (AHR)"): the
      long form decides. A measure is flagged. Otherwise the long form is
      looked up in the entity table as a whole name; if the table does not
      know it, the paper's own abbreviation is kept - the table's entry for
      the same letters may be a different molecule, and the paper has said
      which one it means. Whole names only: "OX40 ligand" contains "OX40",
      the receptor, and is not it.
    - The paper does not define it: the name decides, against the measure
      list and then the table, as before. An ambiguous abbreviation ("AHR")
      is never a measure by name alone.
    """
    name = raw.strip()
    long_form = defined_as(name, abbreviations or {})
    if long_form:
        if is_outcome_measure(long_form, kb, table):
            return ResolvedTarget(name, True, long_form)
        known = kb.lookup(long_form, ("target",))
        if known:
            return ResolvedTarget(known.canonical, False, long_form)
        own = kb.normalize(name, "target")
        # The table knows the letters but not the paper's long form. Keep the
        # table's name only if one of its names for that entity is inside the
        # long form ("calcitonin gene-related peptide receptor" -> CALCA);
        # otherwise the letters are a coincidence and the paper's name stays.
        entity = kb.get(own)
        if entity and any(fold(t) in fold(long_form) for t in (entity.canonical, *entity.terms) if len(t) > 3):
            return ResolvedTarget(own, False, long_form)
        return ResolvedTarget(name, False, long_form)
    if is_outcome_measure(name, kb, table):
        return ResolvedTarget(name, True)
    return ResolvedTarget(kb.normalize(raw, "target"), False)
