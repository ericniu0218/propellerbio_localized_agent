"""Exact entity names, and the code that finds them in text.

The client asked for answers about the exact drug, species or virus in the
question — not about things that look close enough. That is a matching
problem, and it is solved here in code, from a table (`entities.yaml`), for the
same reason quotes are verified in code: a lookup gives the same answer every
time and anybody can read why it gave it.

Three details carry most of the weight:

- **Boundaries.** "SARS-CoV" must not match inside "SARS-CoV-2", "IL-2" must not
  match inside "IL-23", and "RSV" must not match inside "BRSV". A term may not
  be glued to a letter or digit on either side, nor followed by "-<digit>".
  "anti-IL-23" and "psoriasis-like" still match.
- **Longest match wins.** "psoriatic" is a psoriasis term but "psoriatic
  arthritis" is a different disease; "respiratory syncytial virus" is RSV but
  "bovine respiratory syncytial virus" is not. Overlapping matches are
  resolved longest-first, so the near-miss swallows the shorter exact term.
- **Case.** Short acronyms (RSV, PsA, H5N1) match case-sensitively, so "als"
  in a sentence is not ALS. Longer names match in any case.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

TABLE = Path(__file__).parent / "entities.yaml"

KIND_BY_SECTION = {
    "diseases": "disease",
    "pathogens": "pathogen",
    "drugs": "drug",
    "targets": "target",
}

_PUNCT = {
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u2007": " ",
}


def clean(text: str) -> str:
    """Typography out, content in. Applied to every text before matching."""
    text = unicodedata.normalize("NFKC", text or "")
    for bad, good in _PUNCT.items():
        text = text.replace(bad, good)
    return text


_FOLD = re.compile(r"[^0-9a-zα-ω]+")


def fold(name: str) -> str:
    """'IL-23', 'il 23' and 'IL_23' all fold to 'il23'."""
    return _FOLD.sub("", clean(name).lower())


def _case_sensitive(term: str) -> bool:
    upper = sum(1 for ch in term if ch.isupper())
    return upper >= 2 and len(term) <= 8


_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_SEP = r"[\s\-]?"


def term_pattern(term: str) -> str:
    """Regex source for one term, with the boundary rules described above."""
    term = clean(term).strip()
    words = [w for w in re.split(r"[\s\-]+", term) if w]
    pieces = []
    for word in words:
        # "IL23" should find "IL-23" and "H5N1" should find "H5-N1": allow an
        # optional separator wherever letters meet digits.
        sub = _LETTER_DIGIT.split(word)
        pieces.append(_SEP.join(re.escape(s).replace("'", "'?") for s in sub if s))
    body = _SEP.join(pieces)
    last = words[-1] if words else ""
    if last.isalpha() and last.islower() and len(last) >= 4:
        body += "(?:s|es)?"
    if not _case_sensitive(term):
        body = f"(?i:{body})"
    return rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])(?!-\d)"


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    term: str
    kind: str  # "exact" | "near_miss"
    text: str
    label: str = ""


@dataclass
class TermMatcher:
    """Finds exact and near-miss mentions of one entity in text."""

    exact_terms: list[str]
    near_miss_terms: list[str] = field(default_factory=list)
    near_miss_patterns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._compiled: list[tuple[re.Pattern[str], str, str]] = []
        for term in dict.fromkeys(self.exact_terms):
            self._compiled.append((re.compile(term_pattern(term)), term, "exact"))
        for term in dict.fromkeys(self.near_miss_terms):
            self._compiled.append((re.compile(term_pattern(term)), term, "near_miss"))
        for pattern in self.near_miss_patterns:
            wrapped = rf"(?<![A-Za-z0-9])(?:{pattern})(?![A-Za-z0-9])(?!-\d)"
            self._compiled.append((re.compile(wrapped), pattern, "near_miss"))

    def matches(self, text: str) -> list[Match]:
        text = clean(text)
        found = [
            Match(m.start(), m.end(), term, kind, m.group(0))
            for regex, term, kind in self._compiled
            for m in regex.finditer(text)
        ]
        return resolve_overlaps(found)

    def exact_count(self, text: str) -> int:
        return sum(1 for m in self.matches(text) if m.kind == "exact")

    def summary(self, text: str) -> tuple[int, list[str]]:
        """(number of exact mentions, distinct near-miss strings found)."""
        ms = self.matches(text)
        exact = sum(1 for m in ms if m.kind == "exact")
        near = sorted({m.text for m in ms if m.kind == "near_miss"})
        return exact, near


def resolve_overlaps(found: list[Match]) -> list[Match]:
    """Longest match wins; on a tie, an exact match beats a near-miss."""
    found.sort(key=lambda m: (-(m.end - m.start), 0 if m.kind == "exact" else 1, m.start))
    kept: list[Match] = []
    for m in found:
        if all(m.end <= k.start or m.start >= k.end for k in kept):
            kept.append(m)
    kept.sort(key=lambda m: m.start)
    return kept


# --- the table ---------------------------------------------------------------


@dataclass(frozen=True)
class KnownEntity:
    canonical: str
    kind: str
    terms: tuple[str, ...]
    near_miss: tuple[str, ...] = ()
    near_miss_patterns: tuple[str, ...] = ()
    context_terms: tuple[str, ...] = ()
    """Generic names ("F protein") that only mean this entity inside the right
    paper. Used to merge extracted names, never to scan questions or gate."""

    def matcher(self) -> TermMatcher:
        return TermMatcher(list(self.terms), list(self.near_miss), list(self.near_miss_patterns))


class Knowledge:
    """The entity table, loaded once, with lookups over it."""

    def __init__(self, entities: list[KnownEntity]) -> None:
        self.entities = entities
        self._by_canonical = {e.canonical: e for e in entities}
        self._fold_index: dict[str, list[KnownEntity]] = {}
        for entity in entities:
            for term in (entity.canonical, *entity.terms):
                self._fold_index.setdefault(fold(term), []).append(entity)
        self._scan: list[tuple[re.Pattern[str], KnownEntity, str]] = [
            (re.compile(term_pattern(term)), entity, term)
            for entity in entities
            for term in entity.terms
        ]

    @classmethod
    def load(cls, path: Path | None = None) -> "Knowledge":
        raw = yaml.safe_load(Path(path or TABLE).read_text(encoding="utf-8")) or {}
        entities = []
        for section, kind in KIND_BY_SECTION.items():
            for canonical, body in (raw.get(section) or {}).items():
                body = body or {}
                entities.append(
                    KnownEntity(
                        canonical=str(canonical),
                        kind=kind,
                        terms=tuple(str(t) for t in body.get("terms", [])),
                        near_miss=tuple(str(t) for t in body.get("near_miss", [])),
                        near_miss_patterns=tuple(body.get("near_miss_patterns", [])),
                        context_terms=tuple(str(t) for t in body.get("context_terms", [])),
                    )
                )
        return cls(entities)

    def get(self, canonical: str) -> KnownEntity | None:
        return self._by_canonical.get(canonical)

    def lookup(self, name: str, kinds: tuple[str, ...] | None = None) -> KnownEntity | None:
        """Whole-name lookup: 'IL-23p19' -> IL23A. None when unknown."""
        for entity in self._fold_index.get(fold(name), []):
            if kinds is None or entity.kind in kinds:
                return entity
        return None

    def scan(self, text: str, kinds: tuple[str, ...] | None = None) -> list[tuple[KnownEntity, Match]]:
        """Every known entity named in `text`, longest match first, in order."""
        text = clean(text)
        found: list[Match] = []
        owner: dict[tuple[int, int, str], KnownEntity] = {}
        for regex, entity, term in self._scan:
            if kinds is not None and entity.kind not in kinds:
                continue
            for m in regex.finditer(text):
                match = Match(m.start(), m.end(), term, "exact", m.group(0), entity.canonical)
                found.append(match)
                owner[(match.start, match.end, entity.canonical)] = entity
        kept = resolve_overlaps(found)
        return [(owner[(m.start, m.end, m.label)], m) for m in kept]

    def normalize(self, raw: str, kind: str) -> str:
        """Map an extracted name onto the table's canonical name.

        Tries a whole-name match first, then context terms, then looks for
        exactly one known name inside the string ("anti-IL-23p19 antibody").
        Two different names inside it ("IL-23/IL-17 axis") is ambiguous, so the
        raw string is kept rather than guessed at.
        """
        if not raw:
            return raw
        entity = self.lookup(raw, (kind,))
        if entity:
            return entity.canonical
        folded = fold(raw)
        for candidate in self.entities:
            if candidate.kind == kind and any(fold(t) == folded for t in candidate.context_terms):
                return candidate.canonical
        inside = {e.canonical for e, _ in self.scan(raw, (kind,))}
        if len(inside) == 1:
            return inside.pop()
        return raw.strip()

    def alias_table(self, kind: str) -> dict[str, list[str]]:
        return {
            e.canonical: [*e.terms, *e.context_terms] for e in self.entities if e.kind == kind
        }


@lru_cache(maxsize=4)
def load_knowledge(path: str | None = None) -> Knowledge:
    return Knowledge.load(Path(path) if path else None)


# --- families too large to list -----------------------------------------------

_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # influenza subtypes: a question about H5N1 is not about H1N1
    (re.compile(r"^(?:influenza\s+)?(?:A\s*\()?H\d{1,2}N\d{1,2}\)?$", re.I), r"H\d{1,2}N\d{1,2}"),
    # numbered cytokines: IL-23 is not IL-2 or IL-12
    (re.compile(r"^IL-?\d+[A-Z]?$", re.I), r"IL-?\d+[A-Z]?"),
    # coronavirus species
    (re.compile(r"^(?:SARS|MERS|HCoV)-?CoV-?\d?$", re.I), r"(?:SARS|MERS)-CoV(?:-\d)?|HCoV-[A-Z0-9]+"),
]


def family_near_miss_patterns(name: str) -> list[str]:
    """Near-miss regexes for an entity the table does not know, by its shape."""
    name = clean(name).strip()
    return [pattern for shape, pattern in _FAMILY_PATTERNS if shape.match(name)]


# --- does a name look like something one could target? -------------------------
#
# Extraction copies whatever the paper acted on or measured into `target`, and
# a statistics paper supplies "dropout", "discontinuation" and "participants".
# A target row in the report needs a name shaped like a gene, protein, drug,
# cell type or pathway; anything else stays in the evidence but gets no row.

_TARGET_NAME = re.compile(
    r"\b[A-Z][A-Za-z]*-?\d+[A-Za-z0-9]*|"          # IL-23, TYK2, UBE2L3, CD8, HIF-1α
    r"\b(?=\w*[A-Z]\w*[A-Z])\w{2,}\b|"             # VEGF, HuR, PDI, NF-κB
    r"\b[a-z]-[A-Z][a-z]+\b"                       # c-Rel, c-Kit
)
_TARGET_WORD = re.compile(
    r"\w+(?:mab|nib|cept)\b|vaccin|antibod|antigen|\b(?:T|B)[\s-]cells?\b|\bTh17\b|\bTregs?\b|keratinocyte|"
    r"macrophage|dendritic|neutrophil|mast cell|monocyte|fibroblast|pathway|signal(?:l)?ing|receptor|"
    r"cytokine|interleukin|chemokine|kinase|protein|enzyme|channel|ligase|transcription factor|"
    r"glycoprotein|prefusion",
    re.I,
)


# What was measured, not what was acted on. Extraction copies these out of
# results sentences ("RSVPreF3-specific CD4+ T-cell frequencies"), and they
# read as targets in their own right unless code says otherwise.
_MEASUREMENT = re.compile(
    r"\b(markers?|levels?|expression|frequenc(?:y|ies)|ratios?|tit(?:er|re)s?|scores?|counts?|"
    r"rates?|percentages?|proportions?)\b", re.I)
# Longer than this is a clause, not a name.
MAX_TARGET_CHARS = 60


def looks_like_target(name: str) -> bool:
    if len(name) > MAX_TARGET_CHARS or _MEASUREMENT.search(name):
        return False
    return bool(_TARGET_NAME.search(name) or _TARGET_WORD.search(name))
