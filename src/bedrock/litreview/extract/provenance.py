"""Where each field of a finding came from — checked in code, field by field.

Quote verification proves the model's sentence is in the paper. It proves
nothing about the other fields the model filled in beside it: the target it
says the sentence is about, the sample size, who was studied, the one-line
result. A correct quote can carry an invented "n = 40" or "in mice", and until
now the citation guard accepted both as evidence, because they sat on the same
unit as a verified quote.

So each of those fields is looked for in the text the model was shown, in
this order:

    quote    the paper's sentence the finding was taken from
    chunk    the rest of the passage the model read
    methods  the Methods paragraphs it was given as context

and the answer is recorded on the unit (`EvidenceUnit.provenance`). What
happens to a field found nowhere:

    target       the finding is dropped: it is about a molecule the text
                 the model read never names
    intervention, combo_with, disease_raw     set to empty
    species_raw  set to empty; the report shows "not stated"
    n            set to None
    effect_detail  set to empty unless every number and every known name in
                 it is in the quote or the context sentence

Sample size and who-was-studied are usually stated in a Methods sentence,
not in the result sentence itself. When they are, that sentence is found in
the paper and stored as `context_quote` — the paper's own words, anchored
like the quote — and only a field present in the quote or in that sentence
keeps its value. That is what lets the guard check a report sentence's
"in mice" or "n = 24" against text a reader can see.

Names are held to the same standard: the target, agent and combination
partner must be in text the reader is shown. Found in the chunk, a name
counted as located even though the reader saw only the quote - which is how
"Blocking IL-17A reduced ear thickness by 60%" passed the guard citing a quote
about anti-IL-23, in a passage that also named IL-17A. So a name found only in
the chunk or the Methods is looked for, in order, in

    context    the context sentence above
    qualifier  the paper's next sentence, when it qualifies the quote
    naming     otherwise, the passage's sentence that names it, nearest the
               quote and preferring one that also names what the quote
               names ("HDAC6 inhibitors, such as tubastatin A, ..."). Stored
               as `naming_quote`, anchored like the context sentence, and
               shown beside the quote

and that is what the unit's provenance records for it. A target that still
cannot be shown drops the finding; an agent or partner is blanked. Showing
the name is not the same as the quote being about it: the citation guard
still drops a sentence that gives the quote's result to a name the quote does
not state (`report/guard.py`, check 9).

One more check reads beyond the text the model was shown, because the
quote itself can be what overstates: a conclusion sentence that names
people, in a paper whose Methods studied only animals or cells
(`methods_subjects`, at the end of this file).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from bedrock.litreview.extract.normalize import normalize_species
from bedrock.litreview.extract.verify import (
    Located,
    abstract_methods,
    covered,
    locate_quote,
    meaning_words,
    normalize,
    people_named,
    split_sentences,
    studied_in,
)
from bedrock.litreview.knowledge import Knowledge, TermMatcher
from bedrock.litreview.negation import drops_negation
from bedrock.litreview.report.guard import numbers_in

QUOTE, CHUNK, METHODS, NONE = "quote", "chunk", "methods", "none"
# Where a name found outside the quote is shown to the reader.
CONTEXT, QUALIFIER, NAMING = "context", "qualifier", "naming"
# The names that must be in displayed text, and the kind each is looked up as.
_SHOWN_NAMES = (("target", "target"), ("intervention", "drug"), ("combo_with", "target"))

# Words that say a sentence describes who or how many were studied. Also what
# picks the Methods paragraphs for the extraction prompt (`stages/s5_extract`).
SUBJECT_WORDS = re.compile(
    r"\b(mice|mouse|rats?|patients|participants|subjects|volunteers|infants|children|adults|"
    r"animals?|n\s*=|cohort|enrolled|randomi[sz]ed|cells?|strain|C57BL|BALB|cynomolgus|macaques?|women|men)\b",
    re.I,
)
# A number followed by one of these is a dose, a time or a size, not a count.
_NOT_A_UNIT = (r"(?!\s*(?:%|mg|µg|μg|ug|ng|pg|g|kg|ml|mL|µl|μl|l|mm|cm|nm|µm|μm|h|hr|hrs|hours?|days?|weeks?|"
               r"months?|years?|min|mins|minutes?|s|sec|mol|mmol|µmol|mM|µM|μM|nM|pM|IU|U|Gy|°C|x|×|fold|percent|"
               r"per\s*cent|times)\b)")
# Words that may stand between a count and what it counts: "24 female
# C57BL/6 mice", "12 healthy adult volunteers". Not prepositions: "reduced
# by 12 in mice" is not a sample size.
_BETWEEN = r"(?:(?!(?:in|of|for|per|on|at|to|from|with|by|and|or|vs|than)\b)[\w/+-]+\s+){0,3}?"

# The normalised species (`extract/normalize.py`) -> its class in
# `knowledge/meaning_words.yaml`, the same classes the citation guard reads.
_SPECIES_CLASS = {"human": "human", "mouse": "mouse", "rat": "rat", "cyno": "primate", "cell_line": "cells"}
# Words in a model's cell-line species that name no particular cell line.
_GENERIC = {"cell", "cells", "line", "lines", "cultured", "culture", "primary", "human", "derived", "vitro"}


def sentences(text: str) -> list[str]:
    """The same sentence boundaries quote widening uses (abbreviations aware)."""
    return split_sentences(text)


@lru_cache(maxsize=4096)
def _matcher(terms: tuple[str, ...]) -> TermMatcher:
    return TermMatcher(list(terms))


@lru_cache(maxsize=256)
def _norm(text: str) -> str:
    return normalize(text)


def _whole(needle: str, text: str) -> bool:
    """Is `needle` in `text` as whole words? "IL-1" is not in "IL-17A", "RSV"
    is not in "bRSV", "TNF" is not in "TNFR2"."""
    n, t = _norm(needle), _norm(text)
    return bool(n) and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", t) is not None


def name_in(value: str, text: str, kb: Knowledge | None = None, kind: str | None = None) -> bool:
    """Is this name in the text — as written, or as a listed name of the same
    entity? Listed names come from the knowledge table, never the model. Whole
    words only: a name inside a longer, different name does not count."""
    value = (value or "").strip()
    if not value or not text:
        return False
    if _matcher((value,)).exact_count(text) or _whole(value, text):
        return True
    known = kb.lookup(value, (kind,) if kind else None) if kb is not None else None
    if known is None and kb is not None and kind:
        known = kb.lookup(value)
    return bool(known and _matcher((*known.terms, *known.context_terms, known.canonical)).exact_count(text))


def species_in(raw: str, text: str) -> bool:
    """Does the text name the same kind of subject the model's species does?

    Read through `verify.studied_in`, the one reading of who-was-studied the
    guard uses too, so "adult" beside "mice" is not a human. A cell line counts
    when the text says it was work in a dish or names the same line ("HaCaT"),
    never on a bare "cells": "T cells rose in patients" is not a cell line.
    """
    raw = (raw or "").strip()
    if not raw or not text:
        return False
    cls = _SPECIES_CLASS.get(normalize_species(raw))
    if cls is None:
        return _whole(raw, text)
    if cls in studied_in(text):
        return True
    if cls == "cells":
        tokens = [t for t in re.findall(r"[\w-]+", raw.lower()) if len(t) >= 3 and t not in _GENERIC]
        return any(re.search(r"\b" + re.escape(t) + r"\b", text, re.I) for t in tokens)
    return False


@lru_cache(maxsize=1)
def _subject_nouns() -> str:
    """Every noun a sample size can count, from `knowledge/meaning_words.yaml`:
    the who-was-studied words, the life-stage words and `counted`."""
    words = meaning_words()
    terms = {t for terms in words["studied"].values() for t in terms if " " not in t}
    terms |= set(words["ambiguous_human"]) | set(words["counted"])
    return "(?:" + "|".join(sorted((re.escape(t) for t in terms), key=len, reverse=True)) + ")"


def n_in(n: int, text: str) -> bool:
    """Is this number stated as a sample size — "n = 12", "12 mice", "12
    patients with psoriasis" — rather than merely present ("reduced by 12%")?"""
    if n is None or not text:
        return False
    num = re.escape(str(n))
    nouns = _subject_nouns()
    return bool(
        re.search(r"\bn\s*=\s*" + num + r"\b", text, re.I)
        or re.search(r"(?<![\d.,])" + num + r"(?![\d.,]?\d)" + _NOT_A_UNIT + r"\s+" + _BETWEEN
                     + nouns + r"\b", text, re.I)
    )


_WORD = re.compile(r"[^\W\d_]{4,}")
_ANY_WORD = re.compile(r"\w+n't|[^\W\d_]+")


def _stem(word: str) -> str:
    return word[:-1] if word.endswith("s") else word


def summary_supported(detail: str, evidence: str, share: float = 0.6) -> bool:
    """Is a model-written result summary a restatement of the paper's text?

    Numbers and names are checked separately. Here: every meaning word it uses
    ("prevented", "not", "increased") must be in the evidence, and most of its
    other content words too. A summary with no numbers and no names —
    "prophylactic dosing prevented disease onset" — would otherwise pass
    beside a quote that says nothing of the kind.

    And the reverse: a negation in the evidence that governs what the summary
    says must be in the summary. "did not significantly reduce" restated as
    "did significantly reduce" uses only the paper's words and reports the
    opposite result. Read sentence by sentence, so a negation about something
    else ("did not affect body weight, but reduced lesion area") does not
    blank a faithful summary. A bounded cue list, not entailment: see
    `litreview/negation.py`.
    """
    listed = meaning_words()["meaning"]
    # Meaning words of any length ("not", "no", "all") and contractions
    # ("didn't") must be in the evidence; content words are the long ones.
    every = set(_ANY_WORD.findall(_norm(evidence)))
    if any((w in listed or w.endswith("n't")) and w not in every for w in _ANY_WORD.findall(_norm(detail))):
        return False
    if any(drops_negation(detail, s) for s in sentences(evidence)):
        return False
    have = {_stem(w) for w in _WORD.findall(_norm(evidence))}
    words = [w for w in _WORD.findall(_norm(detail))]
    if not words:
        return True
    return sum(_stem(w) in have for w in words) / len(words) >= share


def text_supported(text: str, evidence: str, kb: Knowledge) -> bool:
    """May a line of model-written prose be shown beside this paper text?

    Only if it restates it: every number in it is in the evidence, every
    known entity it names is, every kind of subject it names is, and its
    words pass `summary_supported`. Used for a finding's result summary and
    for the ranking model's one-line reason.
    """
    text = (text or "").strip()
    if not text:
        return True
    if not numbers_in(text) <= numbers_in(evidence):
        return False
    if not all(_matcher((*e.terms, *e.context_terms, e.canonical)).exact_count(evidence) for e, _ in kb.scan(text)):
        return False
    said = studied_in(evidence)
    if not all(covered(kind, said) for kind in studied_in(text)):
        return False
    return summary_supported(text, evidence)


@dataclass
class Provenance:
    where: dict[str, str] = field(default_factory=dict)
    context: Located | None = None
    context_section: str | None = None
    naming: Located | None = None
    """The passage's sentence that names a target or agent the quote, the
    context sentence and the qualifier do not. In `naming_section`."""
    naming_section: str | None = None
    values: dict[str, object] = field(default_factory=dict)
    """The field values to store: the model's where located, blanked where not."""
    drop: str | None = None


def _quote_names(quote: str, hidden: dict[str, tuple[str, str]], kb: Knowledge) -> list[str]:
    """Every known target or drug the quote names, other than the hidden
    names. A sentence naming one of these beside a hidden name is one that
    says how the two are linked."""
    return [m.text for _, m in kb.scan(quote, ("target", "drug"))
            if not any(name_in(value, m.text, kb, kind) for value, kind in hidden.values())]


def naming_sentence(
    hidden: dict[str, tuple[str, str]],
    *,
    partners: list[str],
    quote_names: list[str],
    quote_start: int | None,
    chunk: str,
    methods: str,
    section: str,
    section_text: str,
    methods_section_text: str,
    kb: Knowledge,
) -> tuple[Located, str, str] | None:
    """(sentence, its section, CHUNK or METHODS) that names the `hidden`
    names ({field: (name, kind)}), or None when no sentence can be placed.

    One sentence is stored, so the best one is picked, in this order: it
    names the target, if the target is hidden; it names the most hidden
    names; it also names one of `partners`, the finding's own names that the
    quote does state ("HDAC6 inhibitors, such as tubastatin A" for a quote
    about tubastatin A), or failing that another target or drug the quote
    names; it is in the passage the quote came from rather than the
    Methods; it is nearest the quote. Every candidate is placed in its
    section like the context sentence is, and one that cannot be placed is
    passed over.
    """
    best: tuple[tuple, Located, str, str] | None = None
    for where, text, home, home_section in ((CHUNK, chunk, section_text, section),
                                            (METHODS, methods, methods_section_text, "methods")):
        for sent in sentences(text):
            named = [key for key, (value, kind) in hidden.items() if name_in(value, sent, kb, kind)]
            if not named:
                continue
            located = locate_quote(sent, home)
            if located.status == "failed":
                continue
            near = (abs(located.start - quote_start) if where == CHUNK and quote_start is not None
                    else len(home))
            link = (2 if any(name_in(n, sent, kb) for n in partners)
                    else 1 if any(name_in(n, sent, kb) for n in quote_names) else 0)
            score = ("target" in named, len(named), link, where == CHUNK, -near)
            if best is None or score > best[0]:
                best = (score, located, home_section, where)
    return None if best is None else best[1:]


def trace(
    *,
    quote: str,
    chunk: str,
    methods: str,
    section: str,
    section_text: str,
    methods_section_text: str,
    target: str,
    intervention: str | None,
    combo_with: str | None,
    disease_raw: str,
    species_raw: str,
    n: int | None,
    effect_detail: str,
    kb: Knowledge,
    qualifier: str = "",
    quote_start: int | None = None,
) -> Provenance:
    """Locate every model-written field. See the module docstring for the rules.

    `qualifier` is the paper's next sentence when it qualifies the quote, and
    `quote_start` the quote's offset in `section_text`: both only decide
    where a name found outside the quote is shown."""
    p = Provenance()

    def where_name(value: str | None, kind: str | None) -> str:
        if not value or not value.strip():
            return NONE
        if name_in(value, quote, kb, kind):
            return QUOTE
        if name_in(value, chunk, kb, kind):
            return CHUNK
        if name_in(value, methods, kb, kind):
            return METHODS
        return NONE

    p.where["target"] = where_name(target, "target")
    if p.where["target"] == NONE:
        p.drop = "target_unlocated"
        return p
    for name, value, kind in (("intervention", intervention, "drug"), ("combo_with", combo_with, "target"),
                              ("disease_raw", disease_raw, None)):
        if value and value.strip():
            p.where[name] = where_name(value, kind)

    # Who was studied and how many: the quote, else one context sentence.
    species = (species_raw or "").strip()
    need_species = bool(species) and not species_in(species, quote)
    need_n = n is not None and not n_in(n, quote)
    if need_species or need_n:
        best: tuple[int, str, str] | None = None
        for where, text in ((METHODS, methods), (CHUNK, chunk)):
            for sent in sentences(text):
                score = (2 if need_species and species_in(species, sent) else 0) + \
                        (1 if need_n and n_in(n, sent) else 0)  # type: ignore[arg-type]
                if score and (best is None or score > best[0]):
                    best = (score, where, sent)
            if best and best[0] == (2 if need_species else 0) + (1 if need_n else 0):
                break
        if best is not None:
            _, where, sent = best
            home = methods_section_text if where == METHODS else section_text
            located = locate_quote(sent, home)
            if located.status != "failed":
                p.context, p.context_section = located, ("methods" if where == METHODS else section)
                p.where["context"] = where
    context = p.context.text if p.context else ""

    # A name found only in the passage or the Methods must be shown: in the
    # context sentence, the qualifier, or a naming sentence stored for it.
    values = {"target": target, "intervention": intervention, "combo_with": combo_with}
    hidden: dict[str, tuple[str, str]] = {}
    for name, kind in _SHOWN_NAMES:
        if p.where.get(name) not in (CHUNK, METHODS):
            continue
        value = values[name] or ""
        if context and name_in(value, context, kb, kind):
            p.where[name] = CONTEXT
        elif qualifier and name_in(value, qualifier, kb, kind):
            p.where[name] = QUALIFIER
        else:
            hidden[name] = (value, kind)
    if hidden:
        partners = [values[name] or "" for name, _ in _SHOWN_NAMES if p.where.get(name) == QUOTE]
        found = naming_sentence(
            hidden, partners=partners, quote_names=_quote_names(quote, hidden, kb), quote_start=quote_start,
            chunk=chunk, methods=methods, section=section, section_text=section_text,
            methods_section_text=methods_section_text, kb=kb,
        )
        if found is not None:
            p.naming, p.naming_section, p.where["naming"] = found
        for name, (value, kind) in hidden.items():
            if p.naming is not None and name_in(value, p.naming.text, kb, kind):
                p.where[name] = NAMING
            elif name == "target":
                p.drop = "target_not_shown"
                return p
            else:
                p.where[name] = NONE

    if species:
        if species_in(species, quote):
            p.where["species"] = QUOTE
        elif context and species_in(species, context):
            p.where["species"] = p.where["context"]
        else:
            p.where["species"] = NONE
    if n is not None:
        if n_in(n, quote):
            p.where["n"] = QUOTE
        elif context and n_in(n, context):
            p.where["n"] = p.where["context"]
        else:
            p.where["n"] = NONE

    detail = (effect_detail or "").strip()
    if detail:
        p.where["effect_detail"] = QUOTE if text_supported(detail, f"{quote} {context}", kb) else NONE

    def keep(name: str, value):
        return value if p.where.get(name, NONE) != NONE else None

    p.values = {
        "intervention": keep("intervention", intervention),
        "combo_with": keep("combo_with", combo_with),
        "disease_raw": keep("disease_raw", disease_raw) or "",
        "species_raw": keep("species", species) or "",
        "n": keep("n", n),
        "effect_detail": keep("effect_detail", detail) or "",
    }
    return p


# --- a conclusion's people, against the Methods ---------------------------------
#
# The model copies "patients" from a conclusion sentence, and the species check
# above finds "patients" in the quote: the field is the paper's word, so it is
# kept. But an abstract's conclusion can generalise from a cell line and mice
# to "patients with asthma" (the shape of asthma S3). The Methods say who was
# actually studied. So when a conclusion sentence (`extract/verify.py`
# `conclusion_reason`) names people, Stage 5 reads the paper's Methods - the
# Methods section, and a structured abstract's METHODS part - and if they name
# no human subjects but do name animals or cells, the finding is not human
# evidence, whatever the quote says (`stages/s5_extract.py` `_to_unit`), and
# the citation guard will not let a sentence resting on it say "patients"
# (`report/guard.py`, check 4).
#
# A paper with no Methods text at all - common, since only open-access papers
# come with full text - is left alone. An abstract-only paper says nothing
# about its Methods beyond what its abstract says, and guessing from nothing
# would drop real human findings.
#
# Built to leave a finding alone whenever the Methods might describe people:
# any word below counts, including some that also occur in animal work
# ("cells from donors", "individuals"). A false "people were studied" only
# means the conclusion keeps its word, which is where things stood before.
# S3's own paper is such a case: its Methods recruited patients for induced
# sputum, so its conclusion keeps "patients", though the mechanism it states
# was shown in cells and mice.
#
# Measured on the three stored qwen3:8b corpora (every sentence of every
# abstract, Results and Discussion): 164 conclusion sentences name people, 100
# of them in papers with Methods text, and 7 are marked - all in mouse or rat
# papers generalising to "individuals", "infants" or "patients". Of the 593
# stored findings, 7 are conclusions naming people and one is marked
# (PMID31616172, a mouse paper: "... a therapeutic potential for obese/
# diabetic postmenopausal patients").
# Not "cohort" or "subjects": `knowledge/meaning_words.yaml` keeps those out
# of the human class because mouse studies use them ("two cohorts of mice"),
# and a mouse paper was the one stored case this check exists for.

HUMAN_SUBJECTS = re.compile(
    r"\b(?:enrol(?:l)?(?:ed|ment)|recruit(?:ed|ment)|participants?|patients?|volunteers?|donors?|individuals|"
    r"children|women|men|infants|informed\s+consent|declaration\s+of\s+helsinki|"
    r"institutional\s+review\s+board|questionnaires?|case[- ]control|clinical\s+trials?|"
    r"healthy\s+(?:controls?|subjects)|"
    r"(?:human|patient)\s+(?:subjects?|samples?|biops(?:y|ies)|tissues?|specimens?|sera|serum|blood|sputum)|"
    r"biops(?:y|ies)\s+(?:from|of|were|was))\b",
    re.I,
)
# The non-human classes of `knowledge/meaning_words.yaml` `studied`, in the
# order one is picked to describe the finding: an animal before cells, since
# a paper with both put its claim to the test in the animal. How each is said.
_NON_HUMAN = {"mouse": "mice", "rat": "rats", "primate": "primates", "animal": "animals", "cells": "cells"}


@dataclass
class MethodsSubjects:
    """Who a paper's Methods studied, when they name no people."""

    label: str
    """What the Methods name, for the reader: "mice and cells"."""
    species_raw: str
    """The paper's own word for the first of them ("mice"), from `context`."""
    evidence_type: str
    """in_vivo for an animal, in_vitro for cells."""
    context: Located | None = None
    """The Methods sentence that names it, anchored in `context_section`."""
    context_section: str | None = None


def methods_subjects(quote: str, *, methods_section_text: str, abstract: str) -> MethodsSubjects | None:
    """Who the Methods studied, when `quote` names people and the Methods
    name no human subjects but do name animals or cells; else None.

    The caller decides the quote is a conclusion first. Read from the
    Methods section and from a structured abstract's Methods part; None
    when there is neither (see the block comment above)."""
    if not people_named(quote):
        return None
    parts = [("methods", methods_section_text, 0, len(methods_section_text))] if methods_section_text.strip() else []
    parts += [("abstract", abstract, lo, hi) for lo, hi in abstract_methods(abstract)]
    if not parts:
        return None
    text = "\n\n".join(home[lo:hi] for _, home, lo, hi in parts)
    if HUMAN_SUBJECTS.search(text):
        return None
    said = studied_in(text)
    classes = [c for c in _NON_HUMAN if c in said]
    if "animal" in classes and any(c in classes for c in ("mouse", "rat", "primate")):
        classes.remove("animal")  # "mice and animals" says mice twice
    if not classes:
        return None
    first = classes[0]
    names = [_NON_HUMAN[c] for c in classes]
    label = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    found = MethodsSubjects(label=label, species_raw="", evidence_type="in_vitro" if first == "cells" else "in_vivo")
    # The sentence that says so, shown to the reader as the context sentence,
    # and the paper's own word for the subjects from it: the first word of the
    # class's list in meaning_words.yaml that the sentence uses ("mice" before
    # "c57bl"), as the paper spells it.
    terms = meaning_words()["studied"][first]
    for section, home, lo, hi in parts:
        for sent in split_sentences(home[lo:hi]):
            if first not in studied_in(sent):
                continue
            located = locate_quote(sent, home)
            if located.status == "failed":
                continue
            for term in terms:
                m = re.search(r"\b" + re.escape(term) + r"\b", located.text, re.I)
                if m:
                    found.species_raw = m.group(0)
                    break
            found.context, found.context_section = located, section
            return found
    return found
