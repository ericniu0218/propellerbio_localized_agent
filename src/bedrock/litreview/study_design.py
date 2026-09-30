"""What kind of study a paper reports, read from what the paper itself says. Code only.

The ranking's `evidence_hints` look for words like "patients" or "mice"
anywhere in an abstract. That is fine as a diagnostic and wrong as a
selection rule: a mouse paper that opens "psoriasis affects millions of
patients" is not human evidence. This module answers the narrower question
the selector needs - does the paper *itself* report a human trial, a human
cohort, a genetic association, an animal experiment, cell work? - and keeps
the evidence for every answer: which field it came from (publication type,
title, abstract), the character offsets of the words, and how direct that
evidence is:

  metadata       an indexer's publication type ("Randomized Controlled Trial")
  explicit_text  a design phrase the paper uses about its own work ("patients
                 were randomly assigned", "a prospective cohort of 1,204
                 adults", "C57BL/6 mice were treated")
  weak_hint      a word that often goes with a design but does not establish
                 it ("SNP", "mice" in passing). Recorded, never enough for a
                 reserved slot

Two separate classifications, because they answer different questions:

  labels   study design, several at once: human_rct, human_interventional_other,
           human_observational, human_genetic, animal_interventional, in_vitro,
           plus `mixed` when human and non-human work are both reported, or
           `unknown`. Mixed papers stay mixed here; each finding's own quote
           decides what it is later.
  roles    document type: systematic_review, meta_analysis, narrative_review,
           guideline, editorial_or_notice, primary, unknown. Several may hold
           (a systematic review with a meta-analysis). A primary paper that
           *mentions* an earlier review is still primary: only a review's own
           publication type or its own wording ("this systematic review")
           makes it one.

This never scores quality, and a label is not a finding: it decides which
papers are *looked at* first, and every finding still has to pass the quote,
field and entity checks on its own.
"""

from __future__ import annotations

import re
from typing import Any

from bedrock.litreview.models import Paper

VERSION = "study_design.v1"
GROUNDED = ("metadata", "explicit_text")
HUMAN = ("human_rct", "human_interventional_other", "human_observational", "human_genetic")
NONHUMAN = ("animal_interventional", "in_vitro")
NONPRIMARY_ROLES = ("systematic_review", "meta_analysis", "narrative_review", "guideline", "editorial_or_notice")

# --- publication types (metadata) ---------------------------------------------------------

_TYPE_LABELS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^randomi[sz]ed controlled trial$", re.I), "human_rct"),
    (re.compile(r"^(controlled )?clinical trial(, phase (i{1,3}v?|[1-4]))?$", re.I), "human_interventional_other"),
    (re.compile(r"^pragmatic clinical trial$", re.I), "human_interventional_other"),
    (re.compile(r"^observational study$", re.I), "human_observational"),
    (re.compile(r"^case reports?$|^case-report$", re.I), "human_observational"),
    (re.compile(r"^twin study$", re.I), "human_genetic"),
    (re.compile(r"^genome-wide association study$", re.I), "human_genetic"),
]
_TYPE_ROLES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(network )?meta-analysis$", re.I), "meta_analysis"),
    (re.compile(r"^systematic[ -]review$", re.I), "systematic_review"),
    (re.compile(r"^(review|review-article|scoping review)$", re.I), "narrative_review"),
    (re.compile(r"^(practice guideline|guideline|consensus statement|consensus development conference)$", re.I),
     "guideline"),
    (re.compile(r"^(editorial|comment|news|article-commentary|published erratum|correction|"
                r"retraction of publication|retraction notice|expression of concern|retracted publication)$", re.I),
     "editorial_or_notice"),
]
# Types that mean the paper reports its own study, whatever else it is filed as.
_PRIMARY_TYPES = re.compile(
    r"^(randomi[sz]ed controlled trial|(controlled )?clinical trial.*|observational study|case reports?|case-report|"
    r"comparative study|multicenter study|evaluation study|validation study|twin study|research-article|"
    r"journal article|brief-report)$", re.I)

# --- the paper's own words (explicit_text) ---------------------------------------------------

_HUMANS = r"(patients|participants|subjects|volunteers|adults|children|infants|adolescents|women|men|individuals|people|neonates|pregnant)"
_ANIMALS = (r"(mice|mouse|murine|rats?|zebrafish|macaques?|cynomolgus|monkeys|pigs|piglets|ferrets|cotton rats|"
            r"rabbits|dogs|hamsters|guinea pigs|sheep|calves|non-human primates)")
_NOT_ANIMAL = re.compile(rf"\b{_ANIMALS}\b", re.I)

_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
# A sentence about other people's work: "previous randomized trials showed",
# "has been reported in cohort studies". Its design words describe them, not
# this paper, so an abstract sentence like this establishes nothing.
_BACKGROUND = re.compile(
    r"\b(?:previous(?:ly)?|prior|earlier|other|recent(?:ly)?|existing|published|literature|"
    r"ha(?:ve|s)\s+(?:been\s+)?(?:shown|reported|demonstrated|suggested|linked|associated))\b", re.I)

# (label, pattern, needs a human noun in the same sentence, forbids an animal noun there)
_TEXT_LABELS: list[tuple[str, re.Pattern[str], bool, bool]] = [
    ("human_rct", re.compile(
        r"\brandomi[sz]ed,?\s+(?:double[- ]blind(?:ed)?,?\s+|placebo[- ]controlled,?\s+|multicent(?:er|re),?\s+|"
        r"open[- ]label,?\s+|controlled,?\s+|phase\s+(?:[1-4]|i{1,3}v?)[ab]?,?\s+)*(?:clinical\s+)?trial\b", re.I),
     False, True),
    ("human_rct", re.compile(
        rf"\b{_HUMANS}\b[^.!?]{{0,80}}\b(?:were|was)\s+(?:randomi[sz]ed|randomly\s+(?:assigned|allocated))\b", re.I),
     True, True),
    ("human_interventional_other", re.compile(
        r"\b(?:phase\s+(?:[1-4]|i{1,3}v?)[ab]?|open[- ]label|single[- ]arm|dose[- ]escalation|first[- ]in[- ]human)"
        r"\s+(?:clinical\s+)?(?:study|trial)\b", re.I), False, True),
    ("human_observational", re.compile(
        r"\b(?:(?:prospective|retrospective|population[- ]based|nationwide|longitudinal|birth)\s+"
        r"(?:cohort|registry|study)|case[- ]control\s+study|cross[- ]sectional\s+(?:study|analysis|survey)|"
        r"cohort\s+of\s+[\d,]+\s+" + _HUMANS + r")\b", re.I), True, True),
    ("human_genetic", re.compile(
        r"\b(?:genome[- ]wide\s+association|GWAS|mendelian\s+randomi[sz]ation|exome[- ]wide|"
        r"whole[- ]exome\s+sequencing\s+of|loss[- ]of[- ]function\s+(?:variants?|carriers)\s+in\s+"
        + _HUMANS + r"|polygenic\s+risk\s+scores?)\b", re.I), False, True),
    ("animal_interventional", re.compile(
        rf"\b{_ANIMALS}\b[^.!?]{{0,120}}\b(?:were|was)\s+(?:treated|immuni[sz]ed|infected|challenged|fed|dosed|"
        r"injected|given|administered|vaccinated)\b|\b(?:knock-?out|deficient|transgenic|-/-)\s+" + _ANIMALS + r"\b|"
        + r"\b" + _ANIMALS + r"\s+(?:lacking|deficient\s+in|with\s+(?:a\s+)?(?:targeted\s+)?deletion)\b|"
        r"\b(?:mouse|murine|rat)\s+models?\s+of\b", re.I), False, False),
    ("in_vitro", re.compile(
        r"\b(?:in\s+vitro|cell\s+lines?|cultured\s+(?:human\s+)?(?:cells|keratinocytes|fibroblasts|macrophages|"
        r"epithelial\s+cells|t\s+cells)|organoids?|primary\s+(?:human\s+)?cells)\b", re.I), False, False),
]
_WEAK: list[tuple[str, re.Pattern[str]]] = [
    ("human_genetic", re.compile(r"\b(?:SNPs?|polymorphisms?|risk\s+alleles?)\b", re.I)),
    ("animal_interventional", re.compile(rf"\b{_ANIMALS}\b", re.I)),
    ("human_observational", re.compile(rf"\b{_HUMANS}\b", re.I)),
]
_SELF_ROLE = re.compile(
    r"\b(?:this|the\s+present|our|we\s+(?:conducted|performed|undertook|carried\s+out|report)\s+(?:a|an))\s+"
    r"(systematic\s+review|meta[- ]analysis|narrative\s+review|scoping\s+review|review)\b", re.I)
_TITLE_ROLE = re.compile(r"(?:^|[:\-–—]\s*|\b(?:a|an)\s+)(systematic\s+review|meta[- ]analysis|scoping\s+review|"
                         r"narrative\s+review|review)(?:\s+and\s+(meta[- ]analysis))?\b", re.I)
_ROLE_WORDS = {"systematic review": "systematic_review", "meta-analysis": "meta_analysis",
               "meta analysis": "meta_analysis", "narrative review": "narrative_review",
               "scoping review": "narrative_review", "review": "narrative_review"}


def _signal(label: str, certainty: str, field: str, start: int | None, end: int | None, text: str) -> dict[str, Any]:
    return {"label": label, "certainty": certainty, "field": field, "start": start, "end": end, "text": text}


def _sentences(text: str):
    for m in _SENTENCE.finditer(text):
        yield m.start(), m.group(0)


def design_signals(paper: Paper) -> dict[str, Any]:
    """Every design and role signal in the paper's publication types, title
    and abstract, with where it was found. Deterministic: the same paper
    always gives the same answer, in the same order."""
    signals: list[dict[str, Any]] = []
    roles: list[dict[str, Any]] = []
    types = [t.strip() for t in paper.signals.publication_types if t and t.strip()]
    for t in types:
        for pattern, label in _TYPE_LABELS:
            if pattern.match(t):
                signals.append(_signal(label, "metadata", "publication_types", None, None, t))
        for pattern, role in _TYPE_ROLES:
            if pattern.match(t):
                roles.append(_signal(role, "metadata", "publication_types", None, None, t))

    for field, text in (("title", paper.title or ""), ("abstract", paper.abstract or "")):
        for offset, sentence in _sentences(text):
            background = field == "abstract" and bool(_BACKGROUND.search(sentence))
            for label, pattern, needs_human, no_animal in _TEXT_LABELS:
                if background:
                    break
                for m in pattern.finditer(sentence):
                    if needs_human and not re.search(rf"\b{_HUMANS}\b", sentence, re.I):
                        continue
                    if no_animal and _NOT_ANIMAL.search(sentence):
                        continue
                    signals.append(_signal(label, "explicit_text", field, offset + m.start(), offset + m.end(),
                                           m.group(0)))
            for label, pattern in _WEAK:
                for m in pattern.finditer(sentence):
                    signals.append(_signal(label, "weak_hint", field, offset + m.start(), offset + m.end(), m.group(0)))
        # A review's own wording only - never a mention of someone else's review.
        for m in (_TITLE_ROLE if field == "title" else _SELF_ROLE).finditer(text):
            for group in (m.group(1), m.group(2) if m.lastindex and m.lastindex >= 2 else None):
                if group:
                    role = _ROLE_WORDS.get(re.sub(r"[\s-]+", " ", group.lower()).replace("meta analysis", "meta-analysis"),
                                           "narrative_review")
                    roles.append(_signal(role, "explicit_text", field, m.start(), m.end(), m.group(0)))

    grounded = sorted({s["label"] for s in signals if s["certainty"] in GROUNDED})
    labels = list(grounded)
    if any(x in grounded for x in HUMAN) and any(x in grounded for x in NONHUMAN):
        labels.append("mixed")
    if not labels:
        labels = ["unknown"]

    role_names = sorted({r["label"] for r in roles})
    # A systematic review may carry "Review" as well; the narrower role wins
    # over the generic one, and both are kept in `role_signals`.
    if {"systematic_review", "meta_analysis"} & set(role_names):
        role_names = [r for r in role_names if r != "narrative_review"]
    design_typed = any(s["certainty"] == "metadata" for s in signals)
    if role_names == ["editorial_or_notice"] and design_typed:
        # A comment filed together with a trial or case-report type is a
        # short report of a study, not a notice about one.
        role_names = ["primary"]
    if not role_names:
        own_design = any(x in HUMAN + NONHUMAN for x in grounded)
        role_names = ["primary"] if own_design or any(_PRIMARY_TYPES.match(t) for t in types) else ["unknown"]
    return {
        "version": VERSION,
        "labels": labels,
        "roles": role_names,
        "signals": signals,
        "role_signals": roles,
    }


def grounded_labels(record: dict[str, Any] | None) -> set[str]:
    """Design labels backed by metadata or the paper's own design wording."""
    if not record:
        return set()
    return {s["label"] for s in record.get("signals", []) if s["certainty"] in GROUNDED}


def is_nonprimary(record: dict[str, Any] | None) -> bool:
    return bool(record) and any(r in NONPRIMARY_ROLES for r in record.get("roles", []))


def designs_for(papers: list[Paper]) -> dict[str, dict[str, Any]]:
    return {p.id: design_signals(p) for p in papers}


_DESIGN_ORDER = ("human_rct", "human_interventional_other", "human_genetic", "human_observational")


def design_rank(record: dict[str, Any] | None, paper: Paper | None = None) -> int:
    """Which of two equally relevant papers to read first, by what the paper
    itself says it is: a randomised trial, another trial, human genetics, an
    observational study, then everything unstated, a case report, an animal
    study, cells. A reading order, not a quality score - relevance (the
    grade) always decides first, and every finding is still checked on its own.

    Added after a tirzepatide / liver-fat review graded the randomised trial
    that measured liver fat by MRI (SURPASS-3 MRI) 3/3 and ranked it 21st,
    behind mouse studies of the same grade, so the 16-paper plan never read it.
    Used for efficacy, safety and mechanism questions; target identification
    keeps its lexical order (measured: no gain in reference-target recall on
    the frozen corpora, and a loss on asthma)."""
    title = (paper.title if paper is not None else "") or ""
    types = " ".join(paper.signals.publication_types).lower() if paper is not None else ""
    if "case report" in types or re.search(r"\bcase reports?\b|\ba (?:rare )?case of\b", title, re.I):
        return 5
    labels = grounded_labels(record)
    for i, label in enumerate(_DESIGN_ORDER):
        if label in labels:
            return i
    if "animal_interventional" in labels:
        return 6
    if "in_vitro" in labels:
        return 7
    return 4
