"""Collapse the same finding reported twice.

A paper states its headline result in the abstract and again in the results,
often in the discussion too. Extraction runs per section, so one finding
arrives three times. Left alone, a report claiming "3 studies support this"
would really mean "1 study, counted 3 times" — which is not a cosmetic problem,
it is the report being wrong about the weight of evidence.

We deduplicate *within* a paper only. Two different papers reporting the same
result is the entire point of a literature review, and must never be collapsed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from bedrock.litreview.report.guard import numbers_in

# Preference order when the same finding appears in several sections. Results
# wins because it is where the primary data lives; discussion loses because
# authors restate their findings more strongly than the data supports.
_SECTION_RANK = {"results": 0, "abstract": 1, "methods": 2, "discussion": 3, "other": 4}

# Distinct measurements kept per finding key, per paper. The MedGemma run
# extracted 8 findings from one chunk and the key alone collapsed them to 1.
MAX_PER_KEY = 3


def _key(unit) -> tuple:
    """What makes two findings 'the same finding'.

    Deliberately excludes effect_detail, n and the quote — those are the fields
    that vary between a terse abstract sentence and a detailed results
    sentence, and varying wording is exactly the duplicate we are trying to
    catch. Including them would defeat the purpose.
    """
    return (
        unit.target,
        unit.direction,
        unit.disease,
        unit.evidence_type,
        unit.outcome,
        unit.combo_with,
    )


def _richness(unit) -> tuple:
    """How much a candidate is worth keeping. Higher sorts first.

    An exact quote beats a fuzzy one, a stated sample size beats a missing one,
    and a longer effect_detail usually means real numbers rather than a vague
    restatement.
    """
    return (
        1 if getattr(unit, "quote_status", "") == "exact" else 0,
        1 if getattr(unit, "n", None) is not None else 0,
        len(getattr(unit, "effect_detail", "") or ""),
    )


# Numbers in a sentence that are not measurements: p-values, figure and table
# numbers, years, bracketed reference markers.
_NOT_MEASURED = re.compile(
    r"\b[pP]\s*[<=>≤≥]\s*\d[\d.]*|\b(?:Fig(?:ure)?s?|Tables?|Supp\w*|Ref\w*)\.?\s*[S]?\d+[A-Za-z]?"
    r"|\((?:19|20)\d{2}[a-z]?\)|\[\d+(?:[,\u2013-]\d+)*\]"
)


def _numbers(unit) -> set[str]:
    """The measurements a finding reports: from its effect detail, or — when
    provenance blanked that as unsupported — from its quote, the paper's own
    sentence, without its p-values, figure numbers, years and citations."""
    detail = getattr(unit, "effect_detail", "") or ""
    if detail:
        return numbers_in(detail)
    return numbers_in(_NOT_MEASURED.sub(" ", getattr(unit, "quote", "") or ""))


def dedupe_within_paper(units: Iterable, max_per_key: int = MAX_PER_KEY) -> list:
    """Keep the best version of each distinct finding in one paper.

    Sorting is by section rank first, then richness, so a detailed results-
    section finding beats a terse abstract one, and both beat the discussion's
    restatement of them.

    Within one key, a later candidate survives only if it reports a
    measurement the kept ones do not: "epidermal thickness fell 45%" and
    "IL-17A mRNA fell 60%" are two readouts of one experiment, and the report
    needs both numbers; an abstract's "reduced inflammation" beside them is a
    restatement. This cannot inflate the weight of evidence — targets are
    counted by independent papers (Stage 6), never by findings.
    """
    by_key: dict[tuple, list] = defaultdict(list)
    for unit in units:
        by_key[_key(unit)].append(unit)

    kept = []
    for candidates in by_key.values():
        candidates.sort(
            key=lambda u: (
                _SECTION_RANK.get(getattr(u, "section", "other"), 4),
                # negate richness so that, within a section, richer sorts first
                tuple(-x for x in _richness(u)),
            )
        )
        seen: set[str] = set()
        chosen = []
        for unit in candidates:
            numbers = _numbers(unit)
            if chosen and (not numbers or numbers <= seen):
                continue
            chosen.append(unit)
            seen |= numbers
            if len(chosen) >= max_per_key:
                break
        kept.extend(chosen)
    return kept


def dedupe_run(units: Iterable) -> list:
    """Deduplicate a whole run, paper by paper.

    Cross-paper duplicates are left alone on purpose: independent replication
    is signal, and collapsing it would hide the strongest thing a literature
    review can show.
    """
    by_paper: dict[str, list] = defaultdict(list)
    for unit in units:
        by_paper[unit.paper_id].append(unit)

    out = []
    for paper_units in by_paper.values():
        out.extend(dedupe_within_paper(paper_units))
    return out
