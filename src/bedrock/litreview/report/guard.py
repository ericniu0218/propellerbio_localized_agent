"""The citation guard — quote verification, applied to the report itself.

Quote verification stops a finding being invented. It does not stop the
*writing* model from inventing: given nine real quotes, a model can still
write a fluent sentence with a number none of them contain, or name a drug
from its training data. That is exactly the "outside information" failure
the client sees in general chatbots.

So every sentence the writer produces is checked, in code, against the
evidence it cites:

1. It must cite at least one evidence id, and every id must exist in the
   slice that section was given.
2. Every number in it must appear in the cited evidence: the paper's own
   text (the quote, and the context sentence that gave the subjects or sample
   size) or the paper's year.
3. Every drug, target, disease or pathogen it names must appear in the cited
   evidence — or be the thing the question is about.
4. Who was studied — patients, mice, rats, primates, cells — must be said by
   the cited text itself, not by a label: "in human tissues" citing a quote
   that says "across skin and spleen tissues" is dropped.
5. Which way a result points must match: a sentence may not claim an effect
   its cited quotes report only as null, drop a negation a quote states, or
   say "decreased" where every cited quote says only "increased". Word-level
   checks, not entailment: they catch the flips a probe found, not every one.

The cited evidence is the paper's text, never what the model wrote beside it.
`effect_detail`, `species_raw` and a sample size the paper does not state are
the model's words; a sentence resting on them has nothing a reader can check.
Names the model gave a finding (its target, its agent) count only where
`extract/provenance.py` found them in the text the model read.

A sentence that fails any check is dropped and the reason recorded. The
report shows how many were dropped; the benchmark reads the reasons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bedrock.litreview.extract.verify import covered, split_sentences, studied_in
from bedrock.litreview.knowledge import Knowledge, TermMatcher, clean, fold
from bedrock.litreview.models import EvidenceUnit, QuestionSpec
from bedrock.litreview.negation import drops_negation

_NUMBER = re.compile(r"(?<![A-Za-z0-9.\-/])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![A-Za-z0-9])")


def _norm_number(raw: str) -> str:
    try:
        return "%g" % float(raw.replace(",", ""))
    except ValueError:
        return raw


def numbers_in(text: str) -> set[str]:
    return {_norm_number(m.group(1)) for m in _NUMBER.finditer(clean(text))}


def _located(unit: EvidenceUnit, name: str) -> bool:
    """Did provenance find this field in the paper? A unit with no trace at
    all (written before tracing existed) is treated as not located, except
    for its target, which is what the unit is about."""
    if not unit.provenance:
        return name == "target"
    return unit.provenance.get(name, "none") != "none"


def source_text(unit: EvidenceUnit) -> str:
    """The paper's own words a sentence citing this unit may rest on."""
    return " | ".join(p for p in (unit.quote, unit.context_quote or "") if p)


def unit_text(unit: EvidenceUnit, extra: str = "") -> str:
    """Everything a sentence citing this unit may legitimately draw on: the
    paper's text, plus the names provenance located, plus `extra` (the
    paper's label, so its year is citable)."""
    parts = [unit.quote, unit.context_quote or ""]
    if _located(unit, "target"):
        parts += [unit.target, unit.target_raw]
    if _located(unit, "intervention"):
        parts.append(unit.intervention or "")
    if _located(unit, "combo_with"):
        parts.append(unit.combo_with or "")
    if _located(unit, "disease_raw"):
        parts += [unit.disease, unit.disease_raw]
    parts.append(extra)
    return " | ".join(p for p in parts if p)


@dataclass
class GuardResult:
    kept: bool
    reason: str | None = None


class CitationGuard:
    def __init__(
        self,
        spec: QuestionSpec,
        knowledge: Knowledge,
        units: dict[str, EvidenceUnit],
        *,
        paper_labels: dict[str, str] | None = None,
        vocabulary: list[str] | None = None,
    ) -> None:
        """`units`: short id -> unit, for the slice being written.
        `paper_labels`: paper id -> "Author Year Journal", so a year is citable.
        `vocabulary`: extra entity names seen anywhere in this run's evidence
        (targets, drugs) that the table may not know."""
        self.spec = spec
        self.kb = knowledge
        self.units = units
        self.paper_labels = paper_labels or {}
        self.allowed = {fold(t) for e in spec.entities for t in [e.canonical, e.text, *e.exact_terms]}
        self.question_numbers = numbers_in(spec.text)
        vocab = sorted({v.strip() for v in (vocabulary or []) if v and len(v.strip()) >= 3})
        self._vocab = [(v, TermMatcher([v])) for v in vocab if fold(v) not in self.allowed]
        # The question's own near-misses. `kb.scan` only finds entities the
        # table lists in their own right, so without this a sentence about
        # hMPV in an RSV report passes: the very thing the client asked us to
        # make impossible.
        self._near = [
            TermMatcher(e.exact_terms, e.near_miss_terms, e.near_miss_patterns)
            for e in spec.entities if e.near_miss_terms or e.near_miss_patterns
        ]

    def evidence_blob(self, cited: list[str]) -> str:
        return " || ".join(
            unit_text(self.units[c], self.paper_labels.get(self.units[c].paper_id, ""))
            for c in cited if c in self.units
        )

    def check(self, text: str, citations: list[str]) -> GuardResult:
        if not citations:
            return GuardResult(False, "no citation")
        unknown = [c for c in citations if c not in self.units]
        if unknown:
            return GuardResult(False, f"cites unknown evidence {unknown[0]}")
        blob = self.evidence_blob(citations)
        sentence = clean(text)

        # A near-miss of the question's subject is never an answer to it. The
        # gate excluded those papers and extraction dropped those findings;
        # a sentence naming one has no business in this report either.
        for matcher in self._near:
            for m in matcher.matches(sentence):
                if m.kind == "near_miss":
                    return GuardResult(False, f"names {m.text!r}, which is not what the question asks about")

        # Names first: mask them out before looking for numbers, so the "2" in
        # "type 2 diabetes" or the "17" in "IL 17" is not read as a statistic.
        masked = sentence
        for entity, match in self.kb.scan(sentence):
            masked = masked[: match.start] + " " * (match.end - match.start) + masked[match.end :]
            if fold(match.text) in self.allowed or fold(entity.canonical) in self.allowed:
                continue
            names = [*entity.terms, *entity.context_terms, entity.canonical]
            if not TermMatcher(names).exact_count(blob):
                return GuardResult(False, f"names {match.text!r}, which the cited evidence does not")
        for name, matcher in self._vocab:
            if matcher.exact_count(sentence) and not matcher.exact_count(blob):
                known = self.kb.lookup(name)
                if known and TermMatcher([*known.terms, *known.context_terms]).exact_count(blob):
                    continue
                return GuardResult(False, f"names {name!r}, which the cited evidence does not")

        evidence_numbers = numbers_in(blob) | self.question_numbers
        for number in sorted(numbers_in(masked)):
            if number not in evidence_numbers:
                return GuardResult(False, f"number {number} is not in the cited evidence")

        # A bound keeps its direction. "280.2 (95% UR: 268.8-290.6)" makes
        # 268.8 the LOWER end, and a sentence calling it the upper range is
        # wrong even though every number in it is in the evidence — which is
        # how the RSV report passed this guard with the bounds swapped.
        wrong = bound_misuse(sentence, clean(blob))
        if wrong:
            return GuardResult(False, f"{wrong} is a bound of the opposite end in the cited evidence")

        # Who was studied, said by the paper, not by a label or the writer.
        cited_text = clean(" | ".join(source_text(self.units[c]) for c in citations))
        said = studied_in(cited_text)
        # Read with names masked, so "human papillomavirus" is not a claim
        # about humans.
        for kind, word in studied_in(masked).items():
            if not covered(kind, said):
                return GuardResult(False, f"says {word!r} about who was studied, which the cited evidence does not")

        # Which way a result points. A sentence may say something did not work
        # only if a cited finding is a null result or its words say so: the
        # MedGemma run wrote "inhibition of HIF signaling did not improve
        # outcomes" citing only findings where it did.
        cited = [self.units[c] for c in citations]
        if _NULL_CLAIM.search(sentence) and not any(
            u.outcome in ("no_effect", "mixed") or _NULL_CLAIM.search(clean(u.quote)) for u in cited
        ):
            return GuardResult(False, "claims no effect, which the cited evidence does not")
        # And the reverse, found by an adversarial probe: "Blocking IL-17A
        # reduced epidermal thickness", citing only "Blocking IL-17A had no
        # significant effect on epidermal thickness", was kept - every name
        # and number in it is in the quote.
        if not _NULL_CLAIM.search(sentence) and _EFFECT.search(sentence) \
                and all(_null_only(clean(u.quote)) for u in cited):
            return GuardResult(False, "claims an effect the cited evidence reports as null")
        for u in cited:
            if any(drops_negation(sentence, s) for s in split_sentences(clean(u.quote))):
                return GuardResult(False, "drops a negation in the cited evidence")
        # Which way a measurement moved: "was decreased" citing only quotes
        # that say "was increased" is the opposite finding. Only when the
        # sentence has one direction and no word that reverses one ("prevented
        # the increase"), so a paraphrase is not dropped.
        said = _directions(sentence)
        if len(said) == 1 and not _REVERSAL.search(sentence):
            quoted = [_directions(clean(u.quote)) for u in cited]
            if all(q and not (q & said) for q in quoted):
                return GuardResult(False, f"says {next(iter(said))}, which the cited evidence does not")
        # "Improved outcomes" is the label talking, not the paper: it adds no
        # fact and was how the same run turned a mouse study into a patient result.
        if _LABEL_TALK.search(sentence) and not _LABEL_TALK.search(clean(blob)):
            return GuardResult(False, "restates a label, not the evidence")
        return GuardResult(True)


_NULL_CLAIM = re.compile(
    r"\b(did not|does not|do not|failed to|no significant|not significantly|no effect|no measurable|"
    r"had no|was not|were not|no difference|unchanged)\b", re.I)
_LABEL_TALK = re.compile(
    r"\b(?:improved|improve|improves|worsened|better|worse)\s+(?:patient\s+|clinical\s+)?outcomes?\b", re.I)
# A result a sentence states. Not "block" or "inhibit": those name what was
# done ("Blocking IL-17A ..."), not what happened.
_EFFECT = re.compile(
    r"\b(?:reduc|decreas|lower|attenuat|ameliorat|alleviat|diminish|abolish|abrogat|mitigat|improv|prevent|"
    r"protect|resolv|increas|elevat|worsen|aggravat|exacerbat|promot|accelerat|augment|amplif)\w*", re.I)
_NULL_SCOPE = re.compile(r"(?:" + _NULL_CLAIM.pattern + r")(?:\s+[\w-]+){0,4}", re.I)


def _null_only(text: str) -> bool:
    """Does this text report a null result, and no effect outside the reach
    of its negation? "had no significant effect on epidermal thickness" is;
    "reduced ear thickness but had no effect on scaling" is not."""
    return bool(_NULL_CLAIM.search(text)) and not _EFFECT.search(_NULL_SCOPE.sub(" ", text))


_UP = re.compile(r"\b(?:increas|elevat|upregulat|higher|raised|rose)\w*", re.I)
_DOWN = re.compile(r"\b(?:decreas|reduc|lower|downregulat|diminish|declin|fell)\w*", re.I)
_REVERSAL = re.compile(
    r"\b(?:prevent|block|inhibit|revers|abrogat|abolish|attenuat|blunt|counteract|normali[sz]|restor|suppress|"
    r"protect|rescu|than|versus|vs|compared)\w*", re.I)


def _directions(text: str) -> set[str]:
    """Which ways the text says something moved: {"increased"}, {"decreased"},
    both or neither."""
    return {name for name, pattern in (("increased", _UP), ("decreased", _DOWN)) if pattern.search(text)}


_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[\u2013\u2014-]\s*(\d+(?:\.\d+)?)")
_UPPER = re.compile(r"\b(upper|highest|maximum|at most|top end)\b", re.I)
_LOWER = re.compile(r"\b(lower|lowest|minimum|at least|bottom end)\b", re.I)


def bound_misuse(sentence: str, evidence: str) -> str | None:
    """The number a sentence presents at the wrong end of a range, if any.

    Only fires when the sentence says which end it means, and the evidence
    writes the number as one end of a range: "the upper range" beside a value
    the evidence gives as the low end. Everything else is left alone, because
    a wrong drop is as bad as a wrong keep.
    """
    says_upper, says_lower = bool(_UPPER.search(sentence)), bool(_LOWER.search(sentence))
    if says_upper == says_lower:
        return None
    lows, highs = set(), set()
    for low, high in _RANGE.findall(evidence):
        lows.add(_norm_number(low))
        highs.add(_norm_number(high))
    for number in numbers_in(sentence):
        if says_upper and number in lows - highs:
            return number
        if says_lower and number in highs - lows:
            return number
    return None
