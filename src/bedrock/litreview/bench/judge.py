"""The local judge: entailment, nugget extraction, nugget coverage.

Every call is structured, temperature 0 and cached, like every other model
call in the system. The judge's own reliability is measured separately
(`perturb.py`), because a judge that agrees with everything is worse than no
judge: it turns "unchecked" into "checked".

The judge is a model and code together, split the way the pipeline is. The
model finds: which sentence of the evidence states the claim, which details
it cannot find. Code decides: the sentence must be in the evidence; every
name and number in the claim must be in the evidence; the claim and the
sentence must not say the effect went opposite ways. A small model cannot be
trusted with the verdict itself — MedGemma 4B was too strict when asked for a
verdict and passed nearly everything when asked "is the main point stated?".
"""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from bedrock.litreview.bench.models import (
    CoverageAnswers,
    CoverageVerdict,
    Nugget,
    NuggetList,
    SupportAnswers,
    SupportVerdict,
)
from bedrock.litreview.extract.verify import normalize as normalize_text
from bedrock.litreview.extract.verify import verify_quote
from bedrock.litreview.knowledge import Knowledge, TermMatcher, clean, load_knowledge
from bedrock.litreview.lexical import BM25
from bedrock.litreview.llm import LLMError, ask_json, load_prompt, strict_schema
from bedrock.litreview.models import Paper, stable_id
from bedrock.litreview.negation import (
    _BETTER,
    _NULL,
    _STOP,
    _WORSE,
    SAME_TOPIC,
    _stem_set,
    _stems,
    drops_negation,
)
from bedrock.litreview.report.guard import numbers_in

log = logging.getLogger(__name__)

EVIDENCE_CHARS = 3500
ANSWER_CHARS = 9000

# Something a scientist could act on: a gene or protein name with a number in
# it (IL-23, TYK2, IL17A, CD8), a few names without one (TNF, IFN), a drug by
# its stem (-mab, -nib, -cept), or a word for an agent, cell type or pathway.
_GENE = re.compile(r"\b[A-Z][A-Za-z]*-?\d+[A-Za-z0-9]*\b|\b(?:TNF|IFN|NF-κB|STAT|JAK|RIPK|HLA)")
_ACTION_WORDS = re.compile(
    r"\w+(?:mab|nib|cept)\b|vaccin|antibod|inhibit|blockade|antagonist|agonist|"
    r"\b(?:T|B)[\s-]cells?\b|\bTh17\b|\bTregs?\b|keratinocyte|macrophage|dendritic|neutrophil|mast cell|"
    r"pathway|signal(?:l)?ing|receptor|cytokine|interleukin|kinase|protein|prefusion", re.I)


def actionable(fact: str) -> bool:
    return bool(_GENE.search(fact) or _ACTION_WORDS.search(fact))


class CorpusIndex:
    """Passages of the collected papers — abstracts and results paragraphs —
    for finding where in the corpus a claim could have come from."""

    def __init__(self, papers: list[Paper], passage_chars: int = 900) -> None:
        self.passages: dict[str, str] = {}
        for p in papers:
            if p.abstract:
                self.passages[f"{p.id}#abstract"] = p.abstract
            for name in ("results", "discussion"):
                text = p.sections.get(name, "")
                buf, i = "", 0
                for para in re.split(r"\n\s*\n", text):
                    if buf and len(buf) + len(para) > passage_chars:
                        self.passages[f"{p.id}#{name}{i}"] = buf
                        buf, i = "", i + 1
                    buf = f"{buf}\n{para}".strip()
                if buf:
                    self.passages[f"{p.id}#{name}{i}"] = buf
        self._bm25 = BM25(self.passages) if self.passages else None

    def top(self, query: str, k: int = 4) -> list[tuple[str, str]]:
        if self._bm25 is None:
            return []
        return [(pid, self.passages[pid]) for pid, score in self._bm25.rank(query)[:k] if score > 0]


def _numbered(items: list[str]) -> str:
    out, total = [], 0
    for i, item in enumerate(items, 1):
        item = item.strip()
        if total + len(item) > EVIDENCE_CHARS:
            item = item[: max(200, EVIDENCE_CHARS - total)]
        out.append(f'[{i}] "{item}"')
        total += len(item)
        if total >= EVIDENCE_CHARS:
            break
    return "\n".join(out)


VERBATIM_MIN_CHARS = 40


def verbatim_in(claim: str, evidence: list[str]) -> bool:
    """True when the claim, as a whole, is a passage of one evidence item —
    after the same normalisation the quote check uses. A copied sentence is
    supported by definition, and asking a small judge about it only adds
    noise: MedGemma 4B called copied sentences 'unsupported' while its own
    reasoning said the evidence stated them."""
    text = normalize_text(claim).rstrip(" .")
    return len(text) >= VERBATIM_MIN_CHARS and any(text in normalize_text(e) for e in evidence)


_DRUG = re.compile(r"\b\w+(?:mab|nib|cept)\b", re.I)


def claim_names(text: str, kb: Knowledge) -> list[tuple[str, list[str]]]:
    """The names a claim stands on — known diseases, pathogens, drugs and
    targets, plus anything shaped like a gene or a drug — each with the terms
    that count as it."""
    text = clean(text)
    names: list[tuple[str, list[str]]] = []
    spans: list[tuple[int, int]] = []
    for entity, match in kb.scan(text):
        names.append((match.text, [*entity.terms, *entity.context_terms, entity.canonical]))
        spans.append((match.start, match.end))
    for pattern in (_GENE, _DRUG):
        for m in pattern.finditer(text):
            if any(a <= m.start() < b for a, b in spans):
                continue
            names.append((m.group(0), [m.group(0)]))
            spans.append((m.start(), m.end()))
    seen, out = set(), []
    for label, terms in names:
        if label.lower() not in seen:
            seen.add(label.lower())
            out.append((label, terms))
    return out


MIN_COVERAGE = 0.5


def word_coverage(claim: str, evidence: str) -> float:
    """Share of the claim's content words that appear anywhere in the evidence."""
    wanted = _stem_set(claim)
    return len(wanted & _stem_set(evidence)) / len(wanted) if wanted else 1.0


def opposite(claim: str, sentence: str, evidence: str = "") -> bool:
    """Do the claim and its evidence sentence say the effect went opposite ways?

    Deliberately narrow, because a false "contradicted" is as wrong as a
    false "supported":

    - a claim of no effect contradicts only if no null result is stated
      anywhere in the evidence (the judge may have copied the wrong sentence)
    - a sentence reporting no effect contradicts a claim that uses, without
      the negation, the very effect word the sentence negates
    - otherwise, effect words pointing opposite ways count only when the two
      are about the same thing: most of the claim's other words are in the
      sentence. "Enhanced proliferation" beside "reduced fluorescence" is two
      measurements, not a contradiction.
    """
    claim_null, sentence_null = _NULL.search(claim), _NULL.search(sentence)
    if claim_null and not sentence_null:
        return not _NULL.search(evidence or sentence)
    # "did not receive the maternal vaccine" -> "did receive": the shared
    # rule the result-summary check uses too (`litreview/negation.py`).
    if drops_negation(claim, sentence):
        return True
    bc, bs = _stems(_BETTER, claim), _stems(_BETTER, sentence)
    wc, ws = _stems(_WORSE, claim), _stems(_WORSE, sentence)
    if not ((wc - ws) and (bs - bc)) and not ((bc - bs) and (ws - wc)):
        return False
    effect_words = frozenset(w[:5] for w in wc | ws | bc | bs) | _STOP
    rest = _stem_set(claim, effect_words)
    return bool(rest) and len(rest & _stem_set(sentence, effect_words)) / len(rest) >= SAME_TOPIC


LOCATE_OVERLAP = 0.6
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(normalize_text(text)) if len(w) > 2}


def best_sentence(claim: str, evidence: list[str]) -> tuple[str, float]:
    """The evidence sentence sharing the most of the claim's words, and the
    share. Used when the judge echoes the claim back instead of copying from
    the evidence — MedGemma 4B does this — so a claim the evidence states
    word for word is not lost to the judge's copying."""
    wanted = _words(claim)
    best, score = "", 0.0
    for text in evidence:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            if len(sentence) < 25 or not wanted:
                continue
            share = len(wanted & _words(sentence)) / len(wanted)
            if share > score:
                best, score = sentence.strip(), share
    return best, score


def verdict_from(
    answers: SupportAnswers, claim: str, evidence: list[str], kb: Knowledge | None = None
) -> tuple[SupportVerdict, int, bool]:
    """The verdict, decided in code from the judge's answers.

      unsupported          the judge found no sentence of the evidence that
                           states the claim, or the one it copied is not there,
                           or most of the claim's words, or all of its names,
                           are not in the evidence
      contradicted         the sentence says the effect went the other way
      partially_supported  a name, a number or a detail of the claim is not
                           in the evidence
      supported            otherwise

    A detail the judge calls missing but that is in the evidence is struck
    off, and counted: it measures the judge being wrong. When the judge's
    sentence is not in the evidence, code looks for the sentence sharing most
    of the claim's words (`best_sentence`); the third value says it did.
    """
    kb = kb or load_knowledge()
    joined = "\n".join(evidence)
    ev, cl = normalize_text(joined), normalize_text(claim)
    sentence = answers.evidence_sentence.strip()
    located = bool(sentence) and verify_quote(sentence, joined) != "failed"
    by_code = False
    if sentence and not located:
        # The judge tried to copy a sentence and mangled it — a small model
        # often echoes the claim back instead. Find the sentence it meant.
        # An *empty* answer is not a mistake: it means the judge looked and
        # found nothing, and code must not overrule that, or a judge can
        # never reject anything.
        found, share = best_sentence(claim, evidence)
        if share >= LOCATE_OVERLAP:
            sentence, located, by_code = found, True, True

    names = claim_names(claim, kb)
    missing_names = [label for label, terms in names if not TermMatcher(terms).exact_count(joined)]
    missing_numbers = sorted(numbers_in(claim) - numbers_in(joined))
    missing_details, struck = [], 0
    for item in answers.missing:
        text = normalize_text(item).strip(" .,;:()")
        if len(text) < 2:
            continue
        if text in ev:
            struck += 1
        elif text in cl:
            missing_details.append(item.strip())
    missing = list(dict.fromkeys([*missing_names, *missing_numbers, *missing_details]))

    coverage = word_coverage(claim, joined)
    if not located:
        label, why = "unsupported", "no sentence of the evidence states it"
    elif coverage < MIN_COVERAGE:
        label, why = "unsupported", f"most of what it says is not in the evidence ({coverage:.0%} of its words)"
    elif opposite(claim, sentence, joined):
        label, why = "contradicted", "the evidence sentence says the effect went the other way"
    elif names and len(missing_names) == len(names):
        label, why = "unsupported", "none of its names is in the evidence"
    elif missing:
        label, why = "partially_supported", "not in the evidence: " + "; ".join(missing)
    else:
        label, why = "supported", "stated by the evidence"
    located_by = " (sentence found by code)" if by_code else ""
    return SupportVerdict(
        reasoning=f"{why}{located_by}. Judge: {answers.reasoning}", verdict=label,
        unsupported_part="; ".join(missing) or None,
    ), struck, by_code


def coverage_from(answers: CoverageAnswers, fact: str, answer: str, kb: Knowledge | None = None) -> CoverageVerdict:
    """Coverage of a reference fact, decided in code: the answer's sentence
    must be in the answer, must not say the effect went the other way, and
    must name what the fact names."""
    kb = kb or load_knowledge()
    sentence = answers.answer_sentence.strip()
    if not sentence:
        return CoverageVerdict(reasoning="the answer has no sentence stating it", verdict="not_covered")
    if verify_quote(sentence, answer) == "failed":
        found, share = best_sentence(fact, [answer])
        if share < LOCATE_OVERLAP:
            return CoverageVerdict(reasoning="the answer has no sentence stating it", verdict="not_covered")
        sentence = found
    if opposite(fact, sentence, answer):
        return CoverageVerdict(reasoning="the answer says the effect went the other way", verdict="not_covered")
    names = claim_names(fact, kb)
    found = [label for label, terms in names if TermMatcher(terms).exact_count(sentence)]
    if names and not found:
        return CoverageVerdict(reasoning="the answer's sentence names none of it", verdict="not_covered")
    if len(found) < len(names):
        return CoverageVerdict(reasoning="the answer's sentence names part of it", verdict="partial")
    return CoverageVerdict(reasoning=f"stated: {sentence[:120]}", verdict="covered")


class Judge:
    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model
        self.calls = 0
        self.answered = 0
        """Calls whose answer came back and parsed: the verdicts the judge supplied."""
        self.verbatim = 0
        self.settled_in_code = 0
        """Verdicts no judge supplied: a claim verbatim in its evidence, or with no evidence."""
        self.struck = 0
        self.located_by_code = 0
        self.kb = load_knowledge()

    def support(self, claim: str, evidence: list[str]) -> SupportVerdict | None:
        """Is the claim supported by the evidence? Settled in code when the claim
        is a verbatim passage of the evidence — for every system alike — and by
        the judge model otherwise."""
        if not evidence:
            self.settled_in_code += 1
            return SupportVerdict(reasoning="no evidence", verdict="unsupported")
        if verbatim_in(claim, evidence):
            self.verbatim += 1
            self.settled_in_code += 1
            return SupportVerdict(reasoning="verbatim passage of the evidence (checked in code)", verdict="supported")
        prompt = load_prompt("judge_support").render(claim=claim, evidence=_numbered(evidence))
        try:
            self.calls += 1
            data = ask_json(self.client, prompt, model=self.model,
                            schema=strict_schema(SupportAnswers.model_json_schema()), max_tokens=500)
            answers = SupportAnswers.model_validate(data)
        except (LLMError, ValidationError) as exc:
            log.warning("judge failed on claim %r: %s", claim[:60], exc)
            return None
        self.answered += 1
        verdict, struck, by_code = verdict_from(answers, claim, evidence, self.kb)
        self.struck += struck
        self.located_by_code += int(by_code)
        return verdict

    def nuggets(self, question: str, subject: str, label: str, text: str, max_nuggets: int = 6) -> list[Nugget]:
        prompt = load_prompt("nuggets").render(
            question=question, subject=subject, reference_label=label, reference_text=text[:6000],
            max_nuggets=max_nuggets,
        )
        try:
            self.calls += 1
            data = ask_json(self.client, prompt, model=self.model,
                            schema=strict_schema(NuggetList.model_json_schema()), max_tokens=1200)
            found = NuggetList.model_validate(data).nuggets
        except (LLMError, ValidationError) as exc:
            log.warning("nugget extraction failed for %s: %s", label, exc)
            return []
        self.answered += 1
        out = []
        for n in found[:max_nuggets]:
            # The reference cannot be hallucinated either: a nugget whose
            # sentence is not in the reference is discarded.
            if verify_quote(n.quote, text) == "failed":
                continue
            if not actionable(n.fact):
                # "Psoriasis affects 2-3% of people" is true and answers nothing.
                continue
            out.append(Nugget(id=stable_id(label, n.fact, length=10), fact=n.fact, quote=n.quote,
                              importance=n.importance, source=label))
        return out

    def covered(self, fact: str, answer: str) -> CoverageVerdict | None:
        answer = answer[:ANSWER_CHARS]
        prompt = load_prompt("judge_nugget").render(fact=fact, answer=answer)
        try:
            self.calls += 1
            data = ask_json(self.client, prompt, model=self.model,
                            schema=strict_schema(CoverageAnswers.model_json_schema()), max_tokens=500)
            answers = CoverageAnswers.model_validate(data)
        except (LLMError, ValidationError) as exc:
            log.warning("coverage judge failed: %s", exc)
            return None
        self.answered += 1
        return coverage_from(answers, fact, answer, self.kb)
