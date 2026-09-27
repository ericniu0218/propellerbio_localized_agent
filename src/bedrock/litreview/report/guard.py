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
   text (the quote, the context sentence that gave the subjects or sample
   size, the next sentence when it qualifies the quote, and the sentence
   that names a target or agent the quote does not) or the paper's year.
3. Every drug, target, disease or pathogen it names must appear in the cited
   evidence — or be the thing the question is about.
4. Who was studied — patients, mice, rats, primates, cells — must be said by
   the cited text itself, not by a label: "in human tissues" citing a quote
   that says "across skin and spleen tissues" is dropped. And a conclusion's
   people must be backed by the paper's Methods: a sentence that says
   "patients" (or any word for people) is dropped when every cited finding
   whose text names people is one Stage 5 marked `methods_subjects` - its
   quote is the authors' conclusion, and the Methods studied only animals or
   cells ("... in patients with asthma" from a study of BEAS-2B cells and
   mice only; the shape of asthma S3). Read from that stored field, never
   from the paper, so a re-check agrees.
5. Which way a result points must match: a sentence may not claim no effect
   unless the cited evidence's own displayed text says so (a unit's
   `outcome` label is the model's, not the paper's, and is never enough by
   itself), claim an effect its cited quotes report only as null, drop a
   negation a quote states, drop a hedge the paper's shown text carries
   ("potentially due to X" restated as "X is associated with ..."; asthma
   S25 - checked against the quote, context, qualifier and naming
   sentences together, since the hedge is as often in a qualifier as in the
   quote itself), or say "decreased" where every cited quote says only
   "increased". Word-level checks, not entailment: they catch the flips a
   probe found, not every one.
6. In a findings section (`report/layout.py`'s `SectionSpec.findings`, true
   for every written section unless report.yaml says otherwise), at least one
   cited quote must report a result at all. A hand audit found sentences that
   claim nothing wrong and so pass every check above, yet state no finding: a
   Methods sentence naming the animal model, a trial's enrolment numbers, a
   review's generic scope statement. Each cited only a quote that is itself
   methods or demographics, with no result in it either. This check is what
   drops those - not what they say, but that their evidence never reports
   anything to say it about.
7. Aims, methods and predictions are not findings (`extract/verify.py`
   `not_a_result_reason`). In a findings section a sentence that itself
   only says how a study was done or what it set out to do ("The study used
   a ... mouse model to investigate ...") is dropped, however real the
   result it cites. And a sentence whose every cited quote only states an
   aim, a prediction or a hope ("we investigated the role of ...", "is
   predicted to benefit ...") is dropped in a findings section, and
   anywhere it claims an effect: check 6 reads words, and such quotes often
   use result words to name a model or a hope ("allergen-induced",
   "therapeutics to prevent").
8. A qualifier the paper states is not dropped. When the paper's very next
   sentence says a cited quote's result was not significant ("Although this
   result did not reach statistical significance (p = 0.09) ..."; see
   `extract/verify.py` `qualifier_after`), a sentence restating that result
   with no "not significant", "numerically", "trend" or the qualifier's own
   p value is dropped, unless another cited quote reports a result too. A
   sentence that restates the qualifier's other clause instead ("weight loss
   from baseline was significantly reduced") is kept. Such a qualifier also
   lets a sentence say the result was not significant (check 5).
9. A quote's result stays with what the quote names. When a cited finding's
   target or agent is not named in its quote, but the quote names another
   target or agent, a sentence that names the finding's one and none the
   quote names, while stating the quote's result (a number from the quote,
   or an effect word when its numbers are not another shown sentence's), is
   dropped: "Blocking IL-17A reduced ear thickness by 60%" citing "Anti-IL-23
   treatment reduced ear thickness by 60%", in a passage that also names
   IL-17A. The name itself may be cited - the
   sentence naming it is shown (`naming_quote`) - so "tubastatin A, acting
   on HDAC6, reduced food intake" is kept. Kept too when another cited quote
   names it and reports a result.
10. A sentence's claim rests on the **quote**. The context, qualifier and
    naming sentences are the paper's text too, so checks 2-9 are happy when
    the writer simply restates one of them and cites a unit whose quote says
    something else ("However, whether the CCL2-CCR2 signalling pathway
    contributes to chronic migraine is not clear", citing a quote about
    elevated CCL2 in CSF). Those sentences are shown so they can add a name,
    a limit, a subject or a sample size to the quote's finding - never so
    they can be the whole of it. So a sentence must state one of the quote's
    numbers, or take words from the quote that it could not have taken from
    the sentences shown beside it; if those supplied all of its content, or
    several times as much as the quote did, it is restating one of them and
    is dropped.

The cited evidence is the paper's text, never what the model wrote beside it.
`effect_detail`, `species_raw` and a sample size the paper does not state are
the model's words; a sentence resting on them has nothing a reader can check.
Names the model gave a finding (its target, its agent) count only where
`extract/provenance.py` found them in the text the model read - since issue
#6, in a sentence the reader is shown beside the quote.

A sentence that fails any check is dropped and the reason recorded. The
report shows how many were dropped; the benchmark reads the reasons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bedrock.litreview.extract.verify import (
    KEEPS_NOT_SIGNIFICANT,
    RESULT_WORDS,
    content_words,
    covered,
    not_a_result_reason,
    not_significant_clause,
    people_named,
    reports_result,
    split_sentences,
    studied_in,
)
from bedrock.litreview.knowledge import Knowledge, TermMatcher, clean, fold
from bedrock.litreview.models import EvidenceUnit, QuestionSpec
from bedrock.litreview.negation import drops_hedge, drops_negation

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


def _shown(unit: EvidenceUnit) -> list[str]:
    """The paper's sentences shown beside a unit: the quote, the context
    sentence, the qualifying next sentence and the naming sentence."""
    return [p for p in (unit.quote, unit.context_quote, unit.qualifier_quote, unit.naming_quote) if p]


def source_text(unit: EvidenceUnit) -> str:
    """The paper's own words a sentence citing this unit may rest on: every
    sentence shown beside it."""
    return " | ".join(_shown(unit))


def unit_text(unit: EvidenceUnit, extra: str = "") -> str:
    """Everything a sentence citing this unit may legitimately draw on: the
    paper's text, plus the names provenance located, plus `extra` (the
    paper's label, so its year is citable). A unit stored before names had
    to be shown can still say a name was found in the 'chunk'; it counts
    here, and check 9 is what stops it taking the quote's result."""
    parts = _shown(unit)
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
        findings: bool = True,
    ) -> None:
        """`units`: short id -> unit, for the slice being written.
        `paper_labels`: paper id -> "Author Year Journal", so a year is citable.
        `vocabulary`: extra entity names seen anywhere in this run's evidence
        (targets, drugs) that the table may not know.
        `findings`: whether this section requires every sentence to cite a
        quote that reports a result (see check 6 in the module docstring).
        Callers pass the section's own `SectionSpec.findings` so the same
        section is judged by the same rule everywhere it is checked."""
        self.spec = spec
        self.kb = knowledge
        self.units = units
        self.paper_labels = paper_labels or {}
        self.findings = findings
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

        cited = [self.units[c] for c in citations]
        # The cited quote says "patients" because it is the authors'
        # conclusion; the Methods studied only mice and cells (check 4).
        if (why := unbacked_people(masked, cited)):
            return GuardResult(False, why)
        # The paper's next sentence says the result was not significant, and
        # the sentence states it as if it were (obesity S6: "~0.6 kg/week"
        # restated, "did not reach statistical significance (p = 0.09)" gone).
        # Before the result checks below, so a sentence that fails this and
        # another check is reported for this, the more specific reason.
        if (why := drops_not_significant(sentence, cited)):
            return GuardResult(False, why)
        # The quote is about one agent and the sentence gives its result to
        # another the passage named (the IL-23 / IL-17A probe; check 9).
        if (why := misattributes(sentence, masked, cited, self.kb)):
            return GuardResult(False, why)
        # The sentence is one of the other sentences shown beside the quote -
        # the qualifier, the context or the naming sentence - and the quote it
        # cites says something else (check 10). Every cited unit has to fail
        # this for the sentence to go: one quote it really does rest on is
        # enough, since the sentence may be citing the others for a name.
        if not any(rests_on_quote(sentence, masked, u) for u in cited):
            return GuardResult(False, "restates the context, qualifier or naming sentence, not the quote it cites")

        # Which way a result points. A sentence may say something did not work
        # only if the paper's own displayed text says so: the MedGemma run
        # wrote "inhibition of HIF signaling did not improve outcomes" citing
        # only findings where it did. A unit's `outcome` is never enough by
        # itself - it is the extraction model's label, not the paper's words,
        # and a model can mislabel it: a methods sentence naming a NOX
        # blocker and a NOX2 inhibitor, with nothing else shown beside it,
        # was labelled `no_effect`, and a sentence citing only that quote
        # then claimed "did not significantly alter NOX2 activity or prevent
        # atrial fibrosis" - the opposite of the paper's real result (the
        # blocker abrogated fibrosis), which the guard would have caught had
        # it read the paper's words instead of trusting the label.
        # In a findings section, a sentence with nothing to check above can
        # still be filler: it names nothing wrong because it states nothing
        # at all, just methods, demographics or a generic scope line. Every
        # sentence in a findings section must rest on a quote that reports a
        # result, whether or not the sentence itself uses effect language -
        # unlike the narrower check below, this one does not wait for the
        # sentence to claim an effect first.
        if self.findings and not any(_reports_result(clean(u.quote)) for u in cited):
            return GuardResult(False, "cites no quote that reports a result")
        # The sentence itself says only how the study was done or what it set
        # out to do - "The study used a ... mouse model to investigate the
        # effects of NF-kB inhibition" (migraine S23). It can cite a real
        # result and still claim nothing. Aim and method cues only: a
        # sentence that reports a result and then adds "suggesting a
        # potential role for X" is left alone.
        if self.findings and (not_a_result_reason(sentence) or "").endswith(":aim"):
            return GuardResult(False, "is a methods or aims sentence, not a finding")
        # A sentence resting only on quotes that state an aim, a prediction
        # or a hope has no result under it, however result-like the quote's
        # words: "we investigated the role of the AhR in regulating
        # allergen-induced ROS ..." (asthma S32). Stage 5 no longer keeps such
        # quotes; this is what catches them in a stored run extracted before
        # it did, when `verify-run` re-checks it.
        if (self.findings or _EFFECT.search(sentence)) \
                and all(not_a_result_reason(clean(u.quote)) for u in cited):
            return GuardResult(False, "cites only quotes that state an aim, method or prediction, not a result")
        # A qualifier saying the result was not significant says so too, or
        # the writer, shown it, could never write "did not reach significance".
        # "The cited evidence" is what the reader is shown beside each unit
        # (`source_text`: quote, context, qualifier and naming sentences),
        # never the model-written `outcome` field.
        if _NULL_CLAIM.search(sentence) and not any(
            _NULL_CLAIM.search(clean(source_text(u))) or _not_significant(u)
            for u in cited
        ):
            return GuardResult(False, "claims no effect, which the cited evidence does not")
        # And the reverse, found by an adversarial probe: "Blocking IL-17A
        # reduced epidermal thickness", citing only "Blocking IL-17A had no
        # significant effect on epidermal thickness", was kept - every name
        # and number in it is in the quote.
        if not _NULL_CLAIM.search(sentence) and _EFFECT.search(sentence) \
                and all(_null_only(clean(u.quote)) for u in cited):
            return GuardResult(False, "claims an effect the cited evidence reports as null")
        # A result needs a quote that reports one. Found in a hand audit:
        # "Montelukast reduced asthma exacerbations in children ... 45
        # participants randomized to montelukast and 44 to placebo" cited only
        # the trial's enrolment sentence, and kept every name and number. In a
        # findings section this is already caught above, whether or not the
        # sentence uses effect language; this narrower check is what is left
        # for a section that sets `findings: false`, where a sentence that
        # explicitly claims an effect should still not get away with it.
        if _EFFECT.search(sentence) and not any(_reports_result(clean(u.quote)) for u in cited):
            return GuardResult(False, "states a result its cited evidence does not report")
        for u in cited:
            if any(drops_negation(sentence, s) for s in split_sentences(clean(u.quote))):
                return GuardResult(False, "drops a negation in the cited evidence")
        # A hedge can live in the quote itself (asthma S25's "potentially due
        # to leukotriene mediated airway inflammation" is the quote) or in the
        # qualifier next to it (the obesity TMEM135 "presumably ... weight
        # gain" case), so this reads all the shown text, not only the quote.
        for u in cited:
            if any(drops_hedge(sentence, s) for s in split_sentences(clean(source_text(u)))):
                return GuardResult(False, "drops a hedge the cited evidence carries")
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


# What a quote that reports a result says: one vocabulary, kept beside quote
# verification in `extract/verify.py` so Stage 5's "not a result" rule reads
# results with exactly the same words as this guard.
_RESULT = RESULT_WORDS
_reports_result = reports_result


_UP = re.compile(r"\b(?:increas|elevat|upregulat|higher|raised|rose)\w*", re.I)
_DOWN = re.compile(r"\b(?:decreas|reduc|lower|downregulat|diminish|declin|fell)\w*", re.I)
_REVERSAL = re.compile(
    r"\b(?:prevent|block|inhibit|revers|abrogat|abolish|attenuat|blunt|counteract|normali[sz]|restor|suppress|"
    r"protect|rescu|than|versus|vs|compared)\w*", re.I)


def _directions(text: str) -> set[str]:
    """Which ways the text says something moved: {"increased"}, {"decreased"},
    both or neither."""
    return {name for name, pattern in (("increased", _UP), ("decreased", _DOWN)) if pattern.search(text)}


def _not_significant(unit: EvidenceUnit) -> tuple[str, str] | None:
    """The unit's qualifier split into (not-significant clause, rest), when
    it says the quote's result was not significant."""
    if not unit.qualifier_quote:
        return None
    return not_significant_clause(clean(unit.qualifier_quote), clean(unit.quote))


def drops_not_significant(sentence: str, cited: list[EvidenceUnit]) -> str | None:
    """Why `sentence` drops a "not significant" the paper's next sentence
    states about a cited quote's result, or None. Check 8 in the module
    docstring; `extract/verify.py` decides what a qualifier is.

    Built to keep, not to drop, whenever the sentence may be doing something
    else than restating the qualified result:

    - another cited quote reports a result: the sentence may rest on that one;
    - it says "not significant", "numerically", "a trend", "did not reach",
      or claims no effect at all;
    - it carries a number the qualifier adds, such as its "p = 0.09";
    - it uses a word or number that only the qualifier's other clause has:
      "weight loss from baseline was significantly reduced (p < 0.0001)" is
      the paper's own significant result, and "baseline" or "0.0001" means
      the sentence is reporting that one;
    - it does not restate the quote at all (it shares no number with the
      quote, and fewer than half of its own subject words are the quote's),
      so there is nothing the qualifier limits.
    """
    limited = [(u, parts) for u in cited if (parts := _not_significant(u))]
    if not limited:
        return None
    qualified = {u.id for u, _ in limited}
    if any(u.id not in qualified and _reports_result(clean(u.quote)) for u in cited):
        return None
    if KEEPS_NOT_SIGNIFICANT.search(sentence) or _NULL_CLAIM.search(sentence):
        return None
    said_numbers, said_words = numbers_in(sentence), content_words(sentence)
    restates = False
    for u, (verdict, rest) in limited:
        quote = clean(u.quote)
        quote_numbers, quote_words = numbers_in(quote), content_words(quote)
        if said_numbers & (numbers_in(verdict) - quote_numbers):
            return None
        if said_numbers & (numbers_in(rest) - quote_numbers) or said_words & (content_words(rest) - quote_words):
            return None
        if said_numbers & quote_numbers or (said_words and len(said_words & quote_words) * 2 >= len(said_words)):
            restates = True
    if not restates:
        return None
    return "drops the qualifier the paper states: not significant"


# How much more of a sentence the other displayed sentences may supply than
# the quote does before the sentence is one of them rather than a reading of
# the quote. Five, measured: at four a good sentence that states the quote's
# finding and then borrows the context sentence's detail ("AhR activation
# reprograms the allergen-induced epithelial response, suppressing ROS
# generation ... in bronchial epithelial cells and mouse lungs") is lost; at
# eight a sentence that is the context sentence with one word of the quote's
# in it survives. Five and six give the same answer on all three stored
# reports, which is the only reason to trust the number at all.
QUOTE_SHARE = 5


def rests_on_quote(sentence: str, masked: str, unit: EvidenceUnit) -> bool:
    """Does this sentence's content come from the unit's quote, rather than
    only from the other sentences shown beside it? Check 10 in the module
    docstring.

    The comparison is what each side contributes that the other does not, so
    that words both of them use (usually the target's name) decide nothing:

    - `own`: content words and numbers the sentence shares with the quote
      alone. `content_words` (`extract/verify.py`, the same reading check 8
      uses) is what counts as content: four letters or more, or a name with a
      digit in it, minus filler and result words.
    - `borrowed`: the ones it shares with the context, qualifier and naming
      sentences alone.

    A number is the strongest signal there is that a sentence is reporting
    the quote's result, so any number of the quote's settles it, whether or
    not the other sentences repeat that number: "Montelukast at a dose of
    3.0 mg/kg achieved a potentially therapeutic Cmax in children with ...
    exacerbations" states the quote's own "3.0 mg/kg" result and takes only
    its population from the context. After that: nothing borrowed means the
    other sentences added nothing the quote did not, so there is nothing to
    suspect. Otherwise the quote has to have supplied a fair share of what
    the sentence says - see `QUOTE_SHARE`.

    Numbers are read from `masked`, the sentence with known names blanked,
    so the "2" of "type 2 diabetes" is not read as one of the quote's
    figures. Words are read from the sentence itself, because a name blanked
    out of it is exactly the kind of content this check is weighing.

    One of the shown sentences counts as the quote's own: a qualifier that
    `extract/verify.py` recognises as saying this quote's result was not
    significant. That sentence is, by that test, *about* the quote's result,
    so a sentence reporting the limit is still reporting the quote's finding
    - and check 8 above is what governs those. The qualifiers the four known
    failures restate are not of that kind: they are merely the paper's next
    sentence, reporting something else.
    """
    quote = clean(unit.quote)
    limit = unit.qualifier_quote if _not_significant(unit) else None
    if limit:
        quote += " " + clean(limit)
    others = clean(" ".join(p for p in _shown(unit)[1:] if p != limit))
    said_numbers, said_words = numbers_in(masked), content_words(clean(sentence))
    quote_numbers, quote_words = numbers_in(quote), content_words(quote)
    other_numbers, other_words = numbers_in(others), content_words(others)
    if said_numbers & quote_numbers:
        return True
    own = said_words & (quote_words - other_words)
    borrowed = (said_words & (other_words - quote_words)) | (said_numbers & (other_numbers - quote_numbers))
    if not borrowed:
        return True
    return len(own) * QUOTE_SHARE > len(borrowed)


def unbacked_people(masked: str, cited: list[EvidenceUnit]) -> str | None:
    """Why a sentence's people rest only on conclusions the Methods do not
    back, or None. The second half of check 4 in the module docstring.

    `masked` is the sentence with known names blanked, so "human
    papillomavirus" is not a claim about people. Kept when any cited finding
    not marked `methods_subjects` names people in the text shown beside it:
    the sentence may rest on that one."""
    word = people_named(masked)
    marked = [u for u in cited if u.methods_subjects]
    if not word or not marked:
        return None
    if any(not u.methods_subjects and people_named(clean(source_text(u))) for u in cited):
        return None
    return f"says {word!r}, but the paper's methods studied only {marked[0].methods_subjects}"


def _named(value: str | None, text: str, kb: Knowledge, kind: str | None = None) -> bool:
    """Is this name in the text, as written or as a listed name of the same
    entity? The reading provenance uses (`extract/provenance.py` `name_in`),
    imported here because provenance imports this module."""
    from bedrock.litreview.extract.provenance import name_in

    return bool(value) and name_in(value, text, kb, kind)


def misattributes(sentence: str, masked: str, cited: list[EvidenceUnit], kb: Knowledge) -> str | None:
    """Why `sentence` gives a cited quote's result to a name the quote does
    not state, or None. Check 9 in the module docstring.

    For each cited unit whose target or agent is not in its quote, while the
    quote names a different known target or drug, or states the unit's own
    other name (its agent, when the target is the missing one): the sentence
    is dropped if it names the missing one, names nothing the quote names,
    and states the quote's result - a number from the quote, or an effect
    word when its numbers are not all from the context, qualifier or naming
    sentence instead. It is kept when another cited quote names it and
    reports a result, since the sentence may rest on that one. `masked` is
    the sentence with known names blanked, so the "23" of "IL-23" is not read
    as a number from the quote.

    Measured on the three stored qwen3:8b reports: 21 kept sentences name a
    target or agent their quote does not, and in every one the quote names
    no other agent ("No patient reported treatment-related adverse events
    ..." beside rimegepant), so none is dropped - the rule is for the probe's
    shape, where the quote's result belongs to something else it names.
    """
    said_numbers = numbers_in(masked)
    for u in cited:
        quote = clean(u.quote)
        claimed = [(u.target_raw or u.target, (u.target_raw, u.target), "target"),
                   (u.intervention, (u.intervention,), "drug")]
        for label, names, kind in claimed:
            names = tuple(n for n in names if n)
            if not names or any(_named(n, quote, kb, kind) for n in names):
                continue
            if not any(_named(n, sentence, kb, kind) for n in names):
                continue
            # What the quote does name: the unit's other name when the quote
            # states it, and every known target or drug in it that is not
            # the missing one.
            other = [n for n in (u.intervention, u.target_raw, u.target)
                     if n and n not in names and _named(n, quote, kb)]
            other += [e.canonical for e, m in kb.scan(quote, ("target", "drug"))
                      if not any(kb.lookup(n) is e or _named(n, m.text, kb, kind) for n in names)]
            if not other or any(_named(n, sentence, kb) for n in other):
                continue
            # Stating the quote's result: a number from the quote, or an
            # effect word - unless the sentence's numbers come only from
            # another sentence shown beside it ("IL-17A fell by 40%" from the
            # naming sentence is that sentence's result, not the quote's).
            quote_numbers = numbers_in(quote)
            elsewhere = said_numbers & (numbers_in(" ".join(_shown(u)[1:])) - quote_numbers)
            if not (said_numbers & quote_numbers or (_EFFECT.search(sentence) and not elsewhere)):
                continue
            if any(v is not u and any(_named(n, clean(v.quote), kb, kind) for n in names)
                   and _reports_result(clean(v.quote)) for v in cited):
                continue
            return f"attributes the quote's result to {label}, which the quote does not name"
    return None


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
