"""Issue #3, leftover part b: a sentence drops a hedge the source carries.

The real case, from a hand audit of the asthma benchmark
(evals/litreview/hand_audit/qwen_asthma_after.md, S25/E1): the paper's own
sentence says insufficient response to inhaled beta-agonist and systemic
corticosteroid was "potentially due to leukotriene mediated airway
inflammation" - a guess, not a finding. The kept report sentence dropped
"potentially" and stated it flatly as "is associated with". Before this fix,
none of the existing checks in `CitationGuard.check` caught this: the
sentence's names, numbers and who-was-studied words are all in the quote, and
it does not flip a direction or a null result.

Confirmed by re-reading the audit sheet before choosing what to check: for
S25/E1 the hedge sentence *is* the unit's `quote` itself (no separate
`context_quote` is even recorded for that unit). But the same hedge phrasing
recurs as another unit's (E209's) `context_quote` a few entries later - and
a second real "kept" sentence in the obesity audit
(qwen_obesity_after.md, the TMEM135 body-weight sentence) carries its hedge
("presumably ... rapid ... weight gain") in the *qualifier*, not the quote.
So the guard checks `source_text(u)` - quote, context, qualifier and naming
sentences together - not only `u.quote`, or it would miss the qualifier
case. `test_guard_findings.py` documents the same kind of audit-driven,
minimal-shape test.
"""

from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import EvidenceUnit
from bedrock.litreview.negation import drops_hedge
from bedrock.litreview.report.guard import CitationGuard
from bedrock.litreview.stages.s0_question import parse_question

KB = load_knowledge()

# The real sentences (evals/litreview/hand_audit/qwen_asthma_after.md, S25/E1).
ASTHMA_QUOTE = (
    "Background and Purpose: Children with acute asthma exacerbations are frequently hospitalized "
    "due to insufficient response to inhaled β-agonist and systemic corticosteroid potentially "
    "due to leukotriene mediated airway inflammation."
)
ASTHMA_UNHEDGED_CLAIM = (
    "Leukotriene mediated airway inflammation is associated with insufficient response to inhaled "
    "β-agonist and systemic corticosteroid in children with acute asthma exacerbations."
)

# The real sentences (evals/litreview/hand_audit/qwen_obesity_after.md, a kept
# sentence and the qualifier next to its cited quote).
OBESITY_QUALIFIER = (
    "In contrast to female mice, high fat feeding in male mice did not elicit a significant body "
    "weight difference between the genotypes, presumably because the already rapid diet-induced "
    "weight gain in male C57BL/6J mice could not be further increased by the TMEM135 inactivation "
    "(Supplementary Fig. )."
)
OBESITY_HEDGED_CLAIM = (
    "High fat feeding in male mice did not elicit a significant body weight difference between "
    "TMEM135 knockout and control genotypes, presumably due to the already rapid diet-induced "
    "weight gain in male C57BL/6J mice."
)
OBESITY_QUOTE = (
    "Assessment of glucose homeostasis in these animals revealed that the TMEM135 knockout results "
    "in glucose intolerance, hyperinsulinemia and insulin resistance (Fig. )."
)


def test_drops_hedge_catches_the_real_asthma_case():
    """The report sentence states as settled fact what the paper only
    speculates ("potentially due to") - S25/E1, the case this fix targets."""
    assert drops_hedge(ASTHMA_UNHEDGED_CLAIM, ASTHMA_QUOTE)


def test_drops_hedge_leaves_a_faithfully_hedged_restatement_alone():
    """The real obesity sentence keeps "presumably" over the same governed
    words ("rapid ... diet-induced weight gain") the qualifier hedges, in
    different surrounding words. The guard must never punish this."""
    assert not drops_hedge(OBESITY_HEDGED_CLAIM, OBESITY_QUALIFIER)


def test_drops_hedge_ignores_sentences_with_no_shared_topic():
    """A hedge cue elsewhere, about something the claim never mentions, is
    not a drop - `SAME_TOPIC` guards against coincidental word overlap."""
    unrelated = "Possibly the largest cohort assembled so far, this registry followed 10,000 patients for cardiovascular outcomes."
    assert not drops_hedge(ASTHMA_UNHEDGED_CLAIM, unrelated)


def _unit(**kw) -> EvidenceUnit:
    base = dict(
        id="u1", paper_id="P1", question_id="q", target="leukotriene", target_raw="leukotriene",
        direction="unclear", modality="other", intervention=None, disease="asthma", disease_raw="asthma",
        evidence_type="human_clinical", species="children", species_raw="children", outcome="association",
        effect_detail="", n=None, combo_with=None, target_is_primary=True,
        quote=ASTHMA_QUOTE, quote_status="exact", section="abstract", extracted_by_model="fake",
        prompt_version="1", provenance={"target": "quote"},
    )
    base.update(kw)
    return EvidenceUnit(**base)


def _guard(unit: EvidenceUnit, **kw) -> CitationGuard:
    spec = parse_question("If I want to prevent asthma exacerbations, what should I target?")
    return CitationGuard(spec, KB, {"E1": unit}, paper_labels={"P1": "Smith 2024 Br J Pharmacol"}, **kw)


def test_the_guard_drops_the_real_unhedged_asthma_sentence():
    """End to end: `CitationGuard.check` rejects the S25 sentence, citing the
    new reason - not one of the pre-existing checks (which all pass this
    sentence today, which is exactly the bug)."""
    verdict = _guard(_unit()).check(ASTHMA_UNHEDGED_CLAIM, ["E1"])
    assert not verdict.kept
    assert verdict.reason == "drops a hedge the cited evidence carries"


def test_the_guard_keeps_the_hedge_when_the_sentence_states_it():
    """The inverse, same evidence: writing "potentially due to" (the paper's
    own words) instead of "is associated with" is kept."""
    hedged = (
        "Insufficient response to inhaled β-agonist and systemic corticosteroid in children with "
        "acute asthma exacerbations is potentially due to leukotriene mediated airway inflammation."
    )
    verdict = _guard(_unit()).check(hedged, ["E1"])
    assert verdict.kept, verdict.reason


# A second real unit from the same paper (E209 in qwen_asthma_after.md,
# S32/S34): its *quote* is the Cmax dosing result, and the same hedge
# sentence sits beside it as `context_quote`, not the quote. Real sentences
# recombined into one claim (S32's own kept numbers plus the context's
# claim, unhedged) to exercise the guard end to end without tripping the
# unrelated "does the sentence rest on the quote" check (check 10) - the
# Cmax numbers are the quote's own, which settles that check by themselves,
# so what is left to explain a drop is the hedge check this fix adds.
E209_QUOTE = (
    "The target Cmax was reached in 67%, 80%, and 90% of participants at doses of 2.0, 2.5, "
    "and 3.0 mg/kg, respectively."
)


def _e209(**kw) -> EvidenceUnit:
    return _unit(
        target="montelukast", target_raw="montelukast", direction="block", outcome="improved",
        setting="prevention", quote=E209_QUOTE, context_quote=ASTHMA_QUOTE, **kw,
    )


def test_the_guard_reads_the_context_sentence_for_a_hedge_the_quote_does_not_carry():
    """`u.quote` (the Cmax result) carries no hedge at all; the hedge is only
    in `u.context_quote` (the same real sentence as the asthma S25 case).
    Checking only `u.quote` (as `drops_negation` does) would miss this
    entirely - this pins the guard's wiring to `source_text`, which reads
    the context sentence too."""
    unhedged = (
        "In children with acute asthma exacerbations hospitalized due to insufficient response to "
        "inhaled β-agonist and systemic corticosteroid caused by leukotriene mediated airway "
        "inflammation, the target Cmax was reached in 67%, 80%, and 90% of participants at doses of "
        "2.0, 2.5, and 3.0 mg/kg, respectively."
    )
    verdict = _guard(_e209()).check(unhedged, ["E1"])
    assert not verdict.kept
    assert verdict.reason == "drops a hedge the cited evidence carries"


def test_the_guard_keeps_the_context_sentence_hedge_when_the_claim_states_it():
    """The inverse, same evidence: writing "potentially due to" (the
    context's own words) instead of "caused by" is not penalized."""
    hedged = (
        "In children with acute asthma exacerbations hospitalized due to insufficient response to "
        "inhaled β-agonist and systemic corticosteroid potentially due to leukotriene mediated "
        "airway inflammation, the target Cmax was reached in 67%, 80%, and 90% of participants at "
        "doses of 2.0, 2.5, and 3.0 mg/kg, respectively."
    )
    verdict = _guard(_e209()).check(hedged, ["E1"])
    assert verdict.kept, verdict.reason


# --- What the benchmark found that these unit tests had not -------------------
#
# The hedge rule's only firing on the obesity corpus was a FALSE DROP, found by
# re-running the benchmark rather than by any test here. Both cases below are
# built from that real run:
# evals/litreview/results/obesity-treatment-targets/20260927T070944Z, report
# cb23593c148f1697, section targets:TMEM135, evidence unit E189 (PMID37773161).

E189_QUALIFIER = (
    "In contrast to female mice, high fat feeding in male mice did not elicit a significant "
    "body weight difference between the genotypes, presumably because the already rapid "
    "diet-induced weight gain in male C57BL/6J mice could not be further increased by the "
    "TMEM135 inactivation (Supplementary Fig. )."
)
E189_CLAIM = (
    "TMEM135 knockout in mice led to glucose intolerance, hyperinsulinemia, and insulin "
    "resistance, while high fat feeding in male mice did not elicit a significant body "
    "weight difference between genotypes."
)


def test_repeating_only_the_unhedged_clause_is_not_dropping_a_hedge():
    """The paper states a fact ("did not elicit a significant body weight
    difference") and then speculates about its cause ("presumably because the
    already rapid diet-induced weight gain ..."). This claim repeats only the
    fact and never asserts the guess, so nothing was unhedged. The first
    version of the rule dropped it anyway, because the hedge's governed words
    share the generic word "weight" with the fact it is explaining."""
    assert drops_hedge(E189_CLAIM, E189_QUALIFIER) is False


def test_a_hedge_worded_differently_is_still_a_hedge():
    """The source hedges with "potentially due to leukotriene mediated airway
    inflammation". This claim hedges the same content, but its "potentially"
    governs "associated with" rather than "leukotriene ...". Comparing the two
    hedges' governed words called that a drop; a claim that hedges at all is
    not stating settled fact, so it is left alone."""
    moved = (
        "Leukotriene mediated airway inflammation is potentially associated with insufficient "
        "response to inhaled beta-agonist and systemic corticosteroid in children with acute "
        "asthma exacerbations."
    )
    assert drops_hedge(moved, ASTHMA_QUOTE) is False
