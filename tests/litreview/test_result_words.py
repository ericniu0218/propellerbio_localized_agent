"""`RESULT_WORDS` (`extract/verify.py`), widened after a hand re-audit of the
179 sentences kept across the three stored benchmark reports (obesity,
asthma, migraine) under today's citation guard found real results check 6
("cites no quote that reports a result") wrongly dropped. Every quote below
is verbatim from those stored reports (`evals/litreview/results/.../report.json`);
the short id in each comment is that report's `short_ids` label, for cross-
reference with `evals/litreview/hand_audit/`.

The negatives matter as much as the positives: a false drop throws away real
evidence exactly as a false keep invents it. Several of these are drawn from
sentences that must stay dropped for a reason that has nothing to do with
`RESULT_WORDS` (they cite a quote that is methods, demographics or an aim -
see `test_guard_findings.py` and `test_not_a_result.py`), and one - the
obesity MC4R sentence - stays dropped only because `RESULT_WORDS` is
deliberately scoped not to catch it; see the comment on that test.
"""

import pytest

from bedrock.litreview.extract.verify import reports_result
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import EvidenceUnit
from bedrock.litreview.report.guard import CitationGuard
from bedrock.litreview.stages.s0_question import parse_question

KB = load_knowledge()

# --- real results RESULT_WORDS was missing --------------------------------------

# obesity E214: an effect verb, "led to".
LED_TO = "In a 28-day Phase 1b clinical trial, Setmelanotide led to weight loss in obese MC4R variant carriers."
# obesity E11: an effect verb, "potentiate".
POTENTIATE = ("Mechanistically, FGF21 was found to potentiate mast cell activation through cholesterol "
              "biosynthesis modulation.")
# obesity E157: a null result ("no differences"), stated as such by the paper.
NO_DIFFERENCES = (
    "Consistently, co-culture of adipocytes isolated from HFD-fed AWT or AKO mice with purified CD8⁺ T cells "
    "revealed no differences in the frequencies of IFNγ⁺, granzyme B⁺, or perforin⁺ CD8⁺ T "
    "cells (Supplementary Fig. e–h), indicating that adipocyte-derived OX40L does not directly regulate "
    "CD8⁺ T cell effector function."
)
# obesity E206: an effect verb, "results in".
RESULTS_IN = ("Assessment of glucose homeostasis in these animals revealed that the TMEM135 knockout results in "
              "glucose intolerance, hyperinsulinemia and insulin resistance (Fig. ).")
# asthma E146: a scoped expression comparison, "highly expressed ... compared to".
HIGHLY_EXPRESSED = ("Analysis of immunofluorescence staining and HE staining showed that N-GSDMD was highly "
                    "expressed in the airway epithelium of asthma patients compared to the control group "
                    "(Fig. , Supplementary Fig. ).")
# migraine E64: a scoped comparison, "similar between" two dose arms.
SIMILAR_BETWEEN = ("Volume of distribution and clearance were similar between the 100 mg and 300 mg dose groups, "
                   "and half-life was 22.5−28.1 days.")
# migraine E96: a safety result, "the most frequent AEs were ...".
MOST_FREQUENT_AES = ("The most frequent AEs were nasopharyngitis, oropharyngeal pain, coronavirus infection, upper "
                     "abdominal pain, COVID‐19, constipation, pyrexia, influenza, and back pain (Table ).")
# obesity E70 (issue #1): "elevations ... were observed" - the noun form.
ELEVATIONS_OBSERVED = ("Substantial elevations in ROS levels were observed in DIO mice from baseline, distinct "
                       "from control, Nox2 -KO, and DIO Nox2 -KO mouse atrial myocytes.")
# obesity E130 (issue #1): "normalized".
NORMALIZED = "In obese leptin knockout mice, REGN4461 normalized body weight, food intake, blood glucose, and insulin sensitivity."
# obesity E142 (issue #1): "reversed".
REVERSED = ("Notably, the transfer of mast cells deficient in both Sirt6 and galectin-3 reversed most of the "
           "metabolic disturbances in Kit W-sh/W-sh mice with Sirt6-deficient mast cells, except for total number "
           "of mast cells in EAT.")


@pytest.mark.parametrize("quote", [
    LED_TO, POTENTIATE, NO_DIFFERENCES, RESULTS_IN, HIGHLY_EXPRESSED, SIMILAR_BETWEEN, MOST_FREQUENT_AES,
    ELEVATIONS_OBSERVED, NORMALIZED, REVERSED,
], ids=["led-to", "potentiate", "no-differences", "results-in", "highly-expressed-compared", "similar-between",
        "most-frequent-aes", "elevations-were-observed", "normalized", "reversed"])
def test_real_results_are_recognised(quote):
    assert reports_result(quote)


# --- non-results RESULT_WORDS must keep missing ---------------------------------

# obesity E209: an enrolment sentence - no result of any kind.
ENROLMENT = ("A total of 49 healthy obese (BMI 26.2–40.3 kg/m 2 ) individuals between the ages of 19–53 "
            "years (41 males and 8 females) were enrolled in 5 cohorts and randomized to treatment with either "
            "Setmelanotide or placebo for 14 or 28 days via continuous subcutaneous infusion.")
# obesity E59: describes what was done ("preventive" does not stem-match "prevent").
PREVENTIVE_REGIMEN = "The CCR1‐selective antagonist BX471 was administered to HFD‐fed wild‐type (WT) mice as a preventive regimen."
# asthma E234: describes a drug's approval status, not a finding.
MONTELUKAST_APPROVED = "Montelukast is a potent leukotriene receptor antagonist approved for chronic asthma treatment."
# asthma E110: background about a gene's known biology.
C_MYC_BACKGROUND = ("c-Myc, a well-characterized proto-oncogene and transcription factor, orchestrates diverse "
                    "cellular processes including proliferation, metabolism, and apoptosis.")
# asthma E7: a generic, undated claim about a disease, not a result.
AIRWAY_REMODELING = "Airway remodeling is a critical hallmark of chronic asthma that is very difficult to treat."
# migraine E5: bare "expressed" names where a receptor exists, not a finding -
# the scoped expression pattern requires a comparison word before it.
PAR2_EXPRESSED = ("Protease activated receptor 2 (PAR2) is a G-protein coupled receptor expressed in meningeal "
                  "neurons, fibroblasts and mast cells that may be targeted to treat migraine.")
# migraine E56: demographics only.
FIVE_WOMEN = "Five women with a mean age of 31.8 years (range 22–42 years) provided fully analyzable data."
# migraine E61: enrolment and completion counts, no result.
TWENTY_ENROLLED = ("Twenty participants were enrolled and randomized; 19 completed the trial, but all 20 were "
                   "included in the pharmacokinetic and safety analyses (Fig. ).")
# obesity E212 (issue #1's own example, see test_qualifier.py's S6 cases): a
# real group-mean-difference number, but read as a result only through the
# paper's *next* sentence ("did not reach statistical significance"), which
# is stored as a qualifier on the evidence unit, not a word in this quote.
# `RESULT_WORDS` is deliberately scoped to "no difference(s)", not bare
# "differences", so this sentence keeps being dropped by check 6 ("cites no
# quote that reports a result") exactly as it is today - never by check 8
# ("drops the qualifier"), which only fires when the qualifier was actually
# captured on the cited unit. Reading bare "differences" as a result would
# silently depend on that capture happening, and it does not for every unit
# extracted before qualifiers were stored (see
# test_qualifier.py::test_without_a_stored_qualifier_the_rule_does_not_fire).
MC4R_DIFFERENCES = ("In the MC4R mutation carriers, placebo subtracted group mean differences in weight loss "
                   "(dose 0.01 mg/kg/24 h) of ∼0.6 kg/week were observed ().")


@pytest.mark.parametrize("quote", [
    ENROLMENT, PREVENTIVE_REGIMEN, MONTELUKAST_APPROVED, C_MYC_BACKGROUND, AIRWAY_REMODELING, PAR2_EXPRESSED,
    FIVE_WOMEN, TWENTY_ENROLLED, MC4R_DIFFERENCES,
], ids=["enrolment", "preventive-regimen", "montelukast-approved", "c-myc-background", "airway-remodeling",
        "par2-bare-expressed", "five-women", "twenty-enrolled", "mc4r-differences-no-qualifier"])
def test_non_results_still_report_nothing(quote):
    assert not reports_result(quote)


# --- the full guard, on the trickiest of these ------------------------------------

SPEC = parse_question("If I want to treat obesity, which targets should I pursue?")


def _unit(quote: str, **kw) -> EvidenceUnit:
    base = dict(
        id="u1", paper_id="P1", question_id="q", target="OX40L", target_raw="OX40L", direction="unclear",
        modality="other", intervention=None, disease="obesity", disease_raw="obesity", evidence_type="in_vivo",
        species="other", species_raw="mice", outcome="no_effect", effect_detail="", n=None, combo_with=None,
        target_is_primary=True, quote=quote, quote_status="exact", section="results",
        extracted_by_model="fake", prompt_version="1", provenance={"target": "quote", "species": "quote"},
    )
    base.update(kw)
    return EvidenceUnit(**base)


def _check(sentence: str, unit: EvidenceUnit):
    guard = CitationGuard(SPEC, KB, {"E1": unit}, paper_labels={"P1": "Smith 2024 J"})
    return guard.check(sentence, ["E1"])


def test_guard_keeps_the_null_result_sentence_that_only_the_quotes_no_differences_carries():
    """OX40L (issue #1): the report sentence states the paper's own null
    result, and its only cited quote reports it through "no differences",
    not through any of the older RESULT_WORDS."""
    sentence = "Adipocyte-derived OX40L does not directly regulate CD8⁺ T cell effector function."
    verdict = _check(sentence, _unit(NO_DIFFERENCES))
    assert verdict.kept, verdict.reason


def test_guard_still_drops_the_mc4r_differences_sentence_via_check_6():
    """The sentence must stay dropped (see the comment on MC4R_DIFFERENCES
    above); confirms it is check 6, unchanged, that drops it - not check 8,
    which has no qualifier to work with on this unit."""
    unit = _unit(
        MC4R_DIFFERENCES, target="MC4R", target_raw="MC4R", disease="obesity", disease_raw="obesity",
        evidence_type="human_clinical", species_raw="patients", outcome="association",
    )
    verdict = _check(MC4R_DIFFERENCES.rstrip(" ()."), unit)
    assert not verdict.kept
    assert verdict.reason == "cites no quote that reports a result"


def test_guard_keeps_the_tmem135_results_in_sentence():
    unit = _unit(
        RESULTS_IN, target="TMEM135", target_raw="TMEM135", evidence_type="in_vivo", species_raw="mice",
        outcome="worsened",
        context_quote="Transgenic mice that overexpress TMEM135 (Tmem135 TG) under the control of CAG promoter "
                      "have been previously described.",
        context_section="results",
    )
    sentence = "Knockout of TMEM135 in mice results in glucose intolerance, hyperinsulinemia, and insulin resistance."
    verdict = _check(sentence, unit)
    assert verdict.kept, verdict.reason
