"""Regression cases from report a67d8b766a9a2632 ("Does tirzepatide reduce liver
fat in adults with obesity?", 29 September 2026), made general.

Each case is a quote the report really showed, and the rule that now keeps it
from being counted for something it does not support. The report is evidence
of what the system displayed; the papers were not re-adjudicated here, and the
labels below are engineering expectations written by the developer, not a
scientist's gold standard (see evals/litreview/README.md for that).

Paired cases check both directions: the rule rejects the wrong use and keeps
the right one, so a rule cannot pass by throwing everything away.
"""

from __future__ import annotations

import pytest

from bedrock.litreview.extract.verify import other_work_reason
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import Paper, PaperSignals
from bedrock.litreview.report import claims
from bedrock.litreview.report.guard import CitationGuard, sign_problem, signed_numbers
from bedrock.litreview.sources.europepmc import clean
from bedrock.litreview.stages.s0_question import parse_question
from tests.litreview.test_guard_cache_config import unit

KB = load_knowledge()
SPEC = parse_question("Does tirzepatide reduce liver fat in adults with obesity?", None, knowledge=KB)
TZP_PAPER = Paper(id="p", title="t", abstract="Tirzepatide (TZP), a dual GIP/GLP-1 receptor agonist, was studied.",
                  signals=PaperSignals())


def u(quote, **kw):
    base = dict(paper_id="p", quote=quote, target="liver fat", intervention="tirzepatide", disease="hepatic steatosis",
                disease_raw="MASLD", outcome="improved", effect_detail="", n=None, species_raw="patients")
    base.update(kw)
    return unit(**base)


# --- attribution: who caused it -------------------------------------------------------------------


def test_disease_induction_is_not_the_treatments_harm():  # E75
    induced = u("HFHFr feeding induced hyperglycaemia, increased HOMA-IR, elevated ALT/AST, and marked hepatic "
                "steatosis and inflammatory injury.", intervention="HFHFr feeding", outcome="worsened")
    treated = u("Both Sema and TZP attenuated these histopathological lesions and reduced hepatic lipid accumulation.",
                intervention="Sema and TZP")
    assert not claims.attributable(induced, SPEC, TZP_PAPER)
    assert claims.attributable(treated, SPEC, TZP_PAPER)


def test_another_drugs_result_is_not_the_requested_drugs():
    other = u("Semaglutide reduced liver fat content by 30%.", intervention="semaglutide")
    assert not claims.attributable(other, SPEC, TZP_PAPER)


# --- what was compared ----------------------------------------------------------------------------------


def test_a_genotype_subgroup_null_is_not_a_treatment_null():  # E8
    subgroup = u("There were no statistically significant differences in any of the reported outcomes between any of "
                 "the genotype subgroups (CC vs. CG, CC vs. GG, and CG vs. GG).", outcome="no_effect")
    assert claims.contrast_of(subgroup, SPEC, TZP_PAPER) == "subgroup"


def test_arms_that_all_got_the_drug_are_a_co_intervention_comparison():  # E14, E15
    diets = u("Both groups showed significant reductions in body weight (TZP + LEKT, p = 0.0289; TZP + LCD, "
              "p = 0.0278), with no significant intergroup difference (p = 0.665).")
    assert claims.contrast_of(diets, SPEC, TZP_PAPER) == "co_intervention"
    versus = u("The absolute reduction in LFC was significantly greater for the pooled tirzepatide 10 mg and 15 mg "
               "groups versus the insulin degludec group.")
    assert claims.contrast_of(versus, SPEC, TZP_PAPER) == "between_arm"


def test_change_from_baseline_is_not_a_comparison_with_a_control():  # E56
    within = u("Significant improvements in body weight, liver enzymes, and hemoglobin A1c were found at Week 12 "
               "and were sustained compared with baseline values.")
    assert claims.contrast_of(within, SPEC, TZP_PAPER) == "within_arm"


# --- what was measured ----------------------------------------------------------------------------------


@pytest.mark.parametrize("quote, expected", [
    ("Additionally, TZP was linked to lower risks of all-cause mortality (HR, 0.48), MALO (HR, 0.50).",  # E36
     "clinical_events"),
    ("From an overall mean baseline LFC of 15·71%, the absolute reduction in LFC at week 52 was greater.",  # E12
     "liver_fat_measured"),
    ("TC, LDL-C, triglycerides, FPG, HbA1c, and HSI were significantly reduced.", "liver_fat_estimated"),
    ("At week 36, significant within-group reductions in AST, ALT, and GGT levels were observed.", "liver_enzymes"),
    ("Gastrointestinal adverse events, including nausea, were the most frequently reported events.", "adverse_events"),
])
def test_distinct_endpoints_stay_distinct(quote, expected):
    assert claims.endpoint_of(u(quote)).id == expected


# --- numbers --------------------------------------------------------------------------------------------------

E12 = ("From an overall mean baseline LFC of 15·71% (SD 8·93), the absolute reduction in LFC at week 52 was "
       "significantly greater for the pooled tirzepatide 10 mg and 15 mg groups (-8·09%, SE 0·57) versus the "
       "insulin degludec group (-3·38%, 0·83).")


def test_middle_dot_decimals_and_signs_are_read_as_the_paper_means_them():
    assert signed_numbers(E12)["8.09"] == {"-"} and signed_numbers(E12)["0.57"] == {"+"}
    assert sign_problem("The reduction in LFC was 8.09% vs 3.38% with insulin degludec.", E12) is None
    assert sign_problem("LFC increased by 8.09%.", E12)  # a decrease stated as an increase
    assert sign_problem("The change was 8.09% at week 52.", E12)  # the fall is not said


def test_fixing_the_decimals_does_not_approve_a_sentence_that_overstates_the_population():
    """The originally removed Answer sentence named "adults with type 2 diabetes
    and obesity"; the trial enrolled BMI of at least 25. Numbers now match, and
    the sentence must still fail on what it says about who was studied."""
    e12 = unit(quote=E12, paper_id="p10", species_raw="adults", target="liver fat content",
               context_quote="Eligible participants were adults with type 2 diabetes, a baseline HbA1c 7·0-10·5%, a "
                             "BMI of at least 25 kg/m2, stable weight, were insulin-naive.")
    guard = CitationGuard(SPEC, KB, {"E12": e12})
    right = guard.check("The absolute reduction in liver fat content was greater with pooled tirzepatide 10 mg and "
                        "15 mg (-8.09%, SE 0.57) than with insulin degludec (-3.38%) at week 52, in adults with type 2 "
                        "diabetes.", ["E12"])
    assert right.kept, right.reason
    wrong = guard.check("The absolute reduction in liver fat content was greater with pooled tirzepatide 10 mg and "
                        "15 mg (-8.09%, SE 0.57) than with insulin degludec (-3.38%) at week 52, in adults with type 2 "
                        "diabetes and obesity.", ["E12"])
    assert not wrong.kept and "obesity" in wrong.reason


def test_a_reversed_interval_is_flagged_not_repaired():  # E38
    anomalies = claims.statistical_anomalies("TZP use was associated with a reduced risk of the composite primary "
                                             "outcome compared to TZD (HR, 0.66; 95% CI, 0.58–0.42).")
    assert anomalies and "0.58 to 0.42" in anomalies[0]
    assert claims.statistical_anomalies("(HR, 0.66; 95% CI, 0.54–0.82)") == []


def test_one_paper_giving_two_estimates_for_one_outcome_is_noticed():  # E33 vs E36
    assert claims.estimates("MALOs (HR, 0.64; 95% CI, 0.48–0.86)") == {"MALO": "0.64"}
    assert claims.estimates("MALO (HR, 0.50; 95% CI, 0.36–0.69)") == {"MALO": "0.50"}


def test_a_per_group_sample_size_says_so():  # reference [2]
    e = u("Tirzepatide offers better clinical outcomes.", n=6199,
          context_quote="After PSM, 6199 patients were classified into each group (tirzepatide and BMS).")
    assert claims.n_scope(e) == "per group"
    assert claims.n_scope(u("x", n=190, context_quote="In total 190 participants were randomly assigned.")) == ""


# --- comparators are named; wrong agents are not ------------------------------------------------------------


def _guard_for(quote, naming=None):
    e = unit(quote=quote, paper_id="p8", species_raw="patients", target="gastrointestinal adverse events",
             naming_quote=naming)
    return CitationGuard(SPEC, KB, {"E1": e})


def test_a_comparator_named_by_the_paper_survives():
    guard = _guard_for("Gastrointestinal adverse events were the most frequently reported events and were observed "
                       "most commonly in the liraglutide group.",
                       naming="Constipation was more frequently reported in the liraglutide group compared with "
                              "tirzepatide.")
    kept = guard.check("Gastrointestinal adverse events were observed more commonly with liraglutide than with "
                       "tirzepatide.", ["E1"])
    assert kept.kept, kept.reason


def test_the_near_miss_drug_alone_is_still_not_an_answer():
    guard = _guard_for("Semaglutide reduced liver fat in the semaglutide group.")
    assert not guard.check("Semaglutide reduced liver fat.", ["E1"]).kept


# --- text the parser damaged ---------------------------------------------------------------------------------


def test_a_literal_less_than_no_longer_eats_the_p_value_and_the_heading():  # E13, E56
    text = clean("(95% CI -6·72 to -2·70; p&lt;0·0001).&lt;h4&gt;Interpretation&lt;/h4&gt;Tirzepatide showed a "
                 "significant reduction.")
    assert "p<0·0001). Interpretation: Tirzepatide showed" in text


# --- someone else's claim ------------------------------------------------------------------------------------


def test_a_background_claim_is_not_a_finding():  # E32
    assert other_work_reason("Tirzepatide has been also reported to be beneficial in reducing liver fat content.",
                             "abstract")


# --- found in the re-run of the same question (29 September, same 177 papers) --------------------------------


def test_an_adjustment_list_is_not_the_endpoint():  # E41: weight loss "adjusted for ... hepatic steatosis"
    e = u("After adjustment for sex, age, baseline body mass index, and hepatic steatosis using ANCOVA, weight loss "
          "remained significantly different across treatment groups (F = 12.960, p = 0.001).")
    assert claims.endpoint_of(e).id == "body_size"
    assert claims.endpoint_of(u("Hepatic steatosis improved in the tirzepatide group.")).id == \
        "liver_steatosis_unspecified"


def test_no_agent_and_no_drug_named_is_not_the_drugs_result():  # E1, E77
    population = u("Compared with participants without T2D, those with T2D were significantly older (56.7 vs. 52.4 "
                   "years) and had higher mean HbA1c.", intervention=None, outcome="association")
    model = u("A significant increase in the intracellular lipid content in liver tissue from HFD rats was seen as "
              "compared to that of liver tissue extracted from the control rats.", intervention=None,
              outcome="association")
    plain = u("LFC decreased significantly at week 52 (p < 0.001).", intervention=None)
    assert not claims.attributable(population, SPEC, TZP_PAPER)
    assert not claims.attributable(model, SPEC, TZP_PAPER)
    assert claims.attributable(plain, SPEC, TZP_PAPER)
