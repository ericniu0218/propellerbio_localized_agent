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
    differed = u("Between-group differences were statistically significant for both groups (TZP + LEKT versus "
                 "TZP + LCD) in CAP and LSM.", outcome="association")
    assert claims.contrast_of(differed, SPEC, TZP_PAPER) == "co_intervention"
    # What each arm did from its own baseline is still the drug's within-arm
    # change - shared-treatment studies do not disappear (scientist's review, 1 Oct).
    diets = u("Both groups showed significant reductions in body weight (TZP + LEKT, p = 0.0289; TZP + LCD, "
              "p = 0.0278), with no significant intergroup difference (p = 0.665).")
    assert claims.contrast_of(diets, SPEC, TZP_PAPER) == "within_arm"
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



# --- the scientist's review of the re-run (1 October) ---------------------------------------------------------

ACTIVE_ARMS = Paper(id="p6", title="Real-World Effectiveness and Safety of Tirzepatide, Semaglutide, and Liraglutide",
                    abstract="Adults received liraglutide, semaglutide, or tirzepatide for 36 weeks.", signals=PaperSignals())
DIET_ARMS = Paper(id="p8", title="Tirzepatide Plus Low-Energy Ketogenic Versus Low-Calorie Diet",
                  abstract="Patients on tirzepatide (TZP) were randomised to a ketogenic (LEKT) or low-calorie diet (LCD).",
                  signals=PaperSignals())


def test_arms_on_other_drugs_are_an_active_comparison_not_a_co_intervention():  # [6] vs [8]
    quote = ("At week 36, significant within-group reductions in AST, ALT, and GGT levels were observed across all "
             "treatment groups (all p < 0.01). However, no significant between-group differences were detected.")
    enzymes = u(quote, outcome="association", intervention=None)
    assert claims.contrast_of(enzymes, SPEC, ACTIVE_ARMS) != "co_intervention"
    assert claims.contrast_of(u("Both groups showed no significant between-group difference in CAP (TZP + LEKT "
                                "versus TZP + LCD).", outcome="association"), SPEC, DIET_ARMS) == "co_intervention"


@pytest.mark.parametrize("quote, reason", [
    ("A total of 276 participants were included in this analysis.", "not_a_result"),  # E5
    ("This single-center, retrospective, real-world observational study was conducted at a private internal "
     "medicine clinic in Istanbul between September 2023 and September 2024.", "not_a_result"),  # E55
    ("Data on its hepatic effects in people without diabetes remain limited.", "background"),  # E32
    ("The mean levels of the different analytes are seen in .", "dangling"),  # E82
    ("The expression level of these markers was not significantly different in the other intervention groups.",
     "unresolved"),  # E81
    ("The estimated treatment difference versus insulin degludec was -4·71% (95% CI -6·72 to -2·70; "
     "pInterpretationTirzepatide showed a significant reduction.", "damaged"),  # E13
    ("Treatment with the dual GLP⍰1/GIP receptor agonist tirzepatide ameliorated hepatic steatosis.", "damaged"),
    ("TZP use was associated with a reduced risk of the composite outcome (HR, 0.66; 95% CI, 0.58–0.42).",
     "anomaly"),  # E38
])
def test_a_quote_that_is_not_a_usable_result_is_excluded_with_its_reason(quote, reason):
    a = claims.assess(u(quote), SPEC, TZP_PAPER)
    assert a.status == claims.EXCLUDED and reason in a.reasons, a


def test_a_plain_result_is_counted_and_a_subgroup_is_context():
    counted = claims.assess(u("Tirzepatide reduced liver fat content by 8.09 percentage points versus 3.38 with "
                              "insulin degludec at week 52."), SPEC, TZP_PAPER)
    assert counted.status == claims.COUNTED and counted.reasons == ()
    subgroup = claims.assess(u("There were no statistically significant differences between any of the genotype "
                               "subgroups (CC vs. CG).", outcome="no_effect"), SPEC, TZP_PAPER)  # E8
    assert subgroup.status == claims.CONTEXT and subgroup.reasons == ("subgroup",)


def test_regardless_of_a_subgroup_is_the_drugs_result_not_a_subgroup_comparison():  # E6
    e6 = u("Approximately 30%–40% of tirzepatide-treated participants with baseline LFC ≥ 6% achieved LFC < 6% at "
           "Week 52 regardless of the presence of the G allele.")
    assert claims.contrast_of(e6, SPEC, TZP_PAPER) != "subgroup"


def test_a_label_the_paper_contradicts_is_corrected_not_dropped():  # E28 (mouse study, labelled human), E69
    mouse = u("TZP treatment was associated with reduced abundance of HFHFr-induced proteins.",
              evidence_type="human_clinical", species_raw="HFHFr-fed mice")
    a = claims.assess(mouse, SPEC, TZP_PAPER, tier="animal")
    assert a.evidence_type == "in_vivo" and "design_corrected" in a.reasons
    rats = claims.assess(u("Metabolic analytes decreased in the intervention groups compared to sham.",
                           evidence_type="in_vitro", species_raw="rats"), SPEC, TZP_PAPER, tier="animal")
    assert rats.evidence_type == "in_vivo"


def _ctx(**kw):
    from types import SimpleNamespace

    base = dict(spec=SPEC, papers={}, studies={}, status={}, summary=SimpleNamespace(units=[], short_ids={}))
    base.update(kw)
    return SimpleNamespace(**base)


def test_causal_wording_is_removed_only_when_every_cited_study_is_non_randomised():  # E56, "led to"
    from types import SimpleNamespace

    from bedrock.litreview.stages.s7_write import _causal_from_observational

    cohort = u("Tirzepatide treatment improved liver steatosis.", paper_id="obs")
    trial = u("Tirzepatide reduced LFC versus insulin degludec.", paper_id="rct")
    ctx = _ctx(studies={"obs": SimpleNamespace(tier="observational"), "rct": SimpleNamespace(tier="rct")})
    led = "In patients with T2DM and MASLD, tirzepatide treatment led to significant improvements in steatosis."
    assert _causal_from_observational(led, [cohort], ctx)
    assert _causal_from_observational(led, [trial], ctx) is None
    assert _causal_from_observational("Tirzepatide was associated with improvements in steatosis.", [cohort], ctx) is None


def test_adverse_events_are_reported_as_occurrence_and_comparison_never_as_no_effect():  # E85
    from bedrock.litreview.report import synthesis

    e85 = u("Gastrointestinal adverse events, including nausea, were the most frequently reported events across "
            "treatment groups.", paper_id="p6", outcome="no_effect")
    row = synthesis.SynthesisRow("Adverse events and tolerability", [e85], {"no_effect": {"p6"}}, {"p6": "observational"},
                                 {"no_effect": {"p6"}}, {}, set(), "Non-randomised human studies only", "1 study",
                                 endpoint="adverse_events")
    text = synthesis.render_synthesis([row], {"p6": 6})
    assert "no significant effect" not in text
    assert "reported in 1 study [6]" in text and "no comparison of rates" in text


def test_the_answer_says_which_of_its_studies_are_the_population_asked_about():
    from bedrock.litreview.report import synthesis

    obese = Paper(id="a", title="Tirzepatide in adults with obesity", abstract="x", signals=PaperSignals())
    t2d = Paper(id="b", title="Tirzepatide in type 2 diabetes", abstract="BMI of at least 25.", signals=PaperSignals())
    note = synthesis.cited_population_note(_ctx(papers={"a": obese, "b": t2d}), {"a", "b"}, {"a": 3, "b": 1})
    assert note and "[3] names obesity" in note and "[1] does not" in note
    assert synthesis.cited_population_note(_ctx(papers={"a": obese}), {"a"}, {"a": 3}) is None


def test_an_unread_paper_about_a_harm_is_a_lead_not_a_finding():
    from types import SimpleNamespace

    from bedrock.litreview.stages.s7_write import _unread_safety_leads

    case = Paper(id="c", title="A Rare Case of Tirzepatide-Induced Hepatotoxicity", abstract="", signals=PaperSignals())
    read = Paper(id="r", title="Tirzepatide hepatotoxicity in a cohort", abstract="", signals=PaperSignals())
    other = Paper(id="o", title="Semaglutide-induced pancreatitis", abstract="", signals=PaperSignals())
    ctx = _ctx(papers={p.id: p for p in (case, read, other)},
               ranked=[SimpleNamespace(paper_id=x) for x in ("c", "r", "o")],
               plan=SimpleNamespace(paper_ids=["r"]))
    leads = _unread_safety_leads(ctx)
    assert len(leads) == 1 and "Rare Case of Tirzepatide-Induced Hepatotoxicity" in leads[0]


def test_a_report_stored_before_statuses_existed_still_re_serialises_to_its_own_bytes(tmp_path):
    """`verify-run` re-serialises every stored report and compares bytes; a new
    field must not appear in an old report."""
    from bedrock.litreview.models import EvidenceStatus, Report
    from bedrock.litreview.stages.s8_render import to_json

    old = Report(question_id="q", question="q", sections=[])
    body = to_json(old)
    assert "evidence_status" not in body
    assert to_json(Report.model_validate_json(body)) == body
    new = old.model_copy(update={"evidence_status": {"u1": EvidenceStatus(status="excluded", reasons=["x"])}})
    assert to_json(Report.model_validate_json(to_json(new))) == to_json(new)
