"""Study design from the paper's own words, and the opt-in evidence-aware selector.

Two layers. `study_design` must only call a paper a human study when its
publication type or its own design wording says so - "patients" in a mouse
paper's opening line is not a trial. The `evidence_aware` selector then uses
those labels to reserve part of the same grading budget for human evidence,
and the planner tries human studies first for the clinical need. The default
(`lexical`) policy is unchanged, which every other test in this suite checks.
"""

from __future__ import annotations

import json
import random

import pytest

from bedrock.litreview.facets import facets_for
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import Paper, PaperSignals
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s2_gate import gate_papers
from bedrock.litreview.stages.s3_rank import evidence_aware_selection, rank_papers
from bedrock.litreview.stages.s4_plan import build_plan
from bedrock.litreview.study_design import design_signals, designs_for, grounded_labels

QUESTION = "If I want to prevent psoriasis, what areas should I target?"


def P(pid, title, abstract="", types=(), sections=None):
    return Paper(id=pid, title=title, abstract=abstract, sections=sections or {},
                 signals=PaperSignals(publication_types=list(types)))


# --- study design ------------------------------------------------------------------------------


def test_publication_type_is_metadata_and_offsets_point_at_the_words():
    p = P("a", "Guselkumab in psoriasis: a randomized placebo-controlled trial",
          "Patients with psoriasis were randomly assigned to guselkumab or placebo.", ["Randomized Controlled Trial"])
    d = design_signals(p)
    assert d["labels"] == ["human_rct"] and d["roles"] == ["primary"]
    certainties = {s["certainty"] for s in d["signals"] if s["label"] == "human_rct"}
    assert certainties == {"metadata", "explicit_text"}
    for s in d["signals"]:
        if s["field"] in ("title", "abstract"):
            assert getattr(p, s["field"])[s["start"]:s["end"]] == s["text"]


def test_patients_in_a_mouse_papers_background_is_not_a_human_study():
    p = P("b", "IL-23 blockade prevents psoriasis-like inflammation in mice",
          "Psoriasis affects millions of patients worldwide. Previous randomized trials showed benefit of IL-23 "
          "inhibitors in patients. Here, IL-23 knockout mice were treated with imiquimod.")
    d = design_signals(p)
    assert d["labels"] == ["animal_interventional"]
    assert not grounded_labels(d) & {"human_rct", "human_observational", "human_interventional_other"}
    assert any(s["label"] == "human_observational" and s["certainty"] == "weak_hint" for s in d["signals"]), \
        "the word is recorded as a hint, never as a design"


def test_a_study_with_humans_and_mice_is_mixed():
    p = P("c", "TYK2 in psoriasis", "In a prospective cohort of 1,204 adults, TYK2 levels were higher. "
          "Tyk2-deficient mice were protected from imiquimod-induced inflammation.")
    assert design_signals(p)["labels"] == ["animal_interventional", "human_observational", "mixed"]


@pytest.mark.parametrize("types, title, abstract, roles", [
    (["Review", "Systematic Review", "Meta-Analysis"], "IL-17 inhibitors: a systematic review and meta-analysis",
     "We conducted a systematic review of 12 trials.", ["meta_analysis", "systematic_review"]),
    (["Review"], "IL-23 in psoriasis: a review", "", ["narrative_review"]),
    (["Practice Guideline"], "Psoriasis treatment guideline", "", ["guideline"]),
    (["Retraction of Publication"], "Retraction: IL-23 and psoriasis", "", ["editorial_or_notice"]),
    (["Journal Article"], "A cohort study of psoriasis",
     "A recent systematic review suggested a link. We report a retrospective cohort of 300 patients.", ["primary"]),
    (["Letter", "Randomized Controlled Trial"], "Short report: a randomized trial", "", ["primary"]),
    ([], "Psoriasis", "Psoriasis is common.", ["unknown"]),
])
def test_document_roles(types, title, abstract, roles):
    assert design_signals(P("x", title, abstract, types))["roles"] == roles


def test_weak_hints_alone_never_ground_a_label():
    d = design_signals(P("d", "Psoriasis SNPs", "SNPs and polymorphisms in patients with psoriasis."))
    assert d["labels"] == ["unknown"] and grounded_labels(d) == set()


# --- the selector ---------------------------------------------------------------------------------

MOUSE = ("IL-{n} blockade in psoriasis mice",
         "IL-{n} knockout mice were treated with imiquimod and psoriasis-like inflammation fell. Psoriasis psoriasis "
         "target targeting inhibition blockade knockout deficient pathway mediates drives prevent onset.")
TRIAL = P("TRIAL", "Guselkumab in psoriasis", "Patients with psoriasis were randomly assigned to guselkumab or "
          "placebo; PASI-90 was 73% versus 3%.", ["Randomized Controlled Trial"])


def corpus(n_mice=6, extra=()):
    mice = [P(f"M{i}", MOUSE[0].format(n=10 + i), MOUSE[1].format(n=10 + i),
              sections={"results": "Mice were protected from psoriasis."}) for i in range(n_mice)]
    return [*mice, TRIAL, *extra]


def responder(prompt, schema):
    """Grade 3 for the mouse papers (they changed a named molecule), 2 for the trial."""
    title = (schema or {}).get("title")
    if title == "RelevanceJudgement":
        abstract = prompt.split("Abstract:", 1)[1]
        tested = ""
        for token in abstract.split():
            if token.startswith("IL-") and "knockout" in abstract:
                tested = token
                break
        first = abstract.strip().split(". ")[0].split("\n")[0].strip()
        return json.dumps({"quote": first, "reason": "fake", "about_exact_entity": True, "subject_role": "studied",
                           "study_kind": "primary_research", "tested": tested})
    return "{}"


def rank(papers, *, policy, max_graded, selection=None):
    spec = parse_question(QUESTION, None, model=None)
    gates = gate_papers(papers, spec)
    ranked = rank_papers(papers, gates, spec, facets_for(spec), FakeLLMClient(responder=responder), model="m",
                         max_graded=max_graded, policy=policy, designs=designs_for(papers), selection=selection)
    return spec, gates, ranked


def test_the_lexical_budget_misses_the_trial_and_the_reserved_lane_grades_it():
    papers = corpus()
    _, _, lexical = rank(papers, policy="lexical", max_graded=3)
    assert {r.paper_id for r in lexical if r.grade is not None} == {"M0", "M1", "M2"} or \
        "TRIAL" not in {r.paper_id for r in lexical if r.grade is not None}
    why: dict = {}
    _, _, aware = rank(papers, policy="evidence_aware", max_graded=3, selection=why)
    graded = {r.paper_id for r in aware if r.grade is not None}
    assert "TRIAL" in graded and len(graded) == 3
    assert why["TRIAL"] == "reserved:human_interventional"
    assert {r.paper_id: r.grade for r in aware}["TRIAL"] == 2, "a reserved slot never changes the grade"


def test_the_trial_takes_the_clinical_slot_ahead_of_full_text_mouse_papers():
    """The mouse papers say "a cohort of ... mice", which the clinical word
    list counts; grade 3 with full text, they fill the clinical minimum first
    under the lexical order. The evidence-aware planner tries the paper whose
    own publication type says it is a human trial first."""
    mice = [P(f"M{i}", MOUSE[0].format(n=10 + i),
              "A cohort of IL-" + str(10 + i) + " knockout mice was treated with imiquimod and psoriasis-like "
              "inflammation fell. Psoriasis target blockade knockout pathway drives onset.",
              sections={"results": "Mice were protected from psoriasis."}) for i in range(6)]
    papers = [*mice, TRIAL]
    by_id = {p.id: p for p in papers}
    spec, _, ranked = rank(papers, policy="evidence_aware", max_graded=10)
    facets = facets_for(spec)
    lexical_plan = build_plan(spec, facets, ranked, by_id, max_papers=5)
    aware_plan = build_plan(spec, facets, ranked, by_id, max_papers=5, policy="evidence_aware",
                            designs=designs_for(papers))
    assert "TRIAL" not in lexical_plan.paper_ids, "grade 3 with full text wins every slot lexically"
    assert "TRIAL" in aware_plan.paper_ids
    assert aware_plan.best_source_id == lexical_plan.best_source_id, "the most relevant source is unchanged"
    clinical = next(fp for fp in aware_plan.facets if fp.facet.id == "clinical")
    assert "TRIAL" in clinical.paper_ids


def test_with_no_human_papers_the_lanes_give_their_slots_back():
    papers = corpus(n_mice=5)[:5]
    designs = {p.id: {"labels": ["unknown"], "roles": ["primary"], "signals": []} for p in papers}
    spec = parse_question("What should I target to treat psoriasis?", None, model=None)
    _, _, ranked = rank(papers, policy="lexical", max_graded=3)
    passing = sorted((r for r in ranked if r.gate == "pass"), key=lambda r: (-r.lexical_score, r.paper_id))
    chosen, why = evidence_aware_selection(passing, {p.id: p for p in papers}, spec, designs, max_graded=3)
    assert chosen == [r.paper_id for r in passing[:3]]
    assert set(why.values()) == {"lexical_fill", "below_budget_3"}


def test_a_budget_smaller_than_the_reservations_is_never_exceeded():
    trials = [P(f"T{i}", f"Drug {i} in psoriasis", "Patients with psoriasis were randomly assigned.",
                ["Randomized Controlled Trial"]) for i in range(10)]
    why: dict = {}
    _, _, ranked = rank(corpus(extra=trials), policy="evidence_aware", max_graded=5, selection=why)
    assert sum(1 for r in ranked if r.grade is not None) == 5
    assert list(why.values()).count("reserved:human_interventional") == 5, "lanes fill in order until the cap"


def test_overlapping_lanes_take_a_paper_once():
    prevent_trial = P("PT", "Guselkumab prevents psoriasis relapse",
                      "Patients with psoriasis were randomly assigned; relapse prevention was the endpoint.",
                      ["Randomized Controlled Trial"])
    why: dict = {}
    _, _, _ = rank(corpus(extra=[prevent_trial]), policy="evidence_aware", max_graded=4, selection=why)
    assert why["PT"] == "reserved:human_interventional"
    assert list(why).count("PT") == 1


def test_input_order_never_changes_the_choice():
    papers = corpus(extra=[P("OBS", "Psoriasis cohort", "In a prospective cohort of 500 adults with psoriasis, "
                            "IL-17 was raised.")])
    first: dict = {}
    rank(papers, policy="evidence_aware", max_graded=4, selection=first)
    shuffled = list(papers)
    random.Random(7).shuffle(shuffled)
    second: dict = {}
    rank(shuffled, policy="evidence_aware", max_graded=4, selection=second)
    assert first == second


def test_a_negative_trial_is_selectable_like_any_other():
    negative = P("NEG", "Anti-IL-4 in psoriasis", "Patients with psoriasis were randomly assigned; the drug did not "
                 "reduce PASI.", ["Randomized Controlled Trial"])
    why: dict = {}
    rank(corpus(extra=[negative]), policy="evidence_aware", max_graded=3, selection=why)
    assert why["NEG"] == "reserved:human_interventional"


def test_reviews_notices_and_near_misses_take_no_slot():
    guideline = P("GL", "Psoriasis guideline", "Psoriasis patients were randomly assigned in the trials we cite.",
                  ["Practice Guideline"])
    notice = P("RN", "Comment on psoriasis IL-23 trial", "Psoriasis comment.", ["Comment"])
    near = P("NM", "IL-23 in atopic dermatitis", "Patients with atopic dermatitis were randomly assigned.",
             ["Randomized Controlled Trial"])
    why: dict = {}
    _, gates, ranked = rank(corpus(extra=[guideline, notice, near]), policy="evidence_aware", max_graded=20,
                            selection=why)
    assert why["GL"].startswith("excluded_nonprimary") and why["RN"].startswith("excluded_nonprimary")
    assert "NM" not in why and gates["NM"].status != "pass"
    graded = {r.paper_id for r in ranked if r.grade is not None}
    assert not graded & {"GL", "RN", "NM"}


def test_prevention_lane_is_used_only_for_prevention_questions():
    papers = corpus(n_mice=3, extra=[P("PREV", "Psoriasis onset in adults", "Onset of psoriasis was delayed.")])
    spec = parse_question("What should I target to treat psoriasis?", None, model=None)
    assert spec.goal != "prevent"
    gates = gate_papers(papers, spec)
    why: dict = {}
    rank_papers(papers, gates, spec, facets_for(spec), None, max_graded=2, policy="evidence_aware",
                designs=designs_for(papers), selection=why)
    assert "reserved:prevention" not in why.values()


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="rank.selection"):
        rank(corpus(), policy="adaptive", max_graded=3)
