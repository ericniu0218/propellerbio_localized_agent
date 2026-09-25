"""Relevance: the model answers narrow questions about an abstract; code
decides the grade and which information needs a paper can supply.

Each test pins down a failure seen on the MedGemma 4B benchmark run, where the
model graded 39 of 40 papers 3 and ticked every need on 36 of them.
"""

from bedrock.litreview.facets import facets_for, paper_facets
from bedrock.litreview.llm import strict_schema
from bedrock.litreview.models import Paper, RelevanceJudgement
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s3_rank import check_tested, grade_from

PSO = parse_question("If I want to prevent psoriasis, what should I target?")
RSV = parse_question("What should I target to prevent RSV infection in infants?")


def judgement(**kw) -> RelevanceJudgement:
    base = dict(
        quote="q", reason="r", about_exact_entity=True, subject_role="studied",
        study_kind="primary_research", tested="",
    )
    base.update(kw)
    return RelevanceJudgement(**base)


def paper(abstract: str, title: str = "A study") -> Paper:
    return Paper(id="X", title=title, abstract=abstract)


def test_the_model_is_not_asked_for_a_grade_or_a_list_of_needs():
    schema = strict_schema(RelevanceJudgement.model_json_schema())
    assert "grade" not in schema["properties"] and "facets" not in schema["properties"]
    assert schema["required"][:2] == ["quote", "reason"], "evidence before decisions"


def test_the_grade_follows_the_written_rule():
    assert grade_from(judgement(tested="IL-23"), quote_ok=True, tested="IL-23") == 3
    assert grade_from(judgement(), quote_ok=True, tested=None) == 2
    assert grade_from(judgement(study_kind="review"), quote_ok=True, tested="IL-23") == 1
    assert grade_from(judgement(), quote_ok=False, tested="IL-23") == 1, "an unverifiable quote caps at 1"
    assert grade_from(judgement(about_exact_entity=False), quote_ok=True, tested="IL-23") == 0


def test_a_risk_factor_study_is_not_an_answer():
    """Breast cancer risk in people with psoriasis is about breast cancer."""
    assert grade_from(judgement(subject_role="exposure"), quote_ok=True, tested=None) == 1
    assert grade_from(judgement(subject_role="passing"), quote_ok=True, tested="IL-23") == 1


def test_the_tested_name_must_be_in_the_abstract_and_not_the_subject():
    p = paper("Anti-IL-23 antibody reduced psoriasis-like inflammation in mice.")
    assert check_tested("IL-23", p, PSO) == "IL-23"
    assert check_tested("IL-17A", p, PSO) is None, "not in the abstract"
    assert check_tested("psoriasis", p, PSO) is None, "the subject is not something a paper changed"


def test_a_viral_protein_is_a_target_but_the_virus_is_not():
    p = paper("Antibodies against the RSV F protein prevented RSV infection in infants.")
    assert check_tested("RSV F protein", p, RSV) == "RSV F protein"
    assert check_tested("RSV", p, RSV) is None
    assert check_tested("RSV infection", p, RSV) is None


def test_needs_are_assigned_by_code_and_causal_needs_a_change():
    facets = facets_for(PSO)
    p = paper("IL-23 knockout mice were protected from psoriasis onset; adverse events were not studied.")
    assert "causal" in paper_facets(p, facets), "words alone, when no model graded the paper"
    assert "causal" not in paper_facets(p, facets, tested=False), "the model found nothing changed"
    assert {"prevention", "safety"} <= set(paper_facets(p, facets, tested=True))
    assert "clinical" not in paper_facets(p, facets, tested=True), "a mouse study is not clinical evidence"
