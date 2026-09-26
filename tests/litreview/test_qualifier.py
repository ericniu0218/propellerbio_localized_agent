"""Issue #1: a qualifier in the paper's next sentence is not lost.

The obesity hand audit's S6 (evals/litreview/hand_audit/qwen_obesity.md)
restated a quote faithfully while the paper's very next sentence said the
result "did not reach statistical significance (p = 0.09)". Stage 5 now stores
that sentence beside the quote (`qualifier_quote`, found by
`extract/verify.py` `qualifier_after`), the reader and the writer see it, and
the citation guard drops a sentence that restates the result without the
limit. The paragraphs below are the papers' own text, verbatim from the
corpus. The negatives matter as much: a next sentence that is a separate
finding must not be stored as a qualifier, and a sentence reporting the
qualifier's own significant clause must be kept.
"""

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from bedrock.litreview.extract.verify import is_qualifier, not_significant_clause, qualifier_after
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import EvidenceUnit, ExtractionJob, Facet, Paper
from bedrock.litreview.report.guard import CitationGuard, drops_not_significant
from bedrock.litreview.report.invariants import check_units
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import extract_paper
from bedrock.litreview.stages.s7_write import _evidence_lines
from bedrock.litreview.stages.s8_render import to_markdown

KB = load_knowledge()
OBESITY = parse_question("If I want to treat obesity, which targets should I pursue?")
MIGRAINE = parse_question("If I want to prevent migraine attacks, what areas should I target?")

# --- the evidence cases, verbatim ---------------------------------------------

# obesity S6: PMID29031731, results. E212's quote is the second sentence.
S6_QUOTE = ("In the MC4R mutation carriers, placebo subtracted group mean differences in weight loss (dose "
            "0.01 mg/kg/24 h) of ∼0.6 kg/week were observed ().")
S6_NEXT = ("Although this result did not reach statistical significance (p = 0.09) when compared to placebo, "
           "weight loss from baseline was significantly reduced (p < 0.0001).")
S6_PARAGRAPH = (
    "Although the total number of individuals in each cohort of this Phase 1 trial was small, significant weight "
    "loss was observed in all Obese Control cohorts receiving ≥0.01 mg/kg/24 h of Setmelanotide compared to "
    "placebo (). " + S6_QUOTE + " " + S6_NEXT + " Weight loss was accompanied by reductions in waist "
    "circumference ()."
)
S6_SENTENCE = ("In the MC4R mutation carriers, placebo subtracted group mean differences in weight loss (dose "
               "0.01 mg/kg/24 h) of ∼0.6 kg/week were observed.")

# migraine S9 and the answer section: PMID40542538, results. E78's quote has
# its own "numerically favored ... p = 0.099", and the next sentence says it
# again.
S9_QUOTE = ("The response rates for the first secondary efficacy endpoint (i.e., participants with ≥50% reduction "
            "from the OP in moderate to severe MMDs in the last 4 weeks of the DBT phase) numerically favored "
            "rimegepant versus placebo and were 41.7% (95% CI = 35.5%–47.9%) in the rimegepant group and 34.4% "
            "(95% CI = 28.5%–40.4%) in the placebo group ( p = 0.099; Table ).")
S9_NEXT = ("Because the findings for the first secondary efficacy endpoint were not statistically significant, "
           "formal hypothesis testing did not proceed.")
S9_CONTEXT = ("A total of 496 participants were randomized to rimegepant ( n = 247) or placebo ( n = 249), and all "
              "were treated with study drug (Figure ).")
# Kept in the stored report and graded faithful: "numerically" keeps the limit.
S9_SENTENCE = ("For the first secondary efficacy endpoint, rimegepant numerically favored over placebo, with 41.7% "
               "of participants achieving ≥50% reduction from the OP in moderate to severe MMDs, compared to 34.4% "
               "in the placebo group.")
# Kept in the stored report's answer section: "was higher", limit gone.
ANSWER_SENTENCE = ("The response rate for a 50% reduction in moderate to severe migraine days was higher in patients "
                   "treated with rimegepant than in those receiving placebo.")


# --- which next sentence is a qualifier -----------------------------------------


def test_the_s6_next_sentence_is_found_and_anchored():
    start = S6_PARAGRAPH.index(S6_QUOTE)
    found = qualifier_after(S6_PARAGRAPH, start, start + len(S6_QUOTE))
    assert found is not None and found.text == S6_NEXT
    assert S6_PARAGRAPH[found.start:found.end] == S6_NEXT


def test_the_s6_qualifier_says_not_significant_and_splits_off_its_significant_clause():
    verdict, rest = not_significant_clause(S6_NEXT, S6_QUOTE)
    assert "did not reach statistical significance (p = 0.09)" in verdict
    assert rest.startswith("weight loss from baseline was significantly reduced")


def test_the_migraine_next_sentence_points_back_at_the_same_endpoint():
    assert not_significant_clause(S9_NEXT, S9_QUOTE) is not None


def test_a_next_sentence_that_is_a_new_result_is_not_stored():
    """The sentence after "Although this result ..." in the same paragraph
    reports something else; nothing is stored."""
    start = S6_PARAGRAPH.index(S6_NEXT)
    assert qualifier_after(S6_PARAGRAPH, start, start + len(S6_NEXT)) is None


@pytest.mark.parametrize("quote, sentence", [
    # obesity E216 -> its next sentence: "not statistically significant", but
    # about its own comparison ("this" points inside the sentence).
    ("We did not identify significant genotype–phenotype correlations for most phenotypes after adjusting for "
     "age, gender, and BMI (data not shown).",
     "Of interest, we found that children heterozygous for complete LOF variants consumed slightly more at an ad "
     "libitum test meal (mean 121.1 kJ/kg lean mass ± SEM 12.3) than those with partial LOF mutations (mean 82.2 "
     "kJ/kg of lean mass ± SEM 10.4), although this was not statistically significant (p = 0.07, F)."),
    # asthma E253 -> "non-significant", but about another drug.
    ("H2L5186303 treatment before OVA sensitization or challenge significantly suppressed the elevated IL-4 and "
     "IL-13 levels ().",
     "In addition, GRI977143 treatment before OVA sensitization or challenge suppressed the cytokine levels but to "
     "a non-significant level ()."),
    # obesity E7 -> points back ("This effect"), but narrows nothing.
    ("we found that children heterozygous for complete LOF variants consumed slightly more at an ad libitum test "
     "meal (mean 121.1 kJ/kg lean mass ± SEM 12.3) than those with partial LOF mutations (mean 82.2 kJ/kg of lean "
     "mass ± SEM 10.4), although this was not statistically significant (p = 0.07, F).",
     "This effect was more pronounced when analyzing the data for children under 10 years of age (mean 154.7 "
     "kJ/kg lean mass ± SEM 13.2 in complete LOF, 96.6 ± 15.2 in partial LOF; p = 0.02, G)."),
])
def test_real_next_sentences_that_do_not_qualify_the_quote(quote, sentence):
    assert not is_qualifier(sentence, quote)


def test_a_verdict_on_another_measure_is_not_a_verdict_on_the_quote():
    assert not_significant_clause("The reduction in IL-6 was not significant.",
                                  "TNF levels were reduced by 40% in treated mice.") is None


def test_a_new_paragraph_is_never_a_qualifier():
    text = S6_QUOTE + "\n\n" + S6_NEXT
    assert qualifier_after(text, 0, len(S6_QUOTE)) is None


# --- Stage 5 stores it ----------------------------------------------------------


def test_stage5_stores_the_s6_qualifier_beside_the_quote():
    paper = Paper(id="P1", title="Evaluation of a melanocortin-4 receptor (MC4R) agonist (Setmelanotide) in MC4R "
                  "deficiency.", abstract="MC4R.", sections={"abstract": "MC4R.", "results": S6_PARAGRAPH})
    reply = {"is_review": False, "findings": [{
        "quote": S6_QUOTE, "target": "MC4R", "change": "", "direction": "unclear", "modality": "other",
        "intervention": None, "disease": "obesity", "evidence_type": "human_clinical", "species": "",
        "effect": "", "outcome": "association", "setting": "unclear", "effect_detail": "", "n": None,
        "combo_with": None, "target_is_primary": True, "facet": "causal",
    }]}
    client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
    facets = [Facet(id="causal", title="Causal", need="what changing the target does")]
    job = ExtractionJob(paper_id="P1", sections=["results"], facets=["causal"])
    units = extract_paper(paper, job, OBESITY, facets, client, model="m", knowledge=KB, stats=Counter())
    assert len(units) == 1
    u = units[0]
    assert u.qualifier_quote == S6_NEXT
    assert S6_PARAGRAPH[u.qualifier_start:u.qualifier_end] == S6_NEXT
    assert check_units([u], {"P1": paper}, KB) == []


# --- the guard ------------------------------------------------------------------


def _unit(quote: str, qualifier: str | None, **kw) -> EvidenceUnit:
    base = dict(
        id="u1", paper_id="P1", question_id="q", target="MC4R", target_raw="MC4R", direction="unclear",
        modality="other", intervention=None, disease="obesity", disease_raw="obesity", evidence_type="human_clinical",
        species="other", species_raw="", outcome="association", effect_detail="", n=None, combo_with=None,
        target_is_primary=True, quote=quote, quote_status="exact", section="results",
        extracted_by_model="fake", prompt_version="1", provenance={"target": "quote"},
        qualifier_quote=qualifier, qualifier_start=0 if qualifier else None,
        qualifier_end=len(qualifier) if qualifier else None,
    )
    base.update(kw)
    return EvidenceUnit(**base)


def _e78(qualifier: str | None = S9_NEXT) -> EvidenceUnit:
    return _unit(S9_QUOTE, qualifier, id="u78", target="MMDs", target_raw="MMDs", intervention="rimegepant",
                 disease="migraine", disease_raw="migraine", species_raw="patients", n=496, outcome="improved",
                 direction="block", context_quote=S9_CONTEXT, context_section="results", context_start=0,
                 context_end=len(S9_CONTEXT),
                 provenance={"target": "quote", "intervention": "quote", "disease_raw": "chunk", "context": "chunk",
                             "species": "quote", "n": "chunk", "effect_detail": "none"})


def _check(spec, sentence: str, *units: EvidenceUnit):
    table = {f"E{i}": u for i, u in enumerate(units, 1)}
    return CitationGuard(spec, KB, table, paper_labels={"P1": "Collet 2017 Mol Metab"}).check(sentence, list(table))


def test_guard_drops_the_s6_sentence_that_leaves_out_not_significant():
    verdict = _check(OBESITY, S6_SENTENCE, _unit(S6_QUOTE, S6_NEXT))
    assert not verdict.kept
    assert verdict.reason == "drops the qualifier the paper states: not significant"


def test_without_a_stored_qualifier_the_rule_does_not_fire():
    """A report extracted before qualifiers were stored re-checks the same."""
    assert drops_not_significant(S6_SENTENCE, [_unit(S6_QUOTE, None)]) is None


# E212's quote is a result, but check 6's vocabulary (`RESULT_WORDS`) does not
# read "group mean differences in weight loss ... were observed" as one, so
# every sentence citing only E212 is dropped by check 6 whatever this rule
# says. The keep cases below are therefore asserted on this rule itself.


def test_the_rule_keeps_the_s6_sentence_when_it_carries_the_limit():
    sentence = ("In the MC4R mutation carriers, placebo subtracted group mean differences in weight loss of ∼0.6 "
                "kg/week did not reach statistical significance (p = 0.09).")
    assert drops_not_significant(sentence, [_unit(S6_QUOTE, S6_NEXT)]) is None


def test_the_rule_keeps_the_s6_sentence_when_it_gives_the_qualifiers_p_value():
    sentence = ("In the MC4R mutation carriers, placebo subtracted group mean differences in weight loss of ∼0.6 "
                "kg/week were observed (p = 0.09).")
    assert drops_not_significant(sentence, [_unit(S6_QUOTE, S6_NEXT)]) is None


def test_the_rule_keeps_a_sentence_reporting_the_qualifiers_own_significant_clause():
    """"weight loss from baseline was significantly reduced (p < 0.0001)" is
    the paper's own significant result; the sentence is not the one the
    qualifier limits."""
    for sentence in ("In the MC4R mutation carriers, weight loss from baseline was significantly reduced.",
                     "In the MC4R mutation carriers, weight loss was significantly reduced (p < 0.0001)."):
        assert drops_not_significant(sentence, [_unit(S6_QUOTE, S6_NEXT)]) is None, sentence


def test_the_qualifier_is_citable_paper_text():
    """The paper's own next sentence, shown beside the quote, is text a
    sentence citing the unit may rest on."""
    from bedrock.litreview.report.guard import source_text, unit_text

    u = _unit(S6_QUOTE, S6_NEXT)
    assert S6_NEXT in source_text(u) and S6_NEXT in unit_text(u)
    table = {"E1": _e78()}
    guard = CitationGuard(MIGRAINE, KB, table, paper_labels={"P1": "Sakai 2025"})
    sentence = ("Response rates for the first secondary efficacy endpoint were not statistically significant for "
                "rimegepant versus placebo, so formal hypothesis testing did not proceed.")
    assert guard.check(sentence, ["E1"]).kept


def test_guard_keeps_the_migraine_s9_sentence_that_says_numerically():
    verdict = _check(MIGRAINE, S9_SENTENCE, _e78())
    assert verdict.kept, verdict.reason


def test_guard_drops_the_migraine_answer_sentence_that_says_higher():
    verdict = _check(MIGRAINE, ANSWER_SENTENCE, _e78())
    assert not verdict.kept
    assert verdict.reason == "drops the qualifier the paper states: not significant"
    assert _check(MIGRAINE, ANSWER_SENTENCE, _e78(qualifier=None)).kept


def test_guard_keeps_the_sentence_when_another_cited_quote_reports_a_result():
    """The rule drops only when the qualified unit is the sentence's only
    support: the other half may rest on the Obese Control result."""
    controls = _unit("Although the total number of individuals in each cohort of this Phase 1 trial was small, "
                     "significant weight loss was observed in all Obese Control cohorts receiving ≥0.01 mg/kg/24 h "
                     "of Setmelanotide compared to placebo ().", None, id="u2")
    verdict = _check(OBESITY, S6_SENTENCE, _unit(S6_QUOTE, S6_NEXT), controls)
    assert verdict.kept, verdict.reason


def test_a_contrast_that_says_nothing_of_significance_never_drops_a_sentence():
    """obesity E106: stored (it opens "In contrast"), shown and citable, but
    it is about Sik1 and Sik2, not a verdict on the quote."""
    quote = ("Consistently, we observed similar increases in hypothalamic Sik3 mRNA levels in 12-week-old genetically "
             "induced leptin receptor-deficient ( db/db ) mice when compared to their lean ( db/+ ) counterparts "
             "(Fig. ).")
    nxt = "In contrast, both hypothalamic Sik1 and Sik2 levels remained unchanged across both obesogenic mouse models (Fig. H, )."
    assert is_qualifier(nxt, quote) and not_significant_clause(nxt, quote) is None
    sentence = ("Hypothalamic Sik3 mRNA levels were increased in 12-week-old genetically induced leptin "
                "receptor-deficient (db/db) mice compared to their lean (db/+) counterparts.")
    unit = _unit(quote, nxt, target="SIK3", target_raw="Sik3", evidence_type="in_vivo", species_raw="mice",
                 provenance={"target": "quote", "species": "quote"})
    assert _check(OBESITY, sentence, unit).kept


# --- shown to the reader and the writer, and stored compatibly -------------------


def test_the_writer_sees_the_qualifier_labelled_as_one():
    u = _unit(S6_QUOTE, S6_NEXT)
    ctx = SimpleNamespace(summary=SimpleNamespace(short_ids={u.id: "E1"}), labels={"P1": "Collet 2017"})
    lines = _evidence_lines([u], ctx)
    assert "Qualifier (the paper's next sentence" in lines and S6_NEXT in lines


def test_the_report_shows_the_qualifier_beside_the_quote():
    from bedrock.litreview.models import Report

    u = _unit(S6_QUOTE, S6_NEXT)
    report = Report(question_id="q", question="q", sections=[], evidence=[u], short_ids={u.id: "E1"})
    assert f'qualifier (next sentence): "{S6_NEXT}"' in to_markdown(report)


def test_a_unit_without_a_qualifier_serialises_as_before():
    """Stored runs are verified by re-serialising their report and comparing
    bytes, so a unit with no qualifier must not grow new keys."""
    assert "qualifier" not in _unit(S6_QUOTE, None).model_dump_json()
    u = _unit(S6_QUOTE, S6_NEXT)
    assert EvidenceUnit.model_validate_json(u.model_dump_json()) == u


def test_a_qualifier_that_is_not_the_papers_text_is_caught():
    paper = Paper(id="P1", title="t", abstract="a", sections={"results": S6_PARAGRAPH})
    start = S6_PARAGRAPH.index(S6_QUOTE)
    u = _unit(S6_QUOTE, S6_NEXT.replace("did not reach", "reached"), quote_start=start,
              quote_end=start + len(S6_QUOTE))
    assert any("qualifying sentence" in v for v in check_units([u], {"P1": paper}, KB))
