"""Other people's results are not this paper's own — the audit's asthma S9
case (evals/litreview/hand_audit/qwen_asthma.md), wired end to end through
Stage 5.

A birth-cohort finding from this paper's own Introduction, its reference
number stripped by Europe PMC down to the bracket that held it, was reported
in a benchmark run as if this paper (about MAIT cells and estrogen) had run
that birth cohort. The parser also mis-filed the Introduction paragraph as
"results", because nothing else in the section was labelled that
(parsing/sections.py `_Collector.result`) — these tests prove the
citation-marker check catches the finding even though the section name says
"results", and that the drop is counted where the report's "How this answer
was assembled" section reads it from.
"""

import json
from collections import Counter

from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import ExtractionJob, Facet, Paper
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import extract_paper

KB = load_knowledge()
SPEC = parse_question("If I want to prevent asthma, what should I target?")
FACETS = [Facet(id="causal", title="Causal", need="what changing the target does")]

# The "results" section as the real parser produced it: a lead-in sentence
# citing earlier work, the birth-cohort sentence quoted in the audit (its
# reference number stripped to "[]"), and a genuine result of this paper's
# own, so the section is not uniformly bad.
S9_SECTION = (
    "Previous studies have shown that MAIT cells may modulate asthmatic responses [].\n\n"
    "In a birth cohort study, MAIT cell frequency at 1 year of age was associated with a decreased risk of "
    "asthma by age 7, suggesting that increased MAIT cells early in life could be protective, potentially "
    "reflecting immune programing to T helper 1 (Th1) responses [].\n\n"
    "MAIT cell frequency in peripheral blood was significantly lower in children with asthma than in "
    "healthy controls at age 7."
)


def _extract(quote: str):
    paper = Paper(id="P1", title="Estrogen and MAIT cells in asthma", abstract="MAIT cells and asthma.",
                  sections={"abstract": "MAIT cells and asthma.", "results": S9_SECTION})
    reply = {"is_review": False, "findings": [{
        "quote": quote, "target": "MAIT cells", "change": "", "direction": "unclear", "modality": "other",
        "intervention": None, "disease": "asthma", "evidence_type": "human_clinical", "species": "patients",
        "effect": "decreased risk", "outcome": "improved", "setting": "prevention",
        "effect_detail": "", "n": None, "combo_with": None, "target_is_primary": True, "facet": "causal",
    }]}
    client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
    job = ExtractionJob(paper_id="P1", sections=["results"], facets=["causal"])
    stats: Counter = Counter()
    units = extract_paper(paper, job, SPEC, FACETS, client, model="m", knowledge=KB, stats=stats)
    return units, stats


def test_the_asthma_s9_finding_is_dropped_even_though_its_section_says_results():
    quote = ("In a birth cohort study, MAIT cell frequency at 1 year of age was associated with a decreased risk "
             "of asthma by age 7, suggesting that increased MAIT cells early in life could be protective, "
             "potentially reflecting immune programing to T helper 1 (Th1) responses [].")
    units, stats = _extract(quote)
    assert units == []
    # Dropped for reporting someone else's work, not for failing to verify —
    # the sentence really is in the paper, word for word.
    assert stats["other_work"] == 1
    assert stats["quote_failed"] == 0


def test_a_genuine_result_from_the_same_section_is_kept():
    """The section itself is not radioactive: a clean sentence from it with
    no citation marker or lead-in still comes through."""
    quote = ("MAIT cell frequency in peripheral blood was significantly lower in children with asthma than in "
             "healthy controls at age 7.")
    units, stats = _extract(quote)
    assert len(units) == 1
    assert stats.get("other_work", 0) == 0
