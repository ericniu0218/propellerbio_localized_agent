"""Issue #3: aims, hypotheses, predictions and methods sentences are not
findings - the hand audit's asthma S32, obesity S56, asthma S57 and migraine
S23 cases (evals/litreview/hand_audit/qwen_*.md), each quoted verbatim below.

`extract/verify.py` `not_a_result_reason` decides; Stage 5 drops such a quote
and counts it, and the citation guard drops a report sentence that is itself
a methods sentence, or rests only on such quotes. The negatives are just as
important - a false drop throws away real evidence - and most of them are
real sentences from the same benchmark papers that open with an aim and then
report the result, or hedge a result with "suggest" or "may".
"""

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from bedrock.litreview.extract.verify import not_a_result_reason
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import EvidenceUnit, ExtractionJob, Facet, Paper
from bedrock.litreview.report.guard import CitationGuard
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import extract_paper
from bedrock.litreview.stages.s7_write import _method

KB = load_knowledge()

# --- the evidence cases, verbatim ---------------------------------------------

# asthma S32 (E117): an aim, cited for "AhR regulates allergen-induced ROS
# generation ...". "allergen-induced" is a result word used to name the model.
S32_QUOTE = ("In this study, we investigated the role of the AhR in regulating allergen-induced ROS generation "
             "and cellular senescence in bronchial epithelial cells and its contribution to allergic airway "
             "inflammation.")
S32_SENTENCE = ("AhR regulates allergen-induced ROS generation and cellular senescence in bronchial epithelial "
                "cells, contributing to allergic airway inflammation.")
# obesity S56 (E174): a prediction from mouse work, cited as prevention evidence.
S56_QUOTE = ("Taken together, our findings highlight a therapeutic potential for obese/diabetic postmenopausal "
             "patients; cold exposure is therefore predicted to metabolically benefit those patients.")
S56_SENTENCE = "Cold exposure is predicted to metabolically benefit obese/diabetic postmenopausal patients."
# asthma S57 (E260): background; quote and sentence are the same words.
S57_QUOTE = ("Identifying key pathophysiological determinants in airway epithelium enables novel therapeutics to "
             "prevent asthma recurrence.")
# migraine S23 (E46): the *sentence* is the methods one; its quote is a real result.
S23_SENTENCE = ("The study used a nitroglycerin (NTG)-induced chronic migraine mouse model to investigate the "
                "effects of NF-κB inhibition.")
S23_QUOTE = ("targeted blockade of the NF-κB cascade with PDTC recapitulated the protective behavioral and "
             "biochemical effects of FKBP5 deficiency.")


@pytest.mark.parametrize("text, kind", [
    (S32_QUOTE, "aim"),
    (S56_QUOTE, "hope"),
    (S57_QUOTE, "hope"),
    (S23_SENTENCE, "aim"),
    # More real quotes from the stored benchmark reports that were kept as
    # evidence: methods sentences and open questions.
    ("To assess the functional role of JUN signaling, HDM-challenged Bmal1 -ko mice were treated with the JUN "
     "inhibitor SP600125 (A).", "aim"),
    ("This study was designed to confirm the hypothesis that the pharmacokinetic profile of the "
     "anti-calcitonin gene-related peptide mAb, eptinezumab, is similar in healthy Chinese individuals to that "
     "of healthy non-Asian individuals and non-Asian patients with migraine.", "aim"),
    ("T cells contribute critically to obesity-induced adipose inflammation and insulin resistance, yet the "
     "co-stimulatory signals that govern their activation in adipose tissue remain unclear.", "hope"),
    ("The mechanism of GSDMD in asthma remains unclear.", "hope"),
])
def test_aims_methods_and_predictions_are_not_results(text, kind):
    assert not_a_result_reason(text) == f"not_a_result:{kind}"


@pytest.mark.parametrize("text", [
    S23_QUOTE,
    # An aim, then the result in a clause of its own.
    "To investigate whether IL-23 drives disease, we treated mice with an anti-IL-23 antibody, which reduced "
    "ear thickness by 40%.",
    # Real: the aim sentence is followed by the result sentence in one quote.
    "Lastly, to investigate whether NOX2 increase is associated with increased mRNA PITX2 expression, we treated "
    "BSA-hiPSC-aCMs with 25 μM of hydrogen peroxide (H 2 O 2 ) for 5 days. H 2 O 2 -hiPSC-aCMs showed a similar "
    "increase in PITX2 mRNA expression compared with PA-hiPSC-aCMs, suggesting that PITX2 is indeed upregulated "
    "by ROS ().",
    "To determine the exact molecular basis of this intercellular communication, we analyzed the expression of "
    "classical epithelium-derived alarmins. Quantitative PCR (qPCR) validation confirmed that both HDM and direct "
    "lactate stimulation robustly up-regulated the transcription of key alarmins, most notably TSLP , in BEAS-2B "
    "cells.",
    "Given that c-Myc emerged as a downstream target of AhR, we hypothesized that it contributes to "
    "allergen-induced cellular senescence. HBECs were exposed to CRE (50 μg/mL, 24 h) with or without the c-Myc "
    "inhibitor EN4 (50 μM). Immunostaining revealed that EN4 effectively suppressed CRE-induced c-Myc expression "
    "(A,B).",
    # Plain "suggest", "may" and "potential" hedge a result; they are not cues.
    "These findings suggest that CCL8 blockade effectively mitigates lung inflammatory infiltration and type 2 "
    "immune responses in ITLN1 deficiency asthma, highlighting CCL8 as a potential therapeutic target for "
    "modulating type 2 inflammation in this condition.",
    "The protective effect of semaglutide may be mediated by reduced hypothalamic inflammation.",
    "The causal link between the diacylglycerol DAG (18:1_18:2) and increased migraine risk suggests a potential "
    "dual-hit mechanism.",
    # A result and a hope in one sentence: the hope is set aside, the result stays.
    "Anti-IL-17A treatment reduced epidermal thickness and may offer a new preventive strategy.",
    # A passive methods verb followed by what was found.
    "Serum IL-6 was examined and was significantly elevated in obese mice.",
    # A conclusion that names what the study found keeps it.
    "Thus, Gq proteins play a critical role in airway remodeling, hence pharmacological inhibition of Gq "
    "signaling represents a promising strategy for the treatment of chronic asthma.",
])
def test_results_are_kept_even_beside_an_aim_or_a_hedge(text):
    assert not_a_result_reason(text) is None


# --- Stage 5 ------------------------------------------------------------------

ASTHMA = parse_question("If I want to prevent asthma, what should I target?")
FACETS = [Facet(id="causal", title="Causal", need="what changing the target does")]
# The paper around S32's quote, shortened: the aim sentence, then a result.
S32_RESULTS = (
    S32_QUOTE + " Using both p16 ΔScgb1a1 and AhR ΔScgb1a1 mice along with human bronchial epithelial cells, we "
    "demonstrated that AhR deficiency exacerbated ROS generation and inflammation in allergic airway inflammation."
)


def _extract(quote: str):
    paper = Paper(id="P1", title="Epithelial AhR suppresses allergen-induced oxidative stress", abstract="AhR.",
                  sections={"abstract": "AhR.", "results": S32_RESULTS})
    reply = {"is_review": False, "findings": [{
        "quote": quote, "target": "AhR", "change": "", "direction": "unclear", "modality": "other",
        "intervention": None, "disease": "allergic airway inflammation", "evidence_type": "in_vivo",
        "species": "mice", "effect": "", "outcome": "association", "setting": "unclear",
        "effect_detail": "", "n": None, "combo_with": None, "target_is_primary": True, "facet": "causal",
    }]}
    client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
    job = ExtractionJob(paper_id="P1", sections=["results"], facets=["causal"])
    stats: Counter = Counter()
    units = extract_paper(paper, job, ASTHMA, FACETS, client, model="m", knowledge=KB, stats=stats)
    return units, stats


def test_stage5_drops_the_asthma_s32_aim_and_counts_it():
    units, stats = _extract(S32_QUOTE)
    assert units == []
    # It verified - the paper wrote it - and it is the paper's own work.
    assert stats["not_a_result"] == 1
    assert stats["quote_failed"] == 0 and stats.get("other_work", 0) == 0


def test_stage5_keeps_the_result_sentence_from_the_same_paper():
    units, stats = _extract("we demonstrated that AhR deficiency exacerbated ROS generation and inflammation in "
                            "allergic airway inflammation.")
    assert len(units) == 1
    assert stats.get("not_a_result", 0) == 0


def test_the_drop_is_shown_in_how_this_answer_was_assembled():
    ctx = SimpleNamespace(
        gates={}, ranked=[], papers={}, plan=SimpleNamespace(jobs=[]), summary=SimpleNamespace(units=[]),
        extract_stats={"not_a_result": 3}, sentence_stats={},
    )
    assert "3 discarded because their quote states an aim, a method, a prediction or an open question" \
        in _method(ctx)


# --- the citation guard ---------------------------------------------------------


def _unit(quote: str, **kw) -> EvidenceUnit:
    base = dict(
        id="u1", paper_id="P1", question_id="q", target="AhR", target_raw="AhR", direction="unclear",
        modality="other", intervention=None, disease="asthma", disease_raw="asthma", evidence_type="in_vivo",
        species="other", species_raw="", outcome="association", effect_detail="", n=None, combo_with=None,
        target_is_primary=True, quote=quote, quote_status="exact", section="results",
        extracted_by_model="fake", prompt_version="1", provenance={"target": "quote"},
    )
    base.update(kw)
    return EvidenceUnit(**base)


def _check(spec, sentence: str, *units: EvidenceUnit, findings: bool = True):
    table = {f"E{i}": u for i, u in enumerate(units, 1)}
    guard = CitationGuard(spec, KB, table, paper_labels={"P1": "Smith 2025 J"}, findings=findings)
    return guard.check(sentence, list(table))


def test_guard_drops_the_asthma_s32_sentence_citing_only_an_aim():
    verdict = _check(ASTHMA, S32_SENTENCE, _unit(S32_QUOTE))
    assert not verdict.kept
    assert verdict.reason == "cites only quotes that state an aim, method or prediction, not a result"


def test_guard_drops_the_obesity_s56_prediction():
    spec = parse_question("What are the most promising treatment targets for obesity?")
    unit = _unit(S56_QUOTE, target="cold exposure", target_raw="cold exposure", disease="obesity",
                 disease_raw="obese")
    verdict = _check(spec, S56_SENTENCE, unit)
    assert not verdict.kept
    assert verdict.reason == "cites only quotes that state an aim, method or prediction, not a result"


def test_guard_drops_the_asthma_s57_background_sentence():
    unit = _unit(S57_QUOTE, target="airway epithelium", target_raw="airway epithelium")
    verdict = _check(ASTHMA, S57_QUOTE, unit)
    assert not verdict.kept
    assert verdict.reason == "cites only quotes that state an aim, method or prediction, not a result"


def test_guard_drops_the_migraine_s23_methods_sentence_despite_its_real_result_quote():
    spec = parse_question("If I want to prevent migraine, what should I target?")
    # E46 as stored, with the Methods sentence that gave its subjects.
    unit = _unit(S23_QUOTE, target="NF-κB", target_raw="NF-κB", disease="migraine", disease_raw="migraine",
                 context_quote="Methods We utilized a nitroglycerin (NTG)-induced chronic migraine mouse model.",
                 context_section="abstract", context_start=0, context_end=10)
    verdict = _check(spec, S23_SENTENCE, unit)
    assert not verdict.kept
    assert verdict.reason == "is a methods or aims sentence, not a finding"


def test_guard_keeps_a_result_sentence_when_one_cited_quote_reports_the_result():
    """An aim quote beside a result quote is harmless: the sentence has a
    result under it."""
    result = _unit("Using both p16 ΔScgb1a1 and AhR ΔScgb1a1 mice, we demonstrated that AhR deficiency "
                   "exacerbated ROS generation and inflammation.", id="u2")
    verdict = _check(ASTHMA, "AhR deficiency exacerbated ROS generation and inflammation.", _unit(S32_QUOTE), result)
    assert verdict.kept, verdict.reason


def test_guard_keeps_a_result_sentence_that_adds_a_hedged_interpretation():
    """Only aim and method cues make a sentence itself "not a finding": a
    result followed by "suggesting a potential role" is still a result
    (asthma S23, graded faithful in the audit, has this shape)."""
    quote = ("This study showed higher expression of ERs, and the modulation of MAIT cell function by estrogen in "
             "asthmatics compared to healthy individuals.")
    unit = _unit(quote, target="ERs", target_raw="ERs")
    verdict = _check(ASTHMA, "Higher expression of ERs was found in asthmatics compared to healthy individuals, "
                             "suggesting a potential role for ERs in asthma.", unit)
    assert verdict.kept, verdict.reason
