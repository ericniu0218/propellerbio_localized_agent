"""OPEN_ISSUES item 9: structured-abstract labels leak into displayed text.

Europe PMC keeps a structured abstract's own section headings glued onto the
sentence text: "Key Results: Among 45 participants ..." or "Background and
Purpose: Children with acute asthma exacerbations ...". Quote verification
widens a finding to its whole sentence (`extract/verify.py`
`sentence_bounds`), and with no period separating the label from the text
that follows it, the label comes along for the ride - so both the quote and
the context sentence a reader sees can open with a heading that was never
part of what the authors said.

`extract/verify.py` already had `abstract_labels()`/`ABSTRACT_LABEL`, built
for `conclusion_reason`, which only needs to know a label is *there*, not
remove it. `strip_leading_label()` is the removal: it trims the label (and a
lead-in like "Key " that isn't itself a labelled word - see its docstring)
off the front of a displayed span and slides `start` forward by exactly the
same number of characters, so `section_text[start:end] == text` - the
invariant `report/invariants.py`'s `_anchored()` enforces on every run -
keeps holding.

This is the real case: PMID EPMCPPR1286358 (montelukast, asthma corpus,
evals/litreview/hand_audit/qwen_asthma.md, S41/E233). The quote and context
sentence below are copied verbatim from that file.
"""

import json
from collections import Counter

from bedrock.litreview.extract.verify import strip_leading_label
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import ExtractionJob, Facet, Paper
from bedrock.litreview.report.invariants import check_units
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import extract_paper

KB = load_knowledge()
ASTHMA = parse_question("If I want to prevent asthma exacerbations, what should I target?")
FACETS = [Facet(id="clinical", title="Clinical", need="human evidence of an effect")]

# --- verbatim from evals/litreview/hand_audit/qwen_asthma.md (S41 / E233) ---------

LABELLED_QUOTE = (
    "Key Results: Among 45 participants randomized to montelukast and 44 to placebo, median [IQR] ages were "
    "6.6 [5.4, 10] and 6.7 [5.3, 9.9] years, with high-moderate pre-treatment exacerbation severity."
)
STRIPPED_QUOTE = (
    "Among 45 participants randomized to montelukast and 44 to placebo, median [IQR] ages were "
    "6.6 [5.4, 10] and 6.7 [5.3, 9.9] years, with high-moderate pre-treatment exacerbation severity."
)
LABELLED_CONTEXT = (
    "Background and Purpose: Children with acute asthma exacerbations are frequently hospitalized due to "
    "insufficient response to inhaled β-agonist and systemic corticosteroid potentially due to leukotriene "
    "mediated airway inflammation."
)
STRIPPED_CONTEXT = (
    "Children with acute asthma exacerbations are frequently hospitalized due to insufficient response to "
    "inhaled β-agonist and systemic corticosteroid potentially due to leukotriene mediated airway inflammation."
)

# The rest of the abstract, verbatim from the same file's "paper, around the
# quote" excerpt, reassembled around the two sentences above so a fuller,
# realistic structured abstract can be run through extraction end to end.
ABSTRACT = (
    LABELLED_CONTEXT + " "
    "Montelukast is a potent leukotriene receptor antagonist approved for chronic asthma treatment. "
    "We performed a randomized dose-escalation trial of high-dose oral montelukast to identify the dose of oral "
    "montelukast suitable for future efficacy studies. Experimental Approach: We performed a Phase 2, adaptive, "
    "double-masked randomized controlled trial comparing high-dose oral montelukast plus standard treatment "
    "versus standard alone in children with exacerbations moderate-to-severe exacerbations after initial inhaled "
    "albuterol treatment. Three sequential groups received escalating weight-based dose levels (2.0, 2.5, and 3.0 "
    "mg/kg) with maximum plasma concentration (Cmax) used as the pharmacokinetic target for dose escalation. "
    "We hypothesized that at least one dose level would achieve a target Cmax of 1,700 ng/ml in > 86% of "
    "dose-level participants. " + LABELLED_QUOTE + " "
    "The target Cmax was reached in 67%, 80%, and 90% of participants at doses of 2.0, 2.5, and 3.0 mg/kg, "
    "respectively. Adverse events were infrequent and mild. Conclusion and Implications: Among children with "
    "moderate or severe acute asthma exacerbations not responsive to initial inhaled albuterol, oral montelukast "
    "at a dose of 3.0 mg/kg reliably achieves a potentially therapeutic Cmax. These findings support dose "
    "selection for an adequately powered efficacy trial of montelukast in this population."
)


# --- strip_leading_label itself, direct on the two real sentences ----------------


def test_the_real_quotes_label_is_stripped_and_offsets_still_anchor():
    section_text = "leading filler. " + LABELLED_QUOTE + " trailing filler."
    start = section_text.index(LABELLED_QUOTE)
    end = start + len(LABELLED_QUOTE)
    assert section_text[start:end] == LABELLED_QUOTE  # the anchor before stripping

    new_text, new_start = strip_leading_label(LABELLED_QUOTE, start)

    assert new_text == STRIPPED_QUOTE
    assert not new_text.startswith("Key Results")
    assert new_start > start
    # end never moves; only start does, by exactly the characters removed.
    assert section_text[new_start:end] == new_text


def test_the_real_context_sentences_label_is_stripped_and_offsets_still_anchor():
    section_text = "leading filler. " + LABELLED_CONTEXT + " trailing filler."
    start = section_text.index(LABELLED_CONTEXT)
    end = start + len(LABELLED_CONTEXT)
    assert section_text[start:end] == LABELLED_CONTEXT

    new_text, new_start = strip_leading_label(LABELLED_CONTEXT, start)

    assert new_text == STRIPPED_CONTEXT
    assert not new_text.startswith("Background and Purpose")
    assert new_start > start
    assert section_text[new_start:end] == new_text


def test_text_with_no_label_is_unchanged():
    text = "Montelukast reduced asthma exacerbations significantly in this trial."
    assert strip_leading_label(text, 17) == (text, 17)


def test_a_label_not_at_the_front_is_left_alone():
    """A label appearing mid-sentence is not the sentence's own heading, and
    the words before it are real content, not a bare lead-in - nothing here
    is stripped."""
    text = "The patients received placebo before the trial. Conclusion: it worked well for most."
    assert strip_leading_label(text, 3) == (text, 3)


def test_the_glued_form_is_stripped():
    """As Europe PMC leaves it when the space and colon are lost: no filler
    word before it, so this is the `start == 0` case."""
    glued = "ConclusionDPL combined with BUD can quickly relieve the clinical symptoms."
    new_text, new_start = strip_leading_label(glued, 100)
    assert new_text == "DPL combined with BUD can quickly relieve the clinical symptoms."
    assert new_start == 100 + (len(glued) - len(new_text))


# --- the whole pipeline: extract_paper end to end, on the real abstract ----------


def _finding() -> dict:
    return {
        "quote": LABELLED_QUOTE, "target": "montelukast", "change": "", "direction": "block", "modality": "other",
        "intervention": "montelukast", "disease": "asthma", "evidence_type": "human_clinical", "species": "children",
        "effect": "", "outcome": "worsened", "setting": "treatment", "effect_detail": "", "n": 45,
        "combo_with": None, "target_is_primary": True, "facet": "clinical",
    }


def _paper() -> Paper:
    return Paper(id="EPMCPPR1286358", title="Pharmacokinetic-guided optimization of weight-based dosing for "
                 "high-dose oral montelukast in children with acute asthma exacerbations",
                 abstract=ABSTRACT, sections={"abstract": ABSTRACT})


def _extract_units():
    paper = _paper()
    reply = {"is_review": False, "findings": [_finding()]}
    client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
    job = ExtractionJob(paper_id=paper.id, sections=["abstract"], facets=["clinical"])
    units = extract_paper(paper, job, ASTHMA, FACETS, client, model="m", knowledge=KB, stats=Counter())
    return paper, units


def test_the_label_does_not_leak_into_the_stored_quote_or_context():
    paper, units = _extract_units()
    assert len(units) == 1
    u = units[0]

    assert u.quote == STRIPPED_QUOTE
    assert not u.quote.startswith("Key Results")

    assert u.context_quote is not None
    assert u.context_quote == STRIPPED_CONTEXT
    assert not u.context_quote.startswith("Background and Purpose")


def test_offsets_still_anchor_to_the_papers_own_text():
    paper, units = _extract_units()
    u = units[0]
    section_text = paper.sections["abstract"]

    assert section_text[u.quote_start:u.quote_end] == u.quote
    assert u.context_start is not None and u.context_end is not None
    assert section_text[u.context_start:u.context_end] == u.context_quote


def test_check_units_finds_no_violation():
    """The integrity check (`report/invariants.py`) re-proves every stored
    span against the paper's text at its recorded offsets. If the label had
    been stripped from the text but `start` had not been advanced to match,
    this would fail here on every run that touches a structured abstract."""
    paper, units = _extract_units()
    assert check_units(units, {paper.id: paper}, KB) == []
