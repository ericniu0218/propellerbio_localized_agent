"""Issue #8: filler sentences that pass every existing check because they
claim nothing wrong - they just state nothing at all.

A hand audit of the migraine and asthma benchmark reports found kept
sentences like these:

  S23  "The study used a nitroglycerin (NTG)-induced chronic migraine mouse
       model to investigate the effects of NF-kB inhibition." - a Methods
       sentence, no finding.
  S32  "Five women with a mean age of 31.8 years provided fully analyzable
       data in a study involving rimegepant." - demographics only, no
       finding.
  S57  a generic background sentence in the asthma report's prevention
       section, no finding.

None of these invent a number, a name or a direction, so the checks in
`report/guard.py` before issue #8 all passed them. What they have in common
is that the quote they cite is itself methods, demographics or background -
it never reports a result either. These tests build simplified versions of
exactly that shape (the real papers' quotes happen to also contain a stray
result word from an unrelated part of the abstract; these do not, so the new
check is exercised cleanly) plus one real result sentence, to pin down that
`CitationGuard.check` now drops the first three and keeps the fourth."""

from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import EvidenceUnit
from bedrock.litreview.report.guard import CitationGuard
from bedrock.litreview.report.layout import load_layout
from bedrock.litreview.stages.s0_question import parse_question

KB = load_knowledge()
SPEC = parse_question("If I want to prevent psoriasis, what should I target?")


def _unit(**kw) -> EvidenceUnit:
    base = dict(
        id="u1", paper_id="P1", question_id="q", target="IL-17A", target_raw="IL-17A", direction="block",
        modality="antibody", intervention=None, disease="psoriasis", disease_raw="psoriasis",
        evidence_type="in_vivo", species="other", species_raw="", outcome="improved", effect_detail="",
        n=None, combo_with=None, target_is_primary=True,
        quote="placeholder", quote_status="exact", section="methods", extracted_by_model="fake", prompt_version="1",
        provenance={"target": "quote"},
    )
    base.update(kw)
    return EvidenceUnit(**base)


def _guard(unit: EvidenceUnit, **kw) -> CitationGuard:
    return CitationGuard(SPEC, KB, {"E1": unit}, paper_labels={"P1": "Smith 2020 J Invest Dermatol"}, **kw)


def test_a_methods_sentence_is_dropped_in_a_findings_section():
    """S23's shape: the sentence describes the experimental setup, not what
    happened in it, and the quote it cites is the same Methods sentence.
    ("treated", not "induced": that word is itself in `_RESULT` - a paper
    inducing an effect is reporting one - and would defeat the point of this
    test by making the quote look result-bearing.)"""
    quote = "We used a mouse model of imiquimod-treated skin inflammation to study IL-17A blockade."
    sentence = "The study used a mouse model of imiquimod-treated skin inflammation to investigate IL-17A blockade."
    verdict = _guard(_unit(quote=quote)).check(sentence, ["E1"])
    assert not verdict.kept and verdict.reason == "cites no quote that reports a result"


def test_a_demographics_sentence_is_dropped_in_a_findings_section():
    """S32's shape, from the real hand audit: this exact enrolment sentence
    cites no result of any kind."""
    quote = "Five women with a mean age of 31.8 years (range 22-42 years) provided fully analyzable data."
    sentence = (
        "In a study of patients given guselkumab, five women with a mean age of 31.8 years "
        "provided fully analyzable data."
    )
    unit = _unit(quote=quote, intervention="guselkumab", provenance={"target": "quote", "intervention": "quote"})
    verdict = _guard(unit).check(sentence, ["E1"])
    assert not verdict.kept and verdict.reason == "cites no quote that reports a result"


def test_a_background_sentence_is_dropped_in_a_findings_section():
    """S57's shape: a generic scope statement in the prevention section,
    citing a quote that is itself only scope-setting, no result."""
    quote = "Keratinocyte barrier integrity is a determinant of chronic inflammatory skin disease."
    sentence = (
        "Identifying keratinocyte barrier integrity as a determinant of chronic inflammatory skin disease "
        "may guide future psoriasis prevention strategies."
    )
    verdict = _guard(_unit(quote=quote)).check(sentence, ["E1"])
    assert not verdict.kept and verdict.reason == "cites no quote that reports a result"


def test_a_real_result_sentence_is_kept():
    quote = "Anti-IL-23 antibody given before imiquimod challenge reduced ear thickness by 60% in C57BL/6 mice."
    unit = _unit(
        target="IL-23", target_raw="IL-23", intervention="anti-IL-23 antibody", species="mouse",
        species_raw="C57BL/6 mice", effect_detail="ear thickness reduced by 60%", quote=quote,
        provenance={"target": "quote", "intervention": "quote", "species": "quote"},
    )
    verdict = _guard(unit).check(
        "Anti-IL-23 antibody given before imiquimod challenge reduced ear thickness by 60% in C57BL/6 mice.", ["E1"]
    )
    assert verdict.kept, verdict.reason


def test_findings_false_lets_a_methods_sentence_through():
    """The escape hatch a future non-findings section would use: the same
    methods-only sentence and quote, but the guard built with
    `findings=False` does not require a result at all."""
    quote = "We used a mouse model of imiquimod-treated skin inflammation to study IL-17A blockade."
    sentence = "The study used a mouse model of imiquimod-treated skin inflammation to investigate IL-17A blockade."
    verdict = _guard(_unit(quote=quote), findings=False).check(sentence, ["E1"])
    assert verdict.kept, verdict.reason


def test_every_written_section_in_report_yaml_defaults_to_a_findings_section():
    """The client's report layout draws every written section from extracted
    findings (see the comment in report.yaml), so none of them opts out
    today - `findings: false` exists for a future section that is genuinely
    not about a result, not for any section shipped now."""
    for question_type in ("target_identification", "intervention_efficacy", "default"):
        for spec in load_layout(question_type):
            if spec.kind == "written":
                assert spec.findings, f"{question_type}:{spec.id} should default to a findings section"
