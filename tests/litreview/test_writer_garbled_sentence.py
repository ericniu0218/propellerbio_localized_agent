"""OPEN_ISSUES item 9: a corrupted, duplicated-fragment sentence reaching the
reader. The real case is obesity S13 (evals/litreview/hand_audit/
qwen_obesity_after.md): the paper's quote is clean ("Hypothalamic Sik3 mRNA
levels were significantly increased ..."), but the writer model's drafted
sentence is "Hypoth, hypothalamic Sik3 mRNA levels were significantly
increased ..." - a duplicated, truncated stem of its own first word. Same
family as the "MedGemma wrote a bare 'A.' twice" (MIN_SENTENCE_CHARS) and
"small models loop" (REPEAT_OVERLAP) issues documented beside it in
s7_write.py: a formatting defect, not a citation failure, so it is dropped
silently and counted under its own `sentence_stats` key.

`_garbled_start` is deliberately narrow - anchored to the very start of the
sentence, not a generic "any word is a prefix of the next" scan - so the
second test here throws a battery of real, ordinary kept sentences from the
same hand-audit file at it and checks none of them trip it, including the
one real sentence that opens with a comma right after its first word
("However, fasting plasma levels ...", S29 of the same file) - the closest
ordinary shape to the bug, correctly let through because "however" is not a
prefix of "fasting"."""

import json
from types import SimpleNamespace

from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import EvidencePlan, EvidenceSummary, EvidenceUnit, Paper, QuestionEntity, QuestionSpec
from bedrock.litreview.report.layout import SectionSpec
from bedrock.litreview.stages.s7_write import WriteContext, _garbled_start, _method, write_section_text

# obesity S13, verbatim from evals/litreview/hand_audit/qwen_obesity_after.md:
# the paper's own quote (clean) and the writer's drafted sentence (garbled).
SIK3_QUOTE = (
    "Hypothalamic Sik3 mRNA levels were significantly increased after 10 weeks of high-fat diet (HFD) feeding in "
    "mice compared to their lean chow-fed counterparts (Fig. )."
)
GARBLED_SENTENCE = (
    "Hypoth, hypothalamic Sik3 mRNA levels were significantly increased after 10 weeks of high-fat diet (HFD) "
    "feeding in mice compared to their lean chow-fed counterparts."
)
# obesity S16 of the same file - an ordinary kept sentence, reused here as the
# "normal" sentence a section also drafts alongside the garbled one.
NORMAL_SENTENCE = "Administration of 15-keto-PGE2 improved glucose homeostasis and prevented diet-induced obesity in mice."

# 14 more real kept sentences from evals/litreview/hand_audit/qwen_obesity_after.md
# (every kept sentence in that file except the garbled S13 one above), used
# below purely to prove the detector leaves ordinary writing alone.
REAL_KEPT_SENTENCES = [
    "Human genetic studies have identified GPR75 loss-of-function variants to be strongly protective against "
    "obesity.",
    "TMEM135 gene expression was significantly decreased in abdominal subcutaneous white adipose tissue of female "
    "subjects with obesity compared to lean individuals.",
    "Patients with POMC defects upstream of MC4R show significantly more weight loss with Setmelanotide than "
    "MC4R deficient patients or obese controls.",
    "Human atrial tissue from individuals with a BMI greater than 30 kg/m² showed increased mRNA expression "
    "of NOX2 compared to lean individuals.",
    "In animal studies, shRNA sequences targeting Gpr75, sh517 and sh1109, reduced Gpr75 mRNA and protein levels "
    "in HEK293 cells by approximately 80% and 60%, respectively.",
    "In obese leptin knockout mice, REGN4461 normalized body weight, food intake, blood glucose, and insulin "
    "sensitivity.",
    "A coding variant of TMEM135, resulting in a glycine to glutamic acid replacement at position 369, is "
    "associated with increased BMI specifically in a sub-cohort of 13,000 Hispanic samples.",
    "High fat feeding in male mice did not elicit a significant body weight difference between TMEM135 knockout "
    "and control genotypes, presumably due to the already rapid diet-induced weight gain in male C57BL/6J mice.",
    "Histologic examination showed that TMEM135 overexpression results in markedly smaller adipocytes in the "
    "iWAT depot and a striking protection against obesity-associated whitening of BAT.",
    NORMAL_SENTENCE,
    "The study reported that 15-keto-PGE2 administration reduced body weight gain in mice fed a high-fat diet.",
    "The authors studied acetyl-CoA levels in the liver of ob/ob mice, finding that fasting levels of acetyl-CoA "
    "and gluconeogenic intermediates were reduced in ob/ob HKO mice compared with ob/ob F/F mice.",
    # The closest ordinary shape to the bug: opens with a single word, then a
    # comma, then another word - but "however" is not a prefix of "fasting".
    "However, fasting plasma levels of FFAs and β-hydroxybutyrate were higher and lower, respectively, in "
    "ob/ob HKO mice than in ob/ob F/F mice, suggesting impaired hepatic fatty acid oxidation.",
    "Administering BX471, a selective CCR1 inhibitor, to HFD-fed wild-type mice during HFD feeding reduced body "
    "weight, fat mass, and plasma triglycerides over 8 weeks.",
]


def test_the_real_garbled_sentence_trips_the_detector():
    assert _garbled_start(GARBLED_SENTENCE) is True


def test_ordinary_real_kept_sentences_do_not_trip_the_detector():
    for sentence in REAL_KEPT_SENTENCES:
        assert _garbled_start(sentence) is False, sentence


def test_ordinary_english_openings_do_not_trip_the_detector():
    # The generic "any word is a prefix of the next" shape the task warns
    # against - never anchored with a comma right at the start, so never
    # reached by this narrower check either.
    assert _garbled_start("An animal model was used to study diet-induced obesity in this cohort of mice.") is False
    assert _garbled_start("Over overexpressed genes were noted in the treated group of mice studied here.") is False


def _unit(uid: str, paper_id: str, target: str, quote: str) -> EvidenceUnit:
    return EvidenceUnit(
        id=uid, paper_id=paper_id, question_id="Q1",
        target=target, target_raw=target, direction="unclear", modality="other",
        disease="obesity", disease_raw="obesity", evidence_type="in_vivo",
        species="mice", species_raw="mice", outcome="association",
        setting="unclear", effect_detail="", n=None,
        combo_with=None, target_is_primary=True, facet="mechanism",
        quote=quote, quote_status="exact", section="results", quote_start=0, quote_end=len(quote),
        provenance={"target": "quote"}, extracted_by_model="fake", prompt_version="1",
    )


def _write_context(client: FakeLLMClient, units: list[EvidenceUnit]) -> WriteContext:
    spec = QuestionSpec(
        id="Q1", text="What drives obesity?", question_type="mechanism", goal="understand",
        entities=[QuestionEntity(text="obesity", kind="disease", canonical="obesity", exact_terms=["obesity"])],
    )
    short_ids = {u.id: f"E{i + 1}" for i, u in enumerate(units)}
    summary = EvidenceSummary(question_id="Q1", units=units, short_ids=short_ids)
    plan = EvidencePlan(question_id="Q1", facets=[], jobs=[])
    return WriteContext(
        spec=spec, facets=[], plan=plan, summary=summary, ranked=[],
        papers={"P1": Paper(id="P1", title="An obesity paper", abstract=SIK3_QUOTE + " " + NORMAL_SENTENCE)},
        gates={}, client=client, model="fake-model", knowledge=load_knowledge(),
    )


ANSWER_SPEC = SectionSpec(id="answer", title="Answer", kind="written", source="all")


def test_the_garbled_sentence_is_dropped_and_counted_while_the_normal_one_is_kept():
    """One model call drafts both sentences - the real garbled S13 one and a
    normal one - the same way one section's real call would. The garbled one
    is dropped silently: not in `sentences`, not counted as a citation-guard
    "dropped", counted only under the new `garbled` stat. The normal one
    passes through untouched."""
    sik3 = _unit("U1", "P1", "SIK3", SIK3_QUOTE)
    normal = _unit("U2", "P1", "15-keto-PGE2", NORMAL_SENTENCE)

    def responder(prompt: str, schema: dict | None) -> str:
        ids = schema["properties"]["sentences"]["items"]["properties"]["citations"]["items"]["enum"]
        return json.dumps({
            "sentences": [
                {"citations": [ids[0]], "text": GARBLED_SENTENCE},
                {"citations": [ids[1]], "text": NORMAL_SENTENCE},
            ]
        })

    client = FakeLLMClient(responder=responder, version="fake-1")
    ctx = _write_context(client=client, units=[sik3, normal])

    sentences, note = write_section_text(ctx, ANSWER_SPEC, ctx.summary.units)

    assert [s.text for s in sentences] == [NORMAL_SENTENCE]
    assert all(s.kept for s in sentences)
    assert ctx.sentence_stats["garbled"] == 1
    assert ctx.sentence_stats.get("dropped", 0) == 0, "a formatting defect is not a citation-guard drop"
    assert ctx.sentence_stats.get("too_short", 0) == 0
    assert ctx.sentence_stats.get("repeated", 0) == 0
    assert note is None


def test_written_line_reports_the_garbled_count():
    ctx = SimpleNamespace(
        gates={}, ranked=[], papers={}, plan=SimpleNamespace(jobs=[]), summary=SimpleNamespace(units=[]),
        extract_stats={}, sentence_stats={"kept": 5, "dropped": 1, "repeated": 2, "too_short": 1, "garbled": 3},
    )
    assert "3 removed as a corrupted, duplicated-fragment draft" in _method(ctx)
