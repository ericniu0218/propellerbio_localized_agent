"""Issue #7: the same sentence appearing in two report sections.

`write_section_text` only ever sees one section's own slice of evidence, so
when the same finding is the best evidence for two different sections - the
"answer" and that finding's own "targets:<name>" subsection, most often - two
separate model calls draft the same sentence, and nothing before this test
stopped both copies from being kept. The within-section repeat check in
`s7_write.py` already had the right rule (fuzzy word overlap, not just an
exact string match); it just did not see across a section boundary.

Built directly from the hand-audit evidence: `qwen_obesity.md` keeps the
OX40L sentence as both S4 (the "answer" section) and S23 ("targets:OX40L"),
citing the same finding word for word."""

import json

from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import EvidencePlan, EvidenceSummary, EvidenceUnit, Paper, QuestionEntity, QuestionSpec
from bedrock.litreview.report.layout import SectionSpec
from bedrock.litreview.stages.s7_write import WriteContext, write_section_text

# The obesity hand audit's own sentence, S4 and S23 both.
OX40L_QUOTE = (
    "Adipocytes from obese individuals exhibited a 2.5- to 4-fold increase "
    "in OX40L mRNA levels compared with lean controls."
)


def _unit() -> EvidenceUnit:
    return EvidenceUnit(
        id="U1", paper_id="P1", question_id="Q1",
        target="OX40L", target_raw="OX40L", direction="activate", modality="other",
        disease="obesity", disease_raw="obesity", evidence_type="human_clinical",
        species="human", species_raw="obese individuals", outcome="association",
        setting="unclear", effect_detail="2.5- to 4-fold increase in OX40L mRNA", n=None,
        combo_with=None, target_is_primary=True, facet="mechanism",
        quote=OX40L_QUOTE, quote_status="exact", section="results", quote_start=0, quote_end=len(OX40L_QUOTE),
        provenance={"target": "quote"}, extracted_by_model="fake", prompt_version="1",
    )


def _write_context(client: FakeLLMClient) -> WriteContext:
    spec = QuestionSpec(
        id="Q1", text="What drives obesity?", question_type="mechanism", goal="understand",
        entities=[QuestionEntity(text="obesity", kind="disease", canonical="obesity", exact_terms=["obesity"])],
    )
    summary = EvidenceSummary(question_id="Q1", units=[_unit()], short_ids={"U1": "E1"})
    plan = EvidencePlan(question_id="Q1", facets=[], jobs=[])
    return WriteContext(
        spec=spec, facets=[], plan=plan, summary=summary, ranked=[],
        papers={"P1": Paper(id="P1", title="An obesity paper", abstract=OX40L_QUOTE)},
        gates={}, client=client, model="fake-model", knowledge=load_knowledge(),
    )


def _client_always_drafts(text: str) -> FakeLLMClient:
    """A model that writes the same one sentence for every section it is
    asked to write, whatever evidence it is shown - the same shape of bug the
    hand audit found, where two calls independently reached for the same
    strongest quote."""

    def responder(prompt: str, schema: dict | None) -> str:
        ids = schema["properties"]["sentences"]["items"]["properties"]["citations"]["items"]["enum"]
        return json.dumps({"sentences": [{"citations": [ids[0]], "text": text}]})

    return FakeLLMClient(responder=responder, version="fake-1")


ANSWER_SPEC = SectionSpec(id="answer", title="Answer", kind="written", source="all")
TARGETS_SPEC = SectionSpec(id="targets", title="Candidate targets", kind="written", source="per_target")


def test_a_sentence_kept_in_one_section_is_dropped_from_a_later_one():
    """Two sections, one shared `already_kept` list (what `write_report`
    threads through every section it writes): the second section's identical
    draft is dropped, the same way a same-call repeat is."""
    ctx = _write_context(_client_always_drafts(OX40L_QUOTE))
    already_kept: list[set[str]] = []

    answer_sentences, _ = write_section_text(ctx, ANSWER_SPEC, ctx.summary.units, already_kept=already_kept)
    assert [s.text for s in answer_sentences if s.kept] == [OX40L_QUOTE], "the first section keeps it"

    target_sentences, note = write_section_text(
        ctx, TARGETS_SPEC, ctx.summary.units, subsection="OX40L", already_kept=already_kept
    )
    assert target_sentences == [], "the repeat never becomes a shown sentence, kept or dropped"
    assert note == "Everything drafted here is already stated in an earlier section."
    assert ctx.sentence_stats["repeated"] == 1
    assert ctx.sentence_stats["kept"] == 1, "the sentence is kept exactly once across the two sections"


def test_a_section_left_with_only_repeats_says_so_instead_of_rendering_empty():
    """Follow-up found in review: when every sentence a section drafts is a
    repeat of something kept earlier, `sentences` is `[]` and, before this
    note, so was the note - `s8_render.py` then prints the section's heading
    with nothing under it, and a reader cannot tell "nothing new was found"
    from "the model produced nothing at all". The note distinguishes it. It
    must not fire for the *other* way a section ends up with no sentences at
    all — the model returning nothing, or a call failing outright — since
    those already say so via their own early return, before `already_kept`
    is ever consulted."""
    ctx = _write_context(_client_always_drafts(OX40L_QUOTE))
    already_kept: list[set[str]] = []
    write_section_text(ctx, ANSWER_SPEC, ctx.summary.units, already_kept=already_kept)

    _, note = write_section_text(ctx, TARGETS_SPEC, ctx.summary.units, subsection="OX40L", already_kept=already_kept)
    assert note == "Everything drafted here is already stated in an earlier section."

    # The pre-existing empty-section messages are untouched: no evidence at
    # all, and no model available, both return before `already_kept` matters.
    no_evidence_note = write_section_text(ctx, ANSWER_SPEC, [], already_kept=already_kept)[1]
    assert no_evidence_note == "No evidence for this section in the papers read."
    ctx.client = None
    no_model_note = write_section_text(ctx, ANSWER_SPEC, ctx.summary.units, already_kept=already_kept)[1]
    assert no_model_note == "Not written: no model available."


def test_without_the_shared_list_the_repeat_would_have_been_kept_twice():
    """Sanity check that the fixture actually reproduces the bug: calling
    `write_section_text` the old way, with no cross-section memory, keeps the
    same sentence in both sections - this is issue #7."""
    ctx = _write_context(_client_always_drafts(OX40L_QUOTE))

    answer_sentences, _ = write_section_text(ctx, ANSWER_SPEC, ctx.summary.units)
    target_sentences, _ = write_section_text(ctx, TARGETS_SPEC, ctx.summary.units, subsection="OX40L")

    assert [s.text for s in answer_sentences if s.kept] == [OX40L_QUOTE]
    assert [s.text for s in target_sentences if s.kept] == [OX40L_QUOTE], "the bug: the same sentence twice"


def test_near_duplicate_sentences_are_caught_too():
    """The repeat check is fuzzy word overlap (`REPEAT_OVERLAP`), not an
    exact string match, so a later section rewording the same finding is
    caught the same way a same-call near-duplicate already is."""
    ctx = _write_context(_client_always_drafts(OX40L_QUOTE))
    already_kept: list[set[str]] = []
    write_section_text(ctx, ANSWER_SPEC, ctx.summary.units, already_kept=already_kept)

    reworded = _client_always_drafts(
        "Adipocytes from obese individuals exhibited a 2.5- to 4-fold increase in OX40L mRNA levels, not lean ones."
    )
    ctx.client = reworded
    target_sentences, _ = write_section_text(
        ctx, TARGETS_SPEC, ctx.summary.units, subsection="OX40L", already_kept=already_kept
    )
    assert target_sentences == [], "a near-duplicate wording of an already-kept sentence is still a repeat"
