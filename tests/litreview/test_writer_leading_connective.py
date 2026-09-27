"""OPEN_ISSUES item 3, leftover part (a): strip a leading "However," from a
lifted sentence. ("Moreover," was in scope too and was ruled out by
measurement - see the KNOWN GAP block at the foot of this file.) The real
case is obesity S29
(evals/litreview/hand_audit/qwen_obesity_after.md): the writer paraphrased
the paper's own qualifier sentence, connective included -

    "However, fasting plasma levels of FFAs and β-hydroxybutyrate were
    higher and lower, respectively, in ob/ob HKO mice than in ob/ob F/F
    mice, suggesting impaired hepatic fatty acid oxidation."

The writer is asked for standalone report sentences, never connected
discourse, so a leading "However" is always a leftover connective
from a source sentence the model paraphrased, never something the writer
meant relative to a preceding sentence it cannot see - safe to strip
unconditionally, but only *after* the citation guard has judged the
sentence's real words (never before, or the guard would be judging a
sentence the model never actually wrote)."""

import json

from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import EvidencePlan, EvidenceSummary, EvidenceUnit, Paper, QuestionEntity, QuestionSpec
from bedrock.litreview.report import guard as guard_module
from bedrock.litreview.report.invariants import check_sentences
from bedrock.litreview.report.layout import SectionSpec
from bedrock.litreview.stages.s7_write import WriteContext, _strip_leading_connective, write_report, write_section_text

# obesity S29, verbatim from evals/litreview/hand_audit/qwen_obesity_after.md.
S29_SENTENCE = (
    "However, fasting plasma levels of FFAs and β-hydroxybutyrate were higher and lower, respectively, in "
    "ob/ob HKO mice than in ob/ob F/F mice, suggesting impaired hepatic fatty acid oxidation."
)
S29_STRIPPED = (
    "Fasting plasma levels of FFAs and β-hydroxybutyrate were higher and lower, respectively, in ob/ob HKO "
    "mice than in ob/ob F/F mice, suggesting impaired hepatic fatty acid oxidation."
)


def test_the_real_s29_sentence_has_its_leading_however_stripped():
    assert _strip_leading_connective(S29_SENTENCE) == S29_STRIPPED


def test_a_leading_moreover_is_deliberately_left_alone():
    # "Moreover," looks symmetrical with "However," and is NOT treated the
    # same, on purpose: see the KNOWN GAP block at the foot of this file for
    # the measured verdict flip that ruled it out. This test exists so that
    # anyone who "completes" the symmetry has to read that block first.
    assert (_strip_leading_connective("Moreover, weight loss was observed in the treated group.")
            == "Moreover, weight loss was observed in the treated group.")


def test_a_sentence_with_no_leading_connective_is_untouched():
    ordinary = "Administration of 15-keto-PGE2 improved glucose homeostasis and prevented diet-induced obesity in mice."
    assert _strip_leading_connective(ordinary) == ordinary


def _unit(uid: str, target: str, quote: str) -> EvidenceUnit:
    return EvidenceUnit(
        id=uid, paper_id="P1", question_id="Q1",
        target=target, target_raw=target, direction="unclear", modality="other",
        disease="obesity", disease_raw="obesity", evidence_type="in_vivo",
        species="mice", species_raw="ob/ob HKO mice", outcome="no_effect",
        setting="unclear", effect_detail="", n=None,
        combo_with=None, target_is_primary=True, facet="mechanism",
        quote=quote, quote_status="exact", section="results", quote_start=0, quote_end=len(quote),
        provenance={"target": "quote"}, extracted_by_model="fake", prompt_version="1",
    )


def _write_context(client: FakeLLMClient, unit: EvidenceUnit) -> WriteContext:
    spec = QuestionSpec(
        id="Q1", text="What drives obesity?", question_type="mechanism", goal="understand",
        entities=[QuestionEntity(text="obesity", kind="disease", canonical="obesity", exact_terms=["obesity"])],
    )
    summary = EvidenceSummary(question_id="Q1", units=[unit], short_ids={unit.id: "E1"})
    plan = EvidencePlan(question_id="Q1", facets=[], jobs=[])
    return WriteContext(
        spec=spec, facets=[], plan=plan, summary=summary, ranked=[],
        papers={"P1": Paper(id="P1", title="An obesity paper", abstract=unit.quote)},
        gates={}, client=client, model="fake-model", knowledge=load_knowledge(),
    )


ANSWER_SPEC = SectionSpec(id="answer", title="Answer", kind="written", source="all")


def test_a_kept_sentence_is_stripped_after_the_guard_judges_its_real_words(monkeypatch):
    """Self-citing fixture (the unit's own quote is the drafted sentence
    itself, the same trick `test_report_repeats.py` uses), so the guard
    trivially keeps it - what this test actually pins down is the *order*:
    the guard must see the sentence exactly as the model wrote it, "However,"
    included, and only the stored, reader-facing text has it stripped."""
    unit = _unit("U1", "AMPK", S29_SENTENCE)

    seen_by_guard: list[str] = []
    real_check = guard_module.CitationGuard.check

    def spy_check(self, text: str, citations: list[str]):
        seen_by_guard.append(text)
        return real_check(self, text, citations)

    monkeypatch.setattr(guard_module.CitationGuard, "check", spy_check)

    def responder(prompt: str, schema: dict | None) -> str:
        ids = schema["properties"]["sentences"]["items"]["properties"]["citations"]["items"]["enum"]
        return json.dumps({"sentences": [{"citations": [ids[0]], "text": S29_SENTENCE}]})

    client = FakeLLMClient(responder=responder, version="fake-1")
    ctx = _write_context(client, unit)

    sentences, _ = write_section_text(ctx, ANSWER_SPEC, ctx.summary.units)

    assert seen_by_guard == [S29_SENTENCE], "the guard judged the model's real words, connective included"
    assert len(sentences) == 1
    assert sentences[0].kept is True
    assert sentences[0].text == S29_STRIPPED, "only the stored, reader-facing text has 'However, ' stripped"


def test_the_whole_path_write_then_verify_agrees_on_the_real_s29_sentence():
    """Not just the unit: `report/invariants.py`'s `check_sentences` - what
    `verify_run` calls - re-runs the citation guard on `Sentence.text` as
    *stored*, which is the text after `_strip_leading_connective` already
    ran. If stripping ever changed the guard's verdict, a report that was
    written correctly would come back flagged as broken, forever - the exact
    self-inflicted integrity violation this whole feature must not cause.

    This runs the real `write_report` (drafts the real S29 sentence, the
    guard judges it write-time, "However, " is stripped, it is stored) to
    produce a finished `Report`, then re-runs `check_sentences` against that
    stored report exactly as `verify_run` would, and requires no violation.

    Scoped to "However, " only - the safe half of this rule, confirmed by
    checking `content_words` in `extract/verify.py`: "however" is already in
    its `_FILLER` set, so `content_words(sentence)` is byte-for-byte
    identical with or without it, and every guard check built on it (check 8
    `drops_not_significant`, check 10 `rests_on_quote`) sees an unchanged
    set either way. "Moreover" is *not* in `_FILLER` and is not covered by
    this test - see the module-level flag below."""
    unit = _unit("U1", "AMPK", S29_SENTENCE).model_copy(update={"outcome": "association"})
    # "association" keeps this unit out of the "negative" section's slice
    # (source="negative" wants outcome in no_effect/worsened/mixed) so the
    # default report layout drafts it from exactly one section family
    # ("answer" then "evidence", both source="all"); the second draft is a
    # cross-section repeat of the first and never reaches the guard again -
    # see `test_report_repeats.py` for that mechanism, which is not what this
    # test is about.

    def responder(prompt: str, schema: dict | None) -> str:
        ids = schema["properties"]["sentences"]["items"]["properties"]["citations"]["items"]["enum"]
        return json.dumps({"sentences": [{"citations": [ids[0]], "text": S29_SENTENCE}]})

    client = FakeLLMClient(responder=responder, version="fake-1")
    ctx = _write_context(client, unit)

    report = write_report(ctx, model_versions={}, prompt_versions={})

    kept = report.kept_sentences()
    assert [s.text for _, s in kept] == [S29_STRIPPED], "stored as the stripped, reader-facing text"

    violations = check_sentences(report, ctx.spec, ctx.knowledge, ctx.papers)
    assert violations == [], "a correctly-written report must not come back flagged by its own re-check"


# --- WHY "Moreover," IS OUT OF SCOPE (measured, decided, not a loose end) -----
#
# "Moreover, " is NOT safe to strip post-guard the way "However, " is, so this
# feature strips only "However,". Concretely, empirically checked against the
# real `report/guard.py` and `extract/verify.py`:
#
#   quote = "Weight loss was significantly reduced from baseline."
#   qualifier = "Although this result did not reach statistical significance (p = 0.09)."
#   with_moreover = "Moreover, weight loss occurred in the treated group."
#   without       = "Weight loss occurred in the treated group."
#
#   guard.check(with_moreover, ["E1"]) -> GuardResult(kept=True,  reason=None)
#   guard.check(without,       ["E1"]) -> GuardResult(kept=False, reason=
#       "drops the qualifier the paper states: not significant")
#
# Check 8 (`drops_not_significant`) computes `said_words = content_words(sentence)`
# and keeps the sentence only if the overlap ratio with the quote's own words
# stays below a threshold: `len(said_words & quote_words) * 2 >= len(said_words)`.
# "however" is in `extract/verify.py`'s `_FILLER` set and so is never a member
# of `content_words(...)` either way - but "moreover" is not filtered, so it is
# one more (non-overlapping) word in the denominator `len(said_words)` at
# write time (guard sees "Moreover, ..."), and one fewer after this feature
# strips it for storage (guard re-run at verify time sees "..." without it).
# That is enough, on this real input, to flip the ratio and the verdict: a
# sentence the guard correctly kept at write time is reported as a citation-
# guard violation when `verify_run`/`check_sentences` re-checks the stored,
# stripped text - a bug in the report's own integrity check caused by this
# feature, not a bug in the paper's evidence.
#
# The decision: "Moreover," is dropped from this feature. The fix that would
# make it safe is adding "moreover" to `_FILLER`, but `_FILLER` feeds
# `content_words()`, which drives guard checks 8 and 10 for every sentence of
# every report - an accuracy change that needs measuring on the stored corpora
# in its own right, and not something a cosmetic tidy-up should buy silently.
# Nothing is lost by waiting: no kept sentence in any of the three hand audits
# opens with "Moreover,".
