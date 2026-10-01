"""The optional entailment check (`report/entailment.py`, issue #10): off by
default, and when on, a model's second look at every sentence the citation
guard kept.

What this proves, with a fake model:

- off (the default) no call is made and the report is byte-identical to one
  written with a client that would have dropped everything
- on, "not_supported" and "contradicted" drop a sentence with the written
  reason; "supported" and "partial" keep it; a failed call keeps it; only
  sentences the guard kept are ever asked about
- a dropped sentence is stored as not kept, so the pure-code integrity check
  still passes without a model
- the call goes through `SchedulerClient` at `USER_JOB`, temperature 0, with
  a JSON schema
- the model is shown each quote apart from the sentences beside it, each
  labelled with its role (prompt v2, OPEN_ISSUES #3)
- the agreement tool's confusion matrix, on a tiny report, and the audit
  tool's reading of the committed hand-audit sheets

The sentences and quotes below are verbatim from the stored benchmark reports
(asthma and migraine, 2026-09-26), where the strong judge called them
unsupported, partial or supported. The fake model's answers are scripted: what
these tests prove is the plumbing, not the model - that is what
`evals/litreview/entailment_agreement.py` measures.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from bedrock.inference import Priority
from bedrock.litreview.bench.filejudge import task_key
from bedrock.litreview.bench.judge import _numbered
from bedrock.litreview.bench.run import _unit_evidence
from bedrock.litreview.config import ConfigError, load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.llm import FakeLLMClient, load_prompt
from bedrock.litreview.models import EvidenceUnit, Report, ReportSection, Sentence
from bedrock.litreview.report.entailment import DROP_REASON, EntailmentCheck, evidence_text
from bedrock.litreview.runner import answer_question
from evals.litreview.entailment_agreement import compare, confusion, format_table
from evals.litreview import entailment_audit as audit
from tests.litreview.fixtures import QUESTION, corpus, responder
from tests.litreview.test_bedrock_settings import bedrock
from tests.litreview.test_scheduler_client import ScriptedBackend, in_worker, make

ENTAIL_TITLE = "EntailmentAnswer"


def is_entail(schema) -> bool:
    return (schema or {}).get("title") == ENTAIL_TITLE


def entail_reply(verdict: str, reason: str = "scripted") -> str:
    return json.dumps({"reason": reason, "verdict": verdict})


def client_answering(verdict: str | None) -> FakeLLMClient:
    """The pipeline's usual fake model, plus a scripted entailment answer.
    `None` returns something that is not JSON, as a broken call would."""

    def answer(prompt, schema):
        if is_entail(schema):
            return "not json at all" if verdict is None else entail_reply(verdict)
        return responder(prompt, schema)

    return FakeLLMClient(responder=answer, version="fake-1")


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir = tmp_path / "cache"
    s.out_dir = tmp_path / "out"
    s.db_path = tmp_path / "db.sqlite3"
    return s


def run(settings, tmp_path, client, name="db"):
    return answer_question(
        QUESTION, settings=settings, client=client, papers=corpus(),
        db=Database(tmp_path / f"{name}.sqlite3"), out_dir=tmp_path / f"out-{name}",
    )


def dropped(report: Report) -> list[Sentence]:
    out = []

    def walk(section):
        out.extend(s for s in section.sentences if not s.kept)
        for sub in section.subsections:
            walk(sub)

    for section in report.sections:
        walk(section)
    return out


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


def test_the_check_is_off_by_default():
    assert load_settings(model_set="laptop", env={}, check_endpoint=False).entailment_check is False


def test_off_no_call_is_made_and_the_report_is_unchanged(settings, tmp_path):
    plain = run(settings, tmp_path, client_answering("supported"), name="plain")
    hostile = client_answering("not_supported")
    off = run(settings, tmp_path, hostile, name="off")

    assert not any(is_entail(s) for s in hostile.schemas), "the check made a call while switched off"
    assert off.paths["json"].read_bytes() == plain.paths["json"].read_bytes()
    assert "entail" not in off.report.model_versions and "entail" not in off.report.prompt_versions
    assert not any(k.startswith("sentences_entailment") for k in off.report.stats)
    assert "Entailment check" not in off.paths["md"].read_text()


# ---------------------------------------------------------------------------
# On
# ---------------------------------------------------------------------------


def test_on_not_supported_drops_every_sentence_the_guard_kept(settings, tmp_path):
    guard_only = run(settings, tmp_path, client_answering("supported"), name="guard")
    kept_by_guard = [s.text for _, s in guard_only.report.kept_sentences()]
    assert kept_by_guard, "the fixture must give the guard something to keep"

    settings.entailment_check = True
    client = client_answering("not_supported")
    result = run(settings, tmp_path, client, name="on")
    report = result.report

    assert report.kept_sentences() == []
    removed = [s for s in dropped(report) if s.drop_reason and s.drop_reason.startswith(DROP_REASON)]
    # Every sentence the guard alone kept is among them. There can be more: a
    # dropped sentence is not "already said", so a later section drafting it
    # (or a near-copy) again is checked again rather than skipped as a repeat.
    assert set(kept_by_guard) <= {s.text for s in removed}
    assert all(s.drop_reason == DROP_REASON + "scripted" for s in removed)
    # Asked only about what the guard kept - never about what it dropped.
    assert sum(1 for s in client.schemas if is_entail(s)) == len(removed)
    assert any("number 99" in (s.drop_reason or "") for s in dropped(report)), "the guard still runs first"
    assert report.stats["sentences_entailment_checked"] == len(removed)
    assert report.stats["sentences_entailment_dropped"] == len(removed)
    assert report.stats["sentences_kept"] == 0
    # Recorded like any other stage, so a cached rerun and verify-run know.
    assert report.model_versions["entail"] == "fake-1"
    assert report.prompt_versions["entail"] == load_prompt("entail").version
    # The integrity check is pure code and still passes: the dropped
    # sentences are stored as not kept, so it never needs the model.
    assert report.integrity == []
    assert "Entailment check (optional, not yet checked by a scientist)" in result.paths["md"].read_text()


def test_on_contradicted_drops_too(settings, tmp_path):
    settings.entailment_check = True
    report = run(settings, tmp_path, client_answering("contradicted")).report
    assert report.kept_sentences() == []
    assert report.stats["sentences_entailment_dropped"] > 0


def test_on_supported_keeps_what_the_guard_kept(settings, tmp_path):
    guard_only = run(settings, tmp_path, client_answering("supported"), name="guard")
    settings.entailment_check = True
    report = run(settings, tmp_path, client_answering("supported"), name="on").report
    assert [s.text for _, s in report.kept_sentences()] == [s.text for _, s in guard_only.report.kept_sentences()]
    assert report.stats["sentences_entailment_dropped"] == 0
    assert report.stats["sentences_entailment_checked"] == report.stats["sentences_kept"]


def test_on_partial_is_kept_and_counted(settings, tmp_path):
    settings.entailment_check = True
    report = run(settings, tmp_path, client_answering("partial")).report
    kept = report.kept_sentences()
    assert kept
    assert report.stats["sentences_entailment_partial"] == len(kept)
    assert report.stats["sentences_entailment_dropped"] == 0


def test_a_failed_call_leaves_the_sentence_as_the_guard_decided(settings, tmp_path):
    """The check only removes; being unable to ask it must not remove anything."""
    guard_only = run(settings, tmp_path, client_answering("supported"), name="guard")
    settings.entailment_check = True
    report = run(settings, tmp_path, client_answering(None), name="on").report
    assert len(report.kept_sentences()) == len(guard_only.report.kept_sentences())
    assert report.stats["sentences_entailment_failed"] == len(report.kept_sentences())
    assert report.stats["sentences_entailment_checked"] == 0


# ---------------------------------------------------------------------------
# One sentence, from the real benchmark reports
# ---------------------------------------------------------------------------


def unit(short_id: str, **kw) -> EvidenceUnit:
    base = dict(
        id=f"u-{short_id}", paper_id="P1", question_id="q", target="MONTELUKAST", target_raw="montelukast",
        direction="block", modality="small_molecule", intervention="montelukast", disease="asthma",
        disease_raw="asthma", evidence_type="human_clinical", species="human", species_raw="children",
        outcome="improved", effect_detail="reduced exacerbations", n=None, combo_with=None,
        target_is_primary=True, quote="", quote_status="exact", section="abstract",
        extracted_by_model="fake", prompt_version="1",
    )
    base.update(kw)
    return EvidenceUnit(**base)


# asthma report, clinical section, E233: the judge called it unsupported - the
# quote gives enrolment and ages, not an effect on exacerbations.
MONTELUKAST_SENTENCE = (
    "Montelukast reduced asthma exacerbations in children with high-moderate pre-treatment severity, "
    "as shown in a study of 45 participants randomized to montelukast and 44 to placebo."
)
MONTELUKAST = unit(
    "E233",
    quote="Key Results: Among 45 participants randomized to montelukast and 44 to placebo, median [IQR] ages "
          "were 6.6 [5.4, 10] and 6.7 [5.3, 9.9] years, with high-moderate pre-treatment exacerbation severity.",
    context_quote="Background and Purpose: Children with acute asthma exacerbations are frequently hospitalized "
                  "due to insufficient response to inhaled β-agonist and systemic corticosteroid potentially due "
                  "to leukotriene mediated airway inflammation.",
)

# migraine report, negative section, E92: the judge called it partial - "in a
# study of rimegepant" is not in the quote. A partial answer keeps it.
RIMEGEPANT_SENTENCE = (
    "No patient reported treatment-related adverse events during the observation period in a study of rimegepant."
)
RIMEGEPANT = unit(
    "E92", target="CALCRL", target_raw="rimegepant", intervention="rimegepant", disease="migraine",
    disease_raw="migraine", outcome="no_effect",
    quote="No patient reported treatment-related adverse events during the entire observation period.",
)


def test_the_model_is_shown_the_papers_words_not_the_models():
    client = FakeLLMClient(replies=[entail_reply("not_supported")])
    EntailmentCheck(client, "m").check(MONTELUKAST_SENTENCE, {"E233": MONTELUKAST})
    [prompt] = client.prompts
    assert MONTELUKAST_SENTENCE in prompt
    assert MONTELUKAST.quote in prompt and MONTELUKAST.context_quote in prompt
    assert "[E233] about: montelukast" in prompt
    # The extraction model's own summary is never evidence.
    assert "reduced exacerbations" not in prompt


def test_the_qualifying_next_sentence_is_part_of_what_the_model_reads():
    qualified = unit(
        "E211", target_raw="Setmelanotide",
        quote="Although this result did not reach statistical significance (p = 0.09) when compared to placebo, "
              "weight loss from baseline was significantly reduced (p < 0.0001).",
        qualifier_quote="This is a made-up next sentence that limits the quote.",
    )
    assert qualified.qualifier_quote in evidence_text({"E211": qualified})


# obesity report after the 26 September fixes, S18 citing E190
# (`evals/litreview/hand_audit/qwen_obesity_after.md`), graded weak_citation:
# the sentence states the next sentence's result, and the quote it cites is a
# different one. OPEN_ISSUES #3. Shown as one passage (prompt v1), every word
# of the sentence is in the paper's text.
TMEM135_SENTENCE = (
    "High fat feeding in male mice did not elicit a significant body weight difference between TMEM135 "
    "knockout and control genotypes, presumably due to the already rapid diet-induced weight gain in male "
    "C57BL/6J mice."
)
TMEM135 = unit(
    "E190", target="TMEM135", target_raw="TMEM135", intervention=None, disease="obesity", disease_raw="obesity",
    evidence_type="in_vivo", species="mouse", species_raw="mice", outcome="worsened", section="results",
    quote="Assessment of glucose homeostasis in these animals revealed that the TMEM135 knockout results in "
          "glucose intolerance, hyperinsulinemia and insulin resistance (Fig. ).",
    context_quote="Transgenic mice that overexpress TMEM135 ( Tmem135 TG ) under the control of CAG promoter "
                  "have been previously described .",
    qualifier_quote="In contrast to female mice, high fat feeding in male mice did not elicit a significant body "
                    "weight difference between the genotypes, presumably because the already rapid diet-induced "
                    "weight gain in male C57BL/6J mice could not be further increased by the TMEM135 "
                    "inactivation (Supplementary Fig. ).",
)


def test_the_quote_is_shown_apart_from_the_sentences_beside_it():
    lines = evidence_text({"E190": TMEM135}).splitlines()
    assert lines[:2] == ["[E190] about: TMEM135", f'    Quote: "{TMEM135.quote}"']
    assert f'    - context: "{TMEM135.context_quote}"' in lines[3:]
    assert f'    - next sentence: "{TMEM135.qualifier_quote}"' in lines[3:]
    # The result S18 states is only in the next sentence; the model is asked
    # whether a quote states it, so the two must not run together.
    assert "body weight" not in lines[1]


def test_a_finding_with_only_a_quote_shows_nothing_beside_it():
    assert evidence_text({"E92": RIMEGEPANT}) == f'[E92] about: rimegepant\n    Quote: "{RIMEGEPANT.quote}"'


def test_every_cited_finding_is_shown_in_order():
    text = evidence_text({"E233": MONTELUKAST, "E190": TMEM135})
    assert text.index("[E233]") < text.index("[E190]")
    assert text.count("Quote: ") == 2


def test_the_prompt_says_the_claim_must_be_in_a_quote():
    prompt = load_prompt("entail").render(sentence=TMEM135_SENTENCE, evidence=evidence_text({"E190": TMEM135}))
    assert int(load_prompt("entail").version) >= 2
    rules = " ".join(prompt.split())
    assert 'main claim - its result or finding - must be stated in a "Quote"' in rules
    assert "A result that appears only beside a quote, and not in any quote, is not supported." in rules


def test_a_real_unsupported_sentence_is_dropped_with_the_models_reason():
    reason = "The text gives enrolment and ages, not an effect on exacerbations."
    stats = Counter()
    check = EntailmentCheck(FakeLLMClient(replies=[entail_reply("not_supported", reason)]), "m")
    why = check.review(MONTELUKAST_SENTENCE, {"E233": MONTELUKAST}, stats)
    assert why == DROP_REASON + reason
    assert stats == Counter(entailment_checked=1, entailment_dropped=1)


def test_a_real_partial_sentence_is_kept_and_counted():
    stats = Counter()
    check = EntailmentCheck(FakeLLMClient(replies=[entail_reply("partial")]), "m")
    assert check.review(RIMEGEPANT_SENTENCE, {"E92": RIMEGEPANT}, stats) is None
    assert stats == Counter(entailment_checked=1, entailment_partial=1)


def test_a_long_reason_is_cut_to_one_line():
    stats = Counter()
    check = EntailmentCheck(FakeLLMClient(replies=[entail_reply("contradicted", "word\n" * 200)]), "m")
    why = check.review(MONTELUKAST_SENTENCE, {"E233": MONTELUKAST}, stats)
    assert "\n" not in why and len(why) < len(DROP_REASON) + 210


# ---------------------------------------------------------------------------
# The route to the model
# ---------------------------------------------------------------------------


async def test_the_call_goes_through_the_scheduler_at_user_job_temperature_zero_with_a_schema():
    backend = ScriptedBackend(reply=entail_reply("supported", "stated word for word"))
    client, _ = make(backend)
    check = EntailmentCheck(client, "m")

    verdict = await in_worker(lambda: check.check(RIMEGEPANT_SENTENCE, {"E92": RIMEGEPANT}))

    assert verdict.label == "supported" and not verdict.drops
    assert client._priority is Priority.USER_JOB
    [(model, messages, options)] = backend.calls
    assert model == "m"
    assert options.temperature == 0.0
    assert options.json_schema["title"] == ENTAIL_TITLE
    assert options.json_schema["properties"]["verdict"]["enum"] == [
        "supported", "partial", "not_supported", "contradicted"
    ]
    assert RIMEGEPANT_SENTENCE in messages[0]["content"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_litreview_table_turns_it_on():
    s = load_settings(bedrock(litreview={"entailment_check": True}), model_set="laptop", env={})
    assert s.entailment_check is True


def test_the_environment_overrides_the_table():
    s = load_settings(bedrock(litreview={"entailment_check": True}), model_set="laptop",
                      env={"BEDROCK_LITREVIEW_ENTAILMENT_CHECK": "0"})
    assert s.entailment_check is False
    s = load_settings(bedrock(), model_set="laptop", env={"BEDROCK_LITREVIEW_ENTAILMENT_CHECK": "1"})
    assert s.entailment_check is True


def test_a_quoted_false_is_refused_rather_than_read_as_on():
    """`entailment_check = "false"` is a truthy string in Python."""
    with pytest.raises(ConfigError, match="true or false"):
        load_settings(bedrock(litreview={"entailment_check": "false"}), model_set="laptop", env={})


def test_the_check_has_a_pinned_model():
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    assert s.model_for("entail") == s.models[s.jobs["entail"]].name


# ---------------------------------------------------------------------------
# The agreement tool
# ---------------------------------------------------------------------------


GUSELKUMAB = unit(
    "E1", target="IL23A", target_raw="IL-23", intervention="guselkumab", disease="psoriasis",
    disease_raw="plaque psoriasis", species_raw="patients",
    quote="At week 16, PASI-90 was achieved by 73% of patients receiving guselkumab versus 3% on placebo.",
)
CLAIMS = {
    "judge_supported": "Guselkumab achieved PASI-90 in most patients at week 16 versus placebo.",
    "judge_partial": "Guselkumab achieved PASI-90 in most adolescent patients at week 16 versus placebo.",
    "judge_unsupported": "Guselkumab cleared psoriasis in every patient within one week of treatment.",
    "unjudged": "Guselkumab was compared with placebo over sixteen weeks in patients with psoriasis.",
}


@pytest.fixture()
def tiny(tmp_path):
    """A report with four kept sentences citing one finding, and a judge
    answers file for three of them, keyed exactly as the benchmark keys it."""
    report = Report(
        question_id="q", question=QUESTION,
        sections=[ReportSection(id="answer", title="Answer", kind="written", sentences=[
            Sentence(text=text, citations=["E1"]) for text in CLAIMS.values()
        ])],
        evidence=[GUSELKUMAB], short_ids={GUSELKUMAB.id: "E1"},
    )
    answers = {
        "judge_supported": {"reasoning": "same result", "evidence_sentence": GUSELKUMAB.quote, "missing": []},
        "judge_partial": {"reasoning": "adolescents not stated", "evidence_sentence": GUSELKUMAB.quote,
                          "missing": ["adolescent"]},
        "judge_unsupported": {"reasoning": "not stated", "evidence_sentence": "", "missing": []},
    }
    path = tmp_path / "answers.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for name, answer in answers.items():
            prompt = load_prompt("judge_support").render(
                claim=CLAIMS[name], evidence=_numbered([_unit_evidence(GUSELKUMAB)])
            )
            fh.write(json.dumps({"key": task_key(prompt), "answer": answer}) + "\n")
    return report, path


def test_the_agreement_tool_builds_the_confusion_matrix(tiny):
    report, answers = tiny
    scripted = {
        CLAIMS["judge_supported"]: "not_supported",  # a false drop: evidence lost
        CLAIMS["judge_partial"]: "partial",
        CLAIMS["judge_unsupported"]: "contradicted",
        CLAIMS["unjudged"]: "supported",
    }

    def answer(prompt, schema):
        assert is_entail(schema)
        return entail_reply(next(v for claim, v in scripted.items() if f'"{claim}"' in prompt), "because")

    rows = compare(report, EntailmentCheck(FakeLLMClient(responder=answer), "m"), answers)

    assert confusion(rows) == Counter({
        ("supported", "unsupported"): 1,
        ("partial", "partial"): 1,
        ("unsupported", "unsupported"): 1,
        ("unjudged", "supported"): 1,
    })
    table = format_table(rows, title="tiny")
    assert "| supported | 0 | 0 | 1 | 0 | 1 |" in table
    assert "| unjudged | 1 | 0 | 0 | 0 | 1 |" in table
    assert "Judged by both: 3. Agree: 2 (67%)." in table
    assert "(evidence lost): 1." in table
    assert CLAIMS["judge_supported"] in table.split("read every one:")[1]
    # Reading the judge's file never writes the pending list beside it.
    assert not answers.with_suffix(".pending.jsonl").exists()


# ---------------------------------------------------------------------------
# The audit tool, on the committed hand-audit sheets
# ---------------------------------------------------------------------------


def test_the_audit_reads_every_graded_sentence_from_the_committed_sheets():
    cases = audit.cases()
    assert len(cases) == 105
    assert sum(c.hand == audit.NOT_FAITHFUL for c in cases) == 12
    s18 = next(c for c in cases if (c.report, c.sid) == ("qwen_obesity_after", "S18"))
    assert s18.text == TMEM135_SENTENCE and s18.citations == ["E190"] and s18.grade == "weak_citation"
    e190 = s18.units["E190"]
    assert (e190.quote, e190.context_quote, e190.qualifier_quote) == (
        TMEM135.quote, TMEM135.context_quote, TMEM135.qualifier_quote
    )
    assert e190.target_raw == "TMEM135"


def test_check_10_drops_exactly_the_issue_3_sentences_built_from_beside_the_quote():
    """The four of OPEN_ISSUES #3 that take their content from a sentence
    beside the quote; asthma S25 (a dropped hedge) is the quote's own words,
    so check 10 keeps it. No faithful sentence is dropped."""
    from bedrock.litreview.knowledge import load_knowledge

    kb = load_knowledge()
    cases = audit.cases()
    drops = {(c.report, c.sid) for c in cases if audit.check10(c, kb) == "drops"}
    assert drops == set(audit.ISSUE_3) - {("qwen_asthma_after", "S25")}


def test_the_audit_measures_and_writes_the_sheet_with_a_fake_model(tmp_path):
    cases = [c for c in audit.cases() if c.report == "qwen_obesity_after"]
    target = ("qwen_obesity_after", "S18")

    def answer(prompt, schema):
        assert is_entail(schema)
        return entail_reply("not_supported" if f'"{TMEM135_SENTENCE}"' in prompt else "supported", "because")

    rows = audit.measure(cases, EntailmentCheck(FakeLLMClient(responder=answer), "m"), answers=tmp_path / "none")
    assert [(r.report, r.sid) for r in rows if r.check == "unsupported"] == [target]
    table = audit.format_table(rows, title="obesity")
    assert "| qwen_obesity_after S18 | weak_citation | drops | unjudged | unsupported |" in table
    # The sheet shows the scientist the cited text by role, and not the hand grade.
    sheet = audit.review_sheet(rows, cases)
    assert "1 report sentences" in sheet and TMEM135_SENTENCE in sheet
    assert f'  - next sentence: "{TMEM135.qualifier_quote}"' in sheet
    assert "weak_citation" not in sheet
    before = [{**r.__dict__, "check": "supported"} for r in rows]
    assert audit.changes(before, rows).startswith(f"1 of {len(rows)} sentences changed class.")
