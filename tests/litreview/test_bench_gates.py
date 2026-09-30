"""Issue #27: a gate whose input was never built is "not measured", never PASS.

Obesity's "no distractor leaked" PASSed on 26 September with
`distractors.json` equal to `{}`: that run scored a stored report, which
skips the distractor test, and the gate read the missing leak rate as no
leak. Every gate is now pass, fail or not measured with its reason, the
counts leave unmeasured gates out, and each gate is tested here in every
state it can reach.
"""

from __future__ import annotations

import json

import pytest

from bedrock.litreview.bench.distractors import leak_test, why_none
from bedrock.litreview.bench.history import HistoryRow, gates_text, read_history
from bedrock.litreview.bench.models import Gate, Scorecard, SystemScore
from bedrock.litreview.bench.run import _gate_cell, evaluate, pipeline_gates
from bedrock.litreview.db import Database
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import RankedPaper
from bedrock.litreview.runner import answer_question
from bedrock.litreview.stages.s0_question import parse_question
from tests.litreview.fixtures import QUESTION, corpus
from tests.litreview.test_bench import bench_responder, bench_setup  # noqa: F401 - a fixture

QUOTES = "quotes all verified (1.00)"
CITED = "every sentence cited"
NEAR = "no near-miss claims"
NUMBERS = "numbers all in cited evidence"
DISTRACTOR = "no distractor leaked"
SAME = "same report twice"
INTEGRITY = "integrity check passes"
SUPPORTED = "claims supported by citations >= 0.85"
FLIP = "judge catches flipped effects >= 0.7 (benchmark validity)"


def gates(pipeline=None, score=None, integrity=(), distractors_missing="D-MISSING", perturb_missing="P-MISSING"):
    """Every gate measured and passing, except what the test overrides."""
    base = {"quote_validity": 1.0, "citation_coverage": 1.0, "distractor_leak_rate": 0.0,
            "deterministic": "yes", "judge_catches_flip": 1.0}
    base.update(pipeline or {})
    s = dict(system="pipeline", claims=10, near_miss_claims=0, numeric_fidelity=1.0, cited_support=0.9)
    s.update(score or {})
    return pipeline_gates(base, SystemScore(**s), integrity=list(integrity),
                          distractors_missing=distractors_missing, perturb_missing=perturb_missing)


CASES = [
    # gate, pipeline, score, integrity, status, reason
    (QUOTES, {"quote_validity": 1.0}, {}, (), "pass", None),
    (QUOTES, {"quote_validity": 0.97}, {}, (), "fail", None),
    (QUOTES, {"quote_validity": None}, {}, (), "not_measured", "the report has no evidence units"),
    (CITED, {"citation_coverage": 1.0}, {}, (), "pass", None),
    (CITED, {"citation_coverage": 0.9}, {}, (), "fail", None),
    (CITED, {"citation_coverage": None}, {}, (), "not_measured", "the report kept no sentences"),
    (NEAR, {}, {"near_miss_claims": 0}, (), "pass", None),
    (NEAR, {}, {"near_miss_claims": 1}, (), "fail", None),
    (NEAR, {}, {"claims": 0}, (), "not_measured", "the report has no claims"),
    (NUMBERS, {}, {"numeric_fidelity": 1.0}, (), "pass", None),
    (NUMBERS, {}, {"numeric_fidelity": 0.5}, (), "fail", None),
    (NUMBERS, {}, {"numeric_fidelity": None}, (), "not_measured", "no claim states a number"),
    (NUMBERS, {}, {"numeric_fidelity": None, "claims": 0}, (), "not_measured", "the report has no claims"),
    (DISTRACTOR, {"distractor_leak_rate": 0.0}, {}, (), "pass", None),
    (DISTRACTOR, {"distractor_leak_rate": 0.4}, {}, (), "fail", None),
    (DISTRACTOR, {"distractor_leak_rate": None}, {}, (), "not_measured", "D-MISSING"),
    (SAME, {"deterministic": "yes"}, {}, (), "pass", None),
    (SAME, {"deterministic": "no"}, {}, (), "fail", None),
    (SAME, {"deterministic": "not re-run"}, {}, (), "not_measured",
     "a stored report was scored; the pipeline was not re-run"),
    (INTEGRITY, {}, {}, (), "pass", None),
    (INTEGRITY, {}, {}, ("a quote is not at its offsets",), "fail", None),
    (SUPPORTED, {}, {"cited_support": 0.85}, (), "pass", None),
    (SUPPORTED, {}, {"cited_support": 0.84}, (), "fail", None),
    (SUPPORTED, {}, {"cited_support": None}, (), "not_measured", "no pipeline claim was judged"),
    (FLIP, {"judge_catches_flip": 0.7}, {}, (), "pass", None),
    (FLIP, {"judge_catches_flip": 0.5}, {}, (), "fail", None),
    (FLIP, {"judge_catches_flip": None}, {}, (), "not_measured", "P-MISSING"),
]


@pytest.mark.parametrize("name, pipeline, score, integrity, status, reason", CASES,
                         ids=[f"{c[0][:24]}-{c[4]}-{i}" for i, c in enumerate(CASES)])
def test_every_gate_in_every_state_it_can_reach(name, pipeline, score, integrity, status, reason):
    gate = gates(pipeline, score, integrity)[name]
    assert gate.status == status
    assert gate.reason == reason


def test_the_integrity_gate_is_always_measured():
    """The invariants run on every finished report; there is no input to be missing."""
    assert {c[4] for c in CASES if c[0] == INTEGRITY} == {"pass", "fail"}


def test_every_gate_is_covered_above():
    assert {c[0] for c in CASES} == set(gates())


def test_counts_leave_unmeasured_gates_out():
    card = Scorecard(question_id="q", question="?", report_id="r", created="t", systems=[],
                     gates=gates({"distractor_leak_rate": None, "quote_validity": 0.5}))
    assert card.gates_passed == 7
    assert card.gates_measured == 8
    assert card.gates_not_measured == {DISTRACTOR: "D-MISSING"}


def test_a_gate_cannot_be_read_as_a_bool():
    """`if gate:` would read "not measured" as passed - the bug itself."""
    with pytest.raises(TypeError):
        bool(Gate(status="not_measured", reason="x"))


def test_render_prints_pass_only_for_a_gate_that_measured_something():
    assert _gate_cell(Gate(status="pass")) == "PASS"
    assert _gate_cell(Gate(status="fail")) == "**FAIL**"
    cell = _gate_cell(Gate(status="not_measured", reason="no distractor papers were built"))
    assert cell == "not measured: no distractor papers were built" and "PASS" not in cell


def test_old_scorecards_with_bool_gates_still_load():
    old = {"question_id": "q", "question": "?", "report_id": "r", "created": "t", "systems": [],
           "gates": {QUOTES: True, DISTRACTOR: False}}
    card = Scorecard.model_validate(old)
    assert card.gates[QUOTES].status == "pass" and card.gates[DISTRACTOR].status == "fail"


def test_old_history_rows_still_load_and_say_what_they_counted(tmp_path):
    rows = [
        {"question_id": "q", "created": "1", "report_id": "a", "gates_passed": 9, "gates_total": 9},
        {"question_id": "q", "created": "2", "report_id": "b", "gates_passed": 7, "gates_total": 8,
         "gates_not_measured": {DISTRACTOR: "no distractor papers were built"}},
    ]
    path = tmp_path / "history.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    old, new = read_history(path)
    assert old.gates_not_measured is None and gates_text(old) == "9 of 9 †"
    assert gates_text(new) == "7 of 8 measured, 1 not measured"
    assert gates_text(HistoryRow(question_id="q", created="0", report_id="c")) == "not recorded"


def test_an_empty_distractor_result_is_none_not_zero():
    """What the 26 September obesity run wrote: `{}` - no rate at all. And
    a leak test with nothing to test returns None, not a 0.0 that passes."""
    spec = parse_question(QUESTION)
    assert {}.get("leak_rate") is None
    assert leak_test([], spec, [], client=None, model="m")["leak_rate"] is None


def test_why_no_distractor_was_built():
    spec = parse_question(QUESTION)
    ungraded = [RankedPaper(paper_id="P1", rank=1, gate="pass", grade=1)]
    assert why_none(ungraded, spec) == "no paper was graded relevant enough (2 or more) to copy"
    no_near_miss = spec.model_copy(update={"entities": [
        e.model_copy(update={"near_miss_terms": []}) for e in spec.entities]})
    assert "has no near-miss term" in why_none(ungraded, no_near_miss)


def test_a_run_with_no_distractors_built_says_so(bench_setup, tmp_path):  # noqa: F811
    settings, corpus_dir, bq = bench_setup
    card, out_dir = evaluate(bq, settings=settings, client=FakeLLMClient(responder=bench_responder),
                             corpus_dir=corpus_dir, results_root=tmp_path / "results",
                             skip_baselines=True, skip_distractors=True, skip_perturb=True)
    gate = card.gates[DISTRACTOR]
    assert gate.status == "not_measured"
    assert gate.reason == "no distractor papers were built: skipped with --skip-distractors"
    md = (out_dir / "scorecard.md").read_text()
    assert f"| {DISTRACTOR} | not measured: no distractor papers were built" in md
    assert card.gates[FLIP].reason == "skipped with --skip-perturb"
    assert json.loads((out_dir / "distractors.json").read_text()) == {}
    (row,) = read_history(tmp_path / "results" / "history.jsonl")
    assert row.gates_total == card.gates_measured == len(card.gates) - 2
    assert set(row.gates_not_measured) == {DISTRACTOR, FLIP}
    assert f"{card.gates_passed} of {card.gates_measured} measured checks pass; 2 not measured" in md


def test_a_stored_report_leaves_the_ranking_checks_unmeasured(bench_setup, tmp_path):  # noqa: F811
    """Scoring a stored report cannot test today's ranking or re-run the
    pipeline; both gates say so instead of disappearing or passing."""
    settings, corpus_dir, bq = bench_setup
    client = FakeLLMClient(responder=bench_responder)
    stored = answer_question(QUESTION, settings=settings, client=client, papers=corpus(),
                             db=Database(settings.data_dir / "bench.sqlite3"), out_dir=tmp_path / "stored")
    card, _ = evaluate(bq, settings=settings, client=client, corpus_dir=corpus_dir,
                       results_root=tmp_path / "results", skip_baselines=True, skip_perturb=True,
                       from_run=stored.run.id)
    assert card.gates[DISTRACTOR].reason.startswith("no distractor papers were built: a stored report")
    assert card.gates[SAME].status == "not_measured"
