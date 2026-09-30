"""Issues #25 and #28: every score names what it rests on, and two runs are
compared only when that matches.

The 27 September migraine run judged 30, then 40, of 42 claims, and its 70.0
against an earlier 80.0 was read as a regression; and the answers on file
came from more than one judge. Here: every judged score carries judged/total,
every scorecard lists its raters, and `bench compare` refuses a pair whose
judged fractions or raters differ.
"""

from __future__ import annotations

import json

import pytest

from bedrock.litreview.bench.__main__ import main
from bedrock.litreview.bench.compare import COMPARABLE_WITHIN, compare, refusals
from bedrock.litreview.bench.filejudge import FileClient
from bedrock.litreview.bench.history import read_history
from bedrock.litreview.bench.models import Gate, Scorecard, SystemScore
from bedrock.litreview.bench.run import RESULTS, evaluate
from bedrock.litreview.config import EVALS_DIR
from bedrock.litreview.llm import FakeLLMClient
from tests.litreview.test_bench import bench_responder, bench_setup  # noqa: F401 - a fixture


def card(*, question="q", overall=70.0, judged=(42, 42), raters=None, created="t", gates=None) -> Scorecard:
    return Scorecard(
        question_id=question, question="?", report_id="r", created=created,
        raters={"claude-x": 40, "code": 2} if raters is None else raters,
        systems=[SystemScore(system="pipeline", claims=judged[1], overall=overall, cited_support=0.9,
                             corpus_grounded=0.95, judged={"cited_support": judged, "corpus_grounded": judged})],
        gates=gates or {"every sentence cited": Gate(status="pass")},
    )


def write(tmp_path, name, c: Scorecard):
    d = tmp_path / name
    d.mkdir()
    (d / "scorecard.json").write_text(c.model_dump_json())
    return d


def test_the_tolerance_is_a_named_starting_value_of_five_points():
    import inspect

    import bedrock.litreview.bench.compare as module

    assert COMPARABLE_WITHIN == 0.05
    source = inspect.getsource(module)
    docstring = source.split("COMPARABLE_WITHIN = 0.05\n", 1)[1].split('"""', 2)[1]
    assert "A starting value" in docstring


def test_a_comparable_pair_prints_the_difference(tmp_path, capsys):
    a = write(tmp_path, "a", card(overall=66.6, created="before"))
    b = write(tmp_path, "b", card(overall=70.0, created="after", judged=(41, 42),
                                  raters={"claude-x": 38, "code": 4},
                                  gates={"every sentence cited": Gate(status="not_measured", reason="no sentences")}))
    assert compare(a, b) == 0
    out = capsys.readouterr().out
    assert "| Overall (0–100) | 66.6 | 70.0 | +3.4 | 100% | 98% |" in out
    assert "| Accuracy: claims supported by their citations | 0.90 | 0.90 | +0.00 | 42/42 | 41/42 |" in out
    assert "| every sentence cited | PASS | not measured: no sentences |" in out
    assert "69.4, 69.2 and 69.2" in out, "the noise caveat is printed with every comparison"
    assert "not comparable" not in out


def test_judged_fractions_more_than_five_points_apart_are_refused(tmp_path, capsys):
    a = write(tmp_path, "a", card(overall=80.0, judged=(42, 42)))
    b = write(tmp_path, "b", card(overall=70.0, judged=(30, 42)))
    assert compare(a, b, err=lambda m: print(m)) == 1
    out = capsys.readouterr().out
    assert "refused" in out and "A judged 42/42 (100%), B 30/42 (71%)" in out
    assert "| Overall" not in out, "a refused comparison prints no numbers"

    assert compare(a, b, force=True) == 0
    forced = capsys.readouterr().out
    assert forced.index("Printed anyway (--force)") < forced.index("| Overall")
    assert "A judged 42/42 (100%), B 30/42 (71%)" in forced


def test_exactly_five_points_apart_still_compares():
    assert refusals(card(judged=(100, 100)), card(judged=(95, 100))) == []
    assert refusals(card(judged=(100, 100)), card(judged=(94, 100)))


def test_different_raters_are_refused():
    reasons = refusals(card(raters={"claude-a": 40}), card(raters={"claude-b": 40}))
    assert any("The raters differ" in r and "claude-a 100%" in r and "claude-b 100%" in r for r in reasons)


def test_the_same_raters_in_different_shares_are_refused():
    same = {"claude-a": 50, "scientist:kim": 50}
    assert refusals(card(raters=same), card(raters={"claude-a": 53, "scientist:kim": 47})) == []
    assert refusals(card(raters=same), card(raters={"claude-a": 60, "scientist:kim": 40}))


def test_code_settled_verdicts_are_not_a_rater():
    """Code gives the same verdict every time; more of them is a different report, not a different judge."""
    assert refusals(card(raters={"claude-x": 40, "code": 2}), card(raters={"claude-x": 30, "code": 12})) == []


def test_answers_with_no_recorded_judge_are_refused():
    """Old answers load as "unknown", and item 25 found they came from more than one judge."""
    reasons = refusals(card(raters={"unknown": 40}), card(raters={"unknown": 40}))
    assert len(reasons) == 2 and all("no recorded judge" in r for r in reasons)


def test_a_scorecard_from_before_this_is_refused():
    old = Scorecard.model_validate({"question_id": "q", "question": "?", "report_id": "r", "created": "t",
                                    "systems": [{"system": "pipeline", "claims": 42, "overall": 80.0}],
                                    "gates": {"every sentence cited": True}})
    reasons = refusals(old, card())
    assert any("does not record how many claims" in r for r in reasons)
    assert any("does not record who judged it" in r for r in reasons)


def test_different_questions_are_refused():
    assert any("different questions" in r for r in refusals(card(question="asthma"), card(question="migraine")))


def test_compare_from_the_command_line(tmp_path, capsys):
    a = write(tmp_path, "a", card(judged=(42, 42)))
    b = write(tmp_path, "b", card(judged=(30, 42)))
    assert main(["compare", str(a), str(b)]) == 1
    assert "not comparable" in capsys.readouterr().err
    assert main(["compare", str(a / "scorecard.json"), str(b), "--force"]) == 0
    assert "Printed anyway" in capsys.readouterr().out
    assert main(["compare", str(a), str(tmp_path / "missing")]) == 1


def test_old_answer_rows_load_as_an_unknown_judge(tmp_path):
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"key": "k1", "answer": {"x": 1}}) + "\n"
                       + json.dumps({"key": "k2", "answer": {"x": 2}, "judge": "scientist:kim"}) + "\n")
    client = FileClient(answers)
    assert set(client.answers) == {"k1", "k2"}
    assert client.judges == {"k1": "unknown", "k2": "scientist:kim"}

    path = EVALS_DIR / "judge" / "answers.jsonl"
    real = FileClient(path)
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    assert len(real.answers) == len(set(json.loads(x)["key"] for x in lines))
    assert all(real.judges[k] for k in real.answers)


def test_the_file_judge_counts_the_answers_it_used_per_judge(tmp_path):
    from bedrock.litreview.bench.filejudge import task_key

    answers = tmp_path / "a.jsonl"
    rows = [{"key": task_key("first"), "answer": {}, "judge": "claude-x"}, {"key": task_key("second"), "answer": {}}]
    answers.write_text("".join(json.dumps(r) + "\n" for r in rows))
    client = FileClient(answers)
    for prompt in ("first", "first", "second"):
        client.complete(prompt, model="m")
    assert client.used == {"claude-x": 2, "unknown": 1}


def _answer_pending(pending_path, answers_path, judge: str, keep=lambda i: True) -> int:
    """Answer the pending judging questions the way the fake judge would, as `judge`."""
    rows = [json.loads(line) for line in pending_path.read_text().splitlines()]
    kept = [r for i, r in enumerate(rows) if keep(i)]
    with answers_path.open("a") as fh:
        for r in kept:
            answer = json.loads(bench_responder(r["prompt"], {"title": r["kind"]}))
            fh.write(json.dumps({"key": r["key"], "answer": answer, "judge": judge}) + "\n")
    return len(kept)


def _run(bench_setup, tmp_path, answers, **kw):  # noqa: F811
    settings, corpus_dir, bq = bench_setup
    judge = FileClient(answers, version=f"file:{answers.name}")
    return evaluate(bq, settings=settings, client=FakeLLMClient(responder=bench_responder), corpus_dir=corpus_dir,
                    results_root=tmp_path / "results", skip_baselines=True, skip_distractors=True,
                    judge_client=judge, **kw)


def test_scores_carry_their_judged_fraction_and_raters_end_to_end(bench_setup, tmp_path):  # noqa: F811
    answers = tmp_path / "answers.jsonl"
    empty, _ = _run(bench_setup, tmp_path, answers, skip_perturb=True)
    pipeline = next(s for s in empty.systems if s.system == "pipeline")
    claims = pipeline.claims
    assert pipeline.judged["cited_support"][1] == claims

    # Answer every other question: the score now rests on part of the claims.
    _answer_pending(answers.with_suffix(".pending.jsonl"), answers, "claude-x", keep=lambda i: i == 0)
    partial, out_dir = _run(bench_setup, tmp_path, answers, skip_perturb=True)
    p = next(s for s in partial.systems if s.system == "pipeline")
    judged, total = p.judged["cited_support"]
    assert total == claims and judged < total
    assert partial.raters.get("claude-x") and set(partial.raters) <= {"claude-x", "code"}
    md = (out_dir / "scorecard.md").read_text()
    assert f"({judged}/{total})" in md and "Verdicts by rater: claude-x" in md
    row = read_history(tmp_path / "results" / "history.jsonl")[-1]
    assert tuple(row.judged["pipeline"]["cited_support"]) == (judged, total)
    assert row.raters == partial.raters

    _answer_pending(answers.with_suffix(".pending.jsonl"), answers, "claude-x")
    full, _ = _run(bench_setup, tmp_path, answers, skip_perturb=True)
    f = next(s for s in full.systems if s.system == "pipeline")
    assert f.judged["cited_support"] == (claims, claims)
    if abs(judged / total - 1) > COMPARABLE_WITHIN:
        assert any("judged" in r for r in refusals(partial, full))


def test_an_unjudged_perturbation_is_not_counted_as_missed(bench_setup, tmp_path):  # noqa: F811
    """With no answer for the corrupted claims, the validity gate is not
    measured. It used to count every unanswered one as a miss, and FAIL."""
    answers = tmp_path / "answers.jsonl"
    _run(bench_setup, tmp_path, answers, skip_perturb=True)
    _answer_pending(answers.with_suffix(".pending.jsonl"), answers, "claude-x")
    card_, out_dir = _run(bench_setup, tmp_path, answers)
    rows = json.loads((out_dir / "perturbations.json").read_text())
    flips = [r for r in rows if r["kind"] == "flip"]
    if not flips:
        pytest.skip("the fake report has no claim a flip applies to")
    assert all(r["judge_caught"] is None for r in rows), "no perturbation had an answer on file"
    gate = card_.gates["judge catches flipped effects >= 0.7 (benchmark validity)"]
    assert gate.status == "not_measured" and gate.reason == "no flipped claim was judged"
    assert card_.pipeline["perturbations_judged"] == 0


def test_a_model_judge_is_one_rater(bench_setup, tmp_path):  # noqa: F811
    settings, corpus_dir, bq = bench_setup
    c, _ = evaluate(bq, settings=settings, client=FakeLLMClient(responder=bench_responder, version="fake-1"),
                    corpus_dir=corpus_dir, results_root=tmp_path / "results", skip_baselines=True,
                    skip_distractors=True, skip_perturb=True)
    judges = {k for k in c.raters if k != "code"}
    assert judges == {c.model_versions["judge"]}


def test_the_real_history_has_no_judged_fractions_before_this_change():
    rows = read_history(RESULTS / "history.jsonl")
    assert all(r.judged is None and r.raters is None for r in rows if r.created < "20260930")
