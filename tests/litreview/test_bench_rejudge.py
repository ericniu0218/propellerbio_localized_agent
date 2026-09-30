"""Issue #21: a blind sheet for a scientist to re-judge the benchmark's claims,
and the agreement between their grades and the file judge's verdicts."""

from __future__ import annotations

import json

import pytest

from bedrock.litreview.bench.filejudge import task_key
from bedrock.litreview.bench.judge import _numbered
from bedrock.litreview.bench.run import _unit_evidence
from bedrock.litreview.db import Database
from bedrock.litreview.llm import FakeLLMClient, load_prompt
from bedrock.litreview.runner import answer_question
from evals.litreview.rejudge_sheet import agree, agreement, claims_of, draw, main, write_draw
from tests.litreview.fixtures import QUESTION, corpus
from tests.litreview.test_bench import bench_responder, bench_setup  # noqa: F401 - a fixture


@pytest.fixture()
def two_runs(bench_setup, tmp_path):  # noqa: F811
    """Two stored benchmark runs of the fake pipeline, in the benchmark's database."""
    settings, _, _ = bench_setup
    db = Database(tmp_path / "bench.sqlite3")
    runs = [answer_question(QUESTION, settings=settings, client=FakeLLMClient(responder=bench_responder),
                            papers=corpus(), db=db, out_dir=tmp_path / f"run{i}") for i in range(2)]
    assert runs[0].run.id != runs[1].run.id
    return db, runs


def test_the_sheet_is_blind(two_runs, tmp_path):
    db, runs = two_runs
    sheet = write_draw([r.run.id for r in runs], tmp_path / "sheet", db=db).read_text()
    key = json.loads((tmp_path / "sheet" / "key.json").read_text())
    assert key["rows"], "the fixture report has kept claims"
    identifying = [QUESTION, "pipeline", *(r.run.id for r in runs), *(r.report_id for r in runs),
                   *(r.spec.id for r in runs), *(p.id for p in corpus()),
                   "supported", "contradicted", "verdict"]
    for row in key["rows"]:
        identifying += [row["section"], *row["citations"]]
    leaked = [x for x in identifying if x and x in sheet]
    assert not leaked, f"the sheet shows {leaked}"
    grades = [json.loads(x) for x in (tmp_path / "sheet" / "grades.template.jsonl").read_text().splitlines()]
    assert set(grades[0]) == {"report", "s", "grade", "note", "rater"}
    assert not any(r.run.id in json.dumps(g) for g in grades for r in runs)


def test_each_row_shows_exactly_what_the_judge_sees(two_runs, tmp_path):
    db, runs = two_runs
    sheet = write_draw([runs[0].run.id], tmp_path / "sheet", db=db).read_text()
    report = runs[0].report
    units = {report.short_ids[u.id]: u for u in report.evidence}
    for row in json.loads((tmp_path / "sheet" / "key.json").read_text())["rows"]:
        assert row["evidence"] == [_unit_evidence(units[c]) for c in row["citations"]]
        assert _numbered(row["evidence"]) in sheet


def test_the_draw_is_seeded_shuffled_across_runs_and_even():
    pools = {"a": [{"run_id": "a", "i": i} for i in range(30)], "b": [{"run_id": "b", "i": i} for i in range(4)],
             "c": [{"run_id": "c", "i": i} for i in range(30)]}
    rows = draw(pools, 40, seed=7)
    assert rows == draw(pools, 40, seed=7) and rows != draw(pools, 40, seed=8)
    by_run = {r: sum(1 for x in rows if x["run_id"] == r) for r in pools}
    assert by_run == {"a": 18, "b": 4, "c": 18}, "even shares, a short run giving its all"
    assert [x["run_id"] for x in rows] != sorted(x["run_id"] for x in rows), "shuffled across runs"


def test_agreement_matches_a_hand_calculation():
    # 10 claims: both supported 6, scientist only 1, judge only 1, neither 2.
    pairs = [(True, True)] * 6 + [(True, False)] + [(False, True)] + [(False, False)] * 2
    a = agreement(pairs)
    # p_o = 8/10. Both say supported 7/10 of the time, so
    # p_e = 0.7 * 0.7 + 0.3 * 0.3 = 0.58, and kappa = (0.8 - 0.58) / (1 - 0.58) = 0.5238.
    assert a.pairs == 10 and a.raw == pytest.approx(0.8)
    assert a.kappa == pytest.approx(0.22 / 0.42)
    assert agreement([(True, True), (True, True)]).kappa is None, "undefined when nobody ever disagrees in kind"
    assert agreement([]).raw is None


def _answer(judge_file, rows, supported):
    """File-judge answers: the evidence sentence for claims `supported` says
    are supported, an empty one (the judge found nothing) otherwise."""
    with judge_file.open("w") as fh:
        for row in rows:
            prompt = load_prompt("judge_support").render(claim=row["claim"], evidence=_numbered(row["evidence"]))
            sentence = row["evidence"][0].split(" (Context:")[0] if supported(row) else ""
            fh.write(json.dumps({"key": task_key(prompt), "judge": "claude-x",
                                 "answer": {"reasoning": "r", "evidence_sentence": sentence, "missing": []}}) + "\n")


def test_agreement_on_fixture_grades_matches_a_hand_calculation(two_runs, tmp_path):
    db, runs = two_runs
    a, b = runs[0].run.id, runs[1].run.id
    out = tmp_path / "sheet"
    write_draw([a, b], out, db=db)
    rows = json.loads((out / "key.json").read_text())["rows"]
    assert sorted(r["claim"][:40] for r in rows if r["run_id"] == a) == \
        sorted(r["claim"][:40] for r in rows if r["run_id"] == b), "same report twice: two claims per run"
    assert len(rows) == 4

    # The judge supports the IL-23 claims and finds nothing for the IL-17A ones.
    judge_file = tmp_path / "answers.jsonl"
    _answer(judge_file, rows, supported=lambda r: "IL-17A" not in r["claim"])

    # The scientist agrees on both of run A's, and calls run B's IL-17A claim faithful.
    def grade(r):
        return "unfaithful" if r["run_id"] == a and "IL-17A" in r["claim"] else "faithful"

    (out / "grades.jsonl").write_text("".join(
        json.dumps({"report": out.name, "s": r["s"], "grade": grade(r), "note": "", "rater": "scientist:kim"}) + "\n"
        for r in rows))
    text = agree(out, judge_file=judge_file)

    # Run A: 2 of 2 agree; each side says supported half the time:
    #   p_o = 1, p_e = 0.5*0.5 + 0.5*0.5 = 0.5, kappa = (1 - 0.5) / 0.5 = 1.
    # Run B: 1 of 2 agree; scientist always supported, judge half:
    #   p_o = 0.5, p_e = 1*0.5 + 0*0.5 = 0.5, kappa = 0.
    # All: 3 of 4 agree; scientist 3/4 supported, judge 2/4:
    #   p_e = 0.75*0.5 + 0.25*0.5 = 0.5, kappa = (0.75 - 0.5) / 0.5 = 0.5.
    assert f"| {a} | {runs[0].spec.id} | 2 | 2 | 1.00 | 1.00 |" in text
    assert f"| {b} | {runs[1].spec.id} | 2 | 2 | 0.50 | 0.00 |" in text
    assert "| all runs | | 4 | 4 | 0.75 | 0.50 |" in text
    assert "faithful → supported; misleading, weak_citation, unfaithful → not supported" in text
    assert "partially_supported, unsupported, contradicted → not supported" in text
    assert "Not compared: 0 with no judge answer on file, 0 not graded." in text


def test_claims_with_no_answer_on_file_are_left_out_and_counted(two_runs, tmp_path):
    db, runs = two_runs
    out = tmp_path / "sheet"
    write_draw([runs[0].run.id], out, db=db)
    rows = json.loads((out / "key.json").read_text())["rows"]
    (out / "grades.jsonl").write_text("".join(
        json.dumps({"report": out.name, "s": r["s"], "grade": "faithful", "note": "", "rater": "scientist:kim"})
        + "\n" for r in rows))
    text = agree(out, judge_file=tmp_path / "empty.jsonl")
    assert f"Not compared: {len(rows)} with no judge answer on file, 0 not graded." in text
    assert "| all runs | | 2 | 0 | undefined | undefined |" in text


def test_a_grade_outside_the_vocabulary_is_refused(two_runs, tmp_path, capsys):
    db, runs = two_runs
    out = tmp_path / "sheet"
    write_draw([runs[0].run.id], out, db=db)
    (out / "grades.jsonl").write_text(json.dumps({"report": out.name, "s": "R01", "grade": "supported",
                                                  "note": "", "rater": "scientist:kim"}) + "\n")
    assert main(["agree", str(out), "--judge-file", str(tmp_path / "a.jsonl")]) == 1
    assert "grade 'supported' is not one of faithful" in capsys.readouterr().err


def test_claims_of_reads_a_stored_run(two_runs):
    db, runs = two_runs
    rows = claims_of(db, runs[0].run.id)
    assert [r["claim"] for r in rows] == [s.text for _, s in runs[0].report.kept_sentences()]
