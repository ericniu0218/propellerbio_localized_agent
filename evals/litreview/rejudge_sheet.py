"""A blind re-judging sheet for a scientist, and how far the judge agrees.

No scientist has checked any accuracy number yet (OPEN_ISSUES #21): the
benchmark's judge and the hand audits were all AI models. This puts a sample
of the benchmark's own claims in front of a scientist, exactly as the judge
saw them, and measures the agreement.

    python -m evals.litreview.rejudge_sheet draw RUN_ID [RUN_ID ...] [--n 40] [--seed 0] [--out DIR]
    python -m evals.litreview.rejudge_sheet agree DIR [--grades DIR/grades.jsonl] \\
        [--judge-file evals/litreview/judge/answers.jsonl]

`draw` takes kept claims from stored benchmark runs (the run ids `bench run`
stored in its database, as `--from-run` takes them): N in all (default 40),
shared as evenly as the runs allow, seeded, and shuffled across runs. It
writes to DIR (default evals/litreview/rejudge/<seed>-<n>/):

    sheet.md                one row per claim: the claim, and exactly the evidence the judge
                            is shown (`bench/run._unit_evidence`, numbered and cut as the
                            judge's prompt cuts it)
    grades.template.jsonl   one line per row, shaped like hand_audit/grades.jsonl, plus `rater`
    key.json                which run, report, question and claim each row came from

**The sheet is blind.** It shows no system, no question, no run or report
id, no paper id and no verdict - nothing but the claim and its evidence. The
key is a separate file; the person grading must not open it.

`agree` reads the grades and the key, asks the judge file for its verdict on
each claim (through the benchmark's own `Judge`, so the same checks in code
apply and a claim that is verbatim in its evidence is settled in code as it
is in a run), and prints, per run and overall, raw agreement and Cohen's
kappa. Both vocabularies are collapsed to supported / not supported first,
and the mapping is printed beside the numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bedrock.litreview.bench.claims import pipeline_claims
from bedrock.litreview.bench.filejudge import FileClient
from bedrock.litreview.bench.judge import Judge, _numbered
from bedrock.litreview.bench.run import _unit_evidence
from bedrock.litreview.config import EVALS_DIR, load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.runner import load_result

KEY_VERSION = "rejudge_key.v1"
DEFAULT_N = 40
DEFAULT_JUDGE_FILE = EVALS_DIR / "judge" / "answers.jsonl"

GRADES = {
    "faithful": "the evidence shown states the claim, with nothing added, dropped or changed",
    "misleading": "the evidence says something like it, but the claim overstates it, drops a qualifier "
                  "(\"not significant\", \"in mice\"), or changes who was studied",
    "weak_citation": "the claim may well be true, but this evidence does not state it",
    "unfaithful": "the evidence does not say this, or says the opposite",
}
"""The hand audit's vocabulary (evals/litreview/hand_audit/grades.jsonl)."""

SCIENTIST_MAP = {"faithful": True, "misleading": False, "weak_citation": False, "unfaithful": False}
JUDGE_MAP = {"supported": True, "partially_supported": False, "unsupported": False, "contradicted": False}
"""Both collapsed to supported (True) / not supported (False). Partial support
is not support: a claim the evidence only partly states overstates it, which
is what a scientist grades "misleading"."""


# --- draw ---------------------------------------------------------------------------------------


def claims_of(db: Database, run_id: str) -> list[dict]:
    """Every kept claim of a stored run, with the evidence the judge is shown."""
    result = load_result(db, run_id)
    report = result.report
    units = {report.short_ids[u.id]: u for u in report.evidence}
    return [{
        "run_id": run_id, "report_id": result.report_id, "question_id": result.spec.id,
        "question": result.spec.text, "claim_id": claim.id, "section": claim.section, "claim": claim.text,
        "citations": claim.citations,
        "evidence": [_unit_evidence(units[c]) for c in claim.citations if c in units],
    } for claim in pipeline_claims(report)]


def draw(pools: dict[str, list[dict]], n: int, seed: int) -> list[dict]:
    """N claims, shared as evenly as the runs allow, in one seeded shuffle.
    An even share rather than a share by size, so that every run gets enough
    rows for its own agreement to mean something."""
    rng = random.Random(seed)
    runs = sorted(pools)
    share = {r: 0 for r in runs}
    while sum(share.values()) < n and any(share[r] < len(pools[r]) for r in runs):
        for r in runs:
            if sum(share.values()) < n and share[r] < len(pools[r]):
                share[r] += 1
    picked = [row for r in runs for row in rng.sample(pools[r], share[r])]
    rng.shuffle(picked)
    return picked


def render_sheet(rows: list[dict], name: str) -> str:
    lines = [f"# Re-judging sheet {name}", "",
             f"{len(rows)} sentences from literature-review reports. For each, read the sentence and the "
             "evidence beneath it - the paper's own words the sentence cites - and decide whether the "
             "evidence shown states the sentence. Judge only against the evidence shown, not against what "
             "you know.", "",
             "Grade each row in `grades.template.jsonl` (set `grade`, add a `note` when useful, and put "
             "`scientist:<your name>` in `rater`):", ""]
    lines += [f"- `{grade}`: {meaning}" for grade, meaning in GRADES.items()]
    lines += ["", "Do not open `key.json` until every row is graded.", ""]
    for i, row in enumerate(rows, 1):
        lines += [f"## R{i:02d}", "", f"> {row['claim']}", "", "Evidence:", "", _numbered(row["evidence"])
                  if row["evidence"] else "(none)", ""]
    return "\n".join(lines)


def write_draw(run_ids: list[str], out_dir: Path, *, n: int = DEFAULT_N, seed: int = 0,
               db: Database | None = None) -> Path:
    db = db or Database(load_settings(check_endpoint=False).data_dir / "bench.sqlite3")
    rows = draw({r: claims_of(db, r) for r in run_ids}, n, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_dir.name
    (out_dir / "sheet.md").write_text(render_sheet(rows, name), encoding="utf-8")
    (out_dir / "grades.template.jsonl").write_text("".join(
        json.dumps({"report": name, "s": f"R{i:02d}", "grade": "", "note": "", "rater": ""}) + "\n"
        for i in range(1, len(rows) + 1)), encoding="utf-8")
    key = {"version": KEY_VERSION, "seed": seed, "n": n, "runs": run_ids,
           "rows": [{"s": f"R{i:02d}", **row} for i, row in enumerate(rows, 1)]}
    (out_dir / "key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_dir / "sheet.md"


# --- agree --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    pairs: int
    raw: float | None
    kappa: float | None
    """None when undefined: every grade and verdict on one side is the same."""


def agreement(pairs: list[tuple[bool, bool]]) -> Agreement:
    """Raw agreement and Cohen's kappa for (scientist, judge) pairs, True meaning supported.

    kappa = (p_o - p_e) / (1 - p_e): observed agreement against the agreement
    two raters with these same rates would reach by chance."""
    n = len(pairs)
    if not n:
        return Agreement(0, None, None)
    observed = sum(1 for a, b in pairs if a == b) / n
    a_yes = sum(1 for a, _ in pairs if a) / n
    b_yes = sum(1 for _, b in pairs if b) / n
    chance = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    kappa = None if chance == 1 else (observed - chance) / (1 - chance)
    return Agreement(n, observed, kappa)


def load_grades(path: Path, key_rows: dict[str, dict]) -> dict[str, dict[str, bool]]:
    """Rater -> row -> supported. Blank grades are skipped; anything else
    outside the vocabulary, or a row not on the sheet, is refused."""
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    problems = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        grade = (row.get("grade") or "").strip()
        if not grade:
            continue
        if row.get("s") not in key_rows:
            problems.append(f"line {number}: row {row.get('s')!r} is not on the sheet")
        elif grade not in SCIENTIST_MAP:
            problems.append(f"line {number}, row {row['s']}: grade {grade!r} is not one of {', '.join(GRADES)}")
        elif not (row.get("rater") or "").strip():
            problems.append(f"line {number}, row {row['s']}: no rater; write scientist:<name>")
        else:
            out[row["rater"].strip()][row["s"]] = SCIENTIST_MAP[grade]
    if problems:
        raise ValueError("\n".join(problems))
    return out


def judge_verdicts(key_rows: dict[str, dict], judge_file: Path) -> tuple[dict[str, bool | None], int]:
    """Row -> the file judge's verdict, collapsed; None where no answer is on
    file. Also how many were settled in code (claim verbatim in its evidence)."""
    judge = Judge(FileClient(judge_file), "file")
    out = {}
    for s, row in key_rows.items():
        verdict = judge.support(row["claim"], row["evidence"])
        out[s] = None if verdict is None else JUDGE_MAP[verdict.verdict]
    return out, judge.settled_in_code


def _num(x: float | None) -> str:
    return "undefined" if x is None else f"{x:.2f}"


def render_agreement(key: dict, grades: dict[str, dict[str, bool]], verdicts: dict[str, bool | None],
                     by_code: int, judge_file: Path) -> str:
    rows = {r["s"]: r for r in key["rows"]}
    lines = ["# Scientist against the file judge", "",
             f"Judge: `{judge_file}`. Sheet: {len(rows)} claims from {len(key['runs'])} run(s), seed {key['seed']}.",
             "", "Both collapsed to supported / not supported:", "",
             "- scientist: faithful → supported; misleading, weak_citation, unfaithful → not supported",
             "- judge: supported → supported; partially_supported, unsupported, contradicted → not supported", ""]
    if by_code:
        lines += [f"{by_code} of the judge's verdicts were settled in code (the claim is verbatim in its "
                  "evidence, or it has none), as they are in a benchmark run.", ""]
    for rater, graded in sorted(grades.items()):
        lines += [f"## {rater}", "", "| Run | Question | Claims | Compared | Raw agreement | Cohen's kappa |",
                  "|---|---|---|---|---|---|"]
        everything: list[tuple[bool, bool]] = []
        for run in key["runs"]:
            in_run = [s for s, r in rows.items() if r["run_id"] == run]
            pairs = [(graded[s], verdicts[s]) for s in in_run if s in graded and verdicts[s] is not None]
            everything += pairs
            a = agreement(pairs)
            question = next((rows[s]["question_id"] for s in in_run), "")
            lines.append(f"| {run} | {question} | {len(in_run)} | {a.pairs} | {_num(a.raw)} | {_num(a.kappa)} |")
        a = agreement(everything)
        lines.append(f"| all runs | | {len(rows)} | {a.pairs} | {_num(a.raw)} | {_num(a.kappa)} |")
        unjudged = sum(1 for s in rows if verdicts[s] is None)
        ungraded = sum(1 for s in rows if s not in graded)
        lines += ["", f"Not compared: {unjudged} with no judge answer on file, {ungraded} not graded."]
        disagree = [s for s in sorted(rows) if s in graded and verdicts[s] is not None and graded[s] != verdicts[s]]
        if disagree:
            lines += ["", "Where they disagree:", ""]
            lines += [f"- {s}: scientist {'supported' if graded[s] else 'not supported'}, judge "
                      f"{'supported' if verdicts[s] else 'not supported'} — {rows[s]['claim']}" for s in disagree]
        lines.append("")
    return "\n".join(lines)


def agree(out_dir: Path, *, grades_path: Path | None = None, judge_file: Path = DEFAULT_JUDGE_FILE) -> str:
    key = json.loads((Path(out_dir) / "key.json").read_text(encoding="utf-8"))
    if key.get("version") != KEY_VERSION:
        raise ValueError(f"{out_dir}/key.json is not a {KEY_VERSION} key")
    rows = {r["s"]: r for r in key["rows"]}
    grades = load_grades(grades_path or Path(out_dir) / "grades.jsonl", rows)
    if not grades:
        raise ValueError("no graded rows yet")
    verdicts, by_code = judge_verdicts(rows, judge_file)
    return render_agreement(key, grades, verdicts, by_code, judge_file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.litreview.rejudge_sheet")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draw", help="write a blind sheet from stored benchmark runs")
    d.add_argument("runs", nargs="+", help="run ids stored by `bench run` (as --from-run takes them)")
    d.add_argument("--n", type=int, default=DEFAULT_N)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--db", type=Path, help="the benchmark database (default: <data dir>/bench.sqlite3)")
    d.add_argument("--out", type=Path, help="default: evals/litreview/rejudge/<seed>-<n>/")
    a = sub.add_parser("agree", help="agreement between a scientist's grades and the file judge")
    a.add_argument("dir", type=Path, help="the directory `draw` wrote")
    a.add_argument("--grades", type=Path, help="default: DIR/grades.jsonl")
    a.add_argument("--judge-file", type=Path, default=DEFAULT_JUDGE_FILE)
    args = parser.parse_args(argv)
    try:
        if args.command == "draw":
            out = args.out or EVALS_DIR / "rejudge" / f"{args.seed}-{args.n}"
            print(write_draw(args.runs, out, n=args.n, seed=args.seed, db=Database(args.db) if args.db else None))
        else:
            print(agree(args.dir, grades_path=args.grades, judge_file=args.judge_file), end="")
    except (ValueError, FileNotFoundError) as exc:
        print(f"rejudge_sheet: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
