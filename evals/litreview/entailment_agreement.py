"""How often does the optional entailment check agree with the strong judge?

    python -m evals.litreview.entailment_agreement \\
        evals/litreview/results/<question>/<stamp>/pipeline/<run>/<report>/report.json [...] \\
        --models laptop

The entailment check (`src/bedrock/litreview/report/entailment.py`) is off by
default because nobody has measured it, and a check that drops a sentence the
paper really supports loses real evidence. This is the measurement. For every
kept sentence of a stored report it asks two questions of the same sentence
and the same cited findings:

  judge        the strong judge's verdict, read from
               `evals/litreview/judge/answers.jsonl` through the benchmark's
               own code path (`bench.judge.Judge` over `bench.filejudge.FileClient`,
               evidence built by `bench.run._unit_evidence`), so a sentence
               the benchmark judged gets exactly the benchmark's verdict. A
               sentence with no answer on file is "unjudged"; nothing is
               written to the pending file
  entailment   the check's verdict, from `EntailmentCheck.check` - the very
               code Stage 7 runs - on the local model, through bedrock's
               scheduler and the call cache, like the benchmark. It is shown
               what Stage 7 shows it (`guard.source_text`), which also carries
               a finding's qualifying next sentence; the judge's evidence
               (quote and context sentence) does not

and prints a confusion matrix of the two, folded to three classes:

    supported     judge "supported"            check "supported"
    partial       judge "partially_supported"  check "partial"
    unsupported   judge "unsupported" or       check "not_supported" or
                  "contradicted"               "contradicted"

The two cells that matter most are named under the table: sentences the check
would DROP although the judge calls them supported (evidence lost), and
sentences it would KEEP although the judge calls them unsupported (what it
failed to catch). The judge is a stronger model than the check, not ground
truth; a scientist's review of the disagreements is the other half of the
measurement before the check is switched on anywhere.

The stored report is all this needs. It carries each finding's quote, context
sentence and qualifier - the paper's own text at recorded offsets, which the
integrity check proved when the report was written - and those are exactly
what both the judge and the check are shown.

Tests drive `compare` and `format_table` with `FakeLLMClient`; nothing here
needs a model except `main`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from bedrock.litreview.bench.claims import pipeline_claims
from bedrock.litreview.bench.filejudge import FileClient
from bedrock.litreview.bench.judge import Judge
from bedrock.litreview.bench.run import _unit_evidence
from bedrock.litreview.models import Report
from bedrock.litreview.report.entailment import EntailmentCheck

ANSWERS = Path(__file__).parent / "judge" / "answers.jsonl"

CLASSES = ("supported", "partial", "unsupported")
JUDGE_CLASS = {
    "supported": "supported",
    "partially_supported": "partial",
    "unsupported": "unsupported",
    "contradicted": "unsupported",
}
CHECK_CLASS = {
    "supported": "supported",
    "partial": "partial",
    "not_supported": "unsupported",
    "contradicted": "unsupported",
}
UNJUDGED = "unjudged"
FAILED = "failed"


@dataclass
class Row:
    section: str
    text: str
    citations: list[str]
    judge: str  # a class, or UNJUDGED
    check: str  # a class, or FAILED
    reason: str = ""


def judge_classes(report: Report, answers: Path = ANSWERS) -> list[str]:
    """The strong judge's class for each kept sentence, in `kept_sentences` order."""
    judge = Judge(FileClient(answers, version=f"file:{answers.name}"), model="judge")
    units = {report.short_ids[u.id]: u for u in report.evidence if u.id in report.short_ids}
    out = []
    for claim in pipeline_claims(report):
        cited = [units[c] for c in claim.citations if c in units]
        verdict = judge.support(claim.text, [_unit_evidence(u) for u in cited])
        out.append(JUDGE_CLASS[verdict.verdict] if verdict else UNJUDGED)
    return out


def compare(report: Report, check: EntailmentCheck, answers: Path = ANSWERS) -> list[Row]:
    """One row per kept sentence: the judge's class beside the check's."""
    units = {report.short_ids[u.id]: u for u in report.evidence if u.id in report.short_ids}
    rows = []
    for (section, sentence), judged in zip(report.kept_sentences(), judge_classes(report, answers), strict=True):
        cited = {c: units[c] for c in sentence.citations if c in units}
        verdict = check.check(sentence.text, cited)
        rows.append(Row(
            section=section, text=sentence.text, citations=list(sentence.citations), judge=judged,
            check=CHECK_CLASS[verdict.label] if verdict.label else FAILED, reason=verdict.reason,
        ))
    return rows


def confusion(rows: list[Row]) -> Counter:
    """(judge class, check class) -> number of sentences."""
    return Counter((r.judge, r.check) for r in rows)


def format_table(rows: list[Row], title: str = "") -> str:
    matrix = confusion(rows)
    judged = [r for r in rows if r.judge != UNJUDGED and r.check != FAILED]
    agree = sum(1 for r in judged if r.judge == r.check)
    cols = [*CLASSES, FAILED]
    lines = [f"## {title}" if title else "", "",
             "Rows: the strong judge. Columns: the entailment check.", "",
             "| judge \\ check | " + " | ".join(cols) + " | total |",
             "|---|" + "---|" * (len(cols) + 1)]
    for judge_class in [*CLASSES, UNJUDGED]:
        counts = [matrix[(judge_class, c)] for c in cols]
        lines.append(f"| {judge_class} | " + " | ".join(str(n) for n in counts) + f" | {sum(counts)} |")
    lines += [
        "",
        f"Kept sentences: {len(rows)}. Judged by both: {len(judged)}. "
        f"Agree: {agree}" + (f" ({agree / len(judged):.0%})" if judged else "") + ".",
        f"Would drop although the judge calls them supported (evidence lost): "
        f"{matrix[('supported', 'unsupported')]}.",
        f"Would keep although the judge calls them unsupported (not caught): "
        f"{matrix[('unsupported', 'supported')] + matrix[('unsupported', 'partial')]}.",
    ]
    lost = [r for r in rows if r.judge == "supported" and r.check == "unsupported"]
    if lost:
        lines += ["", "Sentences the check would drop that the judge supports - read every one:"]
        lines += [f"- [{r.section}] {r.text} ({', '.join(r.citations)}) - check: {r.reason}" for r in lost]
    return "\n".join(lines).lstrip("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.litreview.entailment_agreement")
    parser.add_argument("reports", nargs="+", type=Path, help="stored report.json files")
    parser.add_argument("--answers", type=Path, default=ANSWERS, help="the strong judge's answers")
    parser.add_argument("--models", help="model set from defaults/models.yaml (office, laptop, medgemma)")
    args = parser.parse_args(argv)

    from bedrock.config import load_settings as load_bedrock_settings
    from bedrock.litreview.config import load_settings
    from bedrock.litreview.llm import make_client
    from bedrock.litreview.report.entailment import JOB
    from bedrock.litreview.service import standalone_scheduler

    settings = load_settings(model_set=args.models)
    reports = [(path, Report.model_validate_json(path.read_text(encoding="utf-8"))) for path in args.reports]

    async def go() -> list[tuple[Path, list[Row]]]:
        # The same route as the benchmark: a scheduler of its own, every call
        # at USER_JOB through the call cache, the blocking work on a thread.
        async with standalone_scheduler(load_bedrock_settings()) as scheduler:
            client = make_client(settings, scheduler=scheduler, loop=asyncio.get_running_loop())
            check = EntailmentCheck(client, settings.model_for(JOB))
            out = []
            for path, report in reports:
                print(f"checking {path}", file=sys.stderr, flush=True)
                out.append((path, await asyncio.to_thread(compare, report, check, args.answers)))
            return out

    results = asyncio.run(go())
    for path, rows in results:
        print(format_table(rows, title=str(path)) + "\n")
    if len(results) > 1:
        print(format_table([r for _, rows in results for r in rows], title="All reports"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
