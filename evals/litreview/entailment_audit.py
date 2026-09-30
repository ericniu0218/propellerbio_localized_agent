"""How often is the entailment check right, measured against the hand audit?

    python -m evals.litreview.entailment_audit --models laptop \\
        [--rows OUT.jsonl] [--before EARLIER.jsonl] [--sheet OUT.md]

`entailment_agreement.py` measures the optional entailment check
(`src/bedrock/litreview/report/entailment.py`) against the strong judge, on
stored benchmark reports. This measures it against the hand audits in
`evals/litreview/hand_audit/`: 105 report sentences, 15 sampled from each of
seven reports, each graded by a reader who had the paper's text around the
quote and the abstract in front of them (`grades.jsonl`, `grades_after.jsonl`).
It needs nothing but this repository and a model: every sheet shows, for
each cited finding, the quote, the context sentence, the qualifier and the
naming sentence, which are exactly what the check reads.

Why a second reference when the judge exists. The judge is shown the quote
and every sentence beside it as one passage (`bench/run._unit_evidence`), and
a claim that is verbatim anywhere in it is settled "supported" in code. So a
sentence copied from a finding's qualifier while citing a quote that says
something else - OPEN_ISSUES #3, the one failure class the second hand audit
found - is "supported" to the judge by construction. The hand audit read the
sentence against the quote it cites, and graded those `weak_citation` or
`misleading`. That is the failure the narrowed check exists to catch, so it
is measured against the reader who can see it.

For each sentence it prints:

  hand       the audit's grade, folded to faithful / not faithful
  check 10   what guard check 10 (`report/guard.py` `rests_on_quote`) does
             with it today: "kept", or "drops" when the sentence takes its
             content from the sentences beside a quote rather than the quote.
             The guard runs first in Stage 7 and the check only ever sees
             what it kept, so a sentence check 10 drops never reaches it -
             but check 10 is a word count with a tuned threshold, and the
             entailment check is the meaning-level second look behind it
  judge      the strong judge's verdict from `judge/answers.jsonl` through the
             benchmark's own code path, or "unjudged" when the file holds no
             answer for this exact claim and evidence (the older sheets were
             judged with less evidence shown, so most of those are)
  check      the entailment check's class, from `EntailmentCheck.check` - the
             code Stage 7 runs - on the local model, through bedrock's
             scheduler and the call cache

and a confusion matrix of hand grade against check class, over the sentences
check 10 keeps and over all of them. The two numbers that decide whether the
check can be switched on are named under the table: faithful sentences it
would drop (evidence lost) and unfaithful ones it would keep.

`--rows` writes one JSON line per sentence; `--before` reads such a file from
an earlier run (another prompt version, another model) and lists every
sentence whose verdict changed, which is the before-and-after the plan asks
for. `--sheet` writes what the check would drop, one sentence per entry with
the cited text split by role, for a scientist to mark right or wrong.

What this does NOT measure: the hand audits were graded by AI models, not a
scientist (OPEN_ISSUES #21), and 15 sentences a report is a small sample
with 12 unfaithful sentences in it. The sheet is how a scientist closes that.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bedrock.litreview.bench.filejudge import FileClient
from bedrock.litreview.bench.judge import Judge
from bedrock.litreview.bench.run import _unit_evidence
from bedrock.litreview.knowledge import Knowledge, clean, load_knowledge
from bedrock.litreview.models import EvidenceUnit
from bedrock.litreview.report.entailment import BESIDE, EntailmentCheck
from bedrock.litreview.report.guard import rests_on_quote
from evals.litreview.entailment_agreement import CHECK_CLASS, CLASSES, FAILED, JUDGE_CLASS, UNJUDGED

HAND_AUDIT = Path(__file__).parent / "hand_audit"
ANSWERS = Path(__file__).parent / "judge" / "answers.jsonl"
GRADE_FILES = ("grades.jsonl", "grades_after.jsonl")

FAITHFUL = "faithful"
NOT_FAITHFUL = "not faithful"

#: The second hand audit's five non-faithful sentences, all of OPEN_ISSUES #3
#: as it was recorded. Four take their content from a sentence shown beside
#: the quote; asthma S25 is the quote itself with its hedge dropped.
ISSUE_3 = (
    ("qwen_obesity_after", "S6"), ("qwen_obesity_after", "S18"), ("qwen_obesity_after", "S29"),
    ("qwen_migraine_after", "S5"), ("qwen_asthma_after", "S25"),
)

_SENTENCE = re.compile(r"^## (S\d+) \[(.*)\]$")
_UNIT = re.compile(r"^### (E\d+) = (\S+) \((\w+)\)")
_TARGET = re.compile(r"^target=(.*?) outcome=")
_FIELDS = {"quote: ": "quote", "context: ": "context_quote", "qualifier: ": "qualifier_quote"}
_NAMING = re.compile(r"^naming \((\w+)\): (.*)$")
_METHODS = re.compile(r"^methods studied only: (.*) \(the quote is a conclusion naming people\)$")
_SECTIONS = ("abstract", "results", "discussion", "methods", "other")


@dataclass
class Case:
    report: str
    """The sheet's name without `.md`, as the grades files name it."""
    sid: str
    section: str
    text: str
    citations: list[str]
    units: dict[str, EvidenceUnit] = field(default_factory=dict)
    """Short id -> the finding as the sheet shows it. A citation the sheet
    marks NOT IN EVIDENCE has no entry."""
    grade: str = ""
    note: str = ""

    @property
    def hand(self) -> str:
        return FAITHFUL if self.grade == FAITHFUL else NOT_FAITHFUL


def _unit(short_id: str, paper_id: str, section: str, report: str, fields: dict) -> EvidenceUnit:
    """The finding the sheet shows. Only its text fields and its "about" name
    matter to the check; the rest are placeholders the model requires."""
    return EvidenceUnit(
        id=f"{report}:{short_id}", paper_id=paper_id, question_id=report,
        target=fields.get("target_raw") or "?", target_raw=fields.get("target_raw") or "?",
        direction="unclear", modality="other", disease="?", disease_raw="?", evidence_type="in_vivo",
        species="other", species_raw="", outcome="association", effect_detail="", n=None, combo_with=None,
        target_is_primary=True, quote=fields.get("quote", ""), quote_status="exact",
        section=section if section in _SECTIONS else "other",
        context_quote=fields.get("context_quote"), qualifier_quote=fields.get("qualifier_quote"),
        naming_quote=fields.get("naming_quote"), naming_section=fields.get("naming_section"),
        methods_subjects=fields.get("methods_subjects"), extracted_by_model="hand_audit", prompt_version="-",
    )


def read_sheet(path: Path) -> list[Case]:
    """Every sampled sentence on one sheet `hand_audit/audit_sheet.py` wrote,
    with the findings it cites rebuilt from the lines the sheet prints. The
    paper excerpts in code fences are skipped: they are for the reader."""
    report = path.stem
    cases: list[Case] = []
    unit: tuple[str, str, str] | None = None
    fields: dict = {}
    fenced = False

    def close_unit() -> None:
        nonlocal unit, fields
        if unit and cases:
            cases[-1].units[unit[0]] = _unit(*unit, report, fields)
        unit, fields = None, {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if (m := _SENTENCE.match(line)):
            close_unit()
            cases.append(Case(report=report, sid=m[1], section=m[2], text="", citations=[]))
        elif not cases:
            continue
        elif line.startswith("> ") and not cases[-1].text:
            cases[-1].text = line[2:]
        elif line.startswith("cites: "):
            cases[-1].citations = [c.strip() for c in line[len("cites: "):].split(",") if c.strip()]
        elif (m := _UNIT.match(line)):
            close_unit()
            unit = (m[1], m[2], m[3])
        elif unit is None:
            continue
        elif (m := _TARGET.match(line)):
            fields["target_raw"] = ast.literal_eval(m[1])
        elif (m := _NAMING.match(line)):
            fields["naming_section"], fields["naming_quote"] = m[1], m[2]
        elif (m := _METHODS.match(line)):
            fields["methods_subjects"] = m[1]
        else:
            for prefix, name in _FIELDS.items():
                if line.startswith(prefix):
                    fields[name] = line[len(prefix):]
    close_unit()
    return cases


def cases(directory: Path = HAND_AUDIT) -> list[Case]:
    """Every graded sentence of every sheet, with its grade and note. A sheet
    no grades file mentions is not read; a graded sentence missing from its
    sheet is an error, since the sheets and grades are committed together."""
    grades = {}
    for name in GRADE_FILES:
        for line in (directory / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                grades[(row["report"], row["s"])] = row
    out = []
    for report in sorted({r for r, _ in grades}):
        by_sid = {c.sid: c for c in read_sheet(directory / f"{report}.md")}
        for (r, sid), row in grades.items():
            if r != report:
                continue
            if sid not in by_sid:
                raise ValueError(f"{report} {sid} is graded but not on its sheet")
            case = by_sid[sid]
            case.grade, case.note = row["grade"], row.get("note", "")
            out.append(case)
    return sorted(out, key=lambda c: (c.report, int(c.sid[1:])))


def masked(sentence: str, kb: Knowledge) -> str:
    """The sentence with known names blanked, as the guard reads numbers from it."""
    out = clean(sentence)
    for _, match in kb.scan(out):
        out = out[: match.start] + " " * (match.end - match.start) + out[match.end :]
    return out


def check10(case: Case, kb: Knowledge) -> str:
    """"kept" or "drops": guard check 10 alone, as `CitationGuard.check` applies it."""
    cited = [case.units[c] for c in case.citations if c in case.units]
    if not cited:
        return "kept"
    m = masked(case.text, kb)
    return "kept" if any(rests_on_quote(clean(case.text), m, u) for u in cited) else "drops"


def judge_class(case: Case, judge: Judge) -> str:
    cited = [case.units[c] for c in case.citations if c in case.units]
    verdict = judge.support(case.text, [_unit_evidence(u) for u in cited])
    return JUDGE_CLASS[verdict.verdict] if verdict else UNJUDGED


@dataclass
class Row:
    report: str
    sid: str
    section: str
    text: str
    citations: list[str]
    grade: str
    hand: str
    check10: str
    judge: str
    check: str
    reason: str = ""


def measure(all_cases: list[Case], check: EntailmentCheck, *, kb: Knowledge | None = None,
            answers: Path = ANSWERS) -> list[Row]:
    kb = kb or load_knowledge()
    judge = Judge(FileClient(answers, version=f"file:{answers.name}"), model="judge")
    rows = []
    for case in all_cases:
        cited = {c: case.units[c] for c in case.citations if c in case.units}
        verdict = check.check(case.text, cited)
        rows.append(Row(
            report=case.report, sid=case.sid, section=case.section, text=case.text,
            citations=list(case.citations), grade=case.grade, hand=case.hand,
            check10=check10(case, kb), judge=judge_class(case, judge),
            check=CHECK_CLASS[verdict.label] if verdict.label else FAILED, reason=verdict.reason,
        ))
    return rows


def _matrix(rows: list[Row]) -> list[str]:
    counts = Counter((r.hand, r.check) for r in rows)
    cols = [*CLASSES, FAILED]
    lines = ["| hand \\ check | " + " | ".join(cols) + " | total |", "|---|" + "---|" * (len(cols) + 1)]
    for hand in (FAITHFUL, NOT_FAITHFUL):
        cells = [counts[(hand, c)] for c in cols]
        lines.append(f"| {hand} | " + " | ".join(str(n) for n in cells) + f" | {sum(cells)} |")
    return lines


def format_table(rows: list[Row], title: str = "") -> str:
    reached = [r for r in rows if r.check10 == "kept"]
    lost = [r for r in reached if r.hand == FAITHFUL and r.check == "unsupported"]
    missed = [r for r in reached if r.hand == NOT_FAITHFUL and r.check != "unsupported"]
    lines = [f"## {title}" if title else "", "",
             "Rows: the hand audit. Columns: the entailment check.", "",
             f"### The {len(reached)} sentences guard check 10 keeps (what the check would see)", "",
             *_matrix(reached), "",
             f"Would drop although the hand audit calls them faithful (evidence lost): {len(lost)}.",
             f"Would keep although the hand audit calls them unfaithful (not caught): {len(missed)}.", "",
             f"### All {len(rows)} sentences, check 10 or not", "", *_matrix(rows), "",
             "### OPEN_ISSUES #3", "",
             "| sentence | hand | check 10 | judge | check |", "|---|---|---|---|---|"]
    by_key = {(r.report, r.sid): r for r in rows}
    for key in ISSUE_3:
        if key in by_key:
            r = by_key[key]
            lines.append(f"| {r.report} {r.sid} | {r.grade} | {r.check10} | {r.judge} | {r.check} |")
    if lost:
        lines += ["", "Faithful sentences the check would drop - read every one:"]
        lines += [f"- {r.report} {r.sid}: {r.text} - check: {r.reason}" for r in lost]
    if missed:
        lines += ["", "Unfaithful sentences the check would keep:"]
        lines += [f"- {r.report} {r.sid} ({r.grade}, check says {r.check}): {r.text}" for r in missed]
    return "\n".join(lines).lstrip("\n")


def changes(before: list[dict], after: list[Row]) -> str:
    """Every sentence whose check class differs between two runs."""
    earlier = {(r["report"], r["sid"]): r for r in before}
    lines = []
    for r in after:
        old = earlier.get((r.report, r.sid))
        if old and old["check"] != r.check:
            lines.append(f"- {r.report} {r.sid} ({r.grade}): {old['check']} -> {r.check}. {r.text}"
                         + (f" - now: {r.reason}" if r.reason else ""))
    head = f"{len(lines)} of {len(after)} sentences changed class."
    return "\n".join([head, *lines])


def _roles(unit: EvidenceUnit) -> list[str]:
    """The cited text by role, with the labels the check itself is shown."""
    out = [f"  - quote: \"{unit.quote}\""]
    for role, name in BESIDE:
        if getattr(unit, name):
            out.append(f"  - {role}: \"{getattr(unit, name)}\"")
    return out


def review_sheet(rows: list[Row], all_cases: list[Case]) -> str:
    """What the check would drop, for a scientist to mark. No hand grade is
    shown, so the mark is the scientist's own."""
    by_key = {(c.report, c.sid): c for c in all_cases}
    dropped = [r for r in rows if r.check == "unsupported"]
    lines = ["# Entailment check: sentences it would remove", "",
             f"{len(dropped)} report sentences. For each, read the sentence and the paper's text it cites.",
             "Each finding's quote is the sentence the report cites for its result; the context, next and",
             "naming sentences are shown beside it in the report and may add a name, who was studied, a",
             "sample size or a limit.", "",
             "Mark one: **right** (the cited quote does not support the sentence, so removing it is correct) or",
             "**wrong** (the quote supports it and removing it loses real evidence), and a note if useful.", ""]
    for i, r in enumerate(dropped, 1):
        case = by_key[(r.report, r.sid)]
        lines += [f"## {i}. {r.report} {r.sid}", "", f"> {r.text}", ""]
        for c in r.citations:
            if c in case.units:
                lines += [f"- {c}, about {case.units[c].target_raw}:", *_roles(case.units[c])]
        lines += ["", f"The check's reason: {r.reason}", "", "Mark: right / wrong    Note:", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.litreview.entailment_audit")
    parser.add_argument("--models", help="model set from defaults/models.yaml (office, laptop, medgemma)")
    parser.add_argument("--answers", type=Path, default=ANSWERS, help="the strong judge's answers")
    parser.add_argument("--rows", type=Path, help="write one JSON line per sentence here")
    parser.add_argument("--before", type=Path, help="an earlier --rows file: list what changed")
    parser.add_argument("--sheet", type=Path, help="write the scientist's review sheet here")
    args = parser.parse_args(argv)

    from bedrock.config import load_settings as load_bedrock_settings
    from bedrock.litreview.config import load_settings
    from bedrock.litreview.llm import load_prompt, make_client
    from bedrock.litreview.report.entailment import JOB, PROMPT
    from bedrock.litreview.service import standalone_scheduler

    # The judge warns once per sentence it has no answer for; the table
    # already says "unjudged".
    logging.getLogger("bedrock.litreview.bench.judge").setLevel(logging.ERROR)
    settings = load_settings(model_set=args.models)
    all_cases = cases()

    async def go() -> list[Row]:
        # The benchmark's route: a scheduler of its own, every call at
        # USER_JOB through the call cache, the blocking work on a thread.
        async with standalone_scheduler(load_bedrock_settings()) as scheduler:
            client = make_client(settings, scheduler=scheduler, loop=asyncio.get_running_loop())
            check = EntailmentCheck(client, settings.model_for(JOB))
            return await asyncio.to_thread(measure, all_cases, check, answers=args.answers)

    rows = asyncio.run(go())
    title = (f"Entailment check (prompt v{load_prompt(PROMPT).version}, {settings.model_for(JOB)}) "
             f"against the hand audit")
    print(format_table(rows, title=title))
    if args.rows:
        args.rows.write_text("".join(json.dumps(asdict(r)) + "\n" for r in rows), encoding="utf-8")
    if args.before:
        before = [json.loads(line) for line in args.before.read_text(encoding="utf-8").splitlines() if line]
        print("\n## Changed since " + str(args.before) + "\n\n" + changes(before, rows))
    if args.sheet:
        args.sheet.write_text(review_sheet(rows, all_cases), encoding="utf-8")
        print(f"\nwrote {args.sheet}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
