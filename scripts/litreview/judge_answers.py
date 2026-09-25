#!/usr/bin/env python3
"""Turn compact judging answers into the file the benchmark reads.

The benchmark asks the judge to copy the evidence sentence that states a
claim. Retyping a 300-character sentence by hand is how transcription errors
get in, so an answer here names a short *fragment* instead and this script
expands it to the exact sentence from the task's own prompt. A fragment that
matches nothing, or matches more than one sentence, is an error rather than a
guess.

Usage:

    python scripts/judge_answers.py evals/litreview/judge/answers.pending.jsonl \\
        my_answers.json evals/litreview/judge/answers.jsonl

`my_answers.json` maps a task key to one of:

    {"contains": "fragment of the evidence sentence",   # support
     "missing": ["detail", ...],                        # optional
     "reasoning": "one line"}                           # optional

    {"contains": null, "reasoning": "nothing states it"}    # unsupported

    {"nuggets": [{"fact": "...", "contains": "fragment", "importance": "vital"}]}

Answers already in the output file are kept; keys answered again are replaced.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def evidence_block(prompt: str) -> str:
    """The part of the prompt the judge may copy from."""
    for start, end in (
        ("Evidence (quoted from research papers):", "## Answer each field"),
        ("Reference (", "List up to"),
        ("Answer:", "## Answer each field"),
    ):
        if start in prompt and end in prompt:
            body = prompt.split(start, 1)[1].split(end, 1)[0]
            return body.split("):\n", 1)[1] if start == "Reference (" else body
    raise ValueError("cannot find the quotable block in this prompt")


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip the "[1] " label and the trailing "(Result: ...)" annotation,
        # neither of which is part of the paper's sentence.
        line = re.sub(r"^\[\d+\]\s*", "", line).strip().strip('"')
        line = re.sub(r"\s*\(Result:.*?\)\s*$", "", line, flags=re.S)
        out.extend(s.strip() for s in SENTENCE.split(line) if s.strip())
    return out


def expand(fragment: str, prompt: str) -> str:
    """The one sentence of the evidence containing `fragment`, verbatim."""
    hits = [s for s in sentences(evidence_block(prompt)) if fragment in s]
    if not hits:
        raise SystemExit(f"fragment not found in the evidence: {fragment!r}")
    if len({h for h in hits}) > 1:
        raise SystemExit(f"fragment matches {len(hits)} sentences, be more specific: {fragment!r}")
    return hits[0]


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    pending_path, mine_path, out_path = (Path(a) for a in argv[1:])
    tasks = {json.loads(line)["key"]: json.loads(line) for line in pending_path.read_text().splitlines() if line.strip()}
    mine = json.loads(mine_path.read_text())

    existing: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["key"]] = row

    written = 0
    for key, given in mine.items():
        task = tasks.get(key)
        if task is None:
            raise SystemExit(f"no pending task with key {key}")
        prompt = task["prompt"]
        if "nuggets" in given:
            answer = {"nuggets": [
                {"fact": n["fact"], "quote": expand(n["contains"], prompt), "importance": n.get("importance", "okay")}
                for n in given["nuggets"]
            ]}
        elif task["kind"] == "CoverageAnswers":
            answer = {
                "reasoning": given.get("reasoning", ""),
                "answer_sentence": expand(given["contains"], prompt) if given.get("contains") else "",
            }
        else:
            answer = {
                "reasoning": given.get("reasoning", ""),
                "evidence_sentence": expand(given["contains"], prompt) if given.get("contains") else "",
                "missing": given.get("missing", []),
            }
        existing[key] = {"key": key, "answer": answer}
        written += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in existing.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {written} answer(s); {len(existing)} on file at {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
