"""A judge whose answers come from a file instead of a model.

The benchmark's judge was the same 4B model that wrote the reports, which
`BENCHMARK_RESULTS.md` listed as the weakest part of the instrument. This is
the seam for replacing it without weakening anything else: the pipeline still
runs entirely on the local model, and only the judging step reads its answers
from a file that a stronger model — or a scientist — wrote.

Two passes:

    python -m bedrock.litreview.bench run ... --judge-file evals/litreview/judge/answers.jsonl

The first pass finds no answer for anything, records every question it wanted
to ask in `answers.pending.jsonl`, and scores what it can. Somebody answers
those questions. The second pass finds them and scores in full. Answers
already on file are reused, so after a change only the genuinely new questions
come back — the same incremental behaviour as the model call cache.

What is *not* delegated: the checks in `judge.verdict_from` and
`judge.coverage_from` still run over these answers exactly as they run over a
model's. A judge that claims a sentence is in the evidence is still checked
against the evidence. Being a better judge earns no more trust.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bedrock.litreview.llm.client import LLMError

SCHEMA_TITLE = "title"


def task_key(prompt: str) -> str:
    """Stable id for one judging question. The prompt carries the claim and
    the evidence, so the same question keys the same however often it is
    asked, and a changed report only invalidates the claims that changed."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class FileClient:
    """Answers model calls from a JSONL file; records misses as pending.

    Implements the `LLMClient` protocol, so `Judge` cannot tell the
    difference — and neither can the scoring code.
    """

    def __init__(self, path: str | Path, version: str = "file") -> None:
        self.path = Path(path)
        self.pending_path = self.path.with_suffix(".pending.jsonl")
        self._version = version
        self.answers: dict[str, str] = {}
        self.pending: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.answers[row["key"]] = json.dumps(row["answer"])

    def complete(self, prompt: str, *, model: str, schema=None, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        key = task_key(prompt)
        if key in self.answers:
            self.hits += 1
            return self.answers[key]
        self.misses += 1
        self.pending.setdefault(key, {
            "key": key,
            "kind": (schema or {}).get(SCHEMA_TITLE, "unknown"),
            "prompt": prompt,
        })
        raise LLMError(f"no answer on file for judging task {key}")

    def flush(self) -> Path | None:
        """Write the unanswered questions, or delete the file when there are
        none. Leaving yesterday's list behind is how somebody spends an hour
        answering questions that were answered already."""
        if not self.pending:
            self.pending_path.unlink(missing_ok=True)
            return None
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        with self.pending_path.open("w", encoding="utf-8") as fh:
            for row in self.pending.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self.pending_path

    @property
    def model_version(self) -> str:
        return self._version

    def version_of(self, model: str) -> str:
        return self._version
