"""A frozen corpus: papers saved as JSON lines, served as a searcher.

This is the handoff format between the searcher and the pipeline when they run
separately — one `Paper` per line, as `Paper.model_dump_json()` writes it —
and it is how the benchmark's sample corpora are stored, so that a benchmark
run reads exactly the papers it read last time.

`write_papers` also writes `<file>.sha256`, the file's content hash. A
`CorpusSearcher` refuses a file that no longer matches it: a corpus edited
after it was written is not the corpus it claims to be. A file with no
sidecar (older corpora, hand-made files) is read as it is, and its hash is
still recorded on the run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from bedrock.litreview.models import Paper
from bedrock.litreview.sources.base import SearchRequest


def read_papers(path: Path) -> list[Paper]:
    """Every paper in a JSON-lines file, refusing a file changed since it was
    written (its `.sha256` sidecar, when it has one). Every reader — the
    runner's `CorpusSearcher` and the benchmark alike — goes through this."""
    data = Path(path).read_bytes()
    check_sidecar(path, hashlib.sha256(data).hexdigest())
    return _parse(data.decode("utf-8"))


def check_sidecar(path: Path, sha256: str) -> None:
    sidecar = sidecar_path(path)
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").strip()
        if expected != sha256:
            raise ValueError(
                f"corpus file changed since it was written: {path} hashes to {sha256}, "
                f"its {sidecar.name} says {expected}"
            )


def _parse(text: str) -> list[Paper]:
    # Split on newline only: str.splitlines() also breaks on U+2028 and
    # friends, which JSON leaves unescaped inside a string.
    return [Paper.model_validate_json(line) for line in text.split("\n") if line.strip()]


def sidecar_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_papers(papers: list[Paper], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(paper.model_dump_json(exclude_none=True) + "\n" for paper in papers).encode("utf-8")
    path.write_bytes(data)
    sidecar_path(path).write_text(hashlib.sha256(data).hexdigest() + "\n", encoding="utf-8")


class CorpusSearcher:
    """Returns every paper in the file, whatever the question.

    Deliberately no filtering: a frozen corpus stands in for what the real
    searcher returned, and deciding what is relevant is the pipeline's job.
    """

    name = "corpus"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        # Read once: the papers served are the bytes that were hashed.
        self._data = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self._data).hexdigest()
        check_sidecar(self.path, self.sha256)

    def search(self, request: SearchRequest) -> list[Paper]:
        return _parse(self._data.decode("utf-8"))[: request.max_results]
