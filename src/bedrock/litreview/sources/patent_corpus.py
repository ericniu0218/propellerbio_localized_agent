"""A frozen set of patent records, saved as JSON lines — the same handoff
shape `sources/corpus.py` gives papers, kept as its own module because
patents are never mixed into a paper corpus (see `patents_base.py`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from bedrock.litreview.models import PatentRecord
from bedrock.litreview.sources.corpus import check_sidecar, sidecar_path


def read_patents(path: Path) -> list[PatentRecord]:
    data = Path(path).read_bytes()
    check_sidecar(path, hashlib.sha256(data).hexdigest())
    return [PatentRecord.model_validate_json(line)
            for line in data.decode("utf-8").split("\n") if line.strip()]


def write_patents(records: list[PatentRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(r.model_dump_json(exclude_none=True) + "\n" for r in records).encode("utf-8")
    path.write_bytes(data)
    sidecar_path(path).write_text(hashlib.sha256(data).hexdigest() + "\n", encoding="utf-8")
