"""The contract every data-type adapter implements.

This is the same seam idea as ``bedrock.inference.base.InferenceBackend``: the
pipeline only ever talks to a ``SourceAdapter``, so supporting a new kind of
data means adding one file in this package and registering it. The chunker,
the store, the search and the UI never change.

An adapter's job is to turn one source into an ``Extraction``: a deterministic
plain-English ``overview`` plus the ``segments`` worth keeping.

The important design point is that *different data compresses well in
different ways*, and a single "ask a model to summarise it" pass throws away
structure that is already there for free:

* a table's real schema and column statistics beat any prose description,
* a log file's repeated patterns collapse a hundred thousand lines into a few
  hundred templates with counts,
* a module's function signatures are the summary; the bodies are the detail.

So the protocol is generic and the strategy inside each adapter is specific.
That is what makes this work across data types without becoming mush.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bedrock.context.types import Extraction

MAX_DECODE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Source:
    """One thing to ingest, already decoded to text.

    Binary formats that need real parsing (PDF, Parquet, images) are out of
    scope for now and deliberately so - each is its own adapter plus its own
    dependency, and none of them are needed to prove the pipeline works.
    """

    name: str
    text: str
    suffix: str = ""
    byte_size: int = 0

    @classmethod
    def from_path(cls, path: Path | str) -> "Source":
        path = Path(path)
        raw = path.read_bytes()
        if len(raw) > MAX_DECODE_BYTES:
            raise ValueError(
                f"{path.name} is {len(raw) / 1_000_000:.0f} MB, over the "
                f"{MAX_DECODE_BYTES // 1_000_000} MB limit for a single source."
            )
        # errors="replace" so one stray byte in an otherwise fine text file
        # does not fail the whole ingest.
        return cls(
            name=path.name,
            text=raw.decode("utf-8", errors="replace"),
            suffix=path.suffix.lower(),
            byte_size=len(raw),
        )

    @classmethod
    def from_text(cls, text: str, *, name: str = "untitled", suffix: str = "") -> "Source":
        return cls(
            name=name,
            text=text,
            suffix=suffix.lower(),
            byte_size=len(text.encode("utf-8")),
        )

    def lines(self, limit: int | None = None) -> list[str]:
        split = self.text.splitlines()
        return split[:limit] if limit else split


class SourceAdapter(Protocol):
    """Everything the pipeline needs from one data-type handler."""

    name: str

    def can_handle(self, source: Source) -> bool:
        """True if this adapter understands the source. Must not raise."""
        ...

    def extract(self, source: Source) -> Extraction:
        """Produce the overview and segments for this source."""
        ...
