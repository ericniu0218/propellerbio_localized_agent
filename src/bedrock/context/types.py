"""The plain data structures the context pipeline passes around.

Deliberately boring and dependency-free, the same idea as
``bedrock.models.conversation``: adapters produce these, the store writes them
to disk, and the UI renders them. Nothing in this module knows about
networking, Textual, or compression.

The pipeline these flow through is::

    Source  --adapter-->  Extraction (segments + a plain-English overview)
            --chunker-->  [Chunk, ...]
            --store---->  Artifact   (digest + compressed chunks on disk)

The two-layer idea in one sentence: the *digest* is small enough to hand a
model on every turn, and the *chunks* hold the full text for when it needs an
exact answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Roughly how many tokens ``text`` will cost a model.

    Four characters per token is the usual rule of thumb for English prose and
    it is close enough for the only two things this is used for: telling the
    user how much a digest saved them, and keeping a prompt inside a sane
    budget. It is an estimate, not a real tokenizer - do not use it for
    anything that has to be exact.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Segment:
    """One meaningful piece of the source, before it is packed into chunks.

    ``weight`` is how much an adapter thinks this piece matters when we are
    choosing a small sample to show a model. A CSV's schema block is worth far
    more than any individual row, so the tabular adapter gives it a high
    weight; ordinary prose paragraphs all sit at 1.0.
    """

    text: str
    kind: str = "prose"
    label: str | None = None
    weight: float = 1.0


@dataclass
class Extraction:
    """Everything one adapter got out of one source.

    ``overview`` is the important field: a deterministic, plain-English summary
    produced without any language model. For structured data it is usually
    *better* than anything a model would write - a table's real schema and
    column statistics beat a paragraph describing them - and it costs nothing,
    which is why the store is useful even with no inference server running.
    """

    source_kind: str
    overview: str
    segments: list[Segment] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A bounded slice of the source, the unit that gets stored and retrieved."""

    index: int
    text: str
    label: str | None = None

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class Digest:
    """The small thing a model reads instead of the whole source.

    ``generated_by`` records where the gist and key points came from:
    ``"rules"`` when they were derived deterministically, or the model id when
    a local model wrote them. It is worth keeping because the two are not
    equally trustworthy and the UI says which one you are looking at.
    """

    gist: str
    overview: str
    key_points: list[str] = field(default_factory=list)
    generated_by: str = "rules"

    def to_text(self) -> str:
        """Render the digest as the compact Markdown a model is handed."""
        parts = [self.gist.strip()] if self.gist.strip() else []
        if self.overview.strip():
            parts.append(self.overview.strip())
        if self.key_points:
            parts.append("\n".join(f"- {point}" for point in self.key_points))
        return "\n\n".join(parts)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.to_text())

    def to_dict(self) -> dict:
        return {
            "gist": self.gist,
            "overview": self.overview,
            "key_points": list(self.key_points),
            "generated_by": self.generated_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Digest":
        return cls(
            gist=data.get("gist", ""),
            overview=data.get("overview", ""),
            key_points=list(data.get("key_points", [])),
            generated_by=data.get("generated_by", "rules"),
        )


@dataclass(frozen=True)
class ArtifactStats:
    """What the compression actually bought, in both currencies that matter.

    Storage is bytes on disk. Context is tokens a model has to read. They are
    different wins and a design can be good at one and useless at the other -
    plain gzip is excellent at the first and does nothing for the second - so
    the store reports both rather than a single "compression ratio".
    """

    source_bytes: int
    stored_bytes: int
    source_tokens: int
    digest_tokens: int
    chunk_count: int

    @property
    def storage_ratio(self) -> float:
        """How many times smaller the stored form is on disk."""
        if self.stored_bytes <= 0:
            return 0.0
        return self.source_bytes / self.stored_bytes

    @property
    def context_ratio(self) -> float:
        """How many times cheaper the digest is than reading the whole source."""
        if self.digest_tokens <= 0:
            return 0.0
        return self.source_tokens / self.digest_tokens

    def to_dict(self) -> dict:
        return {
            "source_bytes": self.source_bytes,
            "stored_bytes": self.stored_bytes,
            "source_tokens": self.source_tokens,
            "digest_tokens": self.digest_tokens,
            "chunk_count": self.chunk_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArtifactStats":
        return cls(
            source_bytes=data.get("source_bytes", 0),
            stored_bytes=data.get("stored_bytes", 0),
            source_tokens=data.get("source_tokens", 0),
            digest_tokens=data.get("digest_tokens", 0),
            chunk_count=data.get("chunk_count", 0),
        )


@dataclass(frozen=True)
class SearchHit:
    """One chunk that matched a query, with enough context to cite it."""

    artifact_id: str
    artifact_name: str
    chunk_index: int
    score: float
    text: str
    label: str | None = None
