"""Compress any kind of data into something a local model can read cheaply.

The problem this solves: a 2 MB spreadsheet is roughly half a million tokens.
No local model will read that, and compressing it with gzip does not help -
decompressing it just gives you half a million tokens again. Fewer bytes on
disk and fewer tokens in a prompt are two different wins, and this package
goes after both.

**The two layers.** Every source is stored twice over:

* a *digest* - a few hundred tokens, cheap enough to hand a model on every
  turn, and for structured data computed exactly rather than summarised;
* the *chunks* - the full text, compressed one frame at a time and indexed by
  keyword, so an exact detail is one seek and one decompress away.

So the model reads the digest by default and calls back for specifics, instead
of choosing between an expensive full read and a lossy summary.

**Generalising over data types.** ``SourceAdapter`` is the seam, in the same
spirit as ``InferenceBackend``: the pipeline only talks to the protocol, and
each adapter knows how its own data compresses best. Tables become a schema
and column statistics, logs collapse into counted patterns, code becomes its
signatures, prose falls back to a model summary. Supporting a new type is one
new file in ``adapters/``.

Typical use::

    from bedrock.context import ContextStore, ingest

    store = ContextStore()
    artifact = ingest("results.csv", store=store)

    artifact.digest_text()          # the cheap summary
    artifact.search("pH")           # exact chunks, ranked
    store.search("pH", limit=5)     # across everything stored
"""

from bedrock.context.adapters import Source, SourceAdapter
from bedrock.context.chunker import chunk_segments
from bedrock.context.digest import build_digest, summarize
from bedrock.context.ingest import ingest, ingest_and_summarize
from bedrock.context.store import DEFAULT_CONTEXT_DIR, Artifact, ContextStore
from bedrock.context.types import (
    ArtifactStats,
    Chunk,
    Digest,
    Extraction,
    SearchHit,
    Segment,
    estimate_tokens,
)

__all__ = [
    "Artifact",
    "ArtifactStats",
    "Chunk",
    "ContextStore",
    "DEFAULT_CONTEXT_DIR",
    "Digest",
    "Extraction",
    "SearchHit",
    "Segment",
    "Source",
    "SourceAdapter",
    "build_digest",
    "chunk_segments",
    "estimate_tokens",
    "ingest",
    "ingest_and_summarize",
    "summarize",
]
