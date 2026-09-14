"""The one function most callers need.

``ingest`` is deliberately synchronous and model-free: sniff the type, extract,
chunk, compress, store. No network, nothing that can hang, nothing that fails
because a server is down. That makes it easy to test and safe to call from
anywhere.

``ingest_and_summarize`` is the optional second step that asks a local model to
improve the digest. It is a separate function rather than a flag because the
split is the important part of the design: the store is fully useful without a
model, and the model only ever makes it better.
"""

from __future__ import annotations

from pathlib import Path

from bedrock.context.adapters.base import Source, SourceAdapter
from bedrock.context.adapters.registry import select_adapter
from bedrock.context.chunker import DEFAULT_CHUNK_TOKENS, chunk_segments
from bedrock.context.digest import build_digest, summarize
from bedrock.context.store import Artifact, ContextStore


def ingest(
    data: Source | Path | str,
    *,
    store: ContextStore,
    name: str | None = None,
    adapters: tuple[SourceAdapter, ...] | None = None,
    max_chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> Artifact:
    """Compress ``data`` into the store and return a handle on it.

    ``data`` may be a :class:`Source`, a path to a file, or raw text. Passing
    the same content twice returns the existing artifact rather than storing a
    second copy.
    """
    source = _as_source(data, name)
    adapter = select_adapter(source, adapters)
    extraction = adapter.extract(source)
    chunks = chunk_segments(extraction.segments, max_tokens=max_chunk_tokens)
    digest = build_digest(extraction, chunks)
    return store.put(source=source, extraction=extraction, chunks=chunks, digest=digest)


async def ingest_and_summarize(
    data: Source | Path | str,
    *,
    store: ContextStore,
    backend,
    model: str,
    name: str | None = None,
    adapters: tuple[SourceAdapter, ...] | None = None,
) -> Artifact:
    """``ingest``, then upgrade the digest with a local model."""
    artifact = ingest(data, store=store, name=name, adapters=adapters)
    artifact.digest = await summarize(artifact, backend, model, store=store)
    return artifact


def _as_source(data: Source | Path | str, name: str | None) -> Source:
    if isinstance(data, Source):
        return data
    if isinstance(data, Path):
        return Source.from_path(data)
    # A str is ambiguous: it might be a path or it might be the content itself.
    # Treat it as a path only when a file is actually there, so passing a
    # paragraph of text that happens to look path-like still does the right
    # thing.
    candidate = Path(data).expanduser()
    try:
        if len(data) < 4096 and candidate.is_file():
            return Source.from_path(candidate)
    except OSError:
        pass
    return Source.from_text(data, name=name or "untitled")
