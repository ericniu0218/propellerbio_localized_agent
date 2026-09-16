"""The on-disk store: small manifests, compressed chunks, fast lookup.

Layout, one directory per artifact::

    <root>/<artifact_id>/
        manifest.json   small, uncompressed - stats, chunk offsets, word counts
        digest.json     small, uncompressed - what a model reads by default
        chunks.bin      every chunk, compressed one frame at a time

Three decisions worth explaining, because they are what make this fast:

*Why the manifest stays uncompressed.* Listing and searching the store must
not require decompressing anything. The manifest holds each chunk's byte
offset and its word counts, so ranking happens entirely against small JSON and
only the handful of winning chunks are ever decompressed.

*Why one compressed frame per chunk instead of one stream per file.* A single
stream would compress a few percent better, but reading chunk 400 would mean
decompressing the 399 in front of it. Independent frames make any chunk a seek
plus one decompress, which is the "accessed quickly" half of the brief.

*Why the id is a content hash.* Ingesting the same data twice is then free and
cannot create a duplicate - the second call finds the existing artifact and
returns it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bedrock.context import codec
from bedrock.context.adapters.base import Source
from bedrock.context.search import INDEX_VERSION, ScoredDoc, bm25, fold_terms, term_frequencies
from bedrock.context.types import (
    ArtifactStats,
    Chunk,
    Digest,
    Extraction,
    SearchHit,
    estimate_tokens,
)

# The on-disk *format* - which files exist and what the manifest holds. The
# tokeniser has its own version (search.INDEX_VERSION) written alongside it,
# because the two change for unrelated reasons: NEXT_STEPS.md section 6 will
# bump this one for vectors, which must not be read as "the words were
# tokenised differently".
SCHEMA_VERSION = 1
DEFAULT_CONTEXT_DIR = Path.home() / ".local" / "share" / "bedrock" / "context"
_ID_LENGTH = 16


@dataclass
class Artifact:
    """A handle on one stored source: its digest now, its chunks on request."""

    id: str
    name: str
    source_kind: str
    created_at: str
    stats: ArtifactStats
    digest: Digest
    chunk_labels: list[str | None] = field(default_factory=list)
    store: "ContextStore | None" = field(default=None, repr=False, compare=False)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_labels)

    def digest_text(self) -> str:
        return self.digest.to_text()

    def chunk(self, index: int) -> str:
        """The full text of one chunk, decompressed on demand."""
        if self.store is None:
            raise RuntimeError("This artifact is not attached to a store.")
        return self.store.read_chunk(self.id, index)

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        if self.store is None:
            raise RuntimeError("This artifact is not attached to a store.")
        return self.store.search(query, limit=limit, artifact_id=self.id)


class ContextStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root).expanduser() if root else DEFAULT_CONTEXT_DIR
        self._manifest_cache: dict[str, dict] = {}

    # ---- Writing -----------------------------------------------------------

    def put(
        self,
        *,
        source: Source,
        extraction: Extraction,
        chunks: list[Chunk],
        digest: Digest,
    ) -> Artifact:
        """Write one artifact and return a handle on it.

        Re-ingesting identical content is a no-op: the id is the hash of the
        text, so we find the existing directory and hand back what is there.
        """
        artifact_id = content_id(source.text)
        directory = self.root / artifact_id
        if (directory / "manifest.json").exists():
            existing = self.get(artifact_id)
            if existing is not None:
                return existing

        directory.mkdir(parents=True, exist_ok=True)
        chosen_codec = codec.default_codec()

        entries: list[dict] = []
        offset = 0
        with (directory / "chunks.bin").open("wb") as handle:
            for chunk in chunks:
                blob, chosen_codec = codec.compress(chunk.text.encode("utf-8"), chosen_codec)
                handle.write(blob)
                counts = term_frequencies(chunk.text)
                entries.append(
                    {
                        "index": chunk.index,
                        "label": chunk.label,
                        "offset": offset,
                        "length": len(blob),
                        "tokens": chunk.tokens,
                        # Word counts live here so search never touches chunks.bin.
                        "terms": counts,
                        "term_count": sum(counts.values()),
                    }
                )
                offset += len(blob)

        stats = ArtifactStats(
            source_bytes=source.byte_size or len(source.text.encode("utf-8")),
            stored_bytes=offset,
            source_tokens=estimate_tokens(source.text),
            digest_tokens=digest.tokens,
            chunk_count=len(chunks),
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "index_version": INDEX_VERSION,
            "id": artifact_id,
            "name": source.name,
            "source_kind": extraction.source_kind,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "codec": chosen_codec,
            "stats": stats.to_dict(),
            "metadata": extraction.metadata,
            "chunks": entries,
        }
        _write_json(directory / "manifest.json", manifest)
        _write_json(directory / "digest.json", digest.to_dict())
        self._manifest_cache[artifact_id] = manifest
        return self._artifact(manifest, digest)

    def update_digest(self, artifact_id: str, digest: Digest) -> Artifact | None:
        """Replace an artifact's digest, e.g. after the optional model pass."""
        directory = self.root / artifact_id
        if not (directory / "manifest.json").exists():
            return None
        manifest = self._manifest(artifact_id)
        if manifest is None:
            return None
        manifest["stats"]["digest_tokens"] = digest.tokens
        _write_json(directory / "manifest.json", manifest)
        _write_json(directory / "digest.json", digest.to_dict())
        self._manifest_cache[artifact_id] = manifest
        return self._artifact(manifest, digest)

    def delete(self, artifact_id: str) -> bool:
        directory = self.root / artifact_id
        if not directory.is_dir():
            return False
        shutil.rmtree(directory)
        self._manifest_cache.pop(artifact_id, None)
        return True

    # ---- Reading -----------------------------------------------------------

    def get(self, artifact_id: str) -> Artifact | None:
        manifest = self._manifest(artifact_id)
        if manifest is None:
            return None
        digest_path = self.root / artifact_id / "digest.json"
        digest = Digest.from_dict(_read_json(digest_path) or {})
        return self._artifact(manifest, digest)

    def list(self) -> list[Artifact]:
        """Every artifact, newest first. Reads only the small JSON files."""
        if not self.root.is_dir():
            return []
        artifacts = []
        for directory in self.root.iterdir():
            if directory.is_dir():
                artifact = self.get(directory.name)
                if artifact is not None:
                    artifacts.append(artifact)
        artifacts.sort(key=lambda a: a.created_at, reverse=True)
        return artifacts

    def resolve(self, prefix: str) -> Artifact | None:
        """Look up an artifact by an id prefix or by filename.

        Typing a full sixteen-character hash by hand is miserable, so a unique
        prefix is enough - the same convenience git gives you for commits.
        """
        exact = self.get(prefix)
        if exact is not None:
            return exact
        matches = [a for a in self.list() if a.id.startswith(prefix) or a.name == prefix]
        return matches[0] if len(matches) == 1 else None

    def read_chunk(self, artifact_id: str, index: int) -> str:
        manifest = self._manifest(artifact_id)
        if manifest is None:
            raise KeyError(f"No stored artifact {artifact_id!r}")
        entries = manifest["chunks"]
        if not 0 <= index < len(entries):
            raise IndexError(f"Artifact {artifact_id} has no chunk {index}")
        entry = entries[index]
        with (self.root / artifact_id / "chunks.bin").open("rb") as handle:
            handle.seek(entry["offset"])
            blob = handle.read(entry["length"])
        return codec.decompress(blob, manifest.get("codec", codec.LZMA)).decode("utf-8")

    def search(
        self, query: str, *, limit: int = 5, artifact_id: str | None = None
    ) -> list[SearchHit]:
        """Rank chunks against ``query`` and decompress only the winners."""
        manifests = (
            [m for m in [self._manifest(artifact_id)] if m is not None]
            if artifact_id
            else [m for m in (self._manifest(a.id) for a in self.list()) if m is not None]
        )

        docs: list[ScoredDoc] = []
        lookup: dict[tuple, tuple[dict, dict]] = {}
        for manifest in manifests:
            for entry in manifest["chunks"]:
                key = (manifest["id"], entry["index"])
                # An artifact stored before the tokeniser changed still holds
                # its old word counts, and it is never rewritten (ids are
                # content hashes, so re-ingesting returns the existing one).
                # Folding on read keeps those artifacts searchable instead of
                # silently matching nothing.
                terms = entry["terms"]
                if manifest.get("index_version", 1) < INDEX_VERSION:
                    terms = fold_terms(terms)
                docs.append(ScoredDoc(key=key, terms=terms, length=entry["term_count"]))
                lookup[key] = (manifest, entry)

        hits = []
        for key, score in bm25(query, docs, limit=limit):
            manifest, entry = lookup[key]
            hits.append(
                SearchHit(
                    artifact_id=manifest["id"],
                    artifact_name=manifest["name"],
                    chunk_index=entry["index"],
                    score=score,
                    text=self.read_chunk(manifest["id"], entry["index"]),
                    label=entry.get("label"),
                )
            )
        return hits

    def totals(self) -> ArtifactStats:
        """Combined stats across the whole store, for the ``/digest`` listing."""
        artifacts = self.list()
        return ArtifactStats(
            source_bytes=sum(a.stats.source_bytes for a in artifacts),
            stored_bytes=sum(a.stats.stored_bytes for a in artifacts),
            source_tokens=sum(a.stats.source_tokens for a in artifacts),
            digest_tokens=sum(a.stats.digest_tokens for a in artifacts),
            chunk_count=sum(a.stats.chunk_count for a in artifacts),
        )

    # ---- Internals ---------------------------------------------------------

    def _manifest(self, artifact_id: str) -> dict | None:
        cached = self._manifest_cache.get(artifact_id)
        if cached is not None:
            return cached
        manifest = _read_json(self.root / artifact_id / "manifest.json")
        if manifest is not None:
            self._manifest_cache[artifact_id] = manifest
        return manifest

    def _artifact(self, manifest: dict, digest: Digest) -> Artifact:
        return Artifact(
            id=manifest["id"],
            name=manifest["name"],
            source_kind=manifest["source_kind"],
            created_at=manifest["created_at"],
            stats=ArtifactStats.from_dict(manifest["stats"]),
            digest=digest,
            chunk_labels=[entry.get("label") for entry in manifest["chunks"]],
            store=self,
        )


def content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_ID_LENGTH]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
