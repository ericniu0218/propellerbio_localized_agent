"""Internal-document search on bedrock's context store, with access groups.

``rag_search`` ranked by counting words shared with the question. bedrock's
``ContextStore`` (the engine behind ``/digest``) ranks with BM25 on SQLite FTS5
and folds plurals, at no new dependency, so the server searches the company's
documents with it: ``bedrock-server index`` ingests ``knowledge_dir`` into a
store under the server's data directory, and ``ContextStoreIndex`` answers
``rag_search`` from it. Embeddings remain the long-term plan (NEXT_STEPS.md);
this is the same ``VectorIndex`` seam they will use.

**Who may see what** is decided per file by ``access.toml`` in ``knowledge_dir``:
glob patterns mapped to access groups, first match wins, unlisted files are
``general``::

    "clinical/*" = ["clinical"]
    "QA-*.txt" = ["qa", "clinical"]

``rag_search`` then filters hits by the requesting user's groups, as it always
did.

**Which index a server uses** (``default_index``): the built store if there is
one; otherwise the configured folder read directly (the old keyword index), so
setting ``knowledge_dir`` works before anyone runs ``index``; with neither, a
mock server reads the packaged fictional samples and a real server reads
nothing - fiction must never be served as company knowledge.
"""

from __future__ import annotations

import fnmatch
import json
import tomllib
from pathlib import Path

from bedrock.context import ContextStore, ingest
from bedrock.server.config import Settings, private_dir

INDEXED_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".json", ".log"}
ACCESS_FILE = "access.toml"
PATHS_FILE = "paths.json"
"""Artifact id -> path relative to ``knowledge_dir``. The store keeps only a
file's name, and access rules (``clinical/*``) need its folder."""


def load_access(corpus_dir: Path | None) -> list[tuple[str, list[str]]]:
    path = corpus_dir / ACCESS_FILE if corpus_dir else None
    if path is None or not path.is_file():
        return []
    rules = tomllib.loads(path.read_text(encoding="utf-8"))
    return [(str(pattern), [str(g) for g in groups]) for pattern, groups in rules.items()]


def groups_for(name: str, rules: list[tuple[str, list[str]]]) -> list[str]:
    for pattern, groups in rules:
        if fnmatch.fnmatchcase(name, pattern):
            return groups
    return ["general"]


def index_dir(settings: Settings) -> Path:
    return settings.data_dir / "knowledge-index"


def build_index(corpus_dir: Path, store_dir: Path) -> int:
    """Ingest every text-like file under ``corpus_dir``; returns how many."""
    store = ContextStore(private_dir(store_dir))
    paths: dict[str, str] = {}
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in INDEXED_SUFFIXES and path.name != ACCESS_FILE:
            artifact = ingest(path, store=store)
            paths[artifact.id] = path.relative_to(corpus_dir).as_posix()
    (store_dir / PATHS_FILE).write_text(json.dumps(paths, indent=1, sort_keys=True))
    return len(paths)


def load_paths(store_dir: Path) -> dict[str, str]:
    path = store_dir / PATHS_FILE
    return json.loads(path.read_text()) if path.is_file() else {}


class ContextStoreIndex:
    """``rag_search``'s ``VectorIndex`` over a built ``ContextStore``."""

    def __init__(self, store: ContextStore, rules: list[tuple[str, list[str]]]) -> None:
        self._store = store
        self._rules = rules
        self._paths = load_paths(store.root)

    def search(self, query: str, top_k: int):
        from bedrock.server.tools.rag_search import DocChunk

        chunks = []
        for hit in self._store.search(query, limit=top_k):
            relative = self._paths.get(hit.artifact_id, hit.artifact_name)
            chunks.append(DocChunk(
                doc_id=Path(relative).stem,
                section=f"chunk-{hit.chunk_index + 1}",
                text=hit.text,
                access_groups=groups_for(relative, self._rules),
            ))
        return chunks


class EmptyIndex:
    def search(self, query: str, top_k: int):
        return []


def default_index(settings: Settings, sample_dir: Path):
    from bedrock.server.tools.rag_search import KeywordOverlapIndex

    built = index_dir(settings)
    if settings.knowledge_dir and built.is_dir() and ContextStore(built).list():
        return ContextStoreIndex(ContextStore(built), load_access(settings.knowledge_dir))
    if settings.knowledge_dir:
        return KeywordOverlapIndex(settings.knowledge_dir)
    if settings.mode == "real":
        return EmptyIndex()
    return KeywordOverlapIndex(sample_dir)
