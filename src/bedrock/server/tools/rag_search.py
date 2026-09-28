"""Internal knowledge search (RAG) over the company's own documents (SOPs,
study reports, protocols, past submissions).

Every hit must carry a citation (document id + section) — callers that
drop the citation before it reaches the Director will get rejected, per
the hallucination-guardrail requirement in docs/ARCHITECTURE.md section 8.

The real index is pgvector (default) or Qdrant (documented swap) over
embeddings computed at ingest time — that ingest pipeline and the actual
embedding model are out of scope for this pass (no company doc corpus was
provided). What ships here is the tool interface plus a small in-memory
keyword-overlap `VectorIndex` implementation over local text files, so the
tool is genuinely exercisable in mock mode and in tests without a live
vector DB. Swap `VectorIndex` for a pgvector-backed one behind the same
interface for real deployment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel

from bedrock.server.compliance.rbac import RBACFilter, get_rbac_filter
from bedrock.server.config import load_budgets_config
from bedrock.server.tools.base import Tool, ToolResult

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "internal_docs"


class RAGSearchArgs(BaseModel):
    query: str
    user_id: str
    top_k: int | None = None


class DocChunk(BaseModel):
    doc_id: str
    section: str
    text: str
    access_groups: list[str] = ["general"]


class VectorIndex(Protocol):
    def search(self, query: str, top_k: int) -> list[DocChunk]: ...


class KeywordOverlapIndex:
    """Naive local substitute for a real embedding index: scores chunks by
    token overlap with the query. Deterministic and dependency-free, which
    is what mock mode and tests need; not a substitute for real semantic
    search in production."""

    def __init__(self, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> None:
        self._chunks: list[DocChunk] = []
        if corpus_dir.exists():
            for path in sorted(corpus_dir.glob("*.txt")):
                text = path.read_text()
                for i, para in enumerate(p for p in text.split("\n\n") if p.strip()):
                    self._chunks.append(
                        DocChunk(doc_id=path.stem, section=f"para-{i+1}", text=para.strip())
                    )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def search(self, query: str, top_k: int) -> list[DocChunk]:
        q_tokens = self._tokens(query)
        scored = [
            (len(q_tokens & self._tokens(c.text)), c) for c in self._chunks
        ]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda sc: sc[0], reverse=True)
        return [c for _, c in scored[:top_k]]


class RAGSearchTool(Tool):
    name: ClassVar[str] = "rag_search"
    description: ClassVar[str] = (
        "Search the company's internal documents (SOPs, study reports, "
        "protocols, past submissions). Always returns doc_id + section "
        "citations. Results are filtered to the requesting user's RBAC groups."
    )
    args_model: ClassVar[type[BaseModel]] = RAGSearchArgs

    def __init__(self, index: VectorIndex | None = None, rbac: RBACFilter | None = None) -> None:
        self._index = index or KeywordOverlapIndex()
        self._rbac = rbac or get_rbac_filter()

    async def run(self, args: RAGSearchArgs) -> ToolResult:
        top_k = args.top_k or load_budgets_config().rag["top_k_chunks"]
        hits = self._index.search(args.query, top_k=top_k * 2)  # over-fetch, then RBAC-filter
        visible = self._rbac.filter_documents(
            args.user_id, [h.model_dump() for h in hits]
        )[:top_k]

        if not visible:
            return ToolResult(
                summary="No internal documents matched this query (or none were visible to this user).",
                content="",
                citations=[],
            )

        citations = [f"{h['doc_id']}#{h['section']}" for h in visible]
        content = "\n\n".join(f"[{h['doc_id']}#{h['section']}] {h['text']}" for h in visible)
        summary = f"Found {len(visible)} relevant internal document chunk(s)."
        return ToolResult(summary=summary, content=content, citations=citations)
