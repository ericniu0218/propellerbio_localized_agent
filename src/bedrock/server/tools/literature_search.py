"""Scientific literature search over a local mirror/cached export of public
literature (e.g. a PubMed baseline or open-access subset). No live internet
access — the mirror is assumed to already exist on disk (see
docs/ARCHITECTURE.md open question 2); this tool only searches it.

Implementation reuses the same keyword-overlap index as rag_search.py
(swap for real semantic search / a proper PubMed index later) but reads
from a separate corpus directory and has no RBAC filtering, since published
literature isn't access-controlled the way internal documents are.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from bedrock.server.config import load_budgets_config
from bedrock.server.tools.base import Tool, ToolResult
from bedrock.server.tools.rag_search import KeywordOverlapIndex, VectorIndex

# The sample corpus ships inside the package (bedrock/server/sample_data/).
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "literature"


class LiteratureSearchArgs(BaseModel):
    query: str
    top_k: int | None = None


class LiteratureSearchTool(Tool):
    name: ClassVar[str] = "literature_search"
    description: ClassVar[str] = (
        "Search a local mirror of scientific literature (e.g. cached PubMed "
        "export). No live internet access. Returns citations as doc_id#section."
    )
    args_model: ClassVar[type[BaseModel]] = LiteratureSearchArgs

    def __init__(self, index: VectorIndex | None = None) -> None:
        self._index = index or KeywordOverlapIndex(DEFAULT_CORPUS_DIR)

    async def run(self, args: LiteratureSearchArgs) -> ToolResult:
        top_k = args.top_k or load_budgets_config().rag["top_k_chunks"]
        hits = self._index.search(args.query, top_k=top_k)
        if not hits:
            return ToolResult(
                summary="No matching entries in the local literature mirror.",
                content="",
                citations=[],
            )
        citations = [f"{h.doc_id}#{h.section}" for h in hits]
        content = "\n\n".join(f"[{h.doc_id}#{h.section}] {h.text}" for h in hits)
        return ToolResult(
            summary=f"Found {len(hits)} relevant literature excerpt(s).",
            content=content,
            citations=citations,
        )
