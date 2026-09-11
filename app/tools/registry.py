from __future__ import annotations

from functools import lru_cache

from app.tools.base import Tool
from app.tools.chemistry_tools import RDKitTool
from app.tools.doc_templates import DocTemplateTool
from app.tools.lab_connectors import ELNLIMSTool
from app.tools.literature_search import LiteratureSearchTool
from app.tools.meeting_transcripts import MeetingTranscriptSearchTool
from app.tools.rag_search import RAGSearchTool
from app.tools.sandbox import DataAnalysisSandboxTool
from app.tools.sequence_tools import SequenceStructureTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        for tool_cls in (
            RAGSearchTool,
            LiteratureSearchTool,
            SequenceStructureTool,
            RDKitTool,
            DataAnalysisSandboxTool,
            ELNLIMSTool,
            DocTemplateTool,
            MeetingTranscriptSearchTool,
        ):
            tool = tool_cls()
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())


@lru_cache
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry()
