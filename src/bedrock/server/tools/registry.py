from __future__ import annotations

from functools import lru_cache

from bedrock.server.tools.base import Tool
from bedrock.server.tools.chemistry_tools import RDKitTool
from bedrock.server.tools.doc_templates import DocTemplateTool
from bedrock.server.tools.lab_connectors import ELNLIMSTool
from bedrock.server.tools.literature_search import LiteratureSearchTool
from bedrock.server.tools.meeting_transcripts import MeetingTranscriptsTool
from bedrock.server.tools.rag_search import RAGSearchTool
from bedrock.server.tools.sandbox import DataAnalysisSandboxTool
from bedrock.server.tools.sequence_tools import SequenceStructureTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        for tool_cls in (
            RAGSearchTool,
            LiteratureSearchTool,
            MeetingTranscriptsTool,
            SequenceStructureTool,
            RDKitTool,
            DataAnalysisSandboxTool,
            ELNLIMSTool,
            DocTemplateTool,
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
