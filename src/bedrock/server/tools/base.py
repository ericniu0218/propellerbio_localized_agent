"""Base types every tool implements. Keep each tool small, Pydantic-typed,
and documented — see docs/server/ARCHITECTURE.md section 2 (Biotech-Specific
Capabilities)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel


class ToolResult(BaseModel):
    summary: str
    content: str
    content_type: str = "text/plain"
    citations: list[str] = []
    writes_to_external_system: bool = False


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    @abstractmethod
    async def run(self, args: BaseModel) -> ToolResult: ...

    def parse_args(self, raw: dict) -> BaseModel:
        return self.args_model.model_validate(raw)
