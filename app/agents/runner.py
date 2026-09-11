"""Runner: fast/cheap Qwen model. Classification, summarization, entity
extraction, PHI redaction, history compaction, and output formatting — see
docs/ARCHITECTURE.md section 2. Never calls the Director or Specialist
itself; the orchestrator decides what happens with the Runner's output.
"""

from __future__ import annotations

import json

from app.compliance.phi_redaction import RedactionResult, redact_phi
from app.models.client import ChatMessage
from app.schemas.enums import AgentRole, TaskComplexity, TaskDomain
from app.agents.base import BaseAgent

_STATIC_PREFIX = (
    "You are the Runner, a fast, cheap classification/utility agent in a "
    "biotech AI system. You never do open-ended reasoning or drafting — "
    "you classify, summarize, extract, redact, compact, or format exactly "
    "what you are asked, and nothing else. When asked for JSON, respond "
    "with JSON only, no prose."
)


class ClassificationResult:
    def __init__(self, complexity: TaskComplexity, domain: TaskDomain, reason: str) -> None:
        self.complexity = complexity
        self.domain = domain
        self.reason = reason


class Runner(BaseAgent):
    role = AgentRole.RUNNER
    static_prompt_prefix = _STATIC_PREFIX

    async def classify(self, request: str, task_id: str | None = None) -> ClassificationResult:
        prompt = (
            f"Classify this user request.\n\nRequest: {request}\n\n"
            'Respond with JSON: {"complexity": "simple|normal|hard", '
            '"domain": "literature|internal_docs|bioinformatics|chemistry|'
            'data_analysis|regulatory|clinical|lab_data|general", "reason": "..."}'
        )
        raw = await self.call_model(self.build_prompt(prompt), task_id=task_id)
        try:
            data = json.loads(_extract_json(raw))
            return ClassificationResult(
                complexity=TaskComplexity(data.get("complexity", "normal")),
                domain=TaskDomain(data.get("domain", "general")),
                reason=data.get("reason", ""),
            )
        except (ValueError, json.JSONDecodeError):
            # Fail safe toward "normal" rather than silently guessing "simple"
            # — under-escalating a hard/high-stakes request is the worse failure mode.
            return ClassificationResult(TaskComplexity.NORMAL, TaskDomain.GENERAL, "unparsed classifier output")

    async def summarize(self, text: str, task_id: str | None = None, target_tokens: int = 300) -> str:
        prompt = f"Summarize the following in about {target_tokens} tokens, preserving citations verbatim:\n\n{text}"
        return await self.call_model(self.build_prompt(prompt), task_id=task_id, max_tokens=target_tokens + 50)

    async def extract_entities(self, text: str, task_id: str | None = None) -> dict:
        prompt = (
            "Extract structured entities as JSON with keys: genes, proteins, "
            'compound_ids, dates, doses. Use [] for any category with no matches.\n\n'
            f"Text: {text}"
        )
        raw = await self.call_model(self.build_prompt(prompt), task_id=task_id)
        try:
            return json.loads(_extract_json(raw))
        except json.JSONDecodeError:
            return {"genes": [], "proteins": [], "compound_ids": [], "dates": [], "doses": []}

    def redact(self, text: str) -> RedactionResult:
        # Deterministic, no model call needed for the structured-PII baseline
        # — see app/compliance/phi_redaction.py.
        return redact_phi(text)

    async def compact_history(self, turns: list[str], task_id: str | None = None) -> str:
        joined = "\n".join(turns)
        prompt = f"Compress this conversation history into a short summary a future turn can use as context:\n\n{joined}"
        return await self.call_model(self.build_prompt(prompt), task_id=task_id)

    async def format_output(self, content: str, citations: list[str], task_id: str | None = None) -> str:
        citation_block = "\n".join(f"- {c}" for c in citations) if citations else "(no citations)"
        prompt = (
            f"Format this content as clean markdown for a chat UI, and append a "
            f"'Sources' section listing these citations exactly:\n{citation_block}\n\n"
            f"Content:\n{content}"
        )
        return await self.call_model(self.build_prompt(prompt), task_id=task_id)


def _extract_json(raw: str) -> str:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    return raw[start : end + 1]
