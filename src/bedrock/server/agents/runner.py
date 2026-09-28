"""Runner: fast/cheap Qwen model. Classification, summarization, entity
extraction, PHI redaction, history and compaction — see
docs/ARCHITECTURE.md section 2. Never calls the Director or Specialist
itself; the orchestrator decides what happens with the Runner's output.
"""

from __future__ import annotations

import json

from bedrock.server.compliance.phi_redaction import RedactionResult, redact_phi
from bedrock.server.models.client import ChatMessage
from bedrock.server.schemas.enums import AgentRole, TaskComplexity, TaskDomain
from bedrock.server.agents.base import BaseAgent

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

    async def classify(
        self, request: str, task_id: str | None = None, previous: str | None = None
    ) -> ClassificationResult:
        # The question only - never the files sent with it - plus the previous
        # message, so a short follow-up ("and the second one?") classifies with
        # what it follows. Material is for the drafting agent, not the router.
        earlier = (
            f"Earlier message in this conversation, for context only: {previous[:500]}\n\n" if previous else ""
        )
        prompt = (
            f"Classify this user request.\n\nRequest: {request}\n\n{earlier}"
            'Respond with JSON: {"complexity": "simple|normal|hard", '
            '"domain": "literature|internal_docs|bioinformatics|chemistry|'
            'data_analysis|regulatory|clinical|lab_data|meetings|general", "reason": "..."}'
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


def _extract_json(raw: str) -> str:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    return raw[start : end + 1]
