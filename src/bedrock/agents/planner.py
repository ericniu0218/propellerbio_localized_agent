"""Turn an evidenced request into an inert, human-readable plan.

The model may choose among the capabilities in one roster entry, but it does
not get to choose risk, wording, or invent tools.  Those come from the loaded
Capability objects.  Any malformed or unavailable answer becomes a blocked
Plan rather than executable partial output.

The request is a job the employee asked for, so it goes through the
``Scheduler`` as ``Priority.USER_JOB``, with the machine profile's reply limit.
It also sends the plan's shape as a JSON Schema (``plan_schema``), which
servers that support structured output use to constrain the reply to it.
``json_reply`` still parses the answer and ``_build_plan`` still checks every
field, exactly as strictly as before: a server may ignore the schema.
Thinking is left to the server's default, because working out a plan is where
a reasoning model's thinking helps.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from bedrock.agents.contract import Ask, Plan, Step
from bedrock.agents.protocol import Capability
from bedrock.config.profile import DEFAULT_PROFILE, PROFILES
from bedrock.inference import Priority, RequestOptions, Scheduler
from bedrock.inference.base import InferenceBackend
from bedrock.json_reply import decode_json_reply

_SYSTEM_PROMPT = """You turn one evidenced request into a small tool plan.
Return JSON only, with this exact shape:
{"steps":[{"tool":"capability.name","arguments":{"key":"value"}}],"blocked_reason":null}
Use only capability names supplied by the user. Every argument key and value
must be a string. Do not claim that any step has run. If the request cannot be
safely expressed with the supplied capabilities, return an empty steps list
and a short blocked_reason. Never add facts that are absent from the request.
"""


class Planner:
    """Use the configured private inference backend to propose a ``Plan``."""

    def __init__(
        self,
        scheduler: Scheduler | InferenceBackend,
        model: str,
        capabilities: Mapping[str, Capability],
        *,
        max_steps: int = 8,
    ) -> None:
        """``scheduler`` is the app's ``Scheduler``; a plain backend is given
        a scheduler of its own with the default profile."""
        if not model:
            raise ValueError("Planner needs a model name.")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("Planner max_steps must be a positive integer.")
        if not isinstance(scheduler, Scheduler):
            scheduler = Scheduler(scheduler, PROFILES[DEFAULT_PROFILE])
        self.scheduler = scheduler
        self.model = model
        self.capabilities = dict(capabilities)
        self.max_steps = max_steps

    async def plan(self, ask: Ask) -> Plan:
        """Return a plan or a blocked plan; never execute a capability."""
        if not self.capabilities:
            return Plan(ask=ask, blocked_reason="This agent has no rostered capabilities.")

        payload = {
            "request": asdict(ask),
            "available_capabilities": [
                {"name": name, "risk": capability.risk.value}
                for name, capability in sorted(self.capabilities.items())
            ],
            "maximum_steps": self.max_steps,
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        options = RequestOptions(
            max_tokens=self.scheduler.profile.reply_limit,
            json_schema=plan_schema(sorted(self.capabilities), self.max_steps),
        )
        try:
            text = await self.scheduler.complete(
                self.model, messages, priority=Priority.USER_JOB, options=options
            )
            reply = _decode_reply(text)
            return self._build_plan(ask, reply)
        except Exception as exc:
            return Plan(ask=ask, blocked_reason=f"Planner could not produce a valid plan: {exc}")

    def _build_plan(self, ask: Ask, reply: dict[str, Any]) -> Plan:
        raw_steps = reply.get("steps")
        blocked_reason = reply.get("blocked_reason")
        if not isinstance(raw_steps, list):
            raise ValueError("steps must be a list")
        if len(raw_steps) > self.max_steps:
            raise ValueError(f"plan exceeds the {self.max_steps}-step budget")
        if blocked_reason is not None and not isinstance(blocked_reason, str):
            raise ValueError("blocked_reason must be text or null")
        if isinstance(blocked_reason, str):
            blocked_reason = blocked_reason.strip() or None

        steps: list[Step] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                raise ValueError("each step must be an object")
            if raw_step.keys() - {"tool", "arguments"}:
                raise ValueError("a step contained unexpected fields")
            tool = raw_step.get("tool")
            arguments = raw_step.get("arguments")
            if not isinstance(tool, str) or tool not in self.capabilities:
                raise ValueError(f"tool {tool!r} is not in this agent's roster")
            if not isinstance(arguments, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in arguments.items()
            ):
                raise ValueError(f"arguments for {tool!r} must be strings")
            capability = self.capabilities[tool]
            summary = capability.describe(dict(arguments))
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(f"capability {tool!r} did not describe its step")
            steps.append(
                Step(
                    tool=tool,
                    summary=summary.strip(),
                    risk=capability.risk,
                    arguments=dict(arguments),
                )
            )

        if blocked_reason:
            return Plan(ask=ask, steps=tuple(steps), blocked_reason=blocked_reason[:500])
        if not steps:
            return Plan(ask=ask, blocked_reason="The planner proposed no executable steps.")
        return Plan(ask=ask, steps=tuple(steps))


def plan_schema(tools: list[str], max_steps: int) -> dict[str, Any]:
    """The JSON Schema of a planner reply: the same shape ``_build_plan`` checks."""
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "maxItems": max_steps,
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": list(tools)},
                        "arguments": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                    "required": ["tool", "arguments"],
                    "additionalProperties": False,
                },
            },
            "blocked_reason": {"type": ["string", "null"]},
        },
        "required": ["steps", "blocked_reason"],
        "additionalProperties": False,
    }


def _decode_reply(text: str) -> dict[str, Any]:
    try:
        value = decode_json_reply(text)
    except json.JSONDecodeError as exc:
        raise ValueError("model reply was not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("model reply must be a JSON object")
    unexpected = value.keys() - {"steps", "blocked_reason"}
    if unexpected:
        raise ValueError("model reply contained unexpected fields")
    return value
