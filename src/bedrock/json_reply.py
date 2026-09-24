"""Decode JSON replies with the two wrapping styles Bedrock permits."""

from __future__ import annotations

import json
import re
from typing import Any


def decode_json_reply(raw: str, *, allow_surrounding_text: bool = False) -> Any:
    """Decode a model reply, optionally finding an object inside prose.

    Planner replies are strict: the entire reply must be JSON, with an optional
    Markdown fence. Meeting extraction keeps its established tolerance for a
    JSON object surrounded by prose. Keeping the policy flag here removes the
    fence parsing duplicate without silently making planner output looser.
    """
    if allow_surrounding_text:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        candidate = fenced.group(1) if fenced else raw[raw.find("{") : raw.rfind("}") + 1]
        if not candidate.strip():
            raise ValueError("no JSON object in reply")
        return json.loads(candidate)

    candidate = raw.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[0].strip() in ("```", "```json"):
            candidate = "\n".join(lines[1:-1]).strip()
    return json.loads(candidate)
