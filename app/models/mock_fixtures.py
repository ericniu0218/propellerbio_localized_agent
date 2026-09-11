"""Realistic-looking fake replies for MockModelClient.

Without this, the mock model just echoed back whatever prompt it was
given — which, for the Runner's `format_output` call in particular, meant
the literal internal formatting *instructions* ("append a 'Sources'
section listing these citations exactly...") ended up shown to the user as
if it were the answer. That's a bad demo and would confuse a non-technical
user, even though the underlying agent loop was working correctly.

This module recognizes the shape of a handful of known prompts (by the
fixed instruction text each agent method in `app/agents/*.py` always
sends) and returns something that looks like a plausible reply to that
kind of call, instead of parroting the prompt. It is not a language model
— just enough pattern-matching to make mock mode demoable. Real mode
never touches this file.
"""

from __future__ import annotations

import json
import re


def fake_reply(prompt: str, model_name: str) -> str:
    if 'Respond with JSON: {"complexity"' in prompt:
        return _fake_classification(prompt)
    if "Extract structured entities as JSON" in prompt:
        return json.dumps({"genes": [], "proteins": [], "compound_ids": [], "dates": [], "doses": []})
    if "Format this content as clean markdown" in prompt:
        return _fake_format(prompt)
    if "Synthesize a final answer for:" in prompt:
        return _fake_synthesis(prompt)
    if "Compress this conversation history" in prompt:
        return "Earlier in this conversation: the user asked a question and got an answer. Continuing from there."
    if prompt.startswith("Summarize the following"):
        return _trim(_after(prompt, "\n\n"), 400)
    if prompt.startswith("Task:"):
        return _fake_draft(prompt)
    return f"({model_name}, mock mode) placeholder reply to: {_trim(prompt, 100)}"


def _extract_request(prompt: str) -> str:
    """Pulls just the user's actual request out of the classifier prompt —
    the prompt also lists every domain name as part of the instructions
    ("literature|internal_docs|...|general"), so keyword-matching the whole
    prompt would spuriously match all of them every time."""
    match = re.search(r"Request:\s*(.*?)(?:\n\n|$)", prompt, re.S)
    return (match.group(1) if match else prompt).lower()


def _fake_classification(prompt: str) -> str:
    text = _extract_request(prompt)
    if any(k in text for k in ("regulatory", "submission")):
        domain, complexity = "regulatory", "hard"
    elif any(k in text for k in ("clinical", "patient")):
        domain, complexity = "clinical", "hard"
    elif any(k in text for k in ("literature", "paper", "publication")):
        domain, complexity = "literature", "normal"
    elif any(k in text for k in ("sop", "protocol", "study report", "internal")):
        domain, complexity = "internal_docs", "normal"
    elif any(k in text for k in ("sequence", "gene", "protein", "blast")):
        domain, complexity = "bioinformatics", "normal"
    elif any(k in text for k in ("compound", "molecule", "smiles")):
        domain, complexity = "chemistry", "normal"
    elif any(k in text for k in ("csv", "plate", "assay reading", "instrument")):
        domain, complexity = "lab_data", "normal"
    elif len(text.split()) <= 8:
        domain, complexity = "general", "simple"
    else:
        domain, complexity = "general", "normal"
    return json.dumps({"complexity": complexity, "domain": domain, "reason": "mock heuristic, not a real model"})


def _fake_format(prompt: str) -> str:
    content = _after(prompt, "\n\nContent:\n") or prompt
    sources = _between(prompt, "citations exactly:\n", "\n\nContent:")
    text = _trim(content.strip(), 500) or "Nothing to report."
    if sources and sources.strip() and sources.strip() != "(no citations)":
        text += f"\n\n**Sources**\n{sources.strip()}"
    return text


def _fake_synthesis(prompt: str) -> str:
    upstream = _between(prompt, "Upstream research/draft:\n", "\n\nCitations available:")
    citations = _after(prompt, "Citations available:")
    bottom_line = _trim(upstream.strip(), 200) if upstream else "No prior research was available to synthesize."
    text = f"**Bottom line:** {bottom_line}"
    if citations and citations.strip() and citations.strip() != "(none)":
        text += f"\n\nBased on: {citations.strip()}"
    return text


def _fake_draft(prompt: str) -> str:
    tool_result = _after(prompt, "Tool result (")
    if tool_result:
        return _trim(tool_result.split(":\n", 1)[-1].strip(), 500)
    description = _after(prompt, "\n") or prompt
    return f"Draft response addressing: {_trim(description.strip(), 300)}"


def _after(text: str, marker: str) -> str:
    idx = text.find(marker)
    return text[idx + len(marker):] if idx != -1 else ""


def _between(text: str, start: str, end: str) -> str:
    after = _after(text, start)
    if not after:
        return ""
    idx = after.find(end)
    return after[:idx] if idx != -1 else after


def _trim(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"
