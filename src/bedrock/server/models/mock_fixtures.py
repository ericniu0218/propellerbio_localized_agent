"""Realistic-looking fake replies for mock mode (MockBackend).

Without this, the mock model just echoed back whatever prompt it was
given — which, for the Runner's formatting call (since removed), meant
the literal internal formatting *instructions* ("append a 'Sources'
section listing these citations exactly...") ended up shown to the user as
if it were the answer. That's a bad demo and would confuse a non-technical
user, even though the underlying agent loop was working correctly.

This module recognizes the shape of a handful of known prompts (by the
fixed instruction text each agent method in `src/bedrock/server/agents/*.py` always
sends) and returns something that looks like a plausible reply to that
kind of call, instead of parroting the prompt. It is not a language model
— just enough pattern-matching to make mock mode demoable. Real mode
never touches this file.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone


def fake_reply(prompt: str, model_name: str) -> str:
    if 'Respond with JSON: {"complexity"' in prompt:
        return _fake_classification(prompt)
    if "Extract the date/time range" in prompt:
        return _fake_date_range(prompt)
    if "Extract structured entities as JSON" in prompt:
        return json.dumps({"genes": [], "proteins": [], "compound_ids": [], "dates": [], "doses": []})
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
    elif any(k in text for k in ("meeting", "zoom", "standup", "sync", "call")):
        domain, complexity = "meetings", "normal"
    elif any(k in text for k in ("sequence", "gene", "protein", "blast")):
        domain, complexity = "bioinformatics", "normal"
    elif any(k in text for k in ("compound", "molecule", "smiles")):
        domain, complexity = "chemistry", "normal"
    elif any(k in text for k in ("csv", "plate", "assay reading", "instrument")):
        domain, complexity = "lab_data", "normal"
    elif any(k in text for k in ("meeting", "call", "standup", "stand-up", "said", "discuss", "sync")):
        domain, complexity = "meetings", "normal"
    elif len(text.split()) <= 8:
        domain, complexity = "general", "simple"
    else:
        domain, complexity = "general", "normal"
    return json.dumps({"complexity": complexity, "domain": domain, "reason": "mock heuristic, not a real model"})


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _fake_date_range(prompt: str) -> str:
    """Deterministic stand-in for a real model resolving relative dates
    ("last Tuesday", "yesterday") against the reference time the prompt
    provides — same "mock heuristic, not a real model" spirit as
    _fake_classification. Handles the common phrasings; anything it
    doesn't recognize falls back to a bounded 7-day lookback rather than
    an empty range, so a search never silently returns nothing just
    because this heuristic didn't understand the phrasing."""
    reference_match = re.search(
        r"current date and time is (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?)",
        prompt,
    )
    reference = (
        datetime.fromisoformat(reference_match.group(1).replace("Z", "+00:00"))
        if reference_match
        else datetime.now(timezone.utc)
    )
    text = _extract_request(prompt)
    start_of_today = reference.replace(hour=0, minute=0, second=0, microsecond=0)

    if "yesterday" in text:
        start, end = start_of_today - timedelta(days=1), start_of_today
    elif "today" in text:
        start, end = start_of_today, reference
    elif "this week" in text:
        start, end = start_of_today - timedelta(days=start_of_today.weekday()), reference
    elif "last week" in text:
        this_week_start = start_of_today - timedelta(days=start_of_today.weekday())
        start, end = this_week_start - timedelta(days=7), this_week_start
    else:
        weekday_index = next((i for i, name in enumerate(_WEEKDAYS) if name in text), None)
        if weekday_index is not None:
            days_since = (start_of_today.weekday() - weekday_index) % 7
            days_since = days_since or 7  # e.g. "last Monday" said on a Monday means a week ago
            start = start_of_today - timedelta(days=days_since)
            end = start + timedelta(days=1)
        else:
            start, end = reference - timedelta(days=7), reference

    return json.dumps({"from": start.isoformat(), "to": end.isoformat()})


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
