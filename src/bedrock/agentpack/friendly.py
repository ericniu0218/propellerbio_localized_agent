"""Plain-language labels for what an agent is doing.

Kept as pure functions with no networking or Textual imports, same
convention as `bedrock.ui.commands` — trivial to test, and it's the one
place that needs to change if the backend's agents are renamed or a new
one is added. Mirrors the table in the backend's `docs/UX.md` section 2;
if that table changes, update both.

Employees never see "director"/"specialist"/"runner" or a model name in
the default view — only these labels. Nothing here requires the backend
to send anything beyond what it already does (role, status, current task
title); it's a heuristic read of those three fields, not a new contract.
"""

from __future__ import annotations

_DIRECTOR_LABEL = "Planner"
_RESEARCH_KEYWORDS = ("research", "literature", "search", "compare")


def friendly_agent_line(role: str, status: str, task_title: str | None) -> str:
    """One short line for the agent panel, e.g. "Researcher: reading your documents"."""
    title = (task_title or "").lower()

    if role == "director":
        return f"{_DIRECTOR_LABEL}: {_director_action(status, title)}"
    if role == "specialist":
        label = "Researcher" if any(k in title for k in _RESEARCH_KEYWORDS) else "Assistant"
        return f"{label}: {_worker_action(status)}"
    # runner
    return f"Assistant: {_worker_action(status)}"


def _director_action(status: str, title: str) -> str:
    if status == "error":
        return "ran into a problem and stopped"
    if status == "done":
        return "finished checking this"
    if "plan" in title:
        return "breaking your request into steps"
    if "synthesis" in title or "review" in title:
        return "double-checking the results"
    return "working out the best approach"


def _worker_action(status: str) -> str:
    return {
        "idle": "waiting",
        "thinking": "working on it",
        "running_tool": "gathering information",
        "waiting_for_review": "waiting on a check",
        "done": "finished",
        "error": "ran into a problem",
    }.get(status, "working on it")


def friendly_error(message: str) -> str:
    """Best-effort translation of a raw backend error into one plain
    sentence + a next step, per docs/UX.md section 4. Falls back to a
    generic, still-plain message rather than ever showing the raw text
    verbatim (which may contain a Python exception name)."""
    lowered = message.lower()
    if "budget" in lowered:
        return (
            "This task used more thinking than expected and stopped early. "
            "Try breaking it into smaller questions, or ask again."
        )
    if "citation" in lowered:
        return (
            "The answer couldn't be backed up with a clear source, so it was "
            "sent back for another look. You'll see an update shortly."
        )
    if "connection refused" in lowered or "could not" in lowered:
        return "Can't reach the AI system right now. Check your connection and try again."
    return "Something went wrong finishing this. You can try again, or ask for a smaller piece of it."
