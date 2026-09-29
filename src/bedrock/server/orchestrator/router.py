"""Routing rules: simple -> Runner only, normal -> Specialist (+Runner
cleanup), hard/high-stakes -> Director plans+reviews. See
docs/server/ARCHITECTURE.md section 2.

`is_high_stakes` is a deterministic code-level safety net over the Runner's
own classification: certain domains are ALWAYS high-stakes regardless of
what the classifier says, since under-escalating a regulatory/clinical/
safety task is the failure mode the whole compliance layer exists to
prevent.
"""

from __future__ import annotations

from bedrock.server.schemas.enums import TaskComplexity, TaskDomain

_ALWAYS_HIGH_STAKES_DOMAINS = {
    TaskDomain.REGULATORY,
    TaskDomain.CLINICAL,
}

_HIGH_STAKES_KEYWORDS = (
    "regulatory submission",
    "final report",
    "safety conclusion",
    "patient data",
    "clinical data",
)


def is_high_stakes(domain: TaskDomain, request_text: str) -> bool:
    if domain in _ALWAYS_HIGH_STAKES_DOMAINS:
        return True
    lowered = request_text.lower()
    return any(kw in lowered for kw in _HIGH_STAKES_KEYWORDS)


def effective_complexity(classified: TaskComplexity, high_stakes: bool) -> TaskComplexity:
    """A high-stakes request is always treated as at least 'hard', even if
    the Runner classified it as simple/normal."""
    if high_stakes:
        return TaskComplexity.HARD
    return classified
