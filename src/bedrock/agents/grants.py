"""One-plan, one-person, one-use execution grants.

``Grant`` is intentionally a small value in the frozen contract.  This store
supplies the state that the value does not carry: which exact plan was shown
and whether its permission has already been spent.  A hand-constructed Grant
therefore cannot make a plan executable.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections import Counter
from dataclasses import asdict, dataclass

from bedrock.agents.contract import Grant, Plan


class GrantError(ValueError):
    """A grant is missing, forged, stale, or does not cover the requested work."""


def plan_fingerprint(plan: Plan) -> str:
    """Return a stable digest of everything a person reads in a plan."""
    encoded = json.dumps(
        asdict(plan),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _IssuedGrant:
    grant: Grant
    plan_fingerprint: str


class GrantStore:
    """Hold unspent grants in memory for the lifetime of the terminal app."""

    def __init__(self) -> None:
        self._issued: dict[str, _IssuedGrant] = {}
        self._lock = threading.Lock()

    def issue(
        self,
        plan: Plan,
        *,
        granted_by: str,
        steps_allowed: tuple[str, ...] | None = None,
    ) -> Grant:
        """Create permission after a person explicitly approves ``plan``.

        ``steps_allowed`` contains tool names and may select a subset of the
        plan. Repeated names count separately, so approving one ``echo`` step
        does not silently approve every ``echo`` step in the same plan.
        """
        actor = granted_by.strip() if isinstance(granted_by, str) else ""
        if not actor:
            raise GrantError("A grant must name the person who approved it.")
        if plan.blocked_reason:
            raise GrantError("A blocked plan cannot be granted.")
        if not plan.steps:
            raise GrantError("A plan with no steps cannot be granted.")

        planned = tuple(step.tool for step in plan.steps)
        allowed = planned if steps_allowed is None else tuple(steps_allowed)
        if not allowed:
            raise GrantError("A grant must allow at least one plan step.")
        if any(not isinstance(name, str) or not name for name in allowed):
            raise GrantError("Granted step names must be non-empty strings.")
        planned_counts = Counter(planned)
        allowed_counts = Counter(allowed)
        if any(count > planned_counts[name] for name, count in allowed_counts.items()):
            raise GrantError("A grant may only allow steps present in its plan.")

        fingerprint = plan_fingerprint(plan)
        plan_id = f"{fingerprint[:16]}-{uuid.uuid4().hex[:16]}"
        grant = Grant(plan_id=plan_id, granted_by=actor, steps_allowed=allowed)
        with self._lock:
            self._issued[plan_id] = _IssuedGrant(grant, fingerprint)
        return grant

    def consume(self, plan: Plan, grant: Grant) -> tuple[int, ...]:
        """Spend ``grant`` and return the plan-step indexes it authorizes."""
        if not isinstance(grant, Grant):
            raise GrantError("Execution requires an explicit Grant.")
        with self._lock:
            issued = self._issued.get(grant.plan_id)
            if issued is None or issued.grant is not grant:
                raise GrantError("This grant was not issued here or was already used.")
            if issued.plan_fingerprint != plan_fingerprint(plan):
                raise GrantError("The plan changed after it was approved.")
            indexes = _selected_step_indexes(plan, grant.steps_allowed)
            del self._issued[grant.plan_id]
            return indexes

    def revoke(self, grant: Grant) -> None:
        """Make an unspent grant unusable without executing it."""
        with self._lock:
            issued = self._issued.get(grant.plan_id)
            if issued is not None and issued.grant is grant:
                del self._issued[grant.plan_id]


def _selected_step_indexes(plan: Plan, allowed: tuple[str, ...]) -> tuple[int, ...]:
    remaining = Counter(allowed)
    selected: list[int] = []
    for index, step in enumerate(plan.steps):
        if remaining[step.tool] > 0:
            selected.append(index)
            remaining[step.tool] -= 1
    if any(remaining.values()):
        raise GrantError("The grant names a step that is not in the approved plan.")
    return tuple(selected)
