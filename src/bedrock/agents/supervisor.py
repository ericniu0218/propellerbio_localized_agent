"""Execute explicitly granted plan steps concurrently within hard budgets."""

from __future__ import annotations

import asyncio
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bedrock.agents.contract import Grant, Outcome, Plan, Step
from bedrock.agents.grants import GrantError, GrantStore
from bedrock.agents.protocol import Capability


class SupervisorError(RuntimeError):
    """Execution was refused before a capability ran."""


class ActionJournal(Protocol):
    """The append-only subset of ``HistoryStore`` used by the supervisor."""

    def append(
        self,
        operation_id: str,
        *,
        actor_id: str,
        action: str,
        state: str,
        subject_id: str | None = None,
        details: dict | None = None,
    ) -> int: ...


@dataclass(frozen=True)
class ExecutionBudget:
    """Maximum resources one granted plan may consume."""

    max_steps: int = 8
    max_seconds: float = 30.0
    max_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if (
            isinstance(self.max_seconds, bool)
            or not isinstance(self.max_seconds, (int, float))
            or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be a positive number")
        if type(self.max_bytes) is not int or self.max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")


class Supervisor:
    """Run a roster-limited set of capabilities after consuming one Grant."""

    def __init__(
        self,
        capabilities: Mapping[str, Capability],
        grants: GrantStore,
        history: ActionJournal,
        *,
        budget: ExecutionBudget | None = None,
    ) -> None:
        self.capabilities = dict(capabilities)
        self.grants = grants
        self.history = history
        self.budget = budget or ExecutionBudget()

    async def execute(self, plan: Plan, grant: Grant | None) -> Outcome:
        """Execute the granted subset once, or refuse without invoking a tool."""
        self._validate_plan(plan)
        try:
            selected_indexes = self.grants.consume(plan, grant)  # type: ignore[arg-type]
        except GrantError as exc:
            raise SupervisorError(str(exc)) from exc

        assert grant is not None  # ``consume`` accepted an actual Grant.
        selected = tuple(plan.steps[index] for index in selected_indexes)
        await self._record(grant, state="intent")

        if len(selected) > self.budget.max_steps:
            return await self._finish(
                grant,
                Outcome(
                    plan_id=grant.plan_id,
                    failed=f"Plan exceeds the {self.budget.max_steps}-step execution budget.",
                ),
                error_code="step_budget",
            )

        input_bytes = sum(_argument_bytes(step) for step in selected)
        if input_bytes > self.budget.max_bytes:
            return await self._finish(
                grant,
                Outcome(
                    plan_id=grant.plan_id,
                    failed=f"Plan arguments exceed the {self.budget.max_bytes}-byte budget.",
                ),
                error_code="byte_budget",
            )

        try:
            async with asyncio.timeout(float(self.budget.max_seconds)):
                results = await asyncio.gather(
                    *(self._run_step(step) for step in selected),
                    return_exceptions=True,
                )
        except asyncio.CancelledError:
            # Cancellation can arrive after one concurrent capability has
            # committed an effect. Preserve that uncertainty in the journal
            # before allowing the terminal task to unwind.
            try:
                await asyncio.shield(
                    self._record(grant, state="unknown", error_code="cancelled")
                )
            finally:
                raise
        except TimeoutError:
            return await self._finish(
                grant,
                Outcome(
                    plan_id=grant.plan_id,
                    failed=(
                        f"Execution exceeded the {self.budget.max_seconds:g}-second budget; "
                        "some steps may have completed before cancellation."
                    ),
                ),
                error_code="time_budget",
            )

        completed: list[str] = []
        artifacts: list[str] = []
        failures: list[str] = []
        for step, result in zip(selected, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(f"{step.tool}: {_exception_text(result)}")
                continue
            if not isinstance(result, Outcome):
                failures.append(f"{step.tool}: capability returned an invalid outcome")
                continue
            completed.extend(result.completed or (step.summary,))
            artifacts.extend(result.artifacts)
            if result.failed:
                failures.append(f"{step.tool}: {result.failed}")

        outcome = Outcome(
            plan_id=grant.plan_id,
            completed=tuple(completed),
            artifacts=tuple(artifacts),
            failed="; ".join(failures) or None,
        )
        output_bytes = input_bytes + _outcome_bytes(outcome)
        if output_bytes > self.budget.max_bytes:
            budget_failure = (
                f"Execution produced more than the {self.budget.max_bytes}-byte budget."
            )
            outcome = Outcome(
                plan_id=outcome.plan_id,
                completed=outcome.completed,
                artifacts=outcome.artifacts,
                failed=_join_failures(outcome.failed, budget_failure),
            )
        return await self._finish(
            grant,
            outcome,
            error_code="capability_failed" if outcome.failed else None,
        )

    def _validate_plan(self, plan: Plan) -> None:
        if plan.blocked_reason:
            raise SupervisorError("A blocked plan cannot execute.")
        if not plan.steps:
            raise SupervisorError("A plan with no steps cannot execute.")
        for step in plan.steps:
            capability = self.capabilities.get(step.tool)
            if capability is None:
                raise SupervisorError(f"Capability {step.tool!r} is not in this agent's roster.")
            if getattr(capability, "name", None) != step.tool:
                raise SupervisorError(f"Capability binding for {step.tool!r} is invalid.")
            if capability.risk is not step.risk:
                raise SupervisorError(f"Risk for capability {step.tool!r} changed after planning.")

    async def _run_step(self, step: Step) -> Outcome:
        return await self.capabilities[step.tool].run(dict(step.arguments))

    async def _record(
        self,
        grant: Grant,
        *,
        state: str,
        error_code: str | None = None,
    ) -> None:
        details = {
            "task_id": grant.plan_id,
            "decision": "approve",
            "actor_source": "config",
        }
        if error_code:
            details["error_code"] = error_code
        try:
            await asyncio.to_thread(
                self.history.append,
                grant.plan_id,
                actor_id=grant.granted_by,
                action="approval",
                state=state,
                subject_id=grant.plan_id,
                details=details,
            )
        except Exception as exc:
            if state == "intent":
                raise SupervisorError(
                    "Could not journal the approved plan; no capabilities were run."
                ) from exc
            raise SupervisorError("Could not journal the execution outcome.") from exc

    async def _finish(
        self,
        grant: Grant,
        outcome: Outcome,
        *,
        error_code: str | None,
    ) -> Outcome:
        state = "failed" if outcome.failed else "succeeded"
        try:
            await self._record(grant, state=state, error_code=error_code)
        except SupervisorError as exc:
            return Outcome(
                plan_id=outcome.plan_id,
                completed=outcome.completed,
                artifacts=outcome.artifacts,
                failed=_join_failures(outcome.failed, str(exc)),
            )
        return outcome


def _argument_bytes(step: Step) -> int:
    return sum(
        len(key.encode("utf-8")) + len(value.encode("utf-8"))
        for key, value in step.arguments.items()
    )


def _outcome_bytes(outcome: Outcome) -> int:
    size = sum(len(item.encode("utf-8")) for item in outcome.completed)
    size += len(outcome.failed.encode("utf-8")) if outcome.failed else 0
    for artifact in outcome.artifacts:
        size += len(artifact.encode("utf-8"))
        try:
            metadata = Path(artifact).lstat()
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode):
            size += metadata.st_size
    return size


def _exception_text(exc: BaseException) -> str:
    text = str(exc).strip()
    return (text or type(exc).__name__)[:300]


def _join_failures(first: str | None, second: str) -> str:
    return f"{first}; {second}" if first else second
