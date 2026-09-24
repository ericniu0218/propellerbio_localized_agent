from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button

from bedrock.agents.capabilities.echo import CAPABILITY as ECHO
from bedrock.agents.contract import (
    Ask,
    AskKind,
    Evidence,
    Grant,
    Outcome,
    Plan,
    Risk,
    Step,
)
from bedrock.agents.grants import GrantError, GrantStore
from bedrock.agents.planner import Planner
import bedrock.agents.roster as roster_module
from bedrock.agents.roster import RosterError, discover_capability, load_roster
from bedrock.agents.supervisor import ExecutionBudget, Supervisor, SupervisorError
from bedrock.config import load_settings
from bedrock.provenance import HistoryStore
from bedrock.ui.agent_command import AgentApprovalPrompt, render_plan


def an_ask() -> Ask:
    return Ask(
        kind=AskKind.LOOK_UP,
        text="Repeat the status for the notes",
        evidence=Evidence(
            quote="Can you repeat the status for the notes?",
            speaker="Alice",
            timestamp="00:01:12",
        ),
        owner="Alice",
        subject="status",
        confidence=0.95,
    )


def a_plan(*steps: Step) -> Plan:
    return Plan(ask=an_ask(), steps=steps)


class ReplyBackend:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[tuple[str, list[dict[str, str]]]] = []
        self.options: list = []

    async def stream_chat(self, model: str, messages: list[dict[str, str]], options=None, on_usage=None):
        self.requests.append((model, messages))
        self.options.append(options)
        yield self.reply


class RecordingCapability:
    name = "echo"
    risk = Risk.READ_ONLY

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def describe(self, arguments: dict[str, str]) -> str:
        return f"Record {arguments.get('message', '')!r} locally."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        self.calls.append(arguments)
        return Outcome(plan_id="capability-does-not-choose-this", completed=(self.describe(arguments),))


class MemoryJournal:
    def __init__(self, *, fail_intent: bool = False) -> None:
        self.rows: list[dict] = []
        self.fail_intent = fail_intent

    def append(self, operation_id: str, **values) -> int:
        if self.fail_intent and values["state"] == "intent":
            raise RuntimeError("disk unavailable")
        self.rows.append({"operation_id": operation_id, **values})
        return len(self.rows)


def echo_step(message: str = "hello", **arguments: str) -> Step:
    values = {"message": message, **arguments}
    return Step(
        tool="echo",
        summary=f"Record {message!r} locally.",
        risk=Risk.READ_ONLY,
        arguments=values,
    )


async def test_planner_only_uses_rostered_capability_owned_descriptions_and_risk():
    backend = ReplyBackend(
        json.dumps(
            {
                "steps": [{"tool": "echo", "arguments": {"message": "ready"}}],
                "blocked_reason": None,
            }
        )
    )
    planner = Planner(backend, "private-model", {"echo": ECHO})

    plan = await planner.plan(an_ask())

    assert plan.blocked_reason is None
    assert plan.steps == (
        Step(
            tool="echo",
            summary="Echo 'ready' locally.",
            risk=Risk.READ_ONLY,
            arguments={"message": "ready"},
        ),
    )
    model, messages = backend.requests[0]
    assert model == "private-model"
    sent = json.loads(messages[1]["content"])
    assert sent["available_capabilities"] == [{"name": "echo", "risk": "read_only"}]


async def test_planner_blocks_an_invented_capability_instead_of_returning_partial_work():
    backend = ReplyBackend(
        '{"steps":[{"tool":"mail.send","arguments":{}}],"blocked_reason":null}'
    )

    plan = await Planner(backend, "private-model", {"echo": ECHO}).plan(an_ask())

    assert plan.steps == ()
    assert "not in this agent's roster" in plan.blocked_reason


async def test_planner_is_a_user_job_with_a_reply_limit_and_the_plan_schema():
    from bedrock.agents.planner import plan_schema
    from bedrock.config import PROFILES
    from bedrock.inference import Priority, Scheduler

    backend = ReplyBackend('{"steps":[{"tool":"echo","arguments":{"message":"hi"}}],"blocked_reason":null}')
    scheduler = Scheduler(backend, PROFILES["laptop"])
    priorities = []
    original = scheduler.stream

    def recording(model, messages, *, priority, options=None):
        priorities.append(priority)
        return original(model, messages, priority=priority, options=options)

    scheduler.stream = recording
    plan = await Planner(scheduler, "private-model", {"echo": ECHO}, max_steps=3).plan(an_ask())

    assert plan.blocked_reason is None
    assert priorities == [Priority.USER_JOB]
    [options] = backend.options
    assert options.max_tokens == PROFILES["laptop"].reply_limit
    assert options.json_schema == plan_schema(["echo"], 3)
    assert options.json_schema["properties"]["steps"]["items"]["properties"]["tool"]["enum"] == ["echo"]


async def test_planner_still_checks_a_reply_the_server_did_not_constrain():
    """A server may ignore the schema; the parser is exactly as strict as before."""
    backend = ReplyBackend('{"steps":[],"blocked_reason":null,"extra":"field"}')
    plan = await Planner(backend, "private-model", {"echo": ECHO}).plan(an_ask())
    assert "unexpected fields" in plan.blocked_reason


def test_settings_contain_the_complete_agent_block(tmp_path: Path):
    settings = load_settings(config_path=tmp_path / "missing.toml", env={})

    assert settings.agents["roster_path"].endswith("/.config/bedrock/agents.toml")
    assert settings.agents["max_steps"] > 0
    assert settings.agents["max_seconds"] > 0
    assert settings.agents["max_bytes"] > 0
    assert set(settings.agents["capabilities"]) == {
        "echo",
        "files.search",
        "files.read",
        "sources.fetch",
        "digest.query",
        "web.fetch",
        "web.search",
        "calendar.create",
        "mail.draft",
        "mail.send",
        "files.write",
        "shell.run",
    }
    assert all(not value for value in settings.agents["capabilities"].values())


def test_roster_discovers_only_named_module_capabilities(tmp_path: Path):
    path = tmp_path / "agents.toml"
    path.write_text(
        """[agents.note_taker]
capabilities = ["echo"]
max_steps = 2
max_seconds = 1.5
max_bytes = 512
""",
        encoding="utf-8",
    )

    roster = load_roster(path)

    agent = roster["note_taker"]
    assert list(agent.capabilities) == ["echo"]
    assert agent.capabilities["echo"] is ECHO
    assert (agent.max_steps, agent.max_seconds, agent.max_bytes) == (2, 1.5, 512)


def test_roster_does_not_fall_back_to_a_hardcoded_capability(tmp_path: Path):
    path = tmp_path / "agents.toml"
    path.write_text('[agents.bad]\ncapabilities = ["not_installed"]\n', encoding="utf-8")

    with pytest.raises(RosterError, match="unavailable"):
        load_roster(path)


def test_roster_accepts_a_plural_only_capabilities_export(monkeypatch):
    module = SimpleNamespace(CAPABILITIES=(ECHO,))
    monkeypatch.setattr(roster_module.importlib, "import_module", lambda _name: module)

    assert discover_capability("echo") is ECHO


async def test_forged_missing_and_reused_grants_never_execute():
    capability = RecordingCapability()
    grants = GrantStore()
    history = MemoryJournal()
    supervisor = Supervisor({"echo": capability}, grants, history)
    plan = a_plan(echo_step())

    with pytest.raises(SupervisorError, match="explicit Grant"):
        await supervisor.execute(plan, None)
    assert capability.calls == []

    real = grants.issue(plan, granted_by="alice")
    forged = Grant(
        plan_id=real.plan_id,
        granted_by=real.granted_by,
        steps_allowed=real.steps_allowed,
    )
    with pytest.raises(SupervisorError, match="not issued"):
        await supervisor.execute(plan, forged)
    assert capability.calls == []

    outcome = await supervisor.execute(plan, real)
    assert outcome.plan_id == real.plan_id
    assert capability.calls == [{"message": "hello"}]
    assert [row["state"] for row in history.rows] == ["intent", "succeeded"]

    with pytest.raises(SupervisorError, match="already used"):
        await supervisor.execute(plan, real)
    assert len(capability.calls) == 1


async def test_plan_change_and_risk_downgrade_are_refused_before_execution():
    capability = RecordingCapability()
    grants = GrantStore()
    supervisor = Supervisor({"echo": capability}, grants, MemoryJournal())
    original = a_plan(echo_step("one"))
    changed = a_plan(echo_step("two"))
    grant = grants.issue(original, granted_by="alice")

    with pytest.raises(SupervisorError, match="changed"):
        await supervisor.execute(changed, grant)
    assert capability.calls == []

    downgraded = a_plan(
        Step(
            tool="echo",
            summary="Pretend this mutating tool is harmless",
            risk=Risk.CHANGES_THINGS,
            arguments={"message": "one"},
        )
    )
    with pytest.raises(SupervisorError, match="Risk"):
        await supervisor.execute(downgraded, grants.issue(downgraded, granted_by="alice"))
    assert capability.calls == []


async def test_unwritable_intent_journal_stops_every_capability():
    capability = RecordingCapability()
    grants = GrantStore()
    plan = a_plan(echo_step())
    grant = grants.issue(plan, granted_by="alice")
    supervisor = Supervisor({"echo": capability}, grants, MemoryJournal(fail_intent=True))

    with pytest.raises(SupervisorError, match="no capabilities were run"):
        await supervisor.execute(plan, grant)
    assert capability.calls == []


async def test_supervisor_writes_a_real_intent_and_outcome_journal(tmp_path: Path):
    capability = RecordingCapability()
    grants = GrantStore()
    history = HistoryStore(tmp_path / "history")
    plan = a_plan(echo_step())
    grant = grants.issue(plan, granted_by="alice")

    outcome = await Supervisor({"echo": capability}, grants, history).execute(plan, grant)

    assert outcome.failed is None
    events = history.events(limit=10)
    assert [(event["action"], event["state"]) for event in events] == [
        ("approval", "intent"),
        ("approval", "succeeded"),
    ]
    assert all(event["operation_id"] == grant.plan_id for event in events)


async def test_independent_steps_start_together_and_preserve_plan_order():
    class RendezvousCapability(RecordingCapability):
        def __init__(self) -> None:
            super().__init__()
            self.both_started = asyncio.Event()

        async def run(self, arguments: dict[str, str]) -> Outcome:
            self.calls.append(arguments)
            if len(self.calls) == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), timeout=0.2)
            return Outcome(plan_id="", completed=(arguments["message"],))

    capability = RendezvousCapability()
    grants = GrantStore()
    plan = a_plan(echo_step("first"), echo_step("second"))
    grant = grants.issue(plan, granted_by="alice")
    supervisor = Supervisor(
        {"echo": capability},
        grants,
        MemoryJournal(),
        budget=ExecutionBudget(max_steps=2, max_seconds=1, max_bytes=1024),
    )

    outcome = await supervisor.execute(plan, grant)

    assert outcome.failed is None
    assert outcome.completed == ("first", "second")
    assert capability.calls == [{"message": "first"}, {"message": "second"}]


async def test_cancelling_an_in_flight_plan_records_an_unknown_outcome():
    class WaitingCapability(RecordingCapability):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def run(self, arguments: dict[str, str]) -> Outcome:
            self.calls.append(arguments)
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    capability = WaitingCapability()
    grants = GrantStore()
    history = MemoryJournal()
    plan = a_plan(echo_step())
    grant = grants.issue(plan, granted_by="alice")
    task = asyncio.create_task(
        Supervisor({"echo": capability}, grants, history).execute(plan, grant)
    )
    await capability.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [row["state"] for row in history.rows] == ["intent", "unknown"]


async def test_step_and_byte_budgets_refuse_work_before_a_capability_runs():
    capability = RecordingCapability()

    grants = GrantStore()
    too_many = a_plan(echo_step("one"), echo_step("two"))
    step_outcome = await Supervisor(
        {"echo": capability},
        grants,
        MemoryJournal(),
        budget=ExecutionBudget(max_steps=1, max_seconds=1, max_bytes=1024),
    ).execute(too_many, grants.issue(too_many, granted_by="alice"))
    assert "step execution budget" in step_outcome.failed
    assert capability.calls == []

    grants = GrantStore()
    too_large = a_plan(echo_step("a message larger than five bytes"))
    byte_outcome = await Supervisor(
        {"echo": capability},
        grants,
        MemoryJournal(),
        budget=ExecutionBudget(max_steps=1, max_seconds=1, max_bytes=5),
    ).execute(too_large, grants.issue(too_large, granted_by="alice"))
    assert "byte budget" in byte_outcome.failed
    assert capability.calls == []


def test_blocked_and_empty_plans_cannot_receive_grants():
    grants = GrantStore()
    with pytest.raises(GrantError, match="blocked"):
        grants.issue(Plan(ask=an_ask(), blocked_reason="No suitable tool"), granted_by="alice")
    with pytest.raises(GrantError, match="no steps"):
        grants.issue(Plan(ask=an_ask()), granted_by="alice")


class PromptApp(App):
    def __init__(self, plan: Plan, grants: GrantStore) -> None:
        super().__init__()
        self.plan = plan
        self.grants = grants
        self.decisions: list[AgentApprovalPrompt.Decided] = []

    def compose(self) -> ComposeResult:
        yield AgentApprovalPrompt(self.plan, self.grants, granted_by="alice")

    @on(AgentApprovalPrompt.Decided)
    def record_decision(self, message: AgentApprovalPrompt.Decided) -> None:
        self.decisions.append(message)


async def test_textual_approval_click_is_the_only_thing_that_issues_a_grant():
    grants = GrantStore()
    plan = a_plan(echo_step("show me"))
    app = PromptApp(plan, grants)

    async with app.run_test() as pilot:
        prompt = app.query_one(AgentApprovalPrompt)
        assert prompt.grant is None
        prompt.query_one("#approve-agent-plan", Button).press()
        prompt.query_one("#approve-agent-plan", Button).press()
        await pilot.pause()

        assert prompt.grant is not None
        assert prompt.grant.granted_by == "alice"
        assert len(app.decisions) == 1
        assert app.decisions[0].grant is prompt.grant
        assert grants.consume(plan, prompt.grant) == (0,)


def test_rich_plan_render_shows_evidence_risk_arguments_and_safety_boundary():
    console = Console(record=True, width=160)
    console.print(render_plan(a_plan(echo_step("visible"))))

    rendered = console.export_text()
    assert "Can you repeat the status for the notes?" in rendered
    assert "00:01:12" in rendered
    assert "read_only" in rendered
    assert '"message": "visible"' in rendered
    assert "Nothing runs until you approve" in rendered
