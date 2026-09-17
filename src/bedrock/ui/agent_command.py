"""Native terminal approval prompt for an inert agent plan."""

from __future__ import annotations

import json

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Static

from bedrock.agents.contract import Grant, Plan
from bedrock.agents.grants import GrantError, GrantStore
from bedrock.ui.rendering import safe_block, safe_line


def render_plan(plan: Plan) -> RenderableType:
    """Build the complete Rich rendering a person approves or rejects."""
    heading = Text("Agent plan", style="bold")
    request = Text()
    request.append("Request: ", style="bold")
    request.append(safe_line(plan.ask.text, limit=500))

    evidence = Text()
    evidence.append("Evidence: ", style="bold")
    evidence.append(f'“{safe_line(plan.ask.evidence.quote, limit=500)}”')
    if plan.ask.evidence.speaker:
        evidence.append(f" — {safe_line(plan.ask.evidence.speaker)}")
    if plan.ask.evidence.timestamp:
        evidence.append(f" at {safe_line(plan.ask.evidence.timestamp)}")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Tool", no_wrap=True)
    table.add_column("Risk", no_wrap=True)
    table.add_column("What it will do")
    table.add_column("Arguments")
    for index, step in enumerate(plan.steps, start=1):
        arguments = json.dumps(step.arguments, sort_keys=True, ensure_ascii=False)
        table.add_row(
            str(index),
            safe_line(step.tool),
            step.risk.value,
            safe_block(step.summary, limit=800),
            safe_block(arguments, limit=800),
        )

    parts: list[RenderableType] = [heading, request, evidence, table]
    if plan.blocked_reason:
        blocked = Text()
        blocked.append("Blocked: ", style="bold red")
        blocked.append(safe_line(plan.blocked_reason, limit=500))
        parts.append(blocked)
    elif plan.needs_approval:
        parts.append(Text("This plan changes things or reaches people.", style="yellow"))
    parts.append(Text("Nothing runs until you approve this exact plan.", style="dim"))
    return Group(*parts)


class AgentApprovalPrompt(Static):
    """Issue a one-use Grant only in response to an explicit button press."""

    DEFAULT_CSS = """
    AgentApprovalPrompt {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border-left: thick $warning;
    }
    AgentApprovalPrompt #agent-approval-buttons {
        height: auto;
        margin-top: 1;
    }
    AgentApprovalPrompt #agent-approval-buttons Button {
        min-width: 14;
        margin-right: 2;
    }
    AgentApprovalPrompt #agent-approval-status {
        height: auto;
        color: $text-muted;
    }
    """

    class Decided(Message):
        """The employee approved (with a Grant) or rejected the plan."""

        def __init__(
            self,
            prompt: "AgentApprovalPrompt",
            *,
            approved: bool,
            grant: Grant | None,
        ) -> None:
            self.prompt = prompt
            self.plan = prompt.plan
            self.approved = approved
            self.grant = grant
            super().__init__()

    def __init__(
        self,
        plan: Plan,
        grants: GrantStore,
        *,
        granted_by: str,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.grants = grants
        self.granted_by = granted_by
        self.state = "pending"
        self.grant: Grant | None = None

    def compose(self) -> ComposeResult:
        yield Static(render_plan(self.plan), id="agent-plan")
        with Horizontal(id="agent-approval-buttons"):
            yield Button(
                "Approve plan",
                id="approve-agent-plan",
                variant="success",
                disabled=bool(self.plan.blocked_reason or not self.plan.steps),
            )
            yield Button("Reject", id="reject-agent-plan", variant="error")
        yield Static("", id="agent-approval-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.state != "pending":
            return
        approved = event.button.id == "approve-agent-plan"
        self.state = "decided"
        self._set_buttons_enabled(False)
        if approved:
            try:
                self.grant = self.grants.issue(self.plan, granted_by=self.granted_by)
            except GrantError as exc:
                self.state = "pending"
                self._set_buttons_enabled(True)
                self._set_status(str(exc))
                return
            self._set_status("Approved once for this exact plan.")
        else:
            self._set_status("Rejected. Nothing ran.")
        self.post_message(self.Decided(self, approved=approved, grant=self.grant))

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for button in self.query(Button):
            if button.id == "approve-agent-plan" and (
                self.plan.blocked_reason or not self.plan.steps
            ):
                button.disabled = True
            else:
                button.disabled = not enabled

    def _set_status(self, message: str) -> None:
        status = self.query("#agent-approval-status")
        if status:
            status.first(Static).update(safe_line(message, limit=400))


# A readable import name for call sites that think in terms of plans.
PlanApprovalPrompt = AgentApprovalPrompt
