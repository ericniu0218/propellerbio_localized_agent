"""The 'what's happening right now' strip.

Deliberately small and quiet: one line per agent currently working on the
request, in plain language (see `bedrock.agentpack.friendly`), docked
under the status bar. It disappears entirely when nothing is running, so
the idle screen stays as clean as the original chat-only design — the
panel earns its screen space only while there's something to report.
"""

from __future__ import annotations

from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static

from bedrock.agentpack.models import AgentCardState


class AgentPanel(Vertical):
    lines: reactive[tuple[str, ...]] = reactive(())

    def compose(self):
        yield Static("", id="agent-panel-body")

    def watch_lines(self, value: tuple[str, ...]) -> None:
        self.display = bool(value)
        self.query_one("#agent-panel-body", Static).update("\n".join(value))

    def update_agents(self, agents: dict[str, AgentCardState]) -> None:
        # Only show agents still doing something — a finished/idle agent
        # doesn't need a permanent line once the task is over.
        active = [a for a in agents.values() if a.status not in ("done", "idle")]
        self.lines = tuple(f"◆ {a.friendly_line()}" for a in active)

    def clear(self) -> None:
        self.lines = ()
