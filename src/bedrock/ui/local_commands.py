"""Local slash commands shared by both terminal application modes.

The plain model client and the agent-pack client differ in how they answer a
chat message, but these commands all operate on the same local stores.  This
module is the seam: each app delegates the common command names once, while
Textual worker groups, lazy stores, provenance services, and workspace status
stay implemented in one place.
"""

from __future__ import annotations

from textual import work

from bedrock.config import Settings
from bedrock.connectors import SourceHub
from bedrock.context import ContextStore
from bedrock.ui.audit_command import run_audit_command
from bedrock.ui.commands import Command
from bedrock.ui.confidence_view import run_confidence_command
from bedrock.ui.digest_command import run_digest_command
from bedrock.ui.runs_command import run_runs_command
from bedrock.ui.services import ComplianceServices
from bedrock.ui.source_command import run_sources_command
from bedrock.ui.workspace_command import WorkspaceCommandSession
from bedrock.ui.widgets import StatusBar
from bedrock.workspaces import WorkspaceStore


class LocalCommandsMixin:
    """Own the commands whose behavior is identical in both app modes."""

    LOCAL_DIGEST_USES_MODEL = False

    settings: Settings
    services: ComplianceServices
    _context_store: ContextStore | None
    _sources: SourceHub | None
    _workspace_session: WorkspaceCommandSession | None

    def _init_local_commands(
        self,
        *,
        sources: SourceHub | None,
        services: ComplianceServices | None,
    ) -> None:
        self._context_store = None
        self._sources = sources
        self._workspace_session = None
        self.services = services or ComplianceServices(self.settings)

    def run_local_command(self, command: Command) -> bool:
        """Start a shared command and report whether its name was handled."""
        match command.name:
            case "digest":
                self.command_digest(command.args)
            case "sources":
                self.command_sources(command.args)
            case "runs":
                self.command_runs(command.args)
            case "confidence":
                self.command_confidence(command.args)
            case "audit":
                self.command_audit(command.args)
            case "workspace" | "team":
                self.command_workspace(command.args)
            case _:
                return False
        return True

    @work(group="digest")
    async def command_digest(self, args: str) -> None:
        model_options = {}
        if self.LOCAL_DIGEST_USES_MODEL:
            model_options = {"backend": self.backend, "model": self.model_name}
        await run_digest_command(
            args,
            store=self.context_store,
            notify=self.post_notice,
            sources=self.sources,
            history=await self.services.optional_history(),
            actor_id=self.settings.user_id,
            **model_options,
        )

    @property
    def context_store(self) -> ContextStore:
        if self._context_store is None:
            self._context_store = ContextStore(self.settings.context_dir)
        return self._context_store

    @work(group="sources")
    async def command_sources(self, args: str) -> None:
        await run_sources_command(args, hub=self.sources, notify=self.post_notice)

    @property
    def sources(self) -> SourceHub:
        if self._sources is None:
            self._sources = SourceHub.from_settings(self.settings)
        return self._sources

    @work(group="history")
    async def command_runs(self, args: str) -> None:
        await run_runs_command(args, services=self.services, notify=self.post_notice)

    @work(group="history")
    async def command_confidence(self, args: str) -> None:
        await run_confidence_command(args, services=self.services, notify=self.post_notice)

    @work(group="history")
    async def command_audit(self, args: str) -> None:
        await run_audit_command(args, services=self.services, notify=self.post_notice)

    @work(group="workspace")
    async def command_workspace(self, args: str) -> None:
        await self.workspace_session.run(args, self.post_notice)

    @property
    def workspace_session(self) -> WorkspaceCommandSession:
        if self._workspace_session is None:
            store = WorkspaceStore.for_user(
                self.settings.workspace_dir,
                self.settings.user_id,
                self.settings.display_name,
            )
            self._workspace_session = WorkspaceCommandSession(
                store,
                self.context_store,
                selection_changed=self._set_workspace_name,
            )
        return self._workspace_session

    def _set_workspace_name(self, name: str | None) -> None:
        self.query_one(StatusBar).workspace_name = name
