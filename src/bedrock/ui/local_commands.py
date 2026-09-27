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
from bedrock.ui.digest_command import (
    AutoContext,
    auto_context_argument,
    run_auto_context_command,
    run_digest_command,
)
from bedrock.ui.review_command import run_reviews_command
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
    auto_context: AutoContext

    def _init_local_commands(
        self,
        *,
        sources: SourceHub | None,
        services: ComplianceServices | None,
    ) -> None:
        self._context_store = None
        self._sources = sources
        self._workspace_session = None
        self.auto_context = AutoContext(enabled=self.settings.auto_context)
        self.services = services or ComplianceServices(self.settings)

    async def run_local_command(self, command: Command) -> bool:
        """Start a shared command and report whether its name was handled."""
        match command.name:
            case "digest":
                choice = auto_context_argument(command.args)
                if choice is not None:
                    # Apply immediately: a prompt sent next must already follow
                    # the new setting, even though ordinary digest work runs in
                    # a background worker.
                    await run_auto_context_command(
                        choice,
                        auto_context=self.auto_context,
                        notify=self.post_notice,
                    )
                else:
                    self.command_digest(command.args)
            case "sources":
                self.command_sources(command.args)
            case "runs":
                self.command_runs(command.args)
            case "reviews":
                self.command_reviews(command.args)
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
            # Through the scheduler, as a job the employee asked for: it waits
            # for a streaming reply rather than competing with it.
            model_options = {"scheduler": self.scheduler, "model": self.model_name}
        await run_digest_command(
            args,
            store=self.context_store,
            notify=self.post_notice,
            sources=self.sources,
            history=await self.services.optional_history(),
            actor_id=self.settings.user_id,
            auto_context=self.auto_context,
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
    async def command_reviews(self, args: str) -> None:
        """Literature reviews already on this machine. Needs no model, so it
        works in both app modes and with every server switched off.

        The settings loader is imported here rather than at the top: it pulls in
        the review pipeline, which both apps would otherwise pay for on every
        launch. See `bedrock/ui/review_command.py`.
        """
        from bedrock.litreview.config import load_settings as load_review_settings

        await run_reviews_command(
            args,
            # Listing, verifying and forgetting need the database, not a model set.
            settings_for=lambda: load_review_settings(self.settings, check_endpoint=False, require_models=False),
            # `check` is the one verb that does need the model set and the
            # endpoint rule, because what it reports is whether a review could
            # actually start.
            full_settings_for=lambda: load_review_settings(self.settings),
            notify=self.post_notice,
            # The agent-pack app has no scheduler, and neither app has one before
            # it finishes starting; `check` then reports the configuration half
            # and says the server was not asked.
            backend=getattr(getattr(self, "scheduler", None), "backend", None),
        )

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
