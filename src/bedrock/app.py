"""The Textual application: wires the UI to the inference backend.

Responsibilities of this file, and nothing more:

* build the screen (status bar, scrolling conversation, input box)
* turn what the user types into either a slash command or a chat message
* run the streaming request in a background worker and feed the text to the
  widget as it arrives
* keep the connection indicator up to date
* save the conversation as it goes, and reopen the last one on startup
* keep the conversation sidebar in step with what is saved, and fill in the
  short summary titles it shows

It talks to the model *only* through the ``InferenceBackend`` interface, and to
the disk *only* through ``ConversationStore``.
"""

from __future__ import annotations

import asyncio
import re
import sys
import unicodedata
from datetime import datetime

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Input, OptionList

from bedrock.config import Settings, load_settings
from bedrock.context import ContextStore
from bedrock.inference import InferenceBackend, InferenceError, create_backend
from bedrock.models import Conversation, Role
from bedrock.models.conversation import SCHEMA_VERSION, new_conversation_id
from bedrock.storage import ConversationDeleted, ConversationStore, ConversationSummary, StorageConflict, StorageError, create_store
from bedrock.recall import NO_TOPIC, NOTHING_FOUND, RecallResult, build_context, search_history, wants_recall
from bedrock.titles import generate_title, wants_title
from bedrock.ui.audit_command import run_audit_command
from bedrock.ui.commands import HELP_TEXT, Command, parse_command
from bedrock.ui.confidence_view import run_confidence_command
from bedrock.ui.digest_command import run_digest_command
from bedrock.ui.runs_command import run_runs_command
from bedrock.ui.services import ComplianceServices
from bedrock.ui.workspace_command import WorkspaceCommandSession
from bedrock.ui.widgets import (
    NEW_CONVERSATION,
    AssistantMessage,
    ConversationList,
    StatusBar,
    SystemNotice,
    UserMessage,
)
from bedrock.workspaces import WorkspaceStore

CONNECTION_CHECK_SECONDS = 10

HISTORY_DISABLED = (
    "Saving conversations is switched off on this machine "
    "(`BEDROCK_SAVE_HISTORY=0`), so there is no history to show."
)


def markdown_text(value: str) -> str:
    """Escape untrusted text before placing it in a Markdown notice."""
    value = "".join(c for c in value if unicodedata.category(c)[0] != "C")
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|~-])", r"\\\1", value)


def format_when(when: datetime) -> str:
    """Render a saved timestamp in the employee's own timezone.

    Times are stored in UTC; nobody wants to read UTC in a history list.
    """
    local = when.astimezone()
    if local.date() == datetime.now().astimezone().date():
        return f"today {local:%H:%M}"
    return f"{local:%Y-%m-%d %H:%M}"


class BedrockApp(App[None]):
    TITLE = "Bedrock Private AI"
    CSS_PATH = "ui/styles.tcss"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "New"),
        Binding("ctrl+b", "toggle_sidebar", "Chats"),
        Binding("escape", "cancel_stream", "Stop", show=False),
    ]

    def __init__(
        self,
        settings: Settings | None = None,
        backend: InferenceBackend | None = None,
        store: ConversationStore | None = None,
        services: ComplianceServices | None = None,
    ) -> None:
        super().__init__()
        self.theme = "textual-dark"
        self.settings = settings or load_settings()
        # Tests pass a fake backend and a temporary store; real runs build both
        # from settings. ``store`` stays None when saving is switched off.
        self.backend: InferenceBackend = backend or create_backend(self.settings)
        self.store: ConversationStore | None = store if store is not None else create_store(self.settings)
        self.conversation = Conversation(system_prompt=self.settings.system_prompt)
        self.model_name: str | None = self.settings.model
        self._streaming = False
        self._context_store: ContextStore | None = None
        self._workspace_session: WorkspaceCommandSession | None = None
        self._title_worker_running = False
        self._titles_pending = False
        self._title_failures: set[str] = set()
        self._earliest_rendered = 0
        # Recalled excerpts per conversation id, for this session only, so
        # clicking away from a chat and back does not silently drop them.
        self._recalled: dict[str, tuple[str, list[str]]] = {}
        # Run and action-journal browsing. Built on first use, injected by tests.
        self.services = services or ComplianceServices(self.settings)

    # ---- Screen layout ----------------------------------------------------

    def compose(self) -> ComposeResult:
        yield StatusBar()
        with Horizontal(id="body"):
            # No history means nothing to list, so no sidebar at all.
            if self.store is not None:
                yield ConversationList(id="sidebar")
            with Vertical(id="main"):
                yield VerticalScroll(id="chat")
                yield Input(placeholder="Type a message, or /help for commands", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        # Reopen the last conversation first: it may name the model to use, in
        # which case ``pick_default_model`` has nothing left to do.
        await self.restore_last_conversation()
        self.refresh_sidebar()
        self._refresh_status()
        self.set_interval(CONNECTION_CHECK_SECONDS, self.check_connection)
        self.set_interval(2, self.poll_history)
        self.check_connection()
        self.pick_default_model()

    # ---- Connection + model bookkeeping -----------------------------------

    def _refresh_status(self) -> None:
        self.query_one(StatusBar).model_name = self.model_name or "no model"

    @work(exclusive=True, group="connection")
    async def check_connection(self) -> None:
        connected = await self.backend.check_connection()
        self.query_one(StatusBar).connected = connected

    @work(group="models")
    async def pick_default_model(self) -> None:
        """If no model was configured, use the first one the server offers.

        Then start filling in missing conversation titles, which needs a model
        to ask - so it has to wait until this point.
        """
        if not self.model_name:
            try:
                models = await self.backend.list_models()
            except InferenceError:
                return  # Connection indicator already tells the story; try again on /model.
            if not models:
                return
            self.model_name = models[0].id
            self._refresh_status()
        self.write_missing_titles()

    # ---- Input handling ---------------------------------------------------

    @on(Input.Submitted, "#prompt")
    async def handle_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        command = parse_command(text)
        if command is not None:
            await self.run_command(command)
        else:
            await self.send_message(text)

    async def run_command(self, command: Command) -> None:
        match command.name:
            case "help":
                await self.post_notice(HELP_TEXT)
            case "clear" | "new":
                await self.action_clear_chat()
            case "exit" | "quit":
                await self.action_quit()
            case "model":
                await self.command_model(command.args)
            case "digest":
                self.command_digest(command.args)
            case "runs":
                self.command_runs(command.args)
            case "confidence":
                self.command_confidence(command.args)
            case "audit":
                self.command_audit(command.args)
            case "workspace" | "team":
                self.command_workspace(command.args)
            case "history":
                await self.command_history()
            case "resume":
                await self.command_resume(command.args)
            case "forget":
                await self.command_forget()
            case "recall" | "context":
                await self.command_recall(command.args)
            case _:
                await self.post_notice(f"Unknown command `/{command.name}`. Type `/help` for a list.", kind="error")

    async def command_model(self, name: str) -> None:
        if name:
            self.model_name = name
            self._refresh_status()
            await self.post_notice(f"Switched to model `{name}`.")
            return
        try:
            models = await self.backend.list_models()
        except InferenceError as exc:
            await self.post_notice(str(exc), kind="error")
            return
        if not models:
            await self.post_notice("The server reports no models. Pull one first, e.g. `ollama pull llama3.2`.", kind="error")
            return
        lines = ["**Available models** (use `/model <name>` to switch)", ""]
        for model in models:
            marker = "→" if model.id == self.model_name else " "
            lines.append(f"- {marker} `{model.id}`")
        await self.post_notice("\n".join(lines))

    # ---- Stored context ---------------------------------------------------

    @work(group="digest")
    async def command_digest(self, args: str) -> None:
        await run_digest_command(
            args,
            store=self.context_store,
            notify=self.post_notice,
            backend=self.backend,
            model=self.model_name,
            history=await self.services.optional_history(),
            actor_id=self.settings.user_id,
        )

    @property
    def context_store(self) -> ContextStore:
        if self._context_store is None:
            self._context_store = ContextStore(self.settings.context_dir)
        return self._context_store

    # ---- Local history and run browsing -----------------------------------
    #
    # These three read files this machine already has. They never contact the
    # model server, so they keep working when it is down, and they never send
    # what they find to a model: "which runs did project X do last quarter" is
    # a question with a correct answer, and a plausible one would be worse
    # than none. Each runs in a worker because all of the work is disk work.

    @work(group="history")
    async def command_runs(self, args: str) -> None:
        await run_runs_command(args, services=self.services, notify=self.post_notice)

    @work(group="history")
    async def command_confidence(self, args: str) -> None:
        await run_confidence_command(args, services=self.services, notify=self.post_notice)

    @work(group="history")
    async def command_audit(self, args: str) -> None:
        await run_audit_command(args, services=self.services, notify=self.post_notice)

    # ---- Shared workspaces -----------------------------------------------

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

    # ---- Saved conversations ----------------------------------------------

    async def command_history(self) -> None:
        summaries = await self.read_history()
        if summaries is None:
            return
        if not summaries:
            await self.post_notice("No saved conversations yet. This one is saved as soon as you send a message.")
            return
        lines = ["**Saved conversations** (open one with `/resume 2` or `/resume <words from its title>`)", ""]
        for number, summary in enumerate(summaries, start=1):
            marker = " ← open now" if summary.id == self.conversation.id else ""
            # Untitled conversations show their first question, or a list of
            # several would be impossible to tell apart.
            label = f"**{markdown_text(summary.title)}**" if summary.generated_title else f"*Untitled* “{markdown_text(summary.excerpt)}”"
            lines.append(
                f"- `{number}` {label} — {summary.message_count} messages, "
                f"{format_when(summary.updated_at)}{marker}"
            )
        await self.post_notice("\n".join(lines))

    async def command_resume(self, argument: str) -> None:
        """Reopen a saved conversation by its ``/history`` number, id, or title."""
        summaries = await self.read_history()
        if summaries is None:
            return
        if not argument:
            await self.command_history()
            return

        matches = find_summaries(summaries, argument)
        if not matches:
            await self.post_notice(
                f"No saved conversation matches `{argument}`. Use `/history` to see the list.", kind="error"
            )
            return
        if len(matches) > 1:
            # Guessing would open the wrong conversation; show the candidates
            # with their /history numbers so the employee can pick exactly.
            lines = [f"`{argument}` matches more than one conversation. Pick one by number:", ""]
            for summary in matches:
                number = summaries.index(summary) + 1
                lines.append(f"- `{number}` **{markdown_text(summary.title)}** — {format_when(summary.updated_at)}")
            await self.post_notice("\n".join(lines))
            return
        target = matches[0]
        if target.id == self.conversation.id:
            await self.post_notice("That conversation is already open.")
            return

        conversation = await self.open_conversation(target.id)
        if conversation is not None:
            await self.post_notice(f"Resumed **{markdown_text(conversation.title)}** — {len(conversation.messages)} messages.")

    async def open_conversation(self, conversation_id: str) -> Conversation | None:
        """Put a saved conversation on screen. Shared by ``/resume`` and the sidebar.

        Returns None if it could not be read; the employee has been told why.
        """
        if self.store is None:
            return None
        self.save_conversation()  # never lose the one being left behind
        try:
            conversation = await asyncio.to_thread(self.store.load, conversation_id, system_prompt=self.settings.system_prompt)
        except StorageError as exc:
            await self.post_notice(str(exc), kind="error")
            self.refresh_sidebar()
            return None

        self.workers.cancel_group(self, "chat")
        self._streaming = False
        self.conversation = conversation
        if conversation.id in self._recalled:
            conversation.recalled_context, conversation.recalled_from = self._recalled[conversation.id]
        if conversation.model:
            self.model_name = conversation.model
            self._refresh_status()
        # Opening a conversation counts as using it. Save now, which stamps it
        # as the most recent, so it is the one that reopens next launch - even
        # if the employee only reads it and then closes the terminal window,
        # which ends the app without any quit-time save running.
        self.save_conversation(mark_access=True)
        await self.render_conversation(conversation)
        self.write_missing_titles()
        return conversation

    async def command_recall(self, argument: str) -> None:
        """``/recall <words>``: search past chats and pull matches into this one.

        ``/recall`` alone says what is pulled in; ``/recall clear`` drops it.
        """
        if self.store is None:
            await self.post_notice(HISTORY_DISABLED, kind="error")
            return
        conversation = self.conversation
        if not argument:
            if conversation.recalled_context:
                await self.post_notice(
                    f"This chat includes excerpts from {self._recall_sources(conversation.recalled_from)}. "
                    "`/recall clear` removes them."
                )
            else:
                await self.post_notice(
                    "Nothing from past chats is pulled in. Use `/recall <what to look for>`, "
                    "e.g. `/recall deploy workflow`, or just ask \"do you remember when we…\"."
                )
            return
        if argument.casefold() in {"clear", "off", "none"}:
            conversation.recalled_context = None
            conversation.recalled_from = []
            self._recalled.pop(conversation.id, None)
            await self.post_notice("Removed the excerpts from past chats. Your next messages are sent without them.")
            return
        result = await self.recall_into(conversation, argument)
        if result is None:
            return
        if not result.had_terms:
            await self.post_notice("Add a few specific words to look for, e.g. `/recall deploy workflow`.", kind="error")
        elif not result.hits:
            await self.post_notice(f"Nothing in your past chats matches `{markdown_text(argument)}`.{self._partial(result)}", kind="error")
        else:
            await self.post_notice(
                f"Pulled in excerpts from {self._recall_sources(conversation.recalled_from)}. "
                f"They go to the model with your messages in this chat; `/recall clear` removes them.{self._partial(result)}"
            )

    async def recall_into(self, conversation: Conversation, request: str) -> RecallResult | None:
        """Search saved chats off the UI thread and attach the matches to ``conversation``.

        Returns None when history could not be read (the employee has been told).
        The open chat and its linked copies are left out: they are already here.
        """
        exclude = {conversation.id} | ({conversation.family_id} if conversation.family_id else set())
        try:
            result = await asyncio.to_thread(search_history, self.store, request, exclude=exclude)
        except StorageError as exc:
            await self.post_notice(f"Could not search past chats: {exc}", kind="error")
            return None
        if result.hits:
            conversation.recalled_context = build_context(result.hits)
            conversation.recalled_from = [f"{hit.title} ({format_when(hit.updated_at)})" for hit in result.hits]
            self._recalled[conversation.id] = (conversation.recalled_context, conversation.recalled_from)
        return result

    @staticmethod
    def _partial(result: RecallResult) -> str:
        if result.searched >= result.total:
            return ""
        return f" (Searched the {result.searched} most recent of {result.total} chats before the time limit.)"

    @staticmethod
    def _recall_sources(labels: list[str]) -> str:
        names = [f"**{markdown_text(label)}**" for label in labels]
        return f"{len(names)} past chat{'s' if len(names) != 1 else ''}: " + ", ".join(names)

    async def command_forget(self) -> None:
        """Delete the open conversation from disk and start fresh.

        The privacy escape hatch: an employee who pasted something they should
        not have needs a way to remove it without hunting through directories.
        """
        if self.store is None:
            await self.post_notice(HISTORY_DISABLED, kind="error")
            return
        forgotten = self.conversation.id
        self._recalled.pop(forgotten, None)
        forgotten_family = self.conversation.family_id
        # Start the new conversation *first* and without saving, otherwise the
        # conversation being deleted would be written straight back to disk.
        await self.start_new_conversation(save=False)
        try:
            deleted = await asyncio.to_thread(self.store.delete, forgotten, family_id=forgotten_family)
        except StorageError as exc:
            await self.post_notice(str(exc), kind="error")
            return
        self.refresh_sidebar()
        if deleted:
            await self.post_notice("Deleted that conversation and all its linked copies from this machine. Started a new one.")
        else:
            await self.post_notice("Nothing had been saved yet. Started a new conversation.")

    async def read_history(self) -> list[ConversationSummary] | None:
        """Summaries of saved conversations, or None if they cannot be read.

        Returning None means a message has already been shown to the employee,
        so callers just stop.
        """
        if self.store is None:
            await self.post_notice(HISTORY_DISABLED, kind="error")
            return None
        try:
            return await asyncio.to_thread(self.store.list)
        except StorageError as exc:
            await self.post_notice(str(exc), kind="error")
            return None

    async def restore_last_conversation(self) -> None:
        """On startup, reopen the conversation the employee was last using.

        "Last using" is simply the file with the newest ``updated_at``. That
        works because sending a message, receiving a reply, and explicitly
        opening a chat with ``/resume`` record its use. Leaving an unchanged
        transcript does not save it. There is no separate "current conversation"
        pointer to keep in sync.
        """
        if self.store is None:
            return
        try:
            conversation = await asyncio.to_thread(self.store.latest, system_prompt=self.settings.system_prompt)
        except StorageError as exc:
            await self.post_notice(f"Could not read saved conversations: {exc}", kind="error")
            return
        if conversation is None or not conversation.messages:
            return

        self.conversation = conversation
        # Only fall back to the saved model; an explicitly configured one wins.
        if not self.model_name and conversation.model:
            self.model_name = conversation.model
        await self.render_conversation(conversation)
        await self.post_notice(
            f"Resumed **{markdown_text(conversation.title)}** — {len(conversation.messages)} messages, "
            f"last used {format_when(conversation.updated_at)}. "
            "`/new` starts a fresh conversation, `/history` lists the rest."
        )

    async def render_conversation(self, conversation: Conversation) -> None:
        """Replace whatever is on screen with a saved conversation."""
        chat = self.query_one("#chat", VerticalScroll)
        await chat.remove_children()
        self._earliest_rendered = max(0, len(conversation.messages) - 50)
        if self._earliest_rendered:
            await chat.mount(Button(f"Load earlier messages ({self._earliest_rendered} hidden)", id="earlier-messages"))
        widgets = []
        for message in conversation.messages[-50:]:
            if message.role is Role.USER:
                widgets.append(UserMessage(message.content))
            elif message.role is Role.ASSISTANT:
                widgets.append(AssistantMessage(conversation.model or "", message.content))
        if widgets:
            await chat.mount_all(widgets)
        if conversation.source_version > SCHEMA_VERSION:
            await chat.mount(SystemNotice("This conversation is read-only because it was saved by a newer app. Use `/new` to start a writable conversation."))
        self.scroll_chat_to_end()

    @on(Button.Pressed, "#earlier-messages")
    async def load_earlier_messages(self) -> None:
        """Expand history in bounded pages only when requested."""
        chat = self.query_one("#chat", VerticalScroll)
        button = self.query_one("#earlier-messages", Button)
        button.disabled = True
        start = max(0, self._earliest_rendered - 50)
        widgets = [
            UserMessage(message.content) if message.role is Role.USER
            else AssistantMessage(self.conversation.model or "", message.content)
            for message in self.conversation.messages[start:self._earliest_rendered]
            if message.role in (Role.USER, Role.ASSISTANT)
        ]
        if widgets:
            await chat.mount_all(widgets, after=button)
        self._earliest_rendered = start
        if start:
            button.label = f"Load earlier messages ({start} hidden)"
            button.disabled = False
        else:
            await button.remove()
        if widgets:
            self.call_after_refresh(widgets[0].scroll_visible, animate=False, top=True)

    def save_conversation(
        self,
        conversation: Conversation | None = None,
        *,
        touch: bool = True,
        model: str | None = None,
        mark_access: bool = False,
    ) -> bool:
        """Save the active transcript, including during cancelled-worker cleanup.

        Revision conflicts preserve the complete local branch as a separate
        conversation. Unchanged transcripts are skipped unless explicitly
        opened. Other storage failures are reported through a toast.
        """
        conversation = conversation or self.conversation
        if self.store is None or not conversation.messages:
            return True
        if conversation.source_version > SCHEMA_VERSION:
            return False
        messages_changed = self.store.messages_changed(conversation)
        if not messages_changed and not mark_access:
            return True
        conversation.model = self.model_name if model is None else model
        try:
            self.store.save(conversation, touch=touch, wait=False)
        except StorageConflict:
            if not messages_changed:
                # A remote edit raced an explicit open. Reading must never
                # fork or overwrite the newer transcript.
                return True
            # Preserve a complete question/reply branch; never interleave turns.
            old_id, old_revision = conversation.id, conversation.storage_revision
            conversation.id = new_conversation_id()
            conversation.storage_revision = None
            try:
                self.store.save(conversation, touch=touch, wait=False)
            except StorageError as exc:
                conversation.id, conversation.storage_revision = old_id, old_revision
                self.notify(str(exc), severity="error", timeout=8)
                return False
            self.notify("Another window changed this chat. Your conversation was saved as a separate copy.", timeout=8)
        except StorageError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return False
        self.refresh_sidebar()
        return True

    # ---- Sidebar and titles -----------------------------------------------

    @work(exclusive=True, group="history-poll")
    async def poll_history(self) -> None:
        if self.store is None:
            return
        try:
            changed = await asyncio.to_thread(self.store.directory_changed)
        except OSError:
            return
        if changed:
            self.refresh_sidebar()

    @work(exclusive=True, group="history")
    async def refresh_sidebar(self) -> None:
        """Refresh metadata off the UI thread, including other windows' edits."""
        if self.store is None:
            return
        try:
            sidebar = self.query_one(ConversationList)
        except NoMatches:
            return  # the screen is still being built, or already torn down
        try:
            summaries = await asyncio.to_thread(self.store.list)
        except StorageError as exc:
            self.notify(str(exc), severity="error")
            return
        sidebar.show(summaries, self.conversation.id)

    @on(OptionList.OptionSelected, "#sidebar")
    async def handle_sidebar_pick(self, event: OptionList.OptionSelected) -> None:
        """A click (or Enter) on a row in the sidebar."""
        if event.option_id == NEW_CONVERSATION:
            await self.start_new_conversation()
        elif event.option_id and event.option_id != self.conversation.id:
            await self.open_conversation(event.option_id)
        # Straight back to typing, so picking a conversation is one click.
        self.query_one("#prompt", Input).focus()

    def action_toggle_sidebar(self) -> None:
        try:
            sidebar = self.query_one(ConversationList)
        except NoMatches:
            self.notify(HISTORY_DISABLED, timeout=6)
            return
        sidebar.display = not sidebar.display
        if not sidebar.display:
            self.query_one("#prompt", Input).focus()

    @work(group="titles")
    async def write_missing_titles(self) -> None:
        """Title only the open conversation. Never load history for backfill."""
        if self.store is None:
            return
        if self._title_worker_running:
            self._titles_pending = True
            return
        self._title_worker_running = True
        attempted: set[str] = set()
        try:
            while self.model_name:
                conversation = self._next_untitled(attempted)
                if conversation is None:
                    if self._titles_pending:
                        self._titles_pending = False
                        attempted.clear()
                        continue
                    return
                attempted.add(conversation.id)
                while self._streaming:
                    await asyncio.sleep(0.2)
                if conversation is not self.conversation:
                    continue
                checked_at = len(conversation.messages)
                try:
                    title = await generate_title(self.backend, self.model_name, conversation)
                except InferenceError as exc:
                    if exc.unavailable:
                        return
                    self._title_failures.add(conversation.id)
                    continue
                self.apply_title(conversation.id, title, checked_at)
        finally:
            self._title_worker_running = False

    def _next_untitled(self, attempted: set[str]) -> Conversation | None:
        """Only the active transcript is eligible; unopened history stays on disk."""
        if (self.conversation.id not in attempted | self._title_failures
                and self.conversation.source_version <= SCHEMA_VERSION
                and wants_title(self.conversation)):
            return self.conversation
        return None

    def apply_title(self, conversation_id: str, title: str | None, checked_at: int) -> None:
        """Save the result of looking for a topic onto the right conversation.

        ``title`` is None when the model found no concrete topic. The
        conversation stays "Untitled", and ``checked_at`` - how many messages
        the model read - is recorded, so it is only asked again once the
        conversation has grown.

        If the employee switched away, discard the result. Reopening a chat
        later can retry; background title requests never reopen saved history.
        """
        if self.store is None:
            return
        if conversation_id == self.conversation.id:
            conversation = self.conversation
        else:
            return  # Switched away: do not reopen history just to apply a title.
        if conversation.source_version > SCHEMA_VERSION:
            return
        if title:
            conversation.generated_title = title
        conversation.title_checked_at = max(conversation.title_checked_at, checked_at)
        try:
            # touch=False: getting a title is not "using" a conversation, so it
            # must not jump to the top of the list.
            self.store.save(conversation, touch=False, wait=False)
        except StorageError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.refresh_sidebar()

    # ---- Chat -------------------------------------------------------------

    async def send_message(self, text: str) -> None:
        if self.conversation.source_version > SCHEMA_VERSION:
            self.query_one("#prompt", Input).value = text
            await self.post_notice("This conversation is read-only. Your text has not been sent. Use `/new` to continue.", kind="error")
            return
        if self._streaming:
            await self.post_notice("Still responding - press Esc to stop it first.", kind="error")
            return
        if not self.model_name:
            await self.post_notice("No model selected. Use `/model` to see what is available.", kind="error")
            return

        if self.store is not None:
            try:
                self.store.ensure_available(self.conversation)
            except ConversationDeleted as exc:
                self.query_one("#prompt", Input).value = text
                await self.post_notice(str(exc) + " Your draft is still in the input; nothing was sent.", kind="error")
                self.refresh_sidebar()
                return

        question = self.conversation.add_user(text)
        # Save the question before asking: if the app dies mid-answer, the
        # employee still has what they typed.
        if not self.save_conversation():
            self.conversation.messages.remove(question)
            self.query_one("#prompt", Input).value = text
            await self.post_notice("Could not save this message. Your draft is still in the input; nothing was sent. Use `/new` if this chat was deleted.", kind="error")
            return
        await self.mount_message(UserMessage(text))

        conversation, model_name, note = self.conversation, self.model_name, None
        if self.store is not None and wants_recall(text):
            # Plain-English recall ("do you remember when we..."). Block a second
            # send while searching; the reply goes to this chat even if the
            # employee switches away meanwhile.
            self._streaming = True
            try:
                result = await self.recall_into(conversation, text)
            finally:
                self._streaming = False
            if result is None:
                pass
            elif not result.had_terms:
                note = NO_TOPIC
            elif result.hits:
                await self.post_notice(f"Looked back through your chats and pulled in {self._recall_sources(conversation.recalled_from)}.{self._partial(result)}")
            else:
                note = NOTHING_FOUND
                await self.post_notice(f"Looked back through your past chats and found nothing that matches.{self._partial(result)}")
            if conversation is not self.conversation:
                # Switched chats while searching. The question is saved, but its
                # widgets are gone, so say what happened instead of answering
                # into another chat's screen.
                self.notify("You switched chats while your past chats were being searched, so that question was not answered. Reopen the chat and send it again.", timeout=8)
                return

        reply_widget = AssistantMessage(model_name)
        await self.mount_message(reply_widget)
        self.stream_reply(reply_widget, conversation, model_name, note)

    @work(exclusive=True, group="chat")
    async def stream_reply(self, widget: AssistantMessage, conversation: Conversation, model_name: str, note: str | None = None) -> None:
        """Background worker: pull fragments from the backend into the widget.

        Takes the conversation as an argument rather than reading
        ``self.conversation``, so a reply that is still arriving when the
        employee starts a new conversation is saved to the right file.
        """
        prompt_box = self.query_one("#prompt", Input)
        self._streaming = True
        prompt_box.placeholder = "Responding… (Esc to stop)"
        # Serialise the history *before* adding the empty reply placeholder,
        # otherwise the model would see a blank assistant turn at the end.
        api_messages = conversation.to_api(note)
        reply = conversation.add_assistant()
        stream = widget.begin_stream()
        try:
            async for fragment in self.backend.stream_chat(model_name, api_messages):
                reply.content += fragment
                await stream.write(fragment)
                self.scroll_chat_to_end()
        except InferenceError as exc:
            if not reply.content and reply in conversation.messages:
                conversation.messages.remove(reply)
            await self.post_notice(str(exc), kind="error")
        except asyncio.CancelledError:
            reply.content += "\n\n*(stopped)*"
            await stream.write("\n\n*(stopped)*")
            raise
        finally:
            # Runs on success, on error and on Esc, so a half-finished answer is
            # kept rather than thrown away.
            is_open = conversation is self.conversation
            self.save_conversation(conversation, touch=is_open, model=model_name)
            await widget.end_stream()
            self._streaming = False
            prompt_box.placeholder = "Type a message, or /help for commands"
            if is_open:
                self.scroll_chat_to_end()
            self.write_missing_titles()

    async def mount_message(self, widget) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(widget)
        self.scroll_chat_to_end()

    async def post_notice(self, text: str, kind: str = "info") -> None:
        await self.mount_message(SystemNotice(text, kind=kind))

    def scroll_chat_to_end(self) -> None:
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    # ---- Actions (key bindings) -------------------------------------------

    async def start_new_conversation(self, save: bool = True) -> None:
        """Put the current conversation away and begin an empty one."""
        self.workers.cancel_group(self, "chat")
        if save:
            self.save_conversation()
        self.conversation = Conversation(system_prompt=self.settings.system_prompt)
        await self.query_one("#chat", VerticalScroll).remove_children()
        self._streaming = False
        self.refresh_sidebar()

    async def action_clear_chat(self) -> None:
        """``/new``, ``/clear`` and Ctrl+L: the old conversation stays saved."""
        await self.start_new_conversation()

    def action_cancel_stream(self) -> None:
        self.workers.cancel_group(self, "chat")

    async def action_quit(self) -> None:
        self.save_conversation()
        if self._workspace_session is not None:
            self._workspace_session.close()
        await self.backend.close()
        self.exit()


def find_summaries(summaries: list[ConversationSummary], argument: str) -> list[ConversationSummary]:
    """Every saved conversation matching what was typed after ``/resume``.

    Tried in this order, stopping at the first kind that matches anything:

    1. A number is a position in the ``/history`` list: ``/resume 2``.
    2. A conversation id, whole or just the start of one.
    3. Words from the title *or* from the first message, ignoring case:
       ``/resume platypus``. Both are checked because the model renames a
       conversation after its first reply, and people remember what they
       typed, not what the model called it. An exact match beats labels that
       merely contain the words, so ``/resume bees`` opens "bees" rather than
       asking about "bees and wasps".

    Returns a list rather than one result so the caller can tell "no match"
    apart from "several matches" and ask the employee instead of guessing.
    """
    argument = argument.strip()
    if argument.isdigit() and len(argument) < 8:
        index = int(argument) - 1
        return [summaries[index]] if 0 <= index < len(summaries) else []

    by_id = [s for s in summaries if s.id == argument] or [s for s in summaries if s.id.startswith(argument)]
    if by_id:
        return by_id

    wanted = " ".join(argument.lower().split())

    def labels(summary: ConversationSummary) -> list[str]:
        return [" ".join(text.lower().split()) for text in (summary.title, summary.excerpt) if text]

    exact = [s for s in summaries if wanted in labels(s)]
    if exact:
        return exact
    return [s for s in summaries if any(wanted in label for label in labels(s))]


def main() -> None:
    """Run the configured plain-model or agent-pack client."""
    try:
        settings = load_settings()
    except (ValueError, TypeError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    from bedrock.doctor import print_first_run_banner, run_first_run_check

    print_first_run_banner(run_first_run_check(settings))

    if settings.backend == "agentpack":
        from bedrock.agentpack_app import AgentPackApp

        AgentPackApp(settings=settings).run()
    else:
        BedrockApp(settings=settings).run()


if __name__ == "__main__":
    main()
