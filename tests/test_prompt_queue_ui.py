"""Standalone queue UI tests; no execution module or model server needed."""
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, OptionList, Static

from bedrock.ui.prompt_queue import PromptQueuePanel


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    model: str | None = "test-model"


@dataclass(frozen=True)
class Snapshot:
    pending: tuple[Item, ...] = ()
    running: Item | None = None
    paused: bool = False
    conversation_id: str = "chat-a"
    can_skip: bool = False


class QueueHarness(App):
    CSS_PATH = Path(__file__).resolve().parents[1] / "src/bedrock/ui/styles.tcss"
    CSS = "#test-chat { height: 1fr; }"

    def __init__(self, snapshot=Snapshot()):
        super().__init__()
        self.panel = PromptQueuePanel()
        self.panel.show_queue(snapshot)
        self.requests = []

    def compose(self) -> ComposeResult:
        yield Static("Conversation", id="test-chat")
        yield self.panel
        yield Input(id="prompt")

    @on(PromptQueuePanel.RemoveRequested)
    def removed(self, event):
        self.requests.append(("remove", event.item_id))

    @on(PromptQueuePanel.PauseRequested)
    def paused(self):
        self.requests.append(("pause",))

    @on(PromptQueuePanel.ResumeRequested)
    def resumed(self):
        self.requests.append(("resume",))

    @on(PromptQueuePanel.ClearRequested)
    def cleared(self):
        self.requests.append(("clear",))

    @on(PromptQueuePanel.SkipRequested)
    def skipped(self):
        self.requests.append(("skip",))


async def test_empty_queue_and_lifetime_notice():
    app = QueueHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "0 waiting · Idle" in str(app.query_one(".queue-heading", Static).content)
        assert app.query_one(".queue-empty").display
        assert not app.query_one("#queue-pending").display
        assert not app.query_one(".queue-running").display
        assert all(button.disabled for button in app.panel.query(Button))
        notice = str(app.query_one(".queue-hint", Static).content)
        assert "waiting prompts only" in notice and "only while the app is open" in notice


async def test_running_and_pending_are_separate_and_literal():
    snapshot = Snapshot(
        pending=(Item("one", "[bold]literal[/bold] <https://example.test>", "[red]model"),),
        running=Item("active", "Working on first prompt", "running-model"),
    )
    app = QueueHarness(snapshot)
    async with app.run_test() as pilot:
        await pilot.pause()
        options = app.query_one(OptionList)
        assert options.option_count == 1
        assert options.get_option_at_index(0).id == "one"
        assert "[bold]literal[/bold]" in options.get_option_at_index(0).prompt
        assert "[red]model" in options.get_option_at_index(0).prompt
        assert "[bold]literal[/bold]" in options.render_line(0).text
        assert "running-model" in str(app.query_one(".queue-running", Static).content)
        assert "1 waiting · Running" in str(app.query_one(".queue-heading", Static).content)


async def test_keyboard_controls_emit_requests_without_mutating_queue():
    snapshot = Snapshot(pending=(Item("one", "First"), Item("two", "Second")), running=Item("active", "Running"))
    app = QueueHarness(snapshot)
    async with app.run_test() as pilot:
        options = app.query_one(OptionList)
        options.focus()
        await pilot.press("down", "delete")
        assert app.requests == [("remove", "two")]
        app.query_one("#queue-remove", Button).focus()
        await pilot.press("enter")
        app.query_one("#queue-toggle", Button).focus()
        await pilot.press("enter")
        app.query_one("#queue-clear", Button).focus()
        await pilot.press("enter")
        assert app.requests == [("remove", "two"), ("remove", "two"), ("pause",), ("clear",)]
        assert options.option_count == 2
        assert str(app.query_one("#queue-toggle", Button).label) == "Pause"
        assert app.query_one(".queue-running").display
        app.panel.show_queue(replace(snapshot, paused=True))
        app.query_one("#queue-toggle", Button).focus()
        await pilot.pause(0.25)  # Allow the button's normal activation debounce.
        await pilot.press("enter")
        assert app.requests[-1] == ("resume",)
        assert "Paused" in str(app.query_one(".queue-heading", Static).content)


async def test_running_only_allows_pause_but_not_remove_or_clear():
    app = QueueHarness(Snapshot(running=Item("active", "Still running")))
    async with app.run_test() as pilot:
        assert app.query_one("#queue-remove", Button).disabled
        assert app.query_one("#queue-clear", Button).disabled
        assert not app.query_one("#queue-toggle", Button).disabled
        await pilot.click("#queue-toggle")
        assert app.requests == [("pause",)]


async def test_queue_controls_are_reachable_by_tab():
    app = QueueHarness(Snapshot(pending=(Item("one", "Waiting"),)))
    async with app.run_test() as pilot:
        app.query_one(Input).focus()
        visited = set()
        for _ in range(6):
            await pilot.press("tab")
            if app.focused:
                visited.add(app.focused.id)
        assert {"queue-pending", "queue-remove", "queue-toggle", "queue-clear"} <= visited


async def test_refresh_preserves_prompt_draft_and_list_selection_and_scroll():
    snapshot = Snapshot(pending=tuple(Item(str(n), f"Prompt {n}") for n in range(30)))
    app = QueueHarness(snapshot)
    async with app.run_test(size=(70, 28)) as pilot:
        options = app.query_one(OptionList)
        options.focus()
        await pilot.press("end")
        await pilot.pause()
        offset = options.scroll_offset.y
        assert offset > 0
        app.panel.show_queue(replace(snapshot, paused=True))
        await pilot.pause()
        assert app.focused is options
        assert options.highlighted == 29
        assert options.scroll_offset.y == offset
        prompt = app.query_one(Input)
        prompt.focus()
        await pilot.press(*"my unfinished draft")
        app.panel.show_queue(replace(snapshot, pending=snapshot.pending + (Item("new", "New prompt"),)))
        await pilot.pause()
        assert app.focused is prompt
        assert prompt.value == "my unfinished draft"
        assert options.get_option_at_index(options.highlighted).id == "29"
        assert options.scroll_offset.y == offset


async def test_switching_snapshots_removes_old_rows_and_resets_selection():
    snapshot = Snapshot(pending=tuple(Item(str(n), f"Prompt {n}") for n in range(20)))
    app = QueueHarness(snapshot)
    async with app.run_test() as pilot:
        options = app.query_one(OptionList)
        options.focus()
        await pilot.press("end")
        app.panel.show_queue(Snapshot(pending=(Item("other", "Other chat"),), paused=True, conversation_id="chat-b"))
        await pilot.pause()
        assert app.focused is options
        assert options.option_count == 1 and options.highlighted == 0
        assert options.scroll_offset.y == 0
        await pilot.press("delete")
        assert app.requests == [("remove", "other")]
        app.panel.show_queue(Snapshot(conversation_id="chat-c"))
        await pilot.pause()
        await pilot.press("delete")
        assert app.requests == [("remove", "other")]


async def test_minimal_shared_snapshot_without_conversation_id():
    snapshot = SimpleNamespace(
        pending=tuple(Item(str(n), f"Prompt {n}") for n in range(20)), running=None, paused=False,
    )
    app = QueueHarness(snapshot)
    async with app.run_test() as pilot:
        options = app.query_one(OptionList)
        options.focus()
        await pilot.press("end")
        app.panel.show_queue(SimpleNamespace(
            pending=(Item("other-1", "Other first"), Item("other-2", "Other second")),
            running=None, paused=True,
        ))
        await pilot.pause()
        assert options.highlighted == 0 and options.scroll_offset.y == 0
        await pilot.press("delete")
        assert app.requests == [("remove", "other-1")]


@pytest.mark.parametrize("width", [32, 48, 100])
async def test_long_prompts_fit_narrow_terminal_and_controls_remain_accessible(width):
    item = Item("one", "[bold] " + "long prompt\n" * 1000, "model" * 100)
    app = QueueHarness(Snapshot(pending=(item,), running=item))
    async with app.run_test(size=(width, 32)) as pilot:
        await pilot.pause()
        assert app.panel.region.right <= width
        assert app.panel.size.height < 20
        assert len(app.query_one(OptionList).get_option_at_index(0).prompt) < 220
        clear = app.query_one("#queue-clear", Button)
        clear.focus()
        await pilot.pause()
        assert clear.region.right <= app.panel.region.right
        assert clear.region.x >= app.panel.region.x
        await pilot.press("enter")
        assert app.requests == [("clear",)]
        assert app.query_one(Input).region.bottom <= 32


async def test_duplicate_snapshots_skip_render_and_status_changes_keep_rows(monkeypatch):
    snapshot = Snapshot(pending=(Item("waiting", "Waiting prompt"),))
    app = QueueHarness(snapshot)
    async with app.run_test() as pilot:
        await pilot.pause()
        renders = []
        render = app.panel._render_queue

        def counted_render():
            renders.append(True)
            render()

        def unexpected_rebuild(*args, **kwargs):
            pytest.fail("A status-only update must not rebuild waiting rows")

        monkeypatch.setattr(app.panel, "_render_queue", counted_render)
        monkeypatch.setattr(app.query_one(OptionList), "set_options", unexpected_rebuild)
        for _ in range(3):
            app.panel.show_queue(snapshot)
        assert not renders
        app.panel.show_queue(replace(snapshot, running=Item("running", "Working")))
        app.panel.show_queue(replace(snapshot, paused=True))
        assert len(renders) == 2


async def test_pending_changes_do_not_redraw_unchanged_running_prompt(monkeypatch):
    snapshot = Snapshot(pending=(Item("waiting", "Waiting"),), running=Item("active", "Working"))
    app = QueueHarness(snapshot)
    async with app.run_test() as pilot:
        await pilot.pause()

        def unexpected_update(*args, **kwargs):
            pytest.fail("Unchanged text must not be redrawn")

        monkeypatch.setattr(app.query_one(".queue-running", Static), "update", unexpected_update)
        app.panel.show_queue(replace(snapshot, pending=snapshot.pending + (Item("next", "Next"),)))
        monkeypatch.setattr(app.query_one(".queue-heading", Static), "update", unexpected_update)
        app.panel.show_queue(replace(snapshot, pending=(Item("waiting", "Changed preview"), Item("next", "Next"))))
        await pilot.pause()
        assert "Changed preview" in app.query_one(OptionList).get_option_at_index(0).prompt


@pytest.mark.parametrize("can_skip", [False, True])
@pytest.mark.parametrize("has_running", [False, True])
@pytest.mark.parametrize("has_pending", [False, True])
async def test_skip_requires_permission_running_and_pending(can_skip, has_running, has_pending):
    snapshot = Snapshot(
        pending=(Item("next", "Next"),) if has_pending else (),
        running=Item("active", "Working") if has_running else None,
        can_skip=can_skip,
    )
    app = QueueHarness(snapshot)
    async with app.run_test() as pilot:
        button = app.query_one("#queue-skip", Button)
        enabled = can_skip and has_running and has_pending
        assert button.disabled is not enabled
        # Also reject an activation queued before eligibility changed.
        app.panel.handle_queue_button(Button.Pressed(button))
        await pilot.pause()
        assert app.requests == ([("skip",)] if enabled else [])


async def test_old_snapshot_cannot_skip():
    app = QueueHarness(SimpleNamespace(
        pending=(Item("next", "Next"),), running=Item("active", "Working"), paused=False,
    ))
    async with app.run_test():
        assert app.query_one("#queue-skip", Button).disabled


@pytest.mark.parametrize("activation", ["click", "enter"])
@pytest.mark.parametrize("paused", [False, True])
async def test_skip_emits_only_request_without_mutating_snapshot(activation, paused):
    snapshot = Snapshot(
        pending=(Item("next", "Next"),), running=Item("active", "Working"),
        can_skip=True, paused=paused,
    )
    app = QueueHarness(snapshot)
    async with app.run_test() as pilot:
        button = app.query_one("#queue-skip", Button)
        assert "resuming a paused queue" in str(button.tooltip)
        if activation == "click":
            await pilot.click("#queue-skip")
        else:
            app.query_one("#queue-toggle", Button).focus()
            await pilot.press("tab")
            assert app.focused is button
            await pilot.press("enter")
        assert app.requests == [("skip",)]
        assert app.query_one(OptionList).get_option_at_index(0).id == "next"
        assert "Working" in str(app.query_one(".queue-running", Static).content)
        assert str(app.query_one("#queue-toggle", Button).label) == ("Resume" if paused else "Pause")
        assert not button.disabled  # Execution state changes only through a snapshot.


async def test_skip_eligibility_only_update_preserves_draft_and_scroll(monkeypatch):
    snapshot = Snapshot(
        pending=tuple(Item(str(n), f"Prompt {n}") for n in range(30)),
        running=Item("active", "Working"),
    )
    app = QueueHarness(snapshot)
    async with app.run_test(size=(70, 28)) as pilot:
        options = app.query_one(OptionList)
        options.focus()
        await pilot.press("end")
        await pilot.pause()
        offset = options.scroll_offset.y
        prompt = app.query_one(Input)
        prompt.focus()
        prompt.value = "unfinished draft"

        def unexpected_rebuild(*args, **kwargs):
            pytest.fail("Eligibility alone must not rebuild the waiting list")

        monkeypatch.setattr(options, "set_options", unexpected_rebuild)
        app.panel.show_queue(replace(snapshot, can_skip=True))
        await pilot.pause()
        button = app.query_one("#queue-skip", Button)
        assert not button.disabled
        assert app.focused is prompt and prompt.value == "unfinished draft"
        assert options.highlighted == 29 and options.scroll_offset.y == offset
        button.focus()
        await pilot.pause()
        app.panel.show_queue(snapshot)
        await pilot.pause()
        assert button.disabled
        assert app.focused is not button
        app.panel.handle_queue_button(Button.Pressed(button))
        await pilot.pause()
        assert not app.requests


@pytest.mark.parametrize("width", [32, 48, 100])
async def test_skip_button_is_reachable_in_narrow_terminals(width):
    app = QueueHarness(Snapshot(
        pending=(Item("next", "Next"),), running=Item("active", "Working"), can_skip=True,
    ))
    async with app.run_test(size=(width, 32)) as pilot:
        app.query_one("#queue-toggle", Button).focus()
        await pilot.press("tab")
        await pilot.pause()
        button = app.query_one("#queue-skip", Button)
        assert app.focused is button
        assert button.region.x >= app.panel.region.x
        assert button.region.right <= app.panel.region.right <= width
        await pilot.press("enter")
        assert app.requests == [("skip",)]
        assert app.query_one(Input).region.bottom <= 32
