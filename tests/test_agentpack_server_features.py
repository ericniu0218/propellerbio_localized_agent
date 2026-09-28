"""Agent-pack mode against a server that supports more than the original API.

The server lists what it supports on /health (``features``). Against a newer
server the app sends attached files as material apart from the question, can
choose who answers (/tier), can stop a request (/cancel), and shows the full
answer awaiting approval. Against an older one it sends exactly what it always
sent - the tests in test_agentpack_app.py, whose fake client has no features,
keep proving that.
"""

from __future__ import annotations

from bedrock.agentpack_app import AgentPackApp
from bedrock.config import Settings
from bedrock.ui.approval import ApprovalPrompt
from bedrock.ui.widgets import StatusBar, SystemNotice
from textual.widgets import Input

from tests.conftest import make_services
from tests.test_agentpack_app import FakeAgentPackClient

ALL = {"context_blocks", "server_history", "force_tier", "cancel", "approval_result_ref"}


class NewerServerClient(FakeAgentPackClient):
    def __init__(self, features: set[str] = ALL) -> None:
        super().__init__()
        self.feature_set = set(features)
        self.sent_options: list[dict] = []
        self.cancelled: list[str] = []

    async def features(self) -> set[str]:
        return set(self.feature_set)

    async def send_message(self, session_id: str, text: str, **options) -> str:
        self.sent_messages.append((session_id, text))
        self.sent_options.append(options)
        return "task_fake"

    async def cancel_task(self, task_id: str) -> str:
        self.cancelled.append(task_id)
        return "cancelled"


def make_app(client, tmp_path) -> AgentPackApp:
    settings = Settings(user_id="alice", context_dir=str(tmp_path / "ctx"), auto_context=False)
    return AgentPackApp(settings=settings, client=client, services=make_services())


def notices(app) -> list[str]:
    return [str(n.source) for n in app.query(SystemNotice)]


async def send(app, pilot, text: str) -> None:
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await pilot.pause()


async def test_a_file_goes_as_material_to_a_server_that_takes_it(tmp_path):
    plate = tmp_path / "plate.csv"
    plate.write_text("well,od\nA1,0.51\nA2,0.49\n")
    client = NewerServerClient()
    app = make_app(client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await send(app, pilot, f"summarise @{plate}")

    assert client.sent_messages == [("sess_fake", f"summarise @{plate}")]
    [block] = client.sent_options[0]["context"]
    assert block["title"] == "plate.csv" and "A1,0.51" in block["text"] and block["kind"] == "attachment"


async def test_an_older_server_gets_the_file_inside_the_message_as_before(tmp_path):
    plate = tmp_path / "plate.csv"
    plate.write_text("well,od\nA1,0.51\n")
    client = NewerServerClient(features=set())
    app = make_app(client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await send(app, pilot, f"summarise @{plate}")

    assert "A1,0.51" in client.sent_messages[0][1]
    assert client.sent_options == [{}]


async def test_tier_chooses_who_answers_until_set_back_to_auto(tmp_path):
    client = NewerServerClient()
    app = make_app(client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await send(app, pilot, "/tier deep")
        assert app.query_one(StatusBar).model_name == "AI system · Deep Analysis"
        await send(app, pilot, "first question")
        client.push("task.failed", {"task_id": "task_fake", "title": "t", "error": "x"})
        await pilot.pause()
        await send(app, pilot, "/tier auto")
        await send(app, pilot, "/tier sideways")
        assert app.query_one(StatusBar).model_name == "AI system"
        assert any("isn't a tier" in n for n in notices(app))

    assert client.sent_options[0] == {"force_tier": "director"}


async def test_cancel_stops_the_running_request_once(tmp_path):
    client = NewerServerClient()
    app = make_app(client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await send(app, pilot, "/cancel")
        assert any("Nothing is running" in n for n in notices(app))
        await send(app, pilot, "a long question")
        await send(app, pilot, "/cancel")

    assert client.cancelled == ["task_fake"]


async def test_cancel_is_not_offered_by_a_server_that_cannot(tmp_path):
    client = NewerServerClient(features=set())
    app = make_app(client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await send(app, pilot, "a long question")
        await send(app, pilot, "/cancel")
        assert any("can't stop a request" in n for n in notices(app))

    assert client.cancelled == []


async def test_the_full_answer_is_shown_before_asking_for_approval(tmp_path):
    client = NewerServerClient()
    full = "Bottom line: the safety signal is not established. " * 20
    client.results["res_full"] = full
    app = make_app(client, tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        client.push("approval.required", {
            "task_id": "task_gate", "title": "Final synthesis", "reason": "High-stakes output",
            "domain": "regulatory", "preview": full[:500], "result_ref": "res_full",
        }, event_id="e1")
        await pilot.pause()
        await pilot.pause()

        assert any(full.strip() in n for n in notices(app))
        assert len(app.query(ApprovalPrompt)) == 1
