"""Shared test helpers.

``FakeBackend`` implements the InferenceBackend protocol in memory, so the UI
can be tested without any server. This is exactly the swap-ability the
architecture is designed for: the app cannot tell the difference.

``isolated_history`` is autouse and matters just as much: without it a test that
builds a real ``Settings`` would save conversations into the developer's own
home directory, and a later test would cheerfully resume them.
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

import pytest

from bedrock.inference import InferenceError, ModelInfo
from bedrock.storage import ConversationStore
from bedrock.titles import is_title_request, is_topic_check


class FakeBackend:
    """In-memory stand-in for a model server.

    Title requests and YES/NO topic checks (see ``bedrock/titles.py``) are
    recorded in ``title_requests`` and ``topic_checks`` rather than ``requests``,
    so assertions about chat turns stay exact. By default every conversation has
    a topic (``topic_reply="YES"``) and, like a real model, gets its own title -
    "Re <first question>" - unless a fixed ``title_reply`` is supplied.
    """

    def __init__(
        self,
        models: list[str] | None = None,
        reply: str = "Hello from the fake model.",
        connected: bool = True,
        title_reply: str | None = None,
        topic_reply: str = "YES",
    ) -> None:
        self.models = models if models is not None else ["test-model", "other-model"]
        self.reply = reply
        self.connected = connected
        self.title_reply = title_reply
        self.topic_reply = topic_reply
        self.topic_checks: list[tuple[str, list[dict[str, str]]]] = []
        self.fail_titles = False
        self.requests: list[tuple[str, list[dict[str, str]]]] = []
        self.title_requests: list[tuple[str, list[dict[str, str]]]] = []
        self.closed = False
        self.fragment_delay = 0.0

    async def check_connection(self) -> bool:
        return self.connected

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=m) for m in self.models]

    async def stream_chat(self, model: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if is_topic_check(messages):
            self.topic_checks.append((model, list(messages)))
            if self.fail_titles:
                raise InferenceError("Request failed: connection refused.")
            yield self.topic_reply
            return
        if is_title_request(messages):
            self.title_requests.append((model, list(messages)))
            if self.fail_titles:
                raise InferenceError("Request failed: connection refused.")
            yield self.title_reply if self.title_reply is not None else _title_for(messages)
            return
        self.requests.append((model, list(messages)))
        for word in self.reply.split(" "):
            if self.fragment_delay:
                await asyncio.sleep(self.fragment_delay)
            yield word + " "

    async def close(self) -> None:
        self.closed = True


def _title_for(messages: list[dict[str, str]]) -> str:
    """The fake model's title: "Re " plus the first question in the request."""
    first_question = re.search(r"^User: (.+)$", messages[1]["content"], re.MULTILINE)
    return f"Re {first_question.group(1)}" if first_question else "Untitled"


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    """Point the default history directory at a throwaway path, for every test."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "bedrock" / "conversations"


@pytest.fixture
def store(tmp_path) -> ConversationStore:
    """A store in a temporary directory, shared by tests that need one."""
    return ConversationStore(tmp_path / "conversations")
