"""Plain data structures describing a chat conversation.

These are deliberately boring: no UI, no networking. The UI layer renders them
and the inference layer serialises them. Keeping the data model separate means
either side can change without breaking the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def to_api(self) -> dict[str, str]:
        """The shape every OpenAI-compatible server expects."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None

    def add_user(self, content: str) -> Message:
        message = Message(Role.USER, content)
        self.messages.append(message)
        return message

    def add_assistant(self, content: str = "") -> Message:
        message = Message(Role.ASSISTANT, content)
        self.messages.append(message)
        return message

    def clear(self) -> None:
        self.messages.clear()

    def to_api(self) -> list[dict[str, str]]:
        """Full message list to send to the model, including the system prompt."""
        api_messages: list[dict[str, str]] = []
        if self.system_prompt:
            api_messages.append(Message(Role.SYSTEM, self.system_prompt).to_api())
        api_messages.extend(m.to_api() for m in self.messages)
        return api_messages
