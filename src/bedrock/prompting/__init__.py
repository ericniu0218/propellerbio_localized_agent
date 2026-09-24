"""Build bounded chat prompts while keeping the sent prefix stable.

The system text is fixed. Turn-specific material belongs inside that turn's
user message. The in-memory sent-message cache preserves the exact previous
prompt when another turn arrives; it is never part of saved conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING, Callable

from bedrock import attachments

if TYPE_CHECKING:
    from bedrock.attachments import Attachment
    from bedrock.models.conversation import Conversation


def _estimate_tokens(text: str) -> int:
    """Phase 0 replaces this four-character placeholder with the shared estimator."""
    return ceil(len(text) / 4)


@dataclass(frozen=True)
class PromptBudget:
    context_window: int
    reply_reserve: int
    trim_at: float = 0.9
    trim_to: float = 0.6
    attachment_share: float = 0.35

    @property
    def usable(self) -> int:
        return self.context_window - self.reply_reserve


@dataclass(frozen=True)
class ExtraSpec:
    priority: int
    share: float
    render: Callable[[object, int], tuple[str, list[str]]]


def _text(value: object, max_tokens: int) -> tuple[str, list[str]]:
    content = str(value)
    if _estimate_tokens(content) <= max_tokens:
        return content, []
    return content[:max(0, max_tokens * 4)].rstrip() + "…", ["Some added context was shortened to fit the model window."]


def _recall(value: object, max_tokens: int) -> tuple[str, list[str]]:
    content, notices = _text(value, max(0, max_tokens - 16))
    return f"[Excerpts from my earlier chats]\n{content}\n[End of excerpts]", notices


def _attachments(value: object, max_tokens: int) -> tuple[str, list[str]]:
    files: list[Attachment] = list(value)  # type: ignore[arg-type]
    chosen: list[Attachment] = []
    notices: list[str] = []
    omitted = 0
    for file in files:
        if _estimate_tokens(attachments.prompt_block(chosen + [file])) <= max_tokens:
            chosen.append(file)
            continue
        digest = file.as_digest()
        if _estimate_tokens(attachments.prompt_block(chosen + [digest])) <= max_tokens:
            chosen.append(digest)
            continue
        # Shorten only inside the fence; never cut its closing delimiter.
        low, high = 0, len(digest.text)
        fit = None
        while low <= high:
            middle = (low + high) // 2
            candidate = attachments.Attachment(digest.name, digest.path, digest.byte_size,
                                               "digest", digest.text[:middle] + "…")
            if _estimate_tokens(attachments.prompt_block(chosen + [candidate])) <= max_tokens:
                fit = candidate
                low = middle + 1
            else:
                high = middle - 1
        if fit is None:
            omitted += 1
        else:
            chosen.append(fit)
            notices.append(f"The digest for `{file.name}` was shortened to fit the model window.")
    if omitted:
        notices.append(f"{omitted} attachment digests were left out to fit the model window.")
    if not chosen:
        return "", notices
    if notice := attachments.summary(chosen):
        notices.append(notice)
    return attachments.prompt_block(chosen), notices


# A new kind of context needs one registry entry: priority, window share and
# renderer. No branch in the builder is needed.
EXTRA_SPECS: dict[str, ExtraSpec] = {
    "note": ExtraSpec(0, 0.08, _text),
    "recall": ExtraSpec(1, 0.20, _recall),
    "passages": ExtraSpec(2, 0.20, _text),
    "attachments": ExtraSpec(3, 0.35, _attachments),
}


@dataclass(frozen=True)
class PromptResult:
    messages: list[dict[str, str]]
    estimated_tokens: int
    left_out: int
    notices: list[str] = field(default_factory=list)
    trimmed: bool = False


def _size(messages: list[dict[str, str]]) -> int:
    return sum(_estimate_tokens(item["content"]) + 4 for item in messages)


class PromptBuilder:
    """A small stateful policy over an otherwise plain Conversation.

    The cache and trim point live only in memory. Saving a Conversation never
    writes them, so recalled material cannot survive `/forget` through a copy.
    """

    def __init__(self, budget: PromptBudget) -> None:
        if budget.usable <= 16 or not 0 < budget.trim_to < budget.trim_at < 1:
            raise ValueError("A prompt needs room for messages and a reply reserve")
        self.budget = budget

    def build(self, conversation: Conversation, extras: dict[str, object] | None = None) -> PromptResult:
        from bedrock.models.conversation import RECALL_INTRO, Role

        extras = extras or {}
        system = "\n\n".join(part for part in (conversation.system_prompt, RECALL_INTRO) if part)
        prefix = [{"role": "system", "content": system}]
        if _size(prefix) >= self.budget.usable:
            raise ValueError("The system prompt exceeds the model window")
        if not conversation.messages:
            return PromptResult(prefix, _size(prefix), 0)

        cache: dict[int, str] = getattr(conversation, "_prompt_sent", {})
        start: int = getattr(conversation, "_prompt_start", 0)
        previous_start = start
        latest = len(conversation.messages) - 1
        history: list[dict[str, str]] = []
        compacted = False
        for index in range(start, latest):
            message = conversation.messages[index]
            content = cache.get(index, message.content)
            if message.attachments:
                old = attachments.prompt_block(message.attachments)
                new = attachments.prompt_block([file.as_digest() for file in message.attachments])
                if old in content:
                    content = content.replace(old, new)
                    compacted = True
                elif index not in cache:
                    content = attachments.with_attachments(content, [file.as_digest() for file in message.attachments])
            history.append({"role": message.role.value, "content": content})

        newest = conversation.messages[latest]
        typed = newest.content
        notices: list[str] = []
        rendered: list[str] = []
        for name, spec in sorted(EXTRA_SPECS.items(), key=lambda pair: pair[1].priority):
            value = extras.get(name)
            if name == "attachments" and value is None:
                value = newest.attachments
            if not value:
                continue
            cap = int(self.budget.context_window * (self.budget.attachment_share if name == "attachments" else spec.share))
            part, warnings = spec.render(value, cap)
            rendered.append(part)
            notices.extend(warnings)
        content = typed + ("\n\n" + "\n\n".join(rendered) if rendered else "")
        current = {"role": Role.USER.value, "content": content} if newest.role is Role.USER else {"role": newest.role.value, "content": content}
        target = int(self.budget.usable * self.budget.trim_to)
        threshold = int(self.budget.usable * self.budget.trim_at)
        if _size(prefix + history + [current]) > threshold:
            while history and _size(prefix + history + [current]) > target:
                history.pop(0)
                start += 1
        # If the newest turn alone is large, use the same priority registry to
        # shrink low-priority extras before touching the employee's typed text.
        if _size(prefix + history + [current]) > self.budget.usable:
            remaining = self.budget.usable - _size(prefix + history + [{"role": current["role"], "content": typed}])
            rendered = []
            for name, spec in sorted(EXTRA_SPECS.items(), key=lambda pair: pair[1].priority):
                value = extras.get(name)
                if name == "attachments" and value is None:
                    value = newest.attachments
                if not value:
                    continue
                cap = min(int(self.budget.context_window * (self.budget.attachment_share if name == "attachments" else spec.share)), max(0, remaining - 2))
                if cap <= 0:
                    notices.append(f"{name.capitalize()} context was left out to fit the model window.")
                    continue
                part, warnings = spec.render(value, cap)
                rendered.append(part)
                notices.extend(warnings)
                remaining -= _estimate_tokens(part) + 2
            current["content"] = typed + ("\n\n" + "\n\n".join(rendered) if rendered else "")
        if _size(prefix + history + [current]) > self.budget.usable:
            room = self.budget.usable - _size(prefix + history) - 5
            current["content"] = typed[:max(0, room * 4)].rstrip() + "…"
            notices.append("The message was shortened to fit the model window.")
        messages = prefix + history + [current]
        while _size(messages) > self.budget.usable and len(current["content"]) > 1:
            current["content"] = current["content"][:-4]
        if _size(messages) > self.budget.usable:
            raise ValueError("The model window cannot hold this message")
        setattr(conversation, "_prompt_start", start)
        cache[latest] = current["content"]
        for index in list(cache):
            if index < start:
                del cache[index]
        setattr(conversation, "_prompt_sent", cache)
        if start:
            notices.insert(0, f"Left out {start} earlier messages to fit the model window.")
        return PromptResult(messages, _size(messages), start, notices, compacted or start > previous_start)
