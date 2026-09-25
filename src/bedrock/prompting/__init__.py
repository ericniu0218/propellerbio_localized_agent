"""Build bounded chat prompts while keeping the sent prefix stable.

The system text is fixed. Turn-specific material belongs inside that turn's
user message. The in-memory sent-message cache preserves the exact previous
prompt when another turn arrives; it is never part of saved conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from bedrock import attachments
from bedrock.config.profile import MachineProfile
from bedrock.inference.tokens import estimate_tokens as _estimate_tokens

if TYPE_CHECKING:
    from bedrock.attachments import Attachment
    from bedrock.models.conversation import Conversation


def _fit_text(content: str, max_tokens: int) -> str:
    """Find the longest prefix under the current, learned token estimate."""
    low, high = 0, len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if _estimate_tokens(content[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return content[:low]


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

    @classmethod
    def from_profile(cls, profile: MachineProfile) -> PromptBudget:
        """Use the same context and reply limits as the model scheduler."""
        return cls(profile.context_window, profile.reply_limit, attachment_share=profile.attachment_share)


@dataclass(frozen=True)
class ExtraSpec:
    priority: int
    share: float
    render: Callable[[object, int], tuple[str, list[str]]]
    before_question: bool = False


def _text(value: object, max_tokens: int) -> tuple[str, list[str]]:
    content = str(value)
    if _estimate_tokens(content) <= max_tokens:
        return content, []
    return _fit_text(content, max(0, max_tokens - 1)).rstrip() + "…", ["Some added context was shortened to fit the model window."]


def _recall(value: object, max_tokens: int) -> tuple[str, list[str]]:
    opening = "[Excerpts from my earlier chats]\n"
    closing = "\n[End of excerpts]"
    frame_tokens = _estimate_tokens(opening + closing) + 1
    if max_tokens <= frame_tokens:
        return "", ["Recalled excerpts were left out to fit the model window."]
    content, notices = _text(value, max_tokens - frame_tokens)
    return opening + content + closing, notices


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
    "recall": ExtraSpec(1, 0.20, _recall, before_question=True),
    "passages": ExtraSpec(2, 0.20, _text, before_question=True),
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


def _combine(typed: str, rendered: list[tuple[bool, str]]) -> str:
    before = [part for first, part in rendered if first and part]
    after = [part for first, part in rendered if not first and part]
    return "\n\n".join(before + [typed] + after)


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

        cache: dict[int, str | None] = getattr(conversation, "_prompt_sent", {})
        start: int = getattr(conversation, "_prompt_start", 0)
        previous_start = start
        forced_compaction = bool(getattr(conversation, "_prompt_forced_trim", False))
        latest = len(conversation.messages) - 1
        history: list[dict[str, str]] = []
        history_indices: list[int] = []
        compacted = False
        for index in range(start, latest):
            message = conversation.messages[index]
            content = cache[index] if index in cache else message.content
            if content is None:  # folded into a later message of the same role
                continue
            if message.attachments:
                old = attachments.prompt_block(message.attachments)
                new = attachments.prompt_block([file.as_digest() for file in message.attachments])
                if old in content:
                    content = content.replace(old, new)
                    compacted = True
                elif index not in cache:
                    content = attachments.with_attachments(content, [file.as_digest() for file in message.attachments])
            if history and history[-1]["role"] == message.role.value:
                content = history[-1]["content"] + "\n\n" + content
                cache[history_indices[-1]] = None
                history[-1]["content"] = content
                history_indices[-1] = index
                cache[index] = content
            else:
                history.append({"role": message.role.value, "content": content})
                history_indices.append(index)

        newest = conversation.messages[latest]
        typed = newest.content
        notices: list[str] = []
        rendered: list[tuple[bool, str]] = []
        for name, spec in sorted(EXTRA_SPECS.items(), key=lambda pair: pair[1].priority):
            value = extras.get(name)
            if name == "attachments" and value is None:
                value = newest.attachments
            if not value:
                continue
            cap = int(self.budget.context_window * (self.budget.attachment_share if name == "attachments" else spec.share))
            part, warnings = spec.render(value, cap)
            rendered.append((spec.before_question, part))
            notices.extend(warnings)
        content = _combine(typed, rendered)
        current = {"role": Role.USER.value, "content": content} if newest.role is Role.USER else {"role": newest.role.value, "content": content}
        if history and history[-1]["role"] == current["role"]:
            previous_same_role = history.pop()["content"]
            typed = previous_same_role + "\n\n" + typed
            current["content"] = previous_same_role + "\n\n" + current["content"]
            cache[history_indices.pop()] = None
        target = int(self.budget.usable * self.budget.trim_to)
        threshold = int(self.budget.usable * self.budget.trim_at)
        if _size(prefix + history + [current]) > threshold:
            while history and _size(prefix + history + [current]) > target:
                history.pop(0)
                start = history_indices.pop(0) + 1
        # If the newest turn alone is large, use the same priority registry to
        # shrink low-priority extras before touching the employee's typed text.
        if _size(prefix + history + [current]) > self.budget.usable:
            remaining = self.budget.usable - _size(prefix + history + [{"role": current["role"], "content": typed}])
            rendered = []
            notices = []  # report what the rebuilt prompt actually includes
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
                rendered.append((spec.before_question, part))
                notices.extend(warnings)
                remaining -= _estimate_tokens(part) + 2
            current["content"] = _combine(typed, rendered)
        if _size(prefix + history + [current]) > self.budget.usable:
            room = self.budget.usable - _size(prefix + history) - 5
            current["content"] = _fit_text(typed, max(0, room - 1)).rstrip() + "…"
            notices.append("The message was shortened to fit the model window.")
        messages = prefix + history + [current]
        if _size(messages) > self.budget.usable:
            room = self.budget.usable - _size(prefix + history) - 5
            current["content"] = _fit_text(current["content"], max(0, room))
        if _size(messages) > self.budget.usable:
            raise ValueError("The model window cannot hold this message")
        setattr(conversation, "_prompt_start", start)
        setattr(conversation, "_prompt_forced_trim", False)
        cache[latest] = current["content"]
        for index in list(cache):
            if index < start:
                del cache[index]
        setattr(conversation, "_prompt_sent", cache)
        if start:
            notices.insert(0, f"Left out {start} earlier messages to fit the model window.")
        return PromptResult(messages, _size(messages), start, notices,
                            forced_compaction or compacted or start > previous_start)
