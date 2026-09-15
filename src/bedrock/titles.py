"""Short summary titles for conversations, written by the model itself.

A list of conversations is only useful if you can tell them apart. The first
question makes a poor label ("hi", "can you help me with something"), so after
a reply we ask the model for a two-to-five word title, like "Vector Database
Basics".

A conversation only gets a name once it has a concrete topic. Three gates,
cheapest first, and any one of them keeps the conversation "Untitled":

1. ``user_wrote_words``: "1" or "???" contains nothing to name, so the model
   is not even asked.
2. ``has_concrete_topic``: a plain YES/NO question to the model, showing it
   only the user's own messages.
3. The title itself, which may still come back as NONE or as a vague non-title
   such as "Getting Started" (see ``clean_title``).

Why the separate YES/NO step: tested against llama3.2, asking for "a title,
or NONE if there is no topic" in one request did not work - the model invented
names like "Greeting and Introduction" for "hi" nearly every time. Asked the
YES/NO question on its own, it answered correctly 39 times out of 40. Small
models follow a single, simple decision far better than an instruction with an
exception buried inside it.

An untitled conversation is looked at again after its next reply, when a topic
may have appeared - never before, so the same messages are not re-sent.

Privacy: the request goes to the *same* configured inference server as the
chat itself, through the same ``InferenceBackend``. No other service is ever
involved, and the model only sees text from the conversation being titled.

Small local models do not always follow instructions, so everything the model
sends back is cleaned up by ``clean_title`` before it is shown, and a reply
that cannot be turned into a sensible title is rejected rather than displayed.
"""

from __future__ import annotations

import re
import unicodedata

from bedrock.inference import InferenceBackend
from bedrock.models import Conversation, Role

NO_TOPIC = "NONE"
"""What the model is told to answer when there is nothing concrete to name yet."""

TITLE_INSTRUCTION = (
    "You write titles for chat conversations. The title names the specific topic "
    "the user wants to talk about, not how the assistant replied. Use 2 to 5 words "
    "in Title Case, in the same language as the user. Reply with the title only: "
    "no quotes, no full stop, no explanation. If the user has not raised a "
    "concrete topic yet - for example only a greeting, a single letter or number, "
    f"a test message, or small talk - reply with exactly: {NO_TOPIC}"
)
"""Sent as the system message. ``is_title_request`` recognises it by this text.

"The topic the user asked about" is there because, tested against llama3.2,
small models otherwise title the *reply*: a chat asking "do you remember my
phrase" came back as "A Conversation Starts Fresh" instead of being about the phrase."""

TOPIC_CHECK_INSTRUCTION = (
    "Decide whether ANY of the user's messages names a concrete topic: a specific "
    "subject, task or question, in any language, such as \"bees\", \"sea animals\", "
    "\"writing a cover letter\" or \"a Python import error\". Greetings, small talk, tests, "
    "single letters or numbers, and requests for help that do not say with what, are not "
    "concrete topics. If at least one message names a concrete topic, answer YES. "
    "Answer with one word: YES or NO."
)
"""Sent as the system message of the YES/NO request. Only the user's messages
are shown with it, so an assistant reply like "How can I help?" cannot sway it."""

MAX_TOPIC_CHECK_CHARACTERS = 2000
"""YES or NO needs very little; stop reading beyond this."""

OPENING_MESSAGES = 2
RECENT_MESSAGES = 4
"""The model reads the first two messages and the latest four. The opening is
usually the topic; the latest messages are where a topic appears in a
conversation that started with "hi"."""

CHARACTERS_PER_MESSAGE = 600
"""Long pasted documents are trimmed so the title request stays small and fast."""

MAX_REPLY_CHARACTERS = 200
"""Stop reading if the model ignores the instructions and writes an essay."""

MAX_TITLE_CHARACTERS = 40
MAX_TITLE_WORDS = 7

VAGUE_WORDS = {
    # glue words
    "a", "about", "an", "and", "for", "here", "i", "in", "me", "my", "now", "of",
    "on", "our", "some", "the", "to", "us", "we", "with", "you", "your",
    # words about the chat itself rather than a subject
    "anything", "assistance", "basic", "begin", "beginning", "begins", "chat",
    "conversation", "conversations", "discussion", "first", "fresh", "general",
    "get", "getting", "greeting", "greetings", "hello", "help", "helping", "hi",
    "initial", "initiate", "initiating", "inquiry", "introduction", "let's", "lets", "new",
    "no", "none", "question", "questions", "request", "something", "start",
    "started", "starting", "topic", "topics", "unclear", "unknown", "untitled",
}
"""A title made only of these words names nothing: "Unclear Conversation Topic",
"Fresh Conversation Begins", "Getting Help With Something" - all real titles
llama3.2 produced for chats with no topic. Such titles are treated as NONE.
One real word is enough to pass, so "Help Desk Setup" and "Getting Started
with Docker" are kept."""

_PREAMBLE = re.compile(r"^(?:sure[,!.]?\s*)?(?:here(?:'s| is)\b.*?|(?:a\s+)?(?:short\s+)?title)\s*[:\-–]\s*", re.IGNORECASE)
_DECORATION = "\"'“”‘’`*_#[]()「」《》"


def _window(conversation: Conversation, roles: tuple[Role, ...]) -> list:
    """The first and latest messages with the given roles, skipping the middle."""
    chosen = [m for m in conversation.messages if m.role in roles]
    if len(chosen) > OPENING_MESSAGES + RECENT_MESSAGES:
        chosen = chosen[:OPENING_MESSAGES] + chosen[-RECENT_MESSAGES:]
    return chosen


def user_wrote_words(conversation: Conversation) -> bool:
    """False when every user message is only digits, symbols or blank ("1", "???").

    ``isalpha`` counts letters in every script, so "蜂" passes just like "bees".
    """
    return any(
        any(character.isalpha() for character in message.content)
        for message in conversation.messages
        if message.role is Role.USER
    )


def wants_title(conversation: Conversation) -> bool:
    """True if the model should look for a topic now.

    That means: no title yet, at least one question and one answer, and
    something new since the model last looked and found no topic.
    """
    if conversation.generated_title:
        return False
    if len(conversation.messages) <= conversation.title_checked_at:
        return False
    has_question = any(m.role is Role.USER and m.content.strip() for m in conversation.messages)
    has_answer = any(m.role is Role.ASSISTANT and m.content.strip() for m in conversation.messages)
    return has_question and has_answer


def title_request(conversation: Conversation) -> list[dict[str, str]]:
    """The messages to send to the model to get a title back.

    The configured system prompt is deliberately *not* included: it tells the
    model how to chat with employees, which is not the job here.
    """
    lines = []
    for message in _window(conversation, (Role.USER, Role.ASSISTANT)):
        text = " ".join(message.content.split())[:CHARACTERS_PER_MESSAGE]
        lines.append(f"{message.role.value.capitalize()}: {text}")
    transcript = "\n\n".join(lines)
    return [
        {"role": "system", "content": TITLE_INSTRUCTION},
        {"role": "user", "content": f"Conversation:\n\n{transcript}\n\nTitle:"},
    ]


def topic_check_request(conversation: Conversation) -> list[dict[str, str]]:
    """The YES/NO request: does the user's side of the conversation name a topic?"""
    said = "\n".join(
        f"- {' '.join(message.content.split())[:CHARACTERS_PER_MESSAGE]}"
        for message in _window(conversation, (Role.USER,))
    )
    return [
        {"role": "system", "content": TOPIC_CHECK_INSTRUCTION},
        {"role": "user", "content": f"The user's messages:\n{said}\n\nDoes any message name a concrete topic?"},
    ]


def is_title_request(messages: list[dict[str, str]]) -> bool:
    """Lets test doubles tell title requests apart from chat requests."""
    return bool(messages) and messages[0].get("content") == TITLE_INSTRUCTION


def is_topic_check(messages: list[dict[str, str]]) -> bool:
    """Lets test doubles tell YES/NO topic checks apart from chat requests."""
    return bool(messages) and messages[0].get("content") == TOPIC_CHECK_INSTRUCTION


def means_yes(reply: str) -> bool:
    """Read the model's YES/NO answer. Anything that is not clearly YES is NO.

    Erring towards NO is the safe direction: a missed topic only means the
    conversation stays Untitled until the next reply, while a false YES could
    give it a made-up name that never gets corrected.
    """
    visible = visible_reply(reply)
    lines = [line for line in visible.splitlines() if line.strip()]
    words = (lines[-1] if lines else visible).strip().split()
    return bool(words) and words[-1].strip(".,!:;*\"'").upper() == "YES"


def clean_title(raw: str) -> str | None:
    """Turn whatever the model sent back into a short title, or None.

    None means "no concrete topic yet": the model said NONE, wrote a title that
    names nothing, or wrote something that is not a title at all.

    Handles the usual ways small models disobey: wrapping the title in quotes
    or bold, prefixing "Title:" or "Here is a title:", putting the title on
    the line after a preamble, or adding a full stop.
    """
    raw = visible_reply(raw)
    raw = "".join(character for character in raw if unicodedata.category(character)[0] != "C" or character in "\n\t")
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return None

    candidate = lines[0]
    # "Here is a short title:" on its own line, with the title underneath.
    if candidate.rstrip(" *").endswith(":") and len(lines) > 1:
        candidate = lines[1]

    candidate = candidate.strip(_DECORATION + " ")
    candidate = _PREAMBLE.sub("", candidate)
    candidate = candidate.strip(_DECORATION + " ").rstrip(".。!！?？:;,").strip(_DECORATION + " ")
    candidate = " ".join(candidate.split())

    if not candidate or len(candidate.split()) > MAX_TITLE_WORDS:
        return None  # an explanation or a sentence, not a title
    words = [word.strip(".,:;!?-–").lower() for word in candidate.split()]
    if all(word in VAGUE_WORDS for word in words):
        return None  # NONE itself, or a title that names no topic
    if len(candidate) > MAX_TITLE_CHARACTERS:
        candidate = candidate[: MAX_TITLE_CHARACTERS - 1].rstrip() + "…"
    return candidate


async def generate_title(backend: InferenceBackend, model: str, conversation: Conversation) -> str | None:
    """Name the conversation, or return None if it has no concrete topic yet.

    Goes through the three gates described at the top of this module.

    Raises ``InferenceError`` if the server cannot be reached, so the caller
    can tell "try again later" apart from "there is nothing to name yet".
    """
    if not user_wrote_words(conversation):
        return None
    answer = await _ask(backend, model, topic_check_request(conversation), MAX_TOPIC_CHECK_CHARACTERS)
    if not means_yes(answer):
        return None
    return clean_title(await _ask(backend, model, title_request(conversation), MAX_REPLY_CHARACTERS))


async def _ask(backend: InferenceBackend, model: str, messages: list[dict[str, str]], limit: int) -> str:
    """Send one request and collect the reply, stopping after ``limit`` characters."""
    stream = backend.stream_chat(model, messages)
    reply = ""
    try:
        async for fragment in stream:
            reply += fragment
            if len(visible_reply(reply)) >= limit or len(reply) >= 32000:
                break
    finally:
        # Stopping early must close the HTTP stream, not leave it dangling.
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()
    return visible_reply(reply)


def visible_reply(reply: str) -> str:
    """Discard complete or truncated reasoning blocks in either request."""
    return re.sub(r"<think\b[^>]*>.*?(?:</think>|$)", "", reply, flags=re.IGNORECASE | re.DOTALL)
