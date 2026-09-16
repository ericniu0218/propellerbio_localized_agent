"""Pull relevant pieces of past conversations into the one that is open.

Why this exists
---------------
Saved conversations stay on disk and are *not* loaded when the app starts or
when you chat. Loading every old transcript would slow the app down and would
send the model a lot of text it does not need. Instead, past chats are searched
only when the employee asks for it, in one of two ways:

* the ``/recall <what to look for>`` command (``/context`` does the same), or
* plain English that points back at an earlier chat, such as
  "do you remember when I set up the deploy workflow?" (see ``wants_recall``).

How the search works
--------------------
Deliberately simple keyword matching, with no embeddings and no extra service:

1. Turn the request into search words, dropping filler like "remember" or "the".
2. Read saved chats one file at a time, skipping any file that contains none
   of the words before parsing it.
3. Score each chat by which search words it contains, weighting rare words
   above common ones. A request like "remember the Kyoto trip? now write a
   packing list" should find the Kyoto chat, not every chat mentioning "list".
4. Re-read only the few winning chats and keep the question-and-answer pairs
   that matched, cut to a fixed size, so the model receives excerpts instead
   of whole transcripts.

The search stops after ``SEARCH_SECONDS``, newest chats first, so a huge
history cannot stall a reply. Nothing is kept in memory afterwards except the
bounded excerpt text.

The excerpts are sent inside the employee's latest message, not as system
instructions: an old chat may contain pasted text such as "ignore previous
instructions", and that must not gain the authority of the system prompt.
This module never touches Textual; the app decides when to call it.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from bedrock.models.conversation import RECALL_INTRO, Message, Role
from bedrock.storage import ConversationStore, ConversationSummary, StorageError

__all__ = [
    "MAX_CHATS", "MAX_CONTEXT_CHARS", "NO_TOPIC", "NOTHING_FOUND", "RECALL_INTRO", "RecallHit",
    "RecallResult", "build_context", "search_history", "search_terms", "wants_recall",
]

MAX_CHATS = 3
"""How many past conversations one recall may pull from."""

MAX_MESSAGE_CHARS = 700
"""Each quoted message is cut to this length."""

MAX_CHAT_CHARS = 2000
"""Total quoted text taken from any one past conversation."""

MAX_CONTEXT_CHARS = 6000
"""Total quoted text sent to the model for one recall (roughly 1,500 tokens)."""

SEARCH_SECONDS = 3.0
"""Stop searching after this long; the newest chats have been searched first."""

NOTHING_FOUND = (
    "The user may be referring to an earlier conversation. Their saved chats were "
    "searched for this message and nothing matched. If the question depends on an "
    "earlier chat, say you could not find it rather than guessing."
)

NO_TOPIC = (
    "The user referred to an earlier conversation without saying what it was about. "
    "Ask what it was about, or suggest `/recall` followed by a few words from it."
)

# ---- Recognising a request to recall ----------------------------------------

_PAST = r"(?:before|earlier|previously|last\s+(?:time|week|month)|yesterday|the\s+other\s+day|together)"
_DID = (
    r"(?:we|you|u)\s+(?:did|made|built|wrote|said|decided|planned|created|discussed|talked\s+about|"
    r"worked\s+on|went\s+over|helped|gave|showed|suggested|fixed|came\s+up\s+with)"
)

# Kept narrow on purpose: "remember to use type hints", "I don't remember how
# to exit vim" and "last time I checked" must not search the employee's history.
_TRIGGERS = [
    r"\b(?:do|did|can|could|would)\s+(?:you|u)\s+(?:still\s+)?(?:remember|recall)\b",
    r"(?<!n't\s)(?<!n’t\s)(?<!not\s)\bremember\s+(?:when|how|what|where|which|that\s+time|the\s+time|our|last)\b",
    r"\blast\s+time\s+(?:we|you|u)\b",
    r"\blast\s+time\s+i\s+(?:asked|told|talked|mentioned|said|showed)\b",
    r"\b(?:previous|past|earlier|older|old|other|last)\s+(?:chats?|conversations?|sessions?)\b",
    r"\bchat\s+history\b",
    r"\bfrom\s+(?:your|my)\s+memor(?:y|ies)\b",
    r"\bpull\s+(?:up\s+)?(?:the\s+|some\s+)?memor(?:y|ies)\b",
    r"\bwhat\s+(?:did\s+)?i\s+(?:tell|told|say|said|mention(?:ed)?)\s+(?:to\s+)?(?:you|u)\b",
    r"\bwe\s+(?:talked|spoke|chatted)\s+about\b(?!\s+(?:above|so\s+far|in\s+this))",
    rf"\b{_DID}\b.{{0,80}}\b{_PAST}\b",
    rf"\b{_PAST}\b.{{0,80}}\b{_DID}\b",
    r"\b(?:bring|go|get)\s+back\b.{0,60}\bwe\s+(?:said|made|did|built|wrote|decided|discussed|planned)\b",
]
_TRIGGER = re.compile("|".join(_TRIGGERS), re.IGNORECASE)


def wants_recall(text: str) -> bool:
    """True when a message refers back to an earlier conversation."""
    return bool(_TRIGGER.search(text))


# ---- Search words -----------------------------------------------------------

_STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be been before being
    between both but by can could did do does doing done for from get got had has have
    having he her here him his how i i'd i'm i've if in into is it it's its just let
    me more most my myself no not now of on once only or other our ours out over own
    please same she should so some such than that the their them then there these
    they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours yourself u
    remember recall recalled previous previously past earlier older old last time times
    chat chats conversation conversations session sessions talked spoke chatted discussed
    went worked told tell said say asked pull use get bring grab check search look memory
    memories history context thing things something stuff again back together yesterday
    made did helped gave showed wrote built decided planned created fixed mentioned
    """.split()
)

_WORD = re.compile(r"[^\W_][\w'’-]*", re.UNICODE)
_NO_SPACES = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
"""Japanese, Chinese and Korean text has no spaces, so match those words anywhere."""


def _stem(word: str) -> str:
    """Crude word-form folding: "studies" and "study" both become "stud".

    Stems are matched as word *prefixes*, so "deploy" also finds "deployment".
    """
    for suffix in ("ies", "ing", "ed", "es", "s", "y"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


@dataclass(frozen=True)
class SearchTerms:
    words: frozenset[str]
    phrases: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.words or self.phrases)


def search_terms(text: str) -> SearchTerms:
    """The words worth searching for, plus any "quoted phrases"."""
    phrases = tuple(p.strip().casefold() for p in re.findall(r"[\"“”]([^\"“”]{3,})[\"“”]", text) if p.strip())
    words = set()
    for word in _WORD.findall(text.casefold()):
        bare = word.strip("'’-")
        if len(bare) > 1 and bare not in _STOPWORDS:
            words.add(_stem(bare))
    return SearchTerms(words=frozenset(words), phrases=phrases)


class _Matcher:
    """Finds which search words and phrases occur in a piece of text."""

    def __init__(self, terms: SearchTerms) -> None:
        self.terms = terms
        spaced = sorted((w for w in terms.words if not _NO_SPACES.search(w)), key=len, reverse=True)
        self.unspaced = [w for w in terms.words if _NO_SPACES.search(w)]
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, spaced)) + ")", re.IGNORECASE) if spaced else None
        self.ascii_probes = [w for w in terms.words if w.isascii()] + [p for p in terms.phrases if p.isascii()]
        self.needs_parse = len(self.ascii_probes) < len(terms.words) + len(terms.phrases)

    def might_match(self, raw: str) -> bool:
        # Files are written with ensure_ascii, so an ASCII word missing from the
        # raw text cannot be in the chat, and the file need not be parsed.
        lowered = raw.casefold()
        return self.needs_parse or any(probe in lowered for probe in self.ascii_probes)

    def find(self, text: str) -> tuple[set[str], int]:
        lowered = text.casefold()
        found = {m.casefold() for m in self.pattern.findall(text)} if self.pattern else set()
        found.update(w for w in self.unspaced if w in lowered)
        return found, sum(1 for phrase in self.terms.phrases if phrase in lowered)


# ---- Searching --------------------------------------------------------------


@dataclass
class RecallHit:
    """One past conversation that matched, with the excerpt to quote from it."""

    conversation_id: str
    title: str
    updated_at: datetime
    score: float
    excerpt: list[Message] = field(default_factory=list)


@dataclass
class RecallResult:
    hits: list[RecallHit]
    searched: int
    """How many saved chats were searched before finishing or running out of time."""
    total: int
    had_terms: bool = True


@dataclass
class _Candidate:
    summary: ConversationSummary
    covered: set[str]
    phrase_hits: int
    matched_messages: int
    best: list[tuple[int, int]]  # (score, message index), best first


def search_history(
    store: ConversationStore,
    request: str,
    *,
    exclude: set[str] | None = None,
    limit: int = MAX_CHATS,
    seconds: float = SEARCH_SECONDS,
) -> RecallResult:
    """Search saved conversations for ``request``. Slow-ish: call off the UI thread."""
    terms = search_terms(request)
    summaries = [s for s in store.list() if s.id not in (exclude or set())]
    if not terms:
        return RecallResult([], 0, len(summaries), had_terms=False)
    matcher = _Matcher(terms)
    deadline = time.monotonic() + seconds
    candidates: list[_Candidate] = []
    searched = 0
    for summary in summaries:
        if time.monotonic() > deadline:
            break
        searched += 1
        messages = _messages(store, summary.id, matcher, exclude or set())
        if messages is None:
            continue
        covered: set[str] = set()
        phrase_hits = matched = 0
        best: list[tuple[int, int]] = []
        for index, (role, content) in enumerate(messages):
            found, phrases = matcher.find(content)
            if found or phrases:
                covered |= found
                phrase_hits += phrases
                matched += 1
                best = sorted(best + [(len(found) + 2 * phrases, index)], reverse=True)[:3]
        if covered or phrase_hits:
            candidates.append(_Candidate(summary, covered, phrase_hits, matched, best))

    # Rare words count for more: a word found in every chat tells us nothing.
    frequency: dict[str, int] = {}
    for candidate in candidates:
        for word in candidate.covered:
            frequency[word] = frequency.get(word, 0) + 1
    weight = {word: math.log(1 + searched / count) for word, count in frequency.items()}
    available = sum(weight.values()) or 1.0
    top_weight = max(weight.values(), default=1.0)

    def score(candidate: _Candidate) -> float:
        return sum(weight[w] for w in candidate.covered) + 2 * top_weight * candidate.phrase_hits + 0.05 * min(candidate.matched_messages, 9)

    scored = sorted(((score(c), c) for c in candidates), key=lambda pair: (pair[0], pair[1].summary.updated_at), reverse=True)
    best_score = scored[0][0] if scored else 0.0
    hits: list[RecallHit] = []
    for value, candidate in scored:
        if len(hits) >= limit:
            break
        share = sum(weight[w] for w in candidate.covered) / available
        if not candidate.phrase_hits and (share < 0.3 or value < 0.5 * best_score):
            continue
        excerpt = _excerpt(store, candidate, matcher, exclude or set())
        if excerpt:
            summary = candidate.summary
            title = summary.title if summary.generated_title else (summary.excerpt or summary.title)
            hits.append(RecallHit(summary.id, title, summary.updated_at, round(value, 3), excerpt))
    return RecallResult(hits, searched, len(summaries))


def _messages(store: ConversationStore, conversation_id: str, matcher: _Matcher, exclude: set[str]) -> list[tuple[Role, str]] | None:
    """The searchable messages of one saved chat, or None to skip it."""
    try:
        raw = store.read_text(conversation_id)
    except StorageError:
        return None
    if not matcher.might_match(raw):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return None
    if data.get("family_id") in exclude:
        return None  # a linked copy of the open chat is already on screen
    messages = []
    for item in data["messages"]:
        if isinstance(item, dict) and isinstance(item.get("content"), str) and item.get("role") in {"user", "assistant"}:
            messages.append((Role(item["role"]), item["content"]))
    return messages


def _excerpt(store: ConversationStore, candidate: _Candidate, matcher: _Matcher, exclude: set[str]) -> list[Message]:
    """Re-read one winning chat and quote its best-matching exchanges."""
    messages = _messages(store, candidate.summary.id, matcher, exclude)
    if not messages:
        return []
    chosen: set[int] = set()
    for _, index in candidate.best:
        if index >= len(messages):
            continue
        chosen.add(index)
        # Keep each match with its partner, so the model sees the question
        # together with the answer that was given to it.
        partner = index + 1 if messages[index][0] is Role.USER else index - 1
        if 0 <= partner < len(messages):
            chosen.add(partner)
    excerpt: list[Message] = []
    used = 0
    best_first = [index for _, index in candidate.best]
    for index in sorted(chosen, key=lambda i: (min((abs(i - b) for b in best_first), default=0), i)):
        role, content = messages[index]
        text = _cut(content, MAX_MESSAGE_CHARS)
        if used + len(text) > MAX_CHAT_CHARS:
            continue
        used += len(text)
        excerpt.append((index, Message(role, text)))
    return [message for _, message in sorted(excerpt, key=lambda pair: pair[0])]


def _cut(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_context(hits: list[RecallHit]) -> str:
    """The quoted excerpts, bounded to ``MAX_CONTEXT_CHARS``."""
    blocks = []
    total = 0
    for hit in hits:
        lines = [f'--- Past chat: "{hit.title}", last used {hit.updated_at:%Y-%m-%d} ---']
        for message in hit.excerpt:
            speaker = "User" if message.role is Role.USER else "Assistant"
            lines.append(f"{speaker}: {message.content}")
        block = "\n".join(lines)
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        total += len(block)
        blocks.append(block)
    return "\n\n".join(blocks)
