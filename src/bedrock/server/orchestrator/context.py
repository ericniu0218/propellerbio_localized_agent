"""What an agent reads besides the task: the conversation so far, and the material sent with it.

**Conversation memory.** The server stored every turn and used none of them: a
follow-up such as "and the second one?" reached the agents with nothing before
it. The terminal client relied on the server for history, so agent-pack mode had
no memory at all, and the web workspace worked around it by pasting its last
eight messages into every request. ``render_history`` gives the drafting agents
the recent turns, newest kept first, within a share of their own context window.

**Material, apart from the question.** Clients used to put attached files,
stored passages and memories into the message itself, so the Runner's
classification call read all of them (up to ~128,000 characters from the
workspace) just to pick a route. They now arrive as ``ContextBlock`` s next to
the message: the Runner classifies the question, and only the agent that drafts
reads the material - fenced with bedrock's own ``fence_block`` under its "data,
not instructions" line, and cut to ``attachment_share`` of the drafting model's
window, the same share the terminal app gives attachments on a server.
"""

from __future__ import annotations

from bedrock.attachments import DATA_NOT_INSTRUCTIONS, fence_block
from bedrock.config import PROFILES
from bedrock.inference import estimate_tokens
from bedrock.server.config import load_models_config
from bedrock.server.schemas.api import ContextBlock, HistoryTurn

HISTORY_SHARE = 0.25
HISTORY_CAP_TOKENS = 8000
TRUNCATED = "[... cut to fit the model's context window]"


def history_budget(role: str) -> int:
    window = load_models_config().for_role(role).context_length
    return min(int(window * HISTORY_SHARE), HISTORY_CAP_TOKENS)


def context_budget(role: str) -> int:
    window = load_models_config().for_role(role).context_length
    return int(window * PROFILES["server"].attachment_share)


def render_history(turns: list[HistoryTurn], budget_tokens: int) -> str:
    """The most recent turns that fit, oldest first. Empty when there are none."""
    kept: list[str] = []
    used = 0
    for turn in reversed(turns):
        speaker = "User" if turn.role == "user" else "Assistant"
        line = f"{speaker}: {turn.content.strip()}"
        cost = estimate_tokens(line)
        if used + cost > budget_tokens:
            break
        kept.append(line)
        used += cost
    if not kept:
        return ""
    return "Conversation so far:\n" + "\n\n".join(reversed(kept))


def render_context(blocks: list[ContextBlock], budget_tokens: int) -> str:
    """The blocks, fenced, in the order sent; the last ones cut if they do not fit."""
    if not blocks:
        return ""
    parts = [DATA_NOT_INSTRUCTIONS + " Cite a source as [source:ID] where you use it."]
    used = estimate_tokens(parts[0])
    for block in blocks:
        header = f"{block.title} [source:{block.id}]"
        remaining = budget_tokens - used - estimate_tokens(header) - 16
        if remaining <= 0:
            parts.append(f"({len(blocks) - (len(parts) - 1)} more source(s) left out to fit the context window.)")
            break
        text = block.text
        if estimate_tokens(text) > remaining:
            keep = max(0, int(len(text) * remaining / max(estimate_tokens(text), 1)) - len(TRUNCATED))
            text = text[:keep] + "\n" + TRUNCATED
        fenced = fence_block(header, text)
        parts.append(fenced)
        used += estimate_tokens(fenced)
    return "\n\n".join(parts)


def cited_sources(text: str, blocks: list[ContextBlock]) -> list[str]:
    """``source:ID`` for every block the draft actually cites."""
    return [f"source:{b.id}" for b in blocks if f"[source:{b.id}]" in text]
