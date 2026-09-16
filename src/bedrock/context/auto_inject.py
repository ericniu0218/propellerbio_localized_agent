"""Connects the `/digest` store back to an ordinary chat message.

Before this module, compressing a file with `/digest` did nothing for the
*next* message you typed — the store just sat there until someone
remembered it existed and ran `/digest search` by hand. That's a real gap,
not a nuance: the whole point of compressing something is to have it
available later, and "available" meant "available if you remember to ask
for it."

The fix is deliberately the cheap, honest kind: before a message goes to a
model, search the store with the message's own words — the same BM25
search `/digest search` already uses, see `bedrock.context.search`'s
docstring for what that can and can't do — and if anything scores well,
prepend the matching passages, clearly labeled, so the model sees them as
context and the transcript still shows exactly what happened. This is
keyword matching, not comprehension: it will miss a paraphrase, and it
will occasionally attach something that only shares a rare word with the
question. That's the honest cost of doing this without an extra model
call, and it's always visible — nothing is added to what the model sees
without the user also seeing an "Added stored context from ..." line, so
a bad match is easy to notice and easy to ignore.
"""

from __future__ import annotations

from dataclasses import dataclass

from bedrock.context.store import ContextStore
from bedrock.context.types import SearchHit

DEFAULT_LIMIT = 3
PREVIEW_CHARS = 600
# A one-word message ("thanks", "ok") is too weak a signal to search the
# store on — every stored artifact would score on whichever generic word
# happens to match, which is worse than adding nothing.
MIN_QUERY_WORDS = 2


@dataclass(frozen=True)
class InjectedContext:
    """What matched, both as text for the model and something to tell the user."""

    hits: list[SearchHit]
    prompt_block: str

    @property
    def summary(self) -> str:
        names: list[str] = []
        for hit in self.hits:
            if hit.artifact_name not in names:
                names.append(hit.artifact_name)
        listed = ", ".join(f"`{name}`" for name in names)
        return f"Added stored context from {listed}."


def find_relevant_context(
    message: str, store: ContextStore, *, limit: int = DEFAULT_LIMIT
) -> InjectedContext | None:
    """Searches the store with the message itself. None when nothing qualifies.

    Never raises: a search failure here must not be why sending a message
    fails, the same rule `bedrock.context.digest` follows for the optional
    model-summary pass.
    """
    if len(message.split()) < MIN_QUERY_WORDS:
        return None
    try:
        hits = store.search(message, limit=limit)
    except Exception:
        return None
    if not hits:
        return None
    return InjectedContext(hits=hits, prompt_block=_render(hits))


def with_injected_context(message: str, injected: InjectedContext | None) -> str:
    """The text that actually goes to the model. Unchanged if nothing matched."""
    if injected is None:
        return message
    return f"{injected.prompt_block}\n\n---\n\n{message}"


def _render(hits: list[SearchHit]) -> str:
    lines = ["Relevant stored context (compressed earlier with /digest):"]
    for hit in hits:
        where = f"{hit.artifact_name} · {hit.label}" if hit.label else hit.artifact_name
        lines.append("")
        lines.append(f"[{where}]")
        lines.append(_preview(hit.text))
    return "\n".join(lines)


def _preview(text: str) -> str:
    text = text.strip()
    if len(text) <= PREVIEW_CHARS:
        return text
    return text[:PREVIEW_CHARS].rstrip() + "…"
