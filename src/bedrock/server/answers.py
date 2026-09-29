"""The answer a person sees: the drafted text, then its sources. No model involved.

Every answer used to pass through one more Runner call, "format this as clean
markdown and append a Sources section". Two problems with that, one small and
one not:

* It cost a model call on every request, for text the drafting model already
  writes as markdown.
* On a high-stakes request it ran **after** the human approval: a person
  approved one text and a model then rewrote it before anyone saw the final
  answer. For a signed regulatory or clinical answer (21 CFR Part 11), what was
  signed has to be what is delivered.

So formatting is plain code: the text as drafted, and the citations listed
exactly. The hard path stores this text before the approval gate opens and
delivers that same stored text afterwards.
"""

from __future__ import annotations

import re

# A marker the model wrote for material it was never given: an invented id.
_STRAY_MARKER = re.compile(r"\s?\[source:[^\]\s]{1,80}\]")
MEMORY_PREFIX = "User-approved context: "


def source_label(block) -> str:
    """What a person reads for one piece of material: its name, not its id."""
    title = str(block.title)
    if getattr(block, "kind", "") == "memory" or title.startswith(MEMORY_PREFIX):
        return "Memory: " + title.removeprefix(MEMORY_PREFIX)
    if getattr(block, "kind", "") == "artifact":
        return "File: " + title
    return title


def format_answer(content: str, citations: list[str], sources: list | None = None) -> str:
    """The text as drafted, with its citations numbered and named.

    ``citations`` stay what they were for machines (``source:ID`` for material
    sent with the message, a tool's own reference otherwise). For a person,
    each ``[source:ID]`` in the text becomes ``[1]``, ``[2]``... and the list
    below names the file or memory note it points to, rather than a 32-letter
    id nobody can read. ``sources`` is the material the message was sent with.
    """
    body = content.strip()
    labels = {f"source:{block.id}": source_label(block) for block in (sources or [])}
    ordered = list(dict.fromkeys(citations))
    for number, citation in enumerate(ordered, 1):
        if citation in labels:
            body = body.replace(f"[{citation}]", f"[{number}]")
    body = _STRAY_MARKER.sub("", body).strip()
    if not ordered:
        return body
    return body + "\n\n**Sources**\n" + "\n".join(
        f"{number}. {labels.get(citation, citation)}" for number, citation in enumerate(ordered, 1)
    )
