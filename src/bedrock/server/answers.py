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


def format_answer(content: str, citations: list[str]) -> str:
    body = content.strip()
    if not citations:
        return body
    return body + "\n\n**Sources**\n" + "\n".join(f"- {citation}" for citation in citations)
