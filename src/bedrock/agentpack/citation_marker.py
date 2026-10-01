"""Whether a finished agent-pack answer is presenting itself as literature-cited.

In agent-pack mode, a scientist's literature question is answered by the
Director/Specialist, whose "review" (`bedrock.server.agents.director.Director.review` on the
backend) is a deterministic check that a citation is merely *present* -
never that the citation actually says what the answer claims. The one place
in this codebase that verifies a quote against the paper it is attributed to
is the pipeline in `bedrock.litreview`, reached through `/review`. An answer
from the AI system above can look exactly like a reviewed one - PMIDs,
`et al.`, a bracketed reference - with nothing on screen to say the
difference, which makes the unverified answer the path of least resistance.

`looks_cited` decides when to say so, once the answer has finished streaming
(see `PENDING_REVIEW_PREFIX`/`REVIEWING_PREFIX` in `agentpack_app.py` for the
in-flight state this must not interfere with). Following the precedent set
by `bedrock.recall.wants_recall` - a small, pure, regex-based trigger that is
easy to test in both directions - this looks at the *shape of the answer*,
not a guess about the question: PMIDs, DOIs, "et al.", bracketed numeric
citations ("[3]", "[1, 2]") and "according to ... <year>" are all things a
person only writes while presenting a claim as backed by a specific source.

Mentioning a year on its own ("in 2020 the FDA approved...") does not count.
Half of biology has a year in it, and marking every such sentence would teach
scientists to stop reading the marker at all - false positives are cheap
here, but firing on every message is not free.
"""

from __future__ import annotations

import re

CITATION_MARKER = (
    "\n\n---\n\n_The citations above were checked only for presence, not "
    "verified against the papers they name. `/review <question>` runs the "
    "verified literature review._"
)

_PMID = re.compile(r"\bPMID\s*:?\s*\d{4,9}\b", re.IGNORECASE)
_DOI = re.compile(r"\b10\.\d{4,9}/\S+")
_ET_AL = re.compile(r"\bet al\.?\b")
# A bracketed reference number, e.g. "[3]" or "[1, 2]" - never `arr[0]`, which
# has no space (or start-of-text) before the bracket, so the lookbehind rules
# it out.
_BRACKET_CITATION = re.compile(r"(?<![\w])\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]")
_ACCORDING_TO_YEAR = re.compile(r"\baccording to\b[^.]{0,80}\b(?:19|20)\d{2}\b", re.IGNORECASE)

_TRIGGERS = (_PMID, _DOI, _ET_AL, _BRACKET_CITATION, _ACCORDING_TO_YEAR)


def looks_cited(text: str) -> bool:
    """True when ``text`` presents itself as citing sources, not just discussing them."""
    return bool(text) and any(pattern.search(text) for pattern in _TRIGGERS)


def with_citation_notice(text: str) -> str:
    """Append :data:`CITATION_MARKER` to ``text`` when it looks cited, else return it unchanged."""
    return text + CITATION_MARKER if looks_cited(text) else text
