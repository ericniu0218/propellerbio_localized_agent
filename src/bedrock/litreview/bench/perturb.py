"""Checking the checker: corrupt supported claims and see what catches them.

Three corruptions, each a way a report can be wrong while reading well:

  number   a statistic changed ("73%" -> "153%")
  flip     the direction of an effect reversed ("reduced" -> "increased")
  entity   the subject swapped for a near-miss ("psoriasis" -> "atopic dermatitis")

For each, we record whether the citation guard and the judge flag the
corrupted sentence. The guard is expected to catch numbers and entities. A
flipped effect is the judge's to catch: its model must find the evidence
sentence and its code must see that the claim says the effect went the other
way. If the judge misses flips, its "supported" verdicts are not worth much —
and the scorecard says so.
"""

from __future__ import annotations

import re

from bedrock.litreview.bench.distractors import pick_near_miss
from bedrock.litreview.knowledge import TermMatcher, clean
from bedrock.litreview.models import QuestionSpec

_NUM = re.compile(r"(?<![A-Za-z0-9.\-/])(\d+(?:\.\d+)?)(?![A-Za-z0-9])")
_FLIPS = [
    (r"\bdid not\b", "did"), (r"\bno significant\b", "a significant"), (r"\bnot associated\b", "associated"),
    (r"\breduced\b", "increased"), (r"\breduces\b", "increases"), (r"\breduction\b", "increase"),
    (r"\bdecreased\b", "increased"), (r"\bimproved\b", "worsened"), (r"\bimproves\b", "worsens"),
    (r"\bprevented\b", "accelerated"), (r"\bprevents\b", "accelerates"), (r"\bprotected\b", "sensitised"),
    (r"\binhibited\b", "enhanced"), (r"\blower\b", "higher"), (r"\bsuppressed\b", "amplified"),
    (r"\bincreased\b", "decreased"), (r"\bhigher\b", "lower"), (r"\belevated\b", "reduced"),
]


def perturb_number(text: str) -> str | None:
    m = _NUM.search(text)
    if not m:
        return None
    value = float(m.group(1))
    new = value * 2 + 7 if value < 1000 else value + 1234
    rendered = str(int(new)) if m.group(1).isdigit() else f"{new:.2f}"
    return text[: m.start(1)] + rendered + text[m.end(1) :]


def perturb_flip(text: str) -> str | None:
    for pattern, replacement in _FLIPS:
        new, count = re.subn(pattern, replacement, text, count=1, flags=re.I)
        if count:
            return new
    return None


def perturb_entity(text: str, spec: QuestionSpec) -> str | None:
    subject = spec.subject
    near = pick_near_miss(spec)
    if subject is None or near is None:
        return None
    text = clean(text)
    for m in TermMatcher(subject.exact_terms).matches(text):
        if m.kind == "exact":
            return text[: m.start] + near + text[m.end :]
    return None


def perturbations(text: str, spec: QuestionSpec) -> list[tuple[str, str]]:
    out = []
    for kind, fn in (("number", perturb_number), ("flip", perturb_flip)):
        new = fn(text)
        if new and new != text:
            out.append((kind, new))
    new = perturb_entity(text, spec)
    if new and new != text:
        out.append(("entity", new))
    return out
