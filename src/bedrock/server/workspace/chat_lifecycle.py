"""What a chat request says about itself while it runs, when it fails, and what
it is given to read about a saved literature review.

Two jobs, both about telling the truth to the person waiting:

* **Every request ends in a state they can read.** A failure used to leave the
  chat with no message at all (the dots simply vanished), a restart left a run
  "running" for ever, and an answer held for human review looked exactly like
  one still being written. `explain_failure` turns what went wrong into a
  sentence that says what to do, and keeps the kind of failure so the page can
  offer Retry. It never writes an answer: a failed request gets an
  explanation, not a substitute.

* **A follow-up about a saved report reads the parts that answer it.** A saved
  report is tens of thousands of characters; the agents' context budget cuts
  a block from the end, so a question about side effects was answered from
  the report's first screen, and a report whose Safety section said nothing
  was found could read as "no harms". `report_context` picks sections for the
  question within a bounded budget, says which were included and which were
  left out, and states what "no evidence found" means.
"""

from __future__ import annotations

import re

# --- failures ----------------------------------------------------------------------

_KINDS: list[tuple[str, re.Pattern[str], str]] = [
    ("rejected", re.compile(r"rejected by human", re.I),
     "You sent this answer back at review, so it was not posted. Ask again, or rephrase the question."),
    ("context", re.compile(r"context (?:window|length)|too long|maximum context|exceeds? .*tokens|prompt is too", re.I),
     "The question and its attached material are too long for the model's context window. Remove an attachment or "
     "ask about one part of the report, then choose Retry."),
    ("missing_model", re.compile(r"model[^.]{0,80}(?:not found|does not exist|not available|unknown)|404", re.I),
     "The model this workspace is set to use is not available on the model server. Ask IT to load it (or change "
     "the model), then choose Retry. Your message has been kept."),
    ("unavailable", re.compile(r"connect|refused|unavailable|not answering|timed? ?out|timeout|could not list models|"
                               r"request failed|503|502|server failed", re.I),
     "The model server is unavailable or did not answer in time. Your message has been kept. Reconnect the server "
     "and choose Retry."),
    ("budget", re.compile(r"budget", re.I),
     "The request ran over its token budget before it finished. Ask a narrower question, then choose Retry."),
]


def explain_failure(error: str | None, *, unavailable: bool = False) -> tuple[str, str]:
    """(kind, message for the person). The raw error is kept separately, for diagnosis."""
    text = str(error or "")
    if unavailable:
        return "unavailable", _KINDS[3][2]
    for kind, pattern, message in _KINDS:
        if pattern.search(text):
            return kind, message
    short = re.sub(r"\s+", " ", text).strip()[:160]
    reason = f" ({short})" if short else ""
    return "failed", f"The AI system could not finish this request{reason}. Your message has been kept; choose Retry."


STALLED = ("stalled", "Nothing was heard from the model for {minutes} minutes, so the request was stopped. The model "
                      "server may be overloaded or disconnected. Your message has been kept; choose Retry.")
RESTARTED = ("restarted", "The server restarted while this request was running, so it was stopped. Your message has "
                          "been kept; choose Retry.")
EMPTY = ("empty", "The model returned an empty answer. Nothing was posted; choose Retry.")


# --- a saved report, for a question about it -----------------------------------------------

_HEADING = re.compile(r"^(#{1,2}) (.+)$", re.M)
_ALWAYS = ("answer", "evidence at a glance", "how the studies line up", "limitations, gaps and open questions")
_TOPICS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"\b(harm|harmful|safe|safety|side[- ]?effects?|adverse|toxic\w*|risk|danger\w*|tolerab\w*|"
                r"complication)", re.I),
     ("safety", "negative and conflicting results", "key results in context")),
    (re.compile(r"\b(how does|mechanism|pathway|why|moa|signal\w*)\b", re.I), ("mechanism of action", "disease biology")),
    (re.compile(r"\b(mice|mouse|rat|animal|preclinical|in vivo|in vitro|cells?)\b", re.I), ("preclinical efficacy",)),
    (re.compile(r"\b(how much|effect size|magnitude|number|percent|%|dose|reduc\w*|increase\w*|trial|patients?|"
                r"human|clinical)\b", re.I), ("key results in context", "clinical efficacy", "human clinical evidence")),
    (re.compile(r"\b(target|which targets?)\b", re.I), ("candidate targets",)),
    (re.compile(r"\b(prevent\w*|prophyla\w*)\b", re.I), ("prevention-specific evidence",)),
    (re.compile(r"\b(abbreviat\w*|stand for|mean)\b", re.I), ("abbreviations",)),
    (re.compile(r"\b(review articles?|background)\b", re.I), ("context from review articles (not primary evidence)",)),
]
_ABSENCE_NOTE = (
    "How to read this report: a section that says no evidence was found, or that is not included here, means nothing "
    "was found in the papers this review read - not that none exists. Absence of reported harms is not evidence of "
    "safety. Cite the report's reference numbers [n] for any claim."
)


def is_report(artifact: dict) -> bool:
    return bool(artifact.get("researchId"))


def _sections(text: str) -> list[tuple[str, str]]:
    """(lowercased title, the section's own text including its heading) in order."""
    marks = list(_HEADING.finditer(text))
    if not marks:
        return [("", text)]
    out = [("__lead__", text[: marks[0].start()])] if marks[0].start() else []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(2).strip().lower(), text[m.start():end]))
    return out


def report_context(text: str, question: str, budget_chars: int) -> tuple[str, list[str], list[str]]:
    """(text to give the agents, sections included, sections left out).

    The report's opening (question and scope), the sections that answer
    every question, then the sections the question's words call for, then its
    references, each whole, within `budget_chars`. A section that does not fit
    is listed as left out, never cut in the middle."""
    sections = _sections(text)
    main, appendix = [], []
    in_appendix = False
    for title, body in sections:
        if title == "appendix":
            in_appendix = True
        (appendix if in_appendix else main).append((title, body))
    wanted = list(_ALWAYS)
    for pattern, titles in _TOPICS:
        if pattern.search(question or ""):
            wanted += [t for t in titles if t not in wanted]
    order = ["__lead__"] + wanted + ["references"]
    by_title = {title: body for title, body in main}
    chosen: list[str] = []
    parts: list[str] = []
    used = len(_ABSENCE_NOTE) + 200
    for title in order:
        body = by_title.get(title)
        if not body or title in chosen:
            continue
        if used + len(body) > budget_chars:
            continue
        parts.append(body.strip())
        chosen.append(title)
        used += len(body)
    left_out = [t for t, _ in main if t not in chosen and t != "__lead__"] + (["appendix"] if appendix else [])
    shown = [t for t in chosen if t != "__lead__"]
    header = (f"{_ABSENCE_NOTE}\nReport sections included for this question: {', '.join(shown) or 'none'}. "
              f"Left out (not relevant to the question, or no room): {', '.join(left_out) or 'none'}.")
    return header + "\n\n" + "\n\n".join(parts), shown, left_out
