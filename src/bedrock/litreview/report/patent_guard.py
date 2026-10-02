"""The citation guard for patent sentences.

Same idea as `report/guard.py`, for the one kind of statement a patent supports.
Papers report results, and the paper guard spends most of its checks on who was
studied, which way a result points and which qualifier the paper attached. A
patent reports none of that. It says what the filer claims or describes, so the
checks that matter are different, and the paper guard is not reused as it is: most
of it would be asking questions of a filing that has no answer.

Every sentence the writing model drafts is checked, in code, against the filings
it cites. It is dropped when:

1. it cites nothing, or an id that is not in this slice;
2. a number in it is in none of the cited filings' shown text (the quote, what
   it covers, its title, its number, its priority year);
3. it names a target, drug or disease the cited filings do not name, other than
   what the question is itself about;
4. it names a company that is not the filer of a cited filing, or does not
   attribute: nothing says whose claim this is ("claims", "describes", ...);
5. it states an outcome as fact ("is effective", "shows that", "proves");
6. it calls a filing granted when a cited one is only an application.

The words are checked, not their meaning - the same limit the paper guard has,
and the reason the quotes themselves are shown beside every sentence.
"""

from __future__ import annotations

import re

from bedrock.litreview.knowledge import Knowledge
from bedrock.litreview.models import PatentRecord, QuestionSpec
from bedrock.litreview.report.guard import numbers_in
from bedrock.litreview.stages.s6c_patent_extract import PatentUnit, assignee_key, assignee_name

ATTRIBUTION = re.compile(
    r"\b(?:claims?|claimed|recites?|recited|describes?|described|discloses?|disclosed|covers?|covered|"
    r"is directed to|are directed to|directed to|relates? to|seeks?|protects?|names?|filed|files|filing|"
    r"application|applications|patent|patents|publication|publications)\b", re.I)
OUTCOME_AS_FACT = re.compile(
    r"\b(?:proves?|proven|demonstrates?|demonstrated|shows? that|showed that|"
    r"(?:is|are|was|were|be|being)\s+(?:an?\s+)?(?:highly\s+|very\s+)?(?:effective|efficacious|safe|superior)|"
    r"effective\s+(?:treatment|therapy|therapies|treatments|drug|cure)|effectively|works?|safely|cures?|cured|"
    r"eliminates?|outperforms?)\b", re.I)
# "Pfizer's application ...", "a patent filed by Pfizer": the companies a sentence
# names, found by how a sentence names a filer, so that one the run never saw at
# all is caught too (the loop over the run's own filers only catches the wrong one
# of those it knows).
_NAMED_FILER = re.compile(
    r"(?:^|[\s(])([A-Z][A-Za-z0-9&\-]+(?:\s+[A-Z][A-Za-z0-9&\-]+){0,2})(?:'s|\u2019s)\s+(?:[a-z]+\s+){0,2}?"
    r"(?:application|applications|patent|patents|filing|filings|publication|publications|family|families)\b"
    r"|\b(?:filed|held|owned|assigned|applied for)\s+by\s+([A-Z][A-Za-z0-9&\-]+(?:\s+[A-Z][A-Za-z0-9&\-]+){0,2})")
GRANTED = re.compile(r"\bgranted\b|\bissued\b", re.I)


def shown_text(unit: PatentUnit, record: PatentRecord) -> str:
    """What the writing model is shown for a filing, and so all a sentence about it
    may say."""
    year = str(record.priority_date.year) if record.priority_date else ""
    return " ".join(p for p in (unit.quote, unit.covers or "", unit.indication or "", unit.target_raw,
                                record.title, record.id, assignee_name(record.assignee), year,
                                str(record.priority_date or "")) if p)


def is_granted(record: PatentRecord) -> bool:
    return (record.kind_code or "").upper().startswith("B")


class PatentGuard:
    def __init__(self, slice_: dict[str, PatentUnit], records: dict[str, PatentRecord], kb: Knowledge,
                 spec: QuestionSpec, every_assignee: dict[str, str]) -> None:
        self.slice = slice_
        self.records = records
        self.kb = kb
        self.spec = spec
        self.question_names = {e.canonical for e in spec.entities}
        # key -> a word that identifies the company in running text
        self.company_words = {key: key.split()[0] for key in every_assignee if key and len(key.split()[0]) >= 4}

    def check(self, text: str, citations: list[str]) -> str | None:
        """None when the sentence is supported, else why it is dropped."""
        if not citations:
            return "cites nothing"
        unknown = [c for c in citations if c not in self.slice]
        if unknown:
            return f"cites {', '.join(unknown)}, which is not in this section's filings"
        cited = [self.slice[c] for c in citations]
        recs = [self.records[u.patent_id] for u in cited]
        blob = " ".join(shown_text(u, r) for u, r in zip(cited, recs))

        extra = numbers_in(text) - numbers_in(blob)
        if extra:
            return f"number {sorted(extra)[0]} is in none of the cited filings"

        named = {e.canonical for e, _ in self.kb.scan(text)}
        shown = {e.canonical for e, _ in self.kb.scan(blob)} | self.question_names
        invented = sorted(named - shown)
        if invented:
            return f"names {invented[0]}, which the cited filings do not"

        lowered = text.lower()
        own = {assignee_key(r.assignee) for r in recs}
        for key, word in self.company_words.items():
            if key not in own and re.search(rf"\b{re.escape(word)}\b", lowered) \
                    and not any(word in k.split() for k in own):
                return f"names {key.title()}, who is not the filer of a cited filing"

        for match in _NAMED_FILER.finditer(text):
            candidate = (match.group(1) or match.group(2)).split()[0].lower()
            if not any(candidate in key.split() or key.startswith(candidate) for key in own):
                return f"names {candidate.title()}, who is not the filer of a cited filing"

        if not ATTRIBUTION.search(text):
            return "does not say whose claim it is (claims, describes, recites, is directed to ...)"
        if OUTCOME_AS_FACT.search(text):
            return "states an outcome as fact; a filing says what its filer claims, not that it works"
        if GRANTED.search(text) and not all(is_granted(r) for r in recs):
            return "calls a filing granted when a cited one is only an application"
        return None
