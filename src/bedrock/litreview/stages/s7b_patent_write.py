"""Stage 7b - what the competitor landscape says, written from the filings read.

The section had a table and nothing else: who filed, when, and a link-shaped
title. This adds what the filings say they cover, in the same discipline as the
rest of the report - a model drafts, code checks every sentence, and the filing's
own words are shown beside it so a person can check the code.

Per leading target, in this order, and only the first part uses a model:

1. a few sentences about what the filings for that target claim or describe, each
   citing filings (`P1`, `P2`, ...), under `report/patent_guard.py`;
2. a line of counts that is pure code: filings and filers, earliest priority,
   granted and pending, the kinds of thing claimed;
3. the filings cited, each with its own words as a `> quote`. These are the lines
   `report/invariants.py` `check_patents` re-proves against the stored filing.

Not a legal view. Nothing here says a filing is valid, infringed or free to work
around, and a target with no filings found is said to have none *found* by this
search - which is not freedom to operate. The section says so.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date

from pydantic import ValidationError

from bedrock.litreview.llm import LLMError, ask_json, load_prompt, strict_schema
from bedrock.litreview.models import DraftSection, PatentRecord
from bedrock.litreview.report.patent_guard import PatentGuard, is_granted
from bedrock.litreview.stages.s6c_patent_extract import (
    PatentReading,
    PatentUnit,
    assignee_key,
    assignee_name,
    order_key,
)

log = logging.getLogger(__name__)

PROMPT = "patent_write"
MIN_SENTENCE_CHARS = 30
PER_TARGET_CITED = 8

_KIND = {"method_of_treatment": "a method of treatment", "method": "a method", "use": "a use",
         "composition": "a composition", "not_stated": "claim 1 not available"}
_MODALITY = {"antibody": "antibody", "peptide": "peptide", "siRNA": "siRNA", "small_molecule": "small molecule",
             "vaccine": "vaccine", "genetic": "genetic", "other": "other / not stated"}


def evidence_lines(units: list[PatentUnit], reading: PatentReading, records: dict[str, PatentRecord]) -> str:
    out = []
    for u in units:
        r = records[u.patent_id]
        status = "granted" if is_granted(r) else "application"
        out.append(
            f"[{reading.short_ids[u.id]}] {r.id} · Filer: {assignee_name(r.assignee)} · priority "
            f"{r.priority_date or 'unknown'} · {status}\n"
            f"    About: {u.target_raw or u.target} · Kind: {_KIND[u.claim_kind]} · Covers: {u.covers or 'not stated'}\n"
            f'    Quote: "{" ".join(u.quote.split())}"')
    return "\n".join(out)


def _count(ctx, key: str) -> None:
    """Patent sentence counts, kept apart from the papers' `sentence_stats`: the
    report's "sentences removed by the citation check" is about papers, and a
    patent sentence dropped is not one of them."""
    stats = getattr(ctx, "patent_stats", None)
    if stats is not None:
        stats[key] += 1


def write_target_sentences(ctx, target: str, units: list[PatentUnit]) -> list[tuple[str, list[str]]]:
    """One model call and the guard -> kept (sentence, citations). Empty with no model."""
    reading: PatentReading = ctx.patent_reading
    if ctx.client is None or not ctx.model or not units:
        return []
    slice_ = {reading.short_ids[u.id]: u for u in units}
    records = {p.id: p for p in ctx.patents}
    prompt = load_prompt(PROMPT).render(
        question=ctx.spec.text, target=target, evidence=evidence_lines(units, reading, records),
        sentences="2 to 4")
    schema = strict_schema(DraftSection.model_json_schema(), enums={"citations": list(slice_)})
    try:
        draft = DraftSection.model_validate(
            ask_json(ctx.client, prompt, model=ctx.model, schema=schema, max_tokens=1200))
    except (LLMError, ValidationError) as exc:
        log.warning("writing the patent summary for %s failed: %s", target, exc)
        return []
    guard = PatentGuard(slice_, records, ctx.knowledge, ctx.spec,
                        {assignee_key(p.assignee): p.assignee or "" for p in ctx.patents})
    kept: list[tuple[str, list[str]]] = []
    seen: list[set[str]] = []
    from bedrock.litreview.stages.s7_write import REPEAT_OVERLAP, _content_words, _overlap

    for sentence in draft.sentences:
        text = sentence.text.strip()
        if len(text) < MIN_SENTENCE_CHARS:
            continue
        citations = list(dict.fromkeys(sentence.citations))
        words = _content_words(text)
        if any(_overlap(words, earlier) >= REPEAT_OVERLAP for earlier in seen):
            continue
        reason = guard.check(text, citations)
        if reason is not None:
            _count(ctx, "dropped")
            log.info("patent sentence dropped (%s): %s", reason, text[:80])
            continue
        seen.append(words)
        kept.append((text, citations))
    return kept


def _counts_line(target: str, units: list[PatentUnit], records: dict[str, PatentRecord]) -> str:
    recs = [records[u.patent_id] for u in units]
    families = {r.family_id or r.id for r in recs}
    filers = Counter(assignee_name(r.assignee) for r in recs)
    earliest = min((r.priority_date for r in recs if r.priority_date), default=None)
    granted = sum(1 for r in recs if is_granted(r))
    kinds = Counter(_MODALITY[u.modality] for u in units)
    top = ", ".join(f"{name} ({n})" for name, n in filers.most_common(4))
    return (f"**{target}** - {len(families)} patent famil{'y' if len(families) == 1 else 'ies'} directed to it "
            f"({len(recs)} filing{'' if len(recs) == 1 else 's'}); filers: {top}"
            f"{f' and {len(filers) - 4} more' if len(filers) > 4 else ''}; earliest priority {earliest or 'unknown'}; "
            f"{granted} granted, {len(recs) - granted} application{'' if len(recs) - granted == 1 else 's'}; "
            f"claimed as: {', '.join(f'{n} {k}' for k, n in kinds.most_common())}.")


def render_patent_text(ctx) -> str:
    """The markdown that opens the Competitor landscape section, or "" when no
    filing was read (no model, no patents, or none about a target)."""
    reading: PatentReading | None = ctx.patent_reading
    if reading is None or not reading.subjects():
        return ""
    records = {p.id: p for p in ctx.patents}
    target_order = [t.target for t in ctx.summary.targets]
    by_target: dict[str, list[PatentUnit]] = {}
    for u in sorted(reading.subjects(), key=lambda u: order_key(u, records, target_order)):
        by_target.setdefault(u.target, []).append(u)

    blocks = [
        "What the filings say they cover, for the leading targets. **This is what each filer claims or "
        "describes, not evidence that it works, and not a legal view**: nothing here says a filing is valid, "
        "infringed, or can be worked around. Sentences are checked in code against the filings they cite; "
        "each filing's own words are shown beneath."]
    for target, units in by_target.items():
        written = write_target_sentences(ctx, target, units[:PER_TARGET_CITED * 2])
        lines = [_counts_line(target, units, records)]
        cited: list[str] = []
        for text, citations in written:
            # Every filing a sentence cites is listed beneath with its own words
            # (`check_patents` fails the report otherwise), so a sentence that would
            # take the list past PER_TARGET_CITED filings is left out, not its filings.
            if len(set(cited) | set(citations)) > PER_TARGET_CITED:
                _count(ctx, "over_cited")
                continue
            lines.append(f"- {text} " + " ".join(f"[{c}]" for c in citations))
            cited += [c for c in citations if c not in cited]
        if not cited:
            lines.append("_No sentence could be written and verified for this target; the filings are listed below._")
        shown = sorted((u for u in units if reading.short_ids[u.id] in cited),
                       key=lambda u: int(reading.short_ids[u.id][1:])) or units[:3]
        for u in shown:
            r = records[u.patent_id]
            status = "granted" if is_granted(r) else "application"
            src = "abstract and claim 1" if u.basis == "claim_1_and_abstract" else "abstract only"
            lines += ["", f"[{reading.short_ids[u.id]}] {r.id} · {assignee_name(r.assignee)} · priority "
                          f"{r.priority_date or 'unknown'} · {status} · read from {src}",
                      f"> {' '.join(u.quote.split())}"]
        blocks.append("\n".join(lines))

    searched = {t for p in ctx.patents for t in p.matched_targets}
    missing = [t.target for t in ctx.summary.targets[:ctx.max_targets]
               if t.target in searched and t.target not in by_target]
    if missing:
        blocks.append("Searched, and no filing found that is directed to: " + ", ".join(missing)
                      + ". That means this search found none, which is not freedom to operate.")
    mentions = sum(1 for u in reading.units if u.subject == "mention")
    skipped = Counter(v.split(":")[0] for v in reading.skipped.values())
    tail = [f"{mentions} filing(s) only mention a target among others and are not summarised."] if mentions else []
    tail += [f"{n} not read ({why})." for why, n in skipped.items()]
    if tail:
        blocks.append(" ".join(tail))
    return "\n\n".join(blocks)
