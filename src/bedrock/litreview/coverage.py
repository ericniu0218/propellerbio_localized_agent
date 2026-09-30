"""The coverage ledger: where each paper went, and why it stopped there.

A report says what the papers it read support. It does not, on its own, say
what happened to the papers it did *not* use: which failed the gate, which
never got a relevance grade because the budget ran out, which were graded but
not read, which were read and yielded nothing that survived verification, and
which yielded evidence the writer was never given. Without that, "the report
does not mention target X" cannot be told apart from "no paper we had
supports X" - and those call for different fixes.

This module builds that account from what the run already stored, plus three
bookkeeping records the stages keep when asked (grading selection reasons,
read selection reasons, per-chunk extraction outcomes, and the units each
report section was given). It is written as the run's `coverage.v1` artifact.
It is **not** part of the report: the report's bytes do not depend on it, and
building it makes no model call.

Reading the counts. Each count names its denominator in `denominators`. They
are not meant to sum: a paper can be tagged with several facets, a unit can
be given to several sections. "Papers" are records after duplicates were
merged, not independent studies - two papers can report one trial, and
nothing here resolves that. A facet the plan marked `covered` because tagged
papers were selected is `realized` only when verified units for it exist.

What it does not claim. A target "in the answer" here means a kept Answer
sentence cites a verified unit about that target - never that its name
appears in a title or a citation. Nothing here says the literature was
searched completely; `stop_reason` says why reading stopped.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from bedrock.litreview.models import (
    EvidencePlan,
    EvidenceSummary,
    Facet,
    GateResult,
    Paper,
    QuestionSpec,
    RankedPaper,
    Report,
)
from bedrock.litreview.stages.s4_plan import eligible_papers

ARTIFACT = "coverage.v1"
SCHEMA_VERSION = 1

POLICY_KNOBS = (
    "rank.max_graded", "rank.selection", "rank.reservations", "plan.max_papers", "plan.min_grade",
    "extract.sections", "extract.discussion_when_no_results", "extract.chunk_chars",
    "extract.max_chunks_per_paper", "extract.methods_context_chars",
    "write.max_targets", "write.units_per_target", "write.max_units_per_section",
)

DENOMINATORS = {
    "papers_collected": "paper records in the run's pinned set, after duplicates were merged (records, not studies)",
    "full_text_available": "collected papers with a results, discussion or methods section",
    "gate": "collected papers, by exact-entity gate outcome",
    "grading_selected": "gate-passing papers chosen for a relevance grade",
    "graded": "papers with a relevance grade (grading_selected minus grading failures)",
    "grading_failed": "papers chosen for a grade whose grading call failed or returned nothing usable",
    "qualified": "usable papers eligible to be read (graded at least plan.min_grade, or all usable when nothing was graded)",
    "read_selected": "qualified papers chosen for extraction",
    "unread_qualified": "qualified papers not read (budget)",
    "chunks_planned": "text chunks planned across read papers",
    "chunks_attempted": "chunks sent to the model",
    "chunks_failed": "chunks whose call failed (no findings from them)",
    "findings_proposed": "findings the model proposed, before verification",
    "units_accepted": "verified evidence units kept",
    "target_bearing_units": "accepted units whose target has a row in the evidence table",
    "units_sent_to_synthesis": "distinct accepted units given to at least one written section",
    "sentences_over_limit": "guard-passing sentences removed because a section already had its maximum",
    "units_cited_after_guard": "distinct units cited by at least one kept sentence",
    "units_cited_only_by_dropped": "units cited only by sentences the citation guard removed",
    "targets_in_evidence_table": "distinct targets with an evidence-table row",
    "targets_in_narrative": "targets with a unit cited by a kept sentence in any written section",
    "targets_in_answer": "targets with a unit cited by a kept Answer sentence",
    "negative_units": "accepted units whose outcome is no_effect or worsened",
    "negative_units_cited": "negative units cited by a kept sentence",
    "human_units": "accepted units with human clinical or human genetic evidence type",
    "human_numeric_units": "human units whose quote or context sentence contains a digit",
    "human_numeric_units_in_answer": "human numeric units cited by a kept Answer sentence",
}


def acquisition_summary(report, collected: int) -> dict[str, Any]:
    """The searcher's own account of a live search (`SearchReport`), as counts.

    Query strings are not copied - the paper records already say which query
    found them - only how many were sent to each source, how many records
    came back, how many survived merging and the per-question cap, and which
    sources failed. Recorded once, when the papers were collected."""
    out = {
        "status": "recorded",
        "queries_sent": {source: len(qs) for source, qs in sorted(report.queries.items())},
        "records_returned": dict(sorted(report.hits.items())),
        "unique_after_merge": report.unique,
        "kept_after_cap": report.returned,
        "truncated_by_cap": max(0, report.unique - report.returned),
        "collected_after_ingest": collected,
        "full_text_tried": report.full_text_tried,
        "full_text_fetched": report.full_text,
        "title_only": report.title_only,
        "failed_queries": sorted(report.failures),
        "searched_question_text": report.searched_question_text,
    }
    # Only when citation snowballing ran, so a summary without it keeps the
    # bytes it always had.
    seeds = getattr(report, "snowball_seeds", None)
    if seeds:
        out["snowball"] = {"seeds": list(seeds), "links_returned": report.snowball_links,
                           "papers_added": report.snowball_added}
    return out


def policy_fingerprint(knobs: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(knobs, sort_keys=True).encode("utf-8")).hexdigest()


def policy_knobs(settings) -> dict[str, Any]:
    """The selection and budget knobs in effect, as the run read them."""
    out: dict[str, Any] = {}
    for path in POLICY_KNOBS:
        value = settings.knob(path, None)
        out[path] = list(value) if isinstance(value, tuple) else value
    return out


def _cited(report: Report, *, section_ids: set[str] | None = None) -> tuple[set[str], set[str]]:
    """(unit ids cited by kept sentences, unit ids cited by dropped ones)."""
    by_short = {short: uid for uid, short in report.short_ids.items()}
    kept, dropped = set(), set()

    def walk(section, top: str) -> None:
        if section_ids is None or top in section_ids:
            for s in section.sentences:
                ids = {by_short[c] for c in s.citations if c in by_short}
                (kept if s.kept else dropped).update(ids)
        for sub in section.subsections:
            walk(sub, top)

    for section in report.sections:
        walk(section, section.id)
    return kept, dropped


def _has_digit(*texts: str | None) -> bool:
    return any(ch.isdigit() for t in texts if t for ch in t)


def build_coverage(
    *,
    run_id: str,
    corpus_hash: str | None,
    spec: QuestionSpec,
    facets: list[Facet],
    papers: list[Paper],
    gates: dict[str, GateResult],
    ranked: list[RankedPaper],
    plan: EvidencePlan,
    summary: EvidenceSummary,
    report: Report,
    knobs: dict[str, Any],
    extract_stats: dict[str, int] | None,
    used_model: bool,
    grading_reasons: dict[str, str] | None = None,
    read_reasons: dict[str, str] | None = None,
    extract_outcomes: dict[str, dict] | None = None,
    section_inputs: dict[str, list[str]] | None = None,
    sentence_stats: dict[str, int] | None = None,
    design: dict[str, dict] | None = None,
    acquisition: dict[str, Any] | None = None,
    stop_reason: str = "fixed_budget",
) -> dict[str, Any]:
    """The ledger for one run. Pure code over the run's own records.

    Any record passed as None was not kept for this run (a resumed run whose
    earlier stages ran before the ledger existed, say): the fields it would
    have filled are written as `null`, never reconstructed.
    """
    min_grade = int(knobs.get("plan.min_grade") or 2)
    rank_of = {r.paper_id: r for r in ranked}
    graded_any = any(r.grade is not None for r in ranked)
    qualified = {r.paper_id for r in eligible_papers(ranked, min_grade=min_grade, graded=graded_any)}
    read = {job.paper_id: job for job in plan.jobs}

    unit_by_id = {u.id: u for u in summary.units}
    target_of: dict[str, str] = {}
    for t in summary.targets:
        for uid in t.unit_ids:
            target_of[uid] = t.target
    units_by_paper: dict[str, list[str]] = {}
    for u in summary.units:
        units_by_paper.setdefault(u.paper_id, []).append(u.id)

    written = {s.id for s in report.sections if s.kind == "written"}
    cited_all, dropped_all = _cited(report, section_ids=written)
    cited_answer, _ = _cited(report, section_ids={"answer"})
    sent: set[str] = set()
    for ids in (section_inputs or {}).values():
        sent.update(ids)

    # --- per paper -------------------------------------------------------------
    records: dict[str, dict[str, Any]] = {}
    for p in papers:
        gate = gates.get(p.id)
        r = rank_of.get(p.id)
        out = extract_outcomes.get(p.id) if extract_outcomes is not None else None
        chunks = (out or {}).get("chunks") or []
        unit_ids = units_by_paper.get(p.id, [])
        rec: dict[str, Any] = {
            "source_ids": {k: v for k, v in (("doi", p.doi), ("pmid", p.pmid), ("pmcid", p.pmcid)) if v},
            "source": p.source,
            "merged_from": [str(c.get("id")) for c in p.merged_from] or None,
            "text": "full_text" if p.has_full_text else ("abstract_only" if p.abstract else "title_only"),
            "gate": {"status": gate.status, "reason": gate.reason} if gate else None,
            "design": (design or {}).get(p.id),
            "rank": r.rank if r else None,
            "lexical_score": round(r.lexical_score, 4) if r else None,
            "grading": None,
            "qualified": p.id in qualified,
            "read": None,
            "extraction": None,
            "unit_ids": unit_ids,
            "cited_unit_ids": sorted(set(unit_ids) & cited_all),
        }
        if gate and gate.status == "pass":
            reason = (grading_reasons or {}).get(p.id) if grading_reasons is not None else None
            rec["grading"] = {
                "selected_reason": reason,
                "grade": r.grade if r else None,
                "justification_quote": r.quote_status if r else None,
                "study_kind": r.study_kind if r else None,
            }
        if p.id in read:
            job = read[p.id]
            rec["read"] = {
                "reason": (read_reasons or {}).get(p.id) if read_reasons is not None else None,
                "sections": job.sections,
                "abstract_only": job.abstract_only,
            }
            if out is not None:
                rec["extraction"] = {
                    "planned_chunks": out.get("planned"),
                    "chunks": chunks,
                    "review_discarded": out.get("review_discarded", False),
                    "duplicates": out.get("duplicates", 0),
                }
        rec["disposition"] = _disposition(p, gate, r, rec, used_model=used_model, graded_any=graded_any,
                                          min_grade=min_grade, out=out,
                                          sent=sent if section_inputs is not None else None, cited=cited_all)
        records[p.id] = rec

    # --- facets: planned tags against verified units -----------------------------
    facet_rows: dict[str, dict[str, Any]] = {}
    for fp in plan.facets:
        ids = summary.facet_units.get(fp.facet.id, [])
        realized_papers = sorted({unit_by_id[i].paper_id for i in ids if i in unit_by_id})
        n = len(realized_papers)
        realized = "gap" if n == 0 else ("thin" if n < fp.facet.min_sources else "covered")
        facet_rows[fp.facet.id] = {
            "min_sources": fp.facet.min_sources,
            "planned_status": fp.status,
            "planned_papers": len(fp.paper_ids),
            "realized_status": realized if used_model else None,
            "realized_papers": n if used_model else None,
            "realized_unit_ids": ids,
        }
    unmet = [fid for fid, row in facet_rows.items()
             if row["min_sources"] > 0 and row["realized_status"] in ("gap", "thin")]

    # --- targets -------------------------------------------------------------------
    def targets_of(unit_ids: set[str]) -> list[str]:
        return sorted({target_of[u] for u in unit_ids if u in target_of})

    human = [u for u in summary.units if u.evidence_type in ("human_clinical", "human_genetic")]
    human_numeric = [u for u in human if _has_digit(u.quote, u.context_quote)]
    negative = [u for u in summary.units if u.outcome in ("no_effect", "worsened")]
    status = Counter(g.status for g in gates.values())
    chunk_logs = [c for rec in records.values() for c in ((rec.get("extraction") or {}).get("chunks") or [])]
    have_chunks = extract_outcomes is not None
    graded = sum(1 for r in ranked if r.grade is not None)
    selected_for_grade = (
        sum(1 for why in grading_reasons.values() if _selected(why)) if grading_reasons is not None else None
    )

    counts: dict[str, Any] = {
        "papers_collected": len(papers),
        "full_text_available": sum(1 for p in papers if p.has_full_text),
        "gate": dict(sorted(status.items())),
        "grading_selected": selected_for_grade,
        "graded": graded,
        "grading_failed": (selected_for_grade - graded) if (selected_for_grade is not None and used_model) else None,
        "qualified": len(qualified),
        "read_selected": len(plan.jobs),
        "unread_qualified": len(qualified - set(read)),
        "chunks_planned": sum(len((o or {}).get("planned") or []) for o in (extract_outcomes or {}).values())
        if have_chunks else None,
        "chunks_attempted": len(chunk_logs) if have_chunks else None,
        "chunks_failed": sum(1 for c in chunk_logs if c["status"] == "failed") if have_chunks else None,
        "findings_proposed": (extract_stats or {}).get("findings") if extract_stats is not None else None,
        "units_accepted": len(summary.units),
        "target_bearing_units": len(target_of),
        "units_sent_to_synthesis": len(sent) if section_inputs is not None else None,
        "sentences_over_limit": (sentence_stats or {}).get("over_limit", 0) if sentence_stats is not None else None,
        "units_cited_after_guard": len(cited_all),
        "units_cited_only_by_dropped": len(dropped_all - cited_all),
        "targets_in_evidence_table": len(summary.targets),
        "targets_in_narrative": len(targets_of(cited_all)),
        "targets_in_answer": len(targets_of(cited_answer)),
        "negative_units": len(negative),
        "negative_units_cited": sum(1 for u in negative if u.id in cited_all),
        "human_units": len(human),
        "human_numeric_units": len(human_numeric),
        "human_numeric_units_in_answer": sum(1 for u in human_numeric if u.id in cited_answer),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "question_id": spec.id,
        "corpus_manifest_hash": corpus_hash,
        "selection_policy": {"knobs": knobs, "sha256": policy_fingerprint(knobs)},
        "acquisition": acquisition or {"status": "not_recorded",
                                       "why": "papers came from a frozen corpus or an earlier collection"},
        "used_model": used_model,
        "counts": counts,
        "denominators": DENOMINATORS,
        "dispositions": dict(sorted(Counter(r["disposition"] for r in records.values()).items())),
        "papers": records,
        "sections": {
            sid: {"given": ids, "cited": sorted(set(ids) & cited_all)}
            for sid, ids in sorted((section_inputs or {}).items())
        } if section_inputs is not None else None,
        "facets": facet_rows,
        "targets": {
            "evidence_table": [t.target for t in summary.targets],
            "narrative": targets_of(cited_all),
            "answer": targets_of(cited_answer),
        },
        "stop_reason": stop_reason,
        "unmet_requirements": unmet if used_model else None,
        "committed_batches": [],
    }


def _selected(reason: str) -> bool:
    """Was a paper chosen for a grade? (`lexical_top_40`, `reserved:<lane>`, `lexical_fill`)"""
    return reason.startswith(("lexical_top", "reserved:", "lexical_fill"))


def _disposition(p, gate, r, rec, *, used_model, graded_any, min_grade, out, sent, cited) -> str:
    """The furthest stage a paper reached, as one word - its loss reason."""
    if gate is None:
        return "not_gated"
    if gate.status != "pass":
        return f"gate_{gate.status}"
    if r is not None and r.study_kind == "review":
        return "graded_as_review"
    grading = rec.get("grading") or {}
    reason = grading.get("selected_reason")
    if reason is not None and reason.startswith("excluded_nonprimary"):
        return "not_graded_nonprimary"
    if graded_any and (r is None or r.grade is None):
        if reason is not None and reason.startswith(("below_lexical_cutoff", "below_budget")):
            return "not_graded_budget"
        return "grading_failed" if used_model else "not_graded"
    if r is not None and r.grade is not None and r.grade < min_grade:
        return "graded_below_min"
    if not rec["qualified"]:
        return "not_qualified"
    if rec["read"] is None:
        return "qualified_not_read"
    if not used_model:
        return "read_not_extracted"
    if out is not None:
        chunks = out.get("chunks") or []
        if out.get("review_discarded"):
            return "read_discarded_as_review"
        if chunks and all(c["status"] == "failed" for c in chunks):
            return "read_extraction_failed"
    units = rec["unit_ids"]
    if not units:
        return "read_no_verified_units"
    if sent is not None and not any(u in sent for u in units):
        return "units_not_given_to_writer"
    if not any(u in cited for u in units):
        return "units_not_cited"
    return "cited"
