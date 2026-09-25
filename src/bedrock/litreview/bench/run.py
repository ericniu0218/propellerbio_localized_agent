"""Run the benchmark for one question and write its scorecard.

    python -m bedrock.litreview.bench run evals/litreview/questions/psoriasis_prevention.yaml --models laptop

Reads the frozen corpus and references from evals/litreview/corpora/<id>/ (built once
with `build-corpus`), runs the pipeline and both baselines, scores all three
the same way, and writes evals/litreview/results/<id>/<timestamp>/ plus one line in
evals/litreview/results/history.jsonl so scores can be tracked across prompt versions.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from bedrock.litreview.bench.baselines import closed_book, naive_rag
from bedrock.litreview.bench.claims import pipeline_claims, text_claims
from bedrock.litreview.bench.distractors import leak_test, make_distractors
from bedrock.litreview.bench.judge import CorpusIndex, Judge
from bedrock.litreview.bench.models import (
    COVERAGE_SCORE,
    SUPPORT_SCORE,
    BenchQuestion,
    Claim,
    ClaimResult,
    Nugget,
    Scorecard,
    SystemScore,
)
from bedrock.litreview.bench.perturb import perturbations
from bedrock.litreview.bench.references import load_reference_targets, reachable_targets, target_recall
from bedrock.litreview.config import EVALS_DIR, Settings
from bedrock.litreview.db import Database
from bedrock.litreview.extract.verify import verify_quote
from bedrock.litreview.facets import facets_for, subject_label
from bedrock.litreview.knowledge import TermMatcher, load_knowledge
from bedrock.litreview.llm import LLMError, version_of
from bedrock.litreview.models import EvidenceUnit, Paper, QuestionSpec, Report
from bedrock.litreview.report.guard import CitationGuard, numbers_in, unit_text
from bedrock.litreview.runner import RunResult, answer_question, load_result
from bedrock.litreview.sources.corpus import read_papers

log = logging.getLogger(__name__)

# `evals/litreview/` is a source checkout only - it is not in the wheel - so the
# benchmark is the one part of the package that does not run from an install.
CORPORA = EVALS_DIR / "corpora"
RESULTS = EVALS_DIR / "results"
MAX_NUGGETS = 20


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _near_misses(text: str, spec: QuestionSpec) -> list[str]:
    found: set[str] = set()
    for e in spec.required_entities:
        matcher = TermMatcher(e.exact_terms, e.near_miss_terms, e.near_miss_patterns)
        found |= {m.text for m in matcher.matches(text) if m.kind == "near_miss"}
    return sorted(found)


def _unit_evidence(u: EvidenceUnit) -> str:
    """What the judge is shown for a cited unit: the paper's words only. It
    used to see the extraction model's "Result" and "subjects" too, and judged
    "in human tissues" supported because the model had written "human"."""
    return f"{u.quote} (Context: {u.context_quote})" if u.context_quote else u.quote


def pipeline_answer_text(report: Report) -> str:
    parts = []
    for section in report.sections:
        for s in section.sentences:
            if s.kept:
                parts.append(s.text)
        for sub in section.subsections:
            kept = [s.text for s in sub.sentences if s.kept]
            if kept:
                parts.append(f"{sub.title}: " + " ".join(kept))
    return "\n".join(parts)


def score_pipeline_claims(result: RunResult, judge: Judge, index: CorpusIndex, say) -> list[ClaimResult]:
    report, spec = result.report, result.spec
    units = {report.short_ids[u.id]: u for u in report.evidence}
    labels = {p.id: f"{p.signals.year or ''}" for p in result.papers}
    out = []
    claims = pipeline_claims(report)
    for i, claim in enumerate(claims, 1):
        cited = [units[c] for c in claim.citations if c in units]
        verdict = judge.support(claim.text, [_unit_evidence(u) for u in cited])
        blob = " || ".join(unit_text(u, labels.get(u.paper_id, "")) for u in cited)
        claim_numbers = numbers_in(claim.text)
        cr = ClaimResult(
            claim=claim,
            cited_support=verdict.verdict if verdict else None,
            cited_reasoning=verdict.reasoning if verdict else None,
            numbers_ok=(claim_numbers <= numbers_in(blob) | numbers_in(spec.text)) if claim_numbers else None,
            near_misses=_near_misses(claim.text, spec),
        )
        if cr.cited_support in ("supported", "partially_supported"):
            cr.corpus_support = cr.cited_support
        else:
            v = judge.support(claim.text, [t for _, t in index.top(claim.text)])
            cr.corpus_support = v.verdict if v else None
        out.append(cr)
        if i % 10 == 0:
            say(f"  judged {i}/{len(claims)} pipeline claims")
    return out


def score_text_claims(claims: list[Claim], papers: dict[str, Paper], spec: QuestionSpec, judge: Judge, index: CorpusIndex) -> list[ClaimResult]:
    out = []
    for claim in claims:
        cr = ClaimResult(claim=claim, near_misses=_near_misses(claim.text, spec))
        cited_text = [papers[c].head_text() for c in claim.citations if c in papers]
        if cited_text:
            v = judge.support(claim.text, cited_text)
            cr.cited_support = v.verdict if v else None
            cr.cited_reasoning = v.reasoning if v else None
        passages = [t for _, t in index.top(claim.text)]
        if cr.cited_support in ("supported", "partially_supported"):
            cr.corpus_support = cr.cited_support
        else:
            v = judge.support(claim.text, passages)
            cr.corpus_support = v.verdict if v else None
        claim_numbers = numbers_in(claim.text)
        if claim_numbers:
            pool = " ".join(cited_text or passages)
            cr.numbers_ok = claim_numbers <= numbers_in(pool) | numbers_in(spec.text)
        out.append(cr)
    return out


def summarise_system(system: str, results: list[ClaimResult], cited_applicable: bool) -> SystemScore:
    cited = [SUPPORT_SCORE[r.cited_support] for r in results if r.cited_support is not None]
    if cited_applicable:
        # A claim with no citation at all is, for citation accuracy, unsupported.
        cited += [0.0 for r in results if r.cited_support is None and not r.claim.citations]
    corpus = [SUPPORT_SCORE[r.corpus_support] for r in results if r.corpus_support is not None]
    numeric = [1.0 if r.numbers_ok else 0.0 for r in results if r.numbers_ok is not None]
    quantitative = sum(
        1 for r in results
        if r.numbers_ok and (r.cited_support == "supported" or (r.cited_support is None and r.corpus_support == "supported"))
    )
    sources = {c for r in results for c in r.claim.citations}
    return SystemScore(
        system=system,
        claims=len(results),
        cited_support=_mean(cited) if cited_applicable else None,
        corpus_grounded=_mean(corpus),
        numeric_fidelity=_mean(numeric),
        near_miss_claims=sum(1 for r in results if r.near_misses),
        quantitative_supported_claims=quantitative,
        distinct_sources_cited=len(sources),
    )


def overall(score: SystemScore) -> float | None:
    """0-100. Weighted: accuracy .35, groundedness .20, completeness .30, exactness .15.
    Missing components are left out and the weights renormalised."""
    accuracy = score.cited_support if score.cited_support is not None else score.corpus_grounded
    completeness_parts = [x for x in (score.nugget_recall, score.target_recall) if x is not None]
    completeness = sum(completeness_parts) / len(completeness_parts) if completeness_parts else None
    exactness = 1 - score.near_miss_claims / score.claims if score.claims else None
    parts = [(accuracy, 0.35), (score.corpus_grounded, 0.20), (completeness, 0.30), (exactness, 0.15)]
    parts = [(v, w) for v, w in parts if v is not None]
    if not parts:
        return None
    return round(100 * sum(v * w for v, w in parts) / sum(w for _, w in parts), 1)


def evaluate(
    bq: BenchQuestion,
    *,
    settings: Settings,
    client,
    corpus_dir: Path | None = None,
    results_root: Path | None = None,
    skip_baselines: bool = False,
    skip_distractors: bool = False,
    skip_perturb: bool = False,
    from_run: str | None = None,
    judge_client=None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Scorecard, Path]:
    """Run the pipeline and the baselines on a frozen corpus and score them.

    `from_run` scores a stored report instead of producing a new one — the
    way to put a report written by earlier code beside one written by the
    current code. The ranking-dependent checks (distractors, the determinism
    re-run) are skipped then, because they would test today's code.

    `judge_client` judges with something other than the model under test. The
    pipeline always uses `client`; only the judging step changes, so a weak
    local model can be measured by a stronger judge without the report itself
    being touched by it.
    """
    say = progress or (lambda msg: log.info(msg))
    corpus_dir = Path(corpus_dir or CORPORA / bq.id)
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(results_root or RESULTS) / bq.id / created
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = load_knowledge()

    papers = read_papers(corpus_dir / "papers.jsonl")
    reviews = read_papers(corpus_dir / "references.jsonl") if (corpus_dir / "references.jsonl").exists() else []
    ref_targets = load_reference_targets(corpus_dir / "reference_targets.json")
    db = Database(settings.data_dir / "bench.sqlite3")

    # --- the pipeline -------------------------------------------------------------
    if from_run:
        say(f"[{bq.id}] scoring the stored report of run {from_run}")
        result = load_result(db, from_run)
        skip_distractors = True
    else:
        say(f"[{bq.id}] pipeline on {len(papers)} papers")
        result = answer_question(
            bq.question, settings=settings, client=client, papers=papers, db=db,
            out_dir=out_dir / "pipeline", progress=lambda m: say(f"  {m}"),
        )
    report, spec = result.report, result.spec
    by_id = {p.id: p for p in result.papers}
    passing = [p for p in result.papers if result.gates[p.id].status == "pass"]
    index = CorpusIndex(result.papers)
    judge = Judge(judge_client or client, settings.model_for("judge"))
    notes: list[str] = []

    # --- claims -------------------------------------------------------------------
    say(f"[{bq.id}] judging pipeline claims")
    claim_results: dict[str, list[ClaimResult]] = {"pipeline": score_pipeline_claims(result, judge, index, say)}
    answers: dict[str, str] = {"pipeline": pipeline_answer_text(report)}

    if not skip_baselines:
        say(f"[{bq.id}] baselines")
        model = settings.model_for("baseline")
        (out_dir / "baselines").mkdir(exist_ok=True)
        # A baseline that fails is reported, not fatal: an hour of pipeline
        # scoring must not be lost to one chatbot answer.
        try:
            cb = closed_book(client, model, bq.question)
        except LLMError as exc:
            cb = None
            notes.append(f"Closed-book baseline failed and is not scored: {exc}")
        try:
            rag_text, id_map = naive_rag(client, model, bq.question, result.papers)
        except LLMError as exc:
            rag_text, id_map = None, {}
            notes.append(f"Naive-RAG baseline failed and is not scored: {exc}")
        say(f"[{bq.id}] judging baseline claims")
        if cb is not None:
            answers["closed_book"] = cb
            (out_dir / "baselines" / "closed_book.md").write_text(cb, encoding="utf-8")
            claim_results["closed_book"] = score_text_claims(text_claims(cb, "closed_book"), by_id, spec, judge, index)
        if rag_text is not None:
            answers["naive_rag"] = rag_text
            (out_dir / "baselines" / "naive_rag.md").write_text(
                rag_text + "\n\n---\n" + "\n".join(f"{k}: {by_id[v].title}" for k, v in id_map.items()),
                encoding="utf-8",
            )
            claim_results["naive_rag"] = score_text_claims(
                text_claims(rag_text, "naive_rag", id_map), by_id, spec, judge, index
            )

    systems = {
        name: summarise_system(name, rows, cited_applicable=(name != "closed_book"))
        for name, rows in claim_results.items()
    }

    # --- completeness: review nuggets -------------------------------------------------
    nuggets: list[Nugget] = []
    if reviews:
        say(f"[{bq.id}] nuggets from {min(len(reviews), bq.references.reviews_max)} reviews")
        for review in reviews[: bq.references.reviews_max]:
            if review.abstract:
                nuggets += judge.nuggets(bq.question, subject_label(spec), f"review:{review.id}", review.abstract)
        nuggets.sort(key=lambda n: (n.importance != "vital", n.source, n.id))
        nuggets = nuggets[:MAX_NUGGETS]
        for n in nuggets:
            v = judge.support(n.fact, [t for _, t in index.top(n.fact)])
            n.reachable = bool(v and v.verdict in ("supported", "partially_supported"))
            if not n.reachable:
                continue
            for name, text in answers.items():
                c = judge.covered(n.fact, text)
                if c:
                    n.coverage[name] = c.verdict
        reachable = [n for n in nuggets if n.reachable]
        for name, score in systems.items():
            covered = [COVERAGE_SCORE[n.coverage[name]] for n in reachable if name in n.coverage]
            vital = [COVERAGE_SCORE[n.coverage[name]] for n in reachable if name in n.coverage and n.importance == "vital"]
            score.nugget_recall = _mean(covered)
            score.vital_nugget_recall = _mean(vital)
        notes.append(f"{len(nuggets)} reference facts from reviews; {len(reachable)} supported by the collected papers.")
    else:
        notes.append("No review references for this question; nugget recall not measured.")

    # --- completeness: reference targets ------------------------------------------------
    target_detail = {}
    if ref_targets:
        reach = reachable_targets(ref_targets, passing, kb)
        for name, score in systems.items():
            recall, found, missed = target_recall(reach, answers[name], kb)
            score.target_recall = round(recall, 4) if recall is not None else None
            target_detail[name] = {"found": found, "missed": missed}
        raw, _, _ = target_recall(ref_targets, answers["pipeline"], kb)
        notes.append(
            f"{len(ref_targets)} reference targets (Open Targets); {len(reach)} named in the collected papers. "
            f"Pipeline raw recall against all of them: {raw:.2f}." if raw is not None else ""
        )

    for score in systems.values():
        score.overall = overall(score)

    # --- pipeline-only checks -------------------------------------------------------------
    pipeline: dict[str, float | int | str | None] = {}
    papers_all = {p.id: p for p in result.papers}
    # Re-verified independently of extraction: against the section the unit
    # says it came from, then the whole paper. `all_text()` alone leaves out
    # `sections["abstract"]` when it differs from `Paper.abstract` (the JATS
    # Highlights case, audit step 14), which failed quotes that are verbatim
    # in the paper.
    def _valid(u: EvidenceUnit) -> bool:
        paper = papers_all[u.paper_id]
        return any(verify_quote(u.quote, text) != "failed"
                   for text in (paper.sections.get(u.section, ""), paper.all_text()) if text)

    valid = [_valid(u) for u in report.evidence]
    pipeline["quote_validity"] = _mean([1.0 if v else 0.0 for v in valid]) if valid else None
    kept = report.kept_sentences()
    pipeline["citation_coverage"] = _mean([1.0 if s.citations else 0.0 for _, s in kept]) if kept else None
    pipeline["sentences_kept"] = report.stats.get("sentences_kept", 0)
    pipeline["sentences_dropped"] = report.stats.get("sentences_dropped", 0)
    pipeline["evidence_units"] = len(report.evidence)
    pipeline["papers_read"] = report.stats.get("papers_read", 0)
    cited_papers = {u.paper_id for u in report.evidence}
    pipeline["cited_papers_passing_gate"] = _mean(
        [1.0 if result.gates[p].status == "pass" else 0.0 for p in cited_papers]
    ) if cited_papers else None

    distractor_result = {}
    if not skip_distractors:
        say(f"[{bq.id}] distractor papers")
        distractors = make_distractors(by_id, result.ranked, spec, k=5)
        distractor_result = leak_test(
            distractors, spec, facets_for(spec), client, model=settings.model_for("relevance"),
            min_grade=int(settings.knob("plan.min_grade", 2)),
        )
        pipeline["distractor_leak_rate"] = distractor_result.get("leak_rate")

    perturb_rows = []
    if not skip_perturb:
        say(f"[{bq.id}] judge sensitivity")
        units = {report.short_ids[u.id]: u for u in report.evidence}
        labels = {p.id: str(p.signals.year or "") for p in result.papers}
        supported = [r for r in claim_results["pipeline"] if r.cited_support == "supported"][:6]
        for r in supported:
            cited = [units[c] for c in r.claim.citations if c in units]
            guard = CitationGuard(spec, kb, {c: units[c] for c in r.claim.citations if c in units},
                                  paper_labels=labels,
                                  vocabulary=[v for u in report.evidence for v in (u.target_raw, u.intervention or "")])
            for kind, text in perturbations(r.claim.text, spec):
                v = judge.support(text, [_unit_evidence(u) for u in cited])
                perturb_rows.append({
                    "kind": kind, "original": r.claim.text, "perturbed": text,
                    "guard_caught": not guard.check(text, r.claim.citations).kept,
                    # Caught means the judge would not pass the corrupted claim as supported.
                    "judge_caught": bool(v and v.verdict != "supported"),
                    "judge_verdict": v.verdict if v else None,
                })
        by_kind = Counter(row["kind"] for row in perturb_rows)
        for kind in ("number", "flip", "entity"):
            rows = [row for row in perturb_rows if row["kind"] == kind]
            if rows:
                pipeline[f"judge_catches_{kind}"] = _mean([1.0 if x["judge_caught"] else 0.0 for x in rows])
                pipeline[f"guard_or_judge_catches_{kind}"] = _mean(
                    [1.0 if (x["judge_caught"] or x["guard_caught"]) else 0.0 for x in rows])
        pipeline["perturbations"] = sum(by_kind.values())

    if from_run:
        pipeline["deterministic"] = "not re-run"
        notes.append(f"Scored the stored report of run {from_run}; distractor and determinism checks not re-run.")
    else:
        say(f"[{bq.id}] determinism re-run")
        again = answer_question(
            bq.question, settings=settings, client=client, papers=papers, db=db,
            out_dir=out_dir / "pipeline_rerun", progress=lambda m: None,
        )
        pipeline["deterministic"] = "yes" if again.report_id == result.report_id else "no"

    gates = {
        "quotes all verified (1.00)": pipeline.get("quote_validity") in (None, 1.0),
        "every sentence cited": pipeline.get("citation_coverage") in (None, 1.0),
        "no near-miss claims": systems["pipeline"].near_miss_claims == 0,
        "numbers all in cited evidence": systems["pipeline"].numeric_fidelity in (None, 1.0),
        "no distractor leaked": pipeline.get("distractor_leak_rate") in (None, 0.0),
        "same report twice": pipeline["deterministic"] == "yes",
        "integrity check passes": not result.report.integrity,
        "claims supported by citations >= 0.85": (systems["pipeline"].cited_support or 0) >= 0.85,
        "judge catches flipped effects >= 0.7 (benchmark validity)": (
            pipeline.get("judge_catches_flip") is None or (pipeline.get("judge_catches_flip") or 0) >= 0.7
        ),
    }
    if from_run:
        gates.pop("same report twice")
    if judge_client is None:
        notes.append(
            f"The judge is {settings.model_for('judge')}, the same model family that wrote the report; "
            "its sensitivity is measured by the perturbation rows."
        )
    else:
        notes.append(
            f"The judge is {version_of(judge_client, settings.model_for('judge'))}, not the model that wrote "
            "the report. Its answers are "
            "checked by the same code as any model's: the sentence it cites must be in the evidence, and "
            "the claim's names and numbers must be too."
        )
    unanswered = getattr(judge_client, "misses", 0)
    if judge_client is not None:
        pending = judge_client.flush()
    if unanswered:
        notes.append(
            f"{unanswered} judging question(s) had no answer on file and were scored as unjudged"
            + (f"; they are listed in {pending}." if pending else ".")
        )
    notes.append(
        f"{judge.verbatim} support verdict(s), across all systems, were settled in code because the claim "
        f"is a verbatim passage of its evidence; the judge model made {judge.calls} call(s). In "
        f"{judge.located_by_code} the model's sentence was not in the evidence and code found the "
        f"closest one; {judge.struck} detail(s) it called missing were in the evidence and struck off."
    )

    model_versions = dict(result.run.model_versions)
    model_versions["judge"] = version_of(judge_client or client, settings.model_for("judge"))
    card = Scorecard(
        question_id=bq.id, question=bq.question, report_id=result.report_id, created=created,
        model_versions=model_versions, systems=list(systems.values()), pipeline=pipeline, gates=gates,
        notes=[n for n in notes if n],
    )

    # --- write everything -----------------------------------------------------------------
    (out_dir / "scorecard.json").write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with (out_dir / "claims.jsonl").open("w", encoding="utf-8") as fh:
        for rows in claim_results.values():
            for r in rows:
                fh.write(r.model_dump_json() + "\n")
    (out_dir / "nuggets.json").write_text(json.dumps([n.model_dump() for n in nuggets], indent=2), encoding="utf-8")
    (out_dir / "targets.json").write_text(json.dumps(target_detail, indent=2), encoding="utf-8")
    (out_dir / "distractors.json").write_text(json.dumps(distractor_result, indent=2), encoding="utf-8")
    (out_dir / "perturbations.json").write_text(json.dumps(perturb_rows, indent=2), encoding="utf-8")
    (out_dir / "scorecard.md").write_text(
        render_scorecard(card, claim_results, nuggets, target_detail, report), encoding="utf-8"
    )
    history = Path(results_root or RESULTS) / "history.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "question_id": bq.id, "created": created, "report_id": result.report_id,
            "prompt_versions": result.run.prompt_versions, "model_versions": model_versions,
            "overall": {s.system: s.overall for s in card.systems},
            "gates_passed": sum(gates.values()), "gates_total": len(gates),
        }) + "\n")
    db.add_bench_result(result.report_id, spec.id, created, card.model_dump_json())
    say(f"[{bq.id}] judge calls: {judge.calls}; scorecard: {out_dir / 'scorecard.md'}")
    return card, out_dir


def _fmt(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def render_scorecard(card: Scorecard, claims: dict[str, list[ClaimResult]], nuggets: list[Nugget],
                     targets: dict, report: Report) -> str:
    names = {"pipeline": "This pipeline", "naive_rag": "Chatbot + pasted abstracts", "closed_book": "Chatbot, no papers"}
    rows = [
        f"# Benchmark — {card.question}",
        "",
        f"Question id `{card.question_id}` · report `{card.report_id}` · {card.created}",
        "",
        "| Metric | " + " | ".join(names.get(s.system, s.system) for s in card.systems) + " |",
        "|---|" + "---|" * len(card.systems),
    ]
    metrics = [
        ("**Overall (0–100)**", "overall"),
        ("Claims", "claims"),
        ("Accuracy: claims supported by their citations", "cited_support"),
        ("Groundedness: claims supported by the collected papers", "corpus_grounded"),
        ("Numbers present in the evidence", "numeric_fidelity"),
        ("Claims mentioning a near-miss entity", "near_miss_claims"),
        ("Completeness: review facts covered (reachable)", "nugget_recall"),
        ("Completeness: vital review facts covered", "vital_nugget_recall"),
        ("Completeness: reference targets named (reachable)", "target_recall"),
        ("Detail: supported claims with numbers", "quantitative_supported_claims"),
        ("Distinct sources cited", "distinct_sources_cited"),
    ]
    for label, attr in metrics:
        rows.append(f"| {label} | " + " | ".join(_fmt(getattr(s, attr)) for s in card.systems) + " |")

    rows += ["", "## Pipeline checks", "", "| Check | Result |", "|---|---|"]
    for gate, ok in card.gates.items():
        rows.append(f"| {gate} | {'PASS' if ok else '**FAIL**'} |")
    rows += ["", "| Measure | Value |", "|---|---|"]
    for k, v in card.pipeline.items():
        rows.append(f"| {k} | {_fmt(v)} |")

    rows += ["", "## Notes", ""] + [f"- {n}" for n in card.notes]

    rows += ["", "## Where the pipeline lost points", ""]
    bad = [r for r in claims.get("pipeline", []) if r.cited_support in ("unsupported", "contradicted", "partially_supported")]
    if bad:
        rows.append("Claims the judge did not find fully supported by their citations:")
        rows.append("")
        for r in bad:
            rows.append(f"- [{r.cited_support}] ({r.claim.section}) {r.claim.text} — *{r.cited_reasoning or ''}*")
    else:
        rows.append("Every pipeline claim was judged supported by its citations.")
    missed = [n for n in nuggets if n.reachable and n.coverage.get("pipeline") == "not_covered"]
    if missed:
        rows += ["", "Reference facts the collected papers support but the report does not state:", ""]
        rows += [f"- ({n.importance}) {n.fact}" for n in missed]
    if targets.get("pipeline", {}).get("missed"):
        rows += ["", "Reference targets named in the collected papers but not in the report: "
                 + ", ".join(targets["pipeline"]["missed"])]
    dropped = Counter(
        (s.drop_reason or "").split(",")[0].split("'")[0].strip()
        for sec in report.sections for s in [*sec.sentences, *(x for sub in sec.subsections for x in sub.sentences)]
        if not s.kept
    )
    if dropped:
        rows += ["", "Sentences removed by the citation guard, by reason: "
                 + "; ".join(f"{reason or 'other'} ×{n}" for reason, n in dropped.most_common())]
    for name in ("naive_rag", "closed_book"):
        leaks = [r for r in claims.get(name, []) if r.near_misses]
        ungrounded = [r for r in claims.get(name, []) if r.corpus_support in ("unsupported", "contradicted")]
        if leaks or ungrounded:
            rows += ["", f"## {names[name]}: examples of failures", ""]
            for r in ungrounded[:5]:
                rows.append(f"- not in any collected paper: {r.claim.text}")
            for r in leaks[:3]:
                rows.append(f"- mentions a near-miss ({', '.join(r.near_misses)}): {r.claim.text}")
    return "\n".join(rows) + "\n"
