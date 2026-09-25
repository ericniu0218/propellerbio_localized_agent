"""Build a frozen sample corpus for a benchmark question. Needs the network.

It runs the question's own hand-written Europe PMC queries (fixed in
evals/litreview/questions/*.yaml, so a rebuilt corpus asks exactly what the last one
asked) through the searcher's Europe PMC client, fetches open-access full
text as JATS XML, adds a few papers about near-miss entities (a searcher will
return those too, and the gate has to cope), and freezes everything to
evals/litreview/corpora/<id>/papers.jsonl. It also freezes the external references:
review abstracts, and Open Targets' clinically tested targets for the disease.

With a `searcher` (`build-corpus --searcher`), the main papers come from the
paper searcher instead of the hand-written queries — the corpus the pipeline
would really be handed — while the near-miss papers and references are built
as before, so the two kinds of corpus are scored on the same terms.

Run it on a machine that is allowed out; the benchmark itself then runs
offline against the frozen files. Only open-access text is fetched.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import httpx

from bedrock.litreview.bench.models import BenchQuestion
from bedrock.litreview.ingest import ingest
from bedrock.litreview.models import Paper
from bedrock.litreview.sources.base import PaperSearcher
from bedrock.litreview.sources.corpus import write_papers
from bedrock.litreview.sources.europepmc import EuropePMC, paper_from_hit

log = logging.getLogger(__name__)

OPEN_TARGETS = "https://api.platform.opentargets.org/api/v4/graphql"

_STAGE = {
    "APPROVAL": 4.0, "PHASE_4": 4.0, "PHASE_3": 3.0, "PHASE_2_3": 2.5, "PHASE_2": 2.0,
    "PHASE_1_2": 1.5, "PHASE_1": 1.0, "EARLY_PHASE_1": 0.5, "UNKNOWN": 0.0,
}


def open_targets_known_targets(disease_id: str, min_phase: float = 3) -> list[dict]:
    """Human targets of drugs that reached at least `min_phase` for this disease.

    Open Targets' `drugAndClinicalCandidates` (it replaced `knownDrugs` in
    2025); disease ids are MONDO/EFO as the platform's search returns them.
    """
    query = """
    query Candidates($id: String!) {
      disease(efoId: $id) {
        id name
        drugAndClinicalCandidates {
          rows { maxClinicalStage drug { name mechanismsOfAction { rows { targets { approvedSymbol } } } } }
        }
      }
    }"""
    resp = httpx.post(OPEN_TARGETS, json={"query": query, "variables": {"id": disease_id}}, timeout=90.0)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"Open Targets: {data['errors'][0].get('message')}")
    disease = (data.get("data") or {}).get("disease")
    if disease is None:
        raise RuntimeError(f"Open Targets has no disease {disease_id}")
    by_symbol: dict[str, dict] = {}
    for row in (disease.get("drugAndClinicalCandidates") or {}).get("rows", []):
        phase = _STAGE.get(row.get("maxClinicalStage") or "UNKNOWN", 0.0)
        drug = row.get("drug") or {}
        if phase < min_phase:
            continue
        for moa in (drug.get("mechanismsOfAction") or {}).get("rows", []):
            for target in moa.get("targets") or []:
                symbol = target.get("approvedSymbol")
                if not symbol:
                    continue
                entry = by_symbol.setdefault(
                    symbol, {"symbol": symbol, "max_phase": 0.0, "drugs": [], "source": f"Open Targets {disease_id}"}
                )
                entry["max_phase"] = max(entry["max_phase"], phase)
                if drug.get("name") and drug["name"] not in entry["drugs"]:
                    entry["drugs"].append(drug["name"])
    return sorted(by_symbol.values(), key=lambda e: (-e["max_phase"], e["symbol"]))


def build_corpus(
    bq: BenchQuestion, out_dir: Path, *, searcher: PaperSearcher | None = None, progress=print,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    epmc = EuropePMC()
    papers: list[Paper] = []
    seen: set[str] = set()
    queries: list[str] = list(bq.corpus.queries)

    def add(hits: list[dict], query: str) -> int:
        n = 0
        for hit in hits:
            paper = paper_from_hit(hit, query)
            if paper.id in seen or not paper.title:
                continue
            seen.add(paper.id)
            papers.append(paper)
            n += 1
        return n

    if searcher is not None:
        from bedrock.litreview.knowledge import load_knowledge
        from bedrock.litreview.stages.s0_question import parse_question
        from bedrock.litreview.stages.s1_collect import search_request

        spec = parse_question(bq.question, None, model=None, knowledge=load_knowledge())
        for paper in searcher.search(search_request(spec)):
            if paper.id not in seen and paper.title:
                seen.add(paper.id)
                papers.append(paper)
        report = getattr(searcher, "last_report", None)
        queries = [q for qs in (report.queries.values() if report else []) for q in qs]
        progress(f"  searcher {searcher.name} -> {len(papers)} papers")
    else:
        for q in bq.corpus.queries:
            n = add(epmc.search(q, bq.corpus.max_per_query), q)
            progress(f"  query -> {n} new papers: {q[:90]}")
    for q in bq.corpus.near_miss_queries:
        n = add(epmc.search(q, bq.corpus.near_miss_max), f"near_miss:{q}")
        progress(f"  near-miss query -> {n} new papers: {q[:90]}")

    fetched = sum(1 for p in papers if p.full_text)
    for paper in papers:
        if fetched >= bq.corpus.full_text_max:
            break
        if paper.pmcid and paper.signals.open_access and not paper.full_text:
            xml = epmc.full_text(paper.pmcid)
            if xml:
                paper.full_text, paper.full_text_format = xml, "jats"
                fetched += 1
    papers = ingest(papers)
    write_papers(papers, out_dir / "papers.jsonl")
    progress(f"  {len(papers)} papers, {sum(1 for p in papers if p.has_full_text)} with full text")

    reviews: list[Paper] = []
    if bq.references.reviews_query:
        hits = epmc.search(bq.references.reviews_query, bq.references.reviews_max * 3)
        for hit in hits:
            paper = paper_from_hit(hit, "reference_review")
            if paper.abstract and len(paper.abstract) > 400 and paper.id not in seen:
                reviews.append(paper)
            if len(reviews) >= bq.references.reviews_max:
                break
        write_papers(reviews, out_dir / "references.jsonl")
        progress(f"  {len(reviews)} reference reviews")

    targets: list[dict] = []
    if bq.references.open_targets_disease:
        try:
            targets = open_targets_known_targets(bq.references.open_targets_disease, bq.references.open_targets_min_phase)
        except (httpx.HTTPError, RuntimeError) as exc:
            progress(f"  Open Targets unavailable: {exc}")
        (out_dir / "reference_targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
        progress(f"  {len(targets)} reference targets")

    manifest = {
        "question_id": bq.id, "built": date.today().isoformat(), "papers": len(papers),
        "full_text": sum(1 for p in papers if p.has_full_text), "reviews": len(reviews),
        "reference_targets": len(targets), "queries": queries,
        "searcher": searcher.name if searcher is not None else "bench-queries",
        "near_miss_queries": bq.corpus.near_miss_queries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
