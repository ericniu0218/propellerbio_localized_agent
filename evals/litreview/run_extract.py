"""The extraction scoreboard. Run this constantly.

    python -m evals.litreview.run_extract [--goldset PATH] [--models laptop] [--holdout]

Without a number, "the extraction looks pretty good" is the best anyone can
say, and three weeks later a scientist finds a mouse study reported as a human
trial and nobody can tell whether that is rare or everywhere.

The per-field table is the part that tells you what to *do*. An overall score
of 0.84 is not actionable. "evidence_type 0.78" is — it means the model is
confusing in vitro with in vivo, which is a prompt fix you can make in ten
minutes and re-measure immediately.

Two rules for using this:

Fix patterns, not instances. One paper failing oddly is noise. Five failing the
same way is a prompt change.

Never tune against the whole gold set. Five of the twenty papers are held back
(`holdout: true`) and must not be looked at until the end, or you will write a
prompt that is excellent at fifteen specific papers and mediocre at everything
else. By default this scores the tuning papers only; `--holdout` scores the
held-back ones, once, at the end.

It refuses to score a gold set that is not finished (`goldset.load_goldset`):
a placeholder left from the template, an entry with no findings that does not
say `"none_expected": true`, a kind code proposed that nobody confirmed, a
quote that is not in the paper, or any field outside the schema - each named
with its line, its paper and its field.

Each paper goes through Stage 5's own `extract_paper`, with the chunks and
methods context the pipeline would send (`goldset.read_by_pipeline`), through
bedrock's scheduler after `service.preflight`, like every model call.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

from bedrock.litreview.config import ConfigError, Settings, load_settings
from bedrock.litreview.extract.verify import normalize as normalize_text
from bedrock.litreview.extract.verify import verify_quote
from bedrock.litreview.facets import facets_for
from bedrock.litreview.knowledge import Knowledge, load_knowledge
from bedrock.litreview.models import EvidenceUnit, Paper
from bedrock.litreview.stages.s5_extract import extract_paper
from evals.litreview.goldset import (
    GoldFinding,
    GoldPaper,
    GoldsetError,
    extraction_job,
    load_goldset,
    read_corpus,
    read_question,
)

GOLDSET = Path(__file__).parent / "goldset.jsonl"

# The fields we score. `quote` is excluded because it is verified mechanically
# rather than compared, and `effect_detail` because free text will not match
# exactly and scoring it as a string would be meaningless.
SCORED_FIELDS = ("target", "direction", "disease", "evidence_type", "species", "outcome", "n")

SAME_SENTENCE = 0.8
"""Share of the shorter sentence's words the other must contain for an
expected and a predicted finding to be about the same sentence."""


def expected_fields(f: GoldFinding, kb: Knowledge) -> dict:
    """A labelled finding in the form code gives a unit: names normalised by the
    same tables, so "IL-23" and "IL23A" are one target. Species is already one
    of the unit's canonical values - the schema allows nothing else."""
    return {"target": kb.normalize(f.target, "target"), "direction": f.direction,
            "disease": kb.normalize(f.disease, "disease"), "evidence_type": f.evidence_type,
            "species": f.species, "outcome": f.outcome, "n": f.n, "quote": f.quote}


def unit_fields(u: EvidenceUnit) -> dict:
    return {**{field: getattr(u, field) for field in SCORED_FIELDS}, "quote": u.quote}


def _words(text: str) -> set[str]:
    return set(normalize_text(text).lower().split())


def same_sentence(a: str, b: str) -> bool:
    na, nb = normalize_text(a).lower(), normalize_text(b).lower()
    if na in nb or nb in na:
        return True
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= SAME_SENTENCE


def pair(expected: list[dict], predicted: list[dict]) -> list[tuple[dict, dict]]:
    """Expected and predicted findings about the same sentence, one to one.

    Pairing by the sentence, not by the fields being scored: the old key
    (target, direction, disease, evidence type, outcome) paired only findings
    that already agreed on five of the seven scored fields, so their per-field
    accuracy was 1.00 by construction and "evidence_type 0.78" could never
    appear. Two findings from one sentence (two targets) pair by target."""
    free = list(range(len(predicted)))
    pairs = []
    for e in expected:
        same = [i for i in free if same_sentence(e["quote"], predicted[i]["quote"])]
        if not same:
            continue
        best = next((i for i in same if str(predicted[i]["target"]).lower() == str(e["target"]).lower()), same[0])
        free.remove(best)
        pairs.append((e, predicted[best]))
    return pairs


def score(expected: list[dict], predicted: list[dict]) -> dict:
    """Precision, recall, and per-field accuracy over paired findings, for one paper."""
    pairs = pair(expected, predicted)
    field_hits: dict[str, list[int]] = defaultdict(list)
    for e, p in pairs:
        for field in SCORED_FIELDS:
            field_hits[field].append(1 if str(e.get(field)) == str(p.get(field)) else 0)
    paired_e = {id(e) for e, _ in pairs}
    paired_p = {id(p) for _, p in pairs}
    return {
        "matched": len(pairs), "expected": len(expected), "predicted": len(predicted),
        "missed": [e for e in expected if id(e) not in paired_e],
        "spurious": [p for p in predicted if id(p) not in paired_p],
        "field_hits": dict(field_hits),
    }


def combine(per_paper: dict[str, dict]) -> dict:
    """The per-paper scores summed into one table."""
    total = {"matched": 0, "expected": 0, "predicted": 0, "missed": [], "spurious": []}
    hits: dict[str, list[int]] = defaultdict(list)
    for pid, r in per_paper.items():
        for key in ("matched", "expected", "predicted"):
            total[key] += r[key]
        total["missed"] += [{"paper": pid, **_brief(x)} for x in r["missed"]]
        total["spurious"] += [{"paper": pid, **_brief(x)} for x in r["spurious"]]
        for field, values in r["field_hits"].items():
            hits[field] += values
    total["precision"] = total["matched"] / total["predicted"] if total["predicted"] else 0.0
    total["recall"] = total["matched"] / total["expected"] if total["expected"] else 0.0
    total["fields"] = {f: sum(v) / len(v) for f, v in hits.items() if v}
    return total


def _brief(finding: dict) -> dict:
    return {k: finding.get(k) for k in ("target", "outcome", "evidence_type")}


def predict(paper: Paper, entry: GoldPaper, client, settings: Settings, kb: Knowledge) -> list[EvidenceUnit]:
    """What Stage 5 extracts from this paper, as the pipeline would ask it."""
    spec = read_question(entry.question or "")
    facets = facets_for(spec)
    return extract_paper(
        paper, extraction_job(paper, spec, settings), spec, facets, client,
        model=settings.model_for("extract"),
        chunk_chars=int(settings.knob("extract.chunk_chars", 6000)),
        max_chunks=int(settings.knob("extract.max_chunks_per_paper", 6)),
        methods_chars=int(settings.knob("extract.methods_context_chars", 2500)),
        knowledge=kb,
    )


def run(entries: list[GoldPaper], client, settings: Settings) -> dict:
    """Extract from every entry's paper and score it. Returns the combined
    results plus the checks that are not per-field."""
    kb = load_knowledge()
    corpora: dict[str, dict[str, Paper]] = {}
    per_paper, valid, reviews_with_findings, negatives, negatives_ok = {}, [], [], 0, 0
    for entry in entries:
        papers = corpora.setdefault(entry.corpus or "", read_corpus(entry.corpus or ""))
        paper = papers[entry.paper_id]
        units = predict(paper, entry, client, settings, kb)
        texts = [t for t in (paper.title, *paper.sections.values()) if t]
        valid += [any(verify_quote(u.quote, t) != "failed" for t in texts) for u in units]
        per_paper[entry.paper_id] = score([expected_fields(f, kb) for f in entry.findings],
                                          [unit_fields(u) for u in units])
        if entry.kind == "review" and units:
            reviews_with_findings.append(entry.paper_id)
        if entry.kind == "negative_result":
            negatives += 1
            negatives_ok += any(u.outcome == "no_effect" for u in units)
    results = combine(per_paper)
    results.update(
        quote_validity=sum(valid) / len(valid) if valid else 1.0,
        reviews=sum(1 for e in entries if e.kind == "review"), reviews_with_findings=reviews_with_findings,
        negatives=negatives, negatives_with_no_effect=negatives_ok,
    )
    return results


def report(results: dict, out=print) -> None:
    quote_validity = results["quote_validity"]
    out("")
    out(f"  Precision       {results['precision']:.2f}   "
        f"({results['matched']} of {results['predicted']} extracted were right)")
    out(f"  Recall          {results['recall']:.2f}   "
        f"({results['matched']} of {results['expected']} expected were found)")
    out(f"  Quote validity  {quote_validity:.2f}   " +
        ("" if quote_validity >= 1.0 else "<-- MUST be 1.00, investigate before anything else"))
    if results.get("reviews"):
        bad = results["reviews_with_findings"]
        out(f"  Reviews         {len(bad)} of {results['reviews']} produced findings"
            + (f"  <-- must be 0: {', '.join(bad)}" if bad else ""))
    if results.get("negatives"):
        out(f"  Negative papers {results['negatives_with_no_effect']} of {results['negatives']} "
            "produced a no_effect finding")
    out("")
    out("  Per-field accuracy (on findings paired by sentence)")
    fields = results["fields"]
    worst = min(fields, key=fields.get) if fields else None
    for field, accuracy in sorted(fields.items(), key=lambda kv: kv[1]):
        marker = "  <-- worst, fix this first" if field == worst and accuracy < 1.0 else ""
        out(f"    {field:<16} {accuracy:.2f}{marker}")
    out("")
    if results["missed"]:
        out(f"  Missed ({len(results['missed'])}) — recall failures, usually a prompt rule:")
        for item in results["missed"][:5]:
            out(f"    {item}")
    if results["spurious"]:
        out(f"  Spurious ({len(results['spurious'])}) — precision failures, usually missing a skip rule:")
        for item in results["spurious"][:5]:
            out(f"    {item}")
    out("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.litreview.run_extract")
    parser.add_argument("--goldset", type=Path, default=GOLDSET)
    parser.add_argument("--models", help="model set from defaults/models.yaml")
    parser.add_argument("--holdout", action="store_true",
                        help="score the held-back papers instead of the tuning ones - once, at the end")
    args = parser.parse_args(argv)
    say = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731

    try:
        gold = load_goldset(args.goldset)
    except GoldsetError as exc:
        say(f"run_extract: {args.goldset} is not ready to score:")
        for problem in exc.problems:
            say(f"  - {problem}")
        return 1
    held = [e for e in gold if e.holdout]
    tuning = [e for e in gold if not e.holdout]
    print(f"Gold set: {len(tuning)} tuning papers, {len(held)} held back")
    if args.holdout:
        print("Scoring the HELD-BACK papers. Do this once, when tuning is finished.")
    entries = held if args.holdout else tuning

    from bedrock.config import load_settings as load_bedrock_settings
    from bedrock.litreview.llm import make_client
    from bedrock.litreview.service import preflight, standalone_scheduler

    async def go() -> dict:
        settings = load_settings(model_set=args.models)
        async with standalone_scheduler(load_bedrock_settings(), review_settings=settings) as scheduler:
            capacity = await preflight(settings, scheduler.backend)
            client = make_client(settings, scheduler=scheduler, loop=asyncio.get_running_loop(),
                                 context_windows=capacity.windows)
            return await asyncio.to_thread(run, entries, client, settings)

    try:
        results = asyncio.run(go())
    except ConfigError as exc:
        say(str(exc))
        return 1
    report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
