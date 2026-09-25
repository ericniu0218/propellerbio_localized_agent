"""Command line.

    litreview search "If I want to prevent psoriasis, what should I target?" --out papers.jsonl
    litreview ask "If I want to prevent psoriasis, what should I target?" --corpus papers.jsonl
    litreview ask "..." --search            # search and answer in one go (needs the network)
    litreview ask "..." --no-model          # rules + code only; no written sections
    litreview rate REPORT_ID 4 --rater jd --comment "missed IL-36"

`search` is the only command that goes to the internet, and `ask` does so only
with `--search`. On an air-gapped machine, run `search` on one that is allowed
out and carry the JSON-lines file across.

The benchmark has its own entry point: `python -m bedrock.litreview.bench --help`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bedrock.litreview.config import load_settings
from bedrock.litreview.sources.corpus import CorpusSearcher
from bedrock.litreview.sources.literature import SOURCES, LiteratureSearcher, SearchFailed
from bedrock.litreview.stages.s0_question import QuestionError
from bedrock.litreview.stages.s1_collect import NoPapers


def _say(msg: str) -> None:
    print(msg, file=sys.stderr)


def _searcher_overrides(args) -> dict:
    overrides = {}
    if args.sources:
        overrides["sources"] = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    if args.no_full_text:
        overrides["full_text_max"] = 0
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="litreview")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="search the literature and save papers as JSON lines (needs network)")
    search.add_argument("question")
    search.add_argument("--out", type=Path, required=True, help="where to write the papers (.jsonl)")
    search.add_argument("--max-results", type=int, default=200)

    ask = sub.add_parser("ask", help="answer a question with a cited report")
    ask.add_argument("question")
    papers_from = ask.add_mutually_exclusive_group()
    papers_from.add_argument("--corpus", type=Path, help="papers as JSON lines (the searcher's handoff format)")
    papers_from.add_argument("--search", action="store_true", help="search the literature now (needs network)")
    ask.add_argument("--models", help="model set from defaults/models.yaml (office, laptop, medgemma)")
    ask.add_argument("--out", type=Path, help="output directory (default: out/)")
    ask.add_argument("--no-model", action="store_true", help="run without any model")
    ask.add_argument("--refresh", action="store_true", help="re-collect papers instead of reusing the frozen set")
    ask.add_argument("--resume", help="resume a run id from the stage it reached")

    for p in (search, ask):
        p.add_argument("--sources", help=f"comma-separated subset of {','.join(SOURCES)} (default: config/pipeline.yaml)")
        p.add_argument("--no-full-text", action="store_true", help="abstracts only; skip open-access full text")

    verify = sub.add_parser("verify-run", help="re-check a stored run's report against the papers it pinned")
    verify.add_argument("run_id", nargs="?", help="the run to check")
    verify.add_argument("--all", action="store_true", help="check every finished run in the database")
    verify.add_argument("--db", help="database file (default: the configured one; the benchmark uses data/bench.sqlite3)")

    rate = sub.add_parser("rate", help="record a scientist's rating of a report")
    rate.add_argument("report_id")
    rate.add_argument("score", type=int, choices=range(1, 6))
    rate.add_argument("--rater", required=True)
    rate.add_argument("--comment")

    args = parser.parse_args(argv)
    if args.command == "verify-run" and not (args.run_id or args.all):
        parser.error("verify-run needs a run id or --all")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # One line per HTTP request drowns out the pipeline's own progress.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "search":
        from bedrock.litreview.knowledge import load_knowledge
        from bedrock.litreview.sources.corpus import write_papers
        from bedrock.litreview.stages.s0_question import parse_question
        from bedrock.litreview.stages.s1_collect import collect

        # No model: the searcher's machine is the one allowed out, which is
        # exactly the machine the model server is not on.
        settings = load_settings(check_endpoint=False)
        try:
            spec = parse_question(args.question, None, model=None, knowledge=load_knowledge())
            searcher = LiteratureSearcher.from_settings(settings, **_searcher_overrides(args))
            papers = collect(spec, searcher=searcher, max_results=args.max_results)
        except (QuestionError, SearchFailed, ValueError) as exc:
            _say(f"search: {exc}")
            return 1
        write_papers(papers, args.out)
        print(args.out)
        return 0

    if args.command == "ask":
        from bedrock.litreview.runner import answer_question

        settings = load_settings(model_set=args.models, check_endpoint=not args.no_model)
        searcher = None
        if args.corpus:
            try:
                searcher = CorpusSearcher(args.corpus)
            except (OSError, ValueError) as exc:
                _say(f"ask: {exc}")
                return 1
        elif args.search:
            searcher = LiteratureSearcher.from_settings(settings, **_searcher_overrides(args))
        try:
            result = answer_question(
                args.question,
                settings=settings,
                use_model=not args.no_model,
                searcher=searcher,
                refresh_papers=args.refresh,
                out_dir=args.out,
                resume_run_id=args.resume,
                progress=_say,
            )
        except NoPapers:
            _say("ask: no papers for this question yet — pass --corpus FILE, or --search")
            return 1
        except (QuestionError, SearchFailed) as exc:
            _say(f"ask: {exc}")
            return 1
        print(result.paths["html"])
        if result.report.integrity:
            _say(f"ask: the report failed its integrity check ({len(result.report.integrity)} violation(s)); "
                 "it opens with a warning")
            return 1
        return 0

    if args.command == "verify-run":
        from bedrock.litreview.db import Database
        from bedrock.litreview.report.invariants import verify_run

        db = Database(args.db or load_settings(check_endpoint=False).db_path)
        if args.all:
            # Only runs that finished a report; an interrupted run has nothing to verify.
            run_ids = db.finished_run_ids()
        else:
            run_ids = [args.run_id]
        failed = 0
        for run_id in run_ids:
            try:
                violations = verify_run(db, run_id)
            except Exception as exc:  # a missing pinned paper or an unreadable artifact is a failure too
                violations = [f"cannot be verified: {type(exc).__name__}: {exc}"]
            print(f"{run_id}: {'OK' if not violations else f'{len(violations)} violation(s)'}")
            for v in violations:
                print(f"  - {v}")
            failed += bool(violations)
        return 1 if failed else 0

    if args.command == "rate":
        from bedrock.litreview.api import rate_report

        rating = rate_report(args.report_id, args.score, rater=args.rater, comments=args.comment)
        print(rating.model_dump_json())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
