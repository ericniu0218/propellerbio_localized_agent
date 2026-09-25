"""python -m bedrock.litreview.bench build-corpus|run QUESTION.yaml [...]"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from bedrock.litreview.bench.models import BenchQuestion
from bedrock.litreview.config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bedrock.litreview.bench")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-corpus", help="fetch and freeze a sample corpus (needs network)")
    build.add_argument("questions", nargs="+", type=Path)
    build.add_argument("--searcher", action="store_true",
                       help="take the main papers from the paper searcher, not the question's hand-written queries")
    build.add_argument("--out", type=Path, help="corpora root (default: evals/litreview/corpora)")

    run = sub.add_parser("run", help="run the pipeline and baselines on a frozen corpus and score them")
    run.add_argument("questions", nargs="+", type=Path)
    run.add_argument("--models", help="model set from defaults/models.yaml (office, laptop, medgemma)")
    run.add_argument("--skip-baselines", action="store_true")
    run.add_argument("--skip-distractors", action="store_true")
    run.add_argument("--skip-perturb", action="store_true")
    run.add_argument("--from-run", help="score the stored report of this run id instead of running the pipeline")
    run.add_argument("--corpora", type=Path, help="corpora root to read (default: evals/litreview/corpora)")
    run.add_argument(
        "--judge-file",
        type=Path,
        help="judge with answers from this JSONL file instead of the local model; questions with no "
        "answer are written to <file>.pending.jsonl",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    say = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731

    if args.command == "build-corpus":
        from bedrock.litreview.bench.corpus import build_corpus
        from bedrock.litreview.bench.run import CORPORA
        from bedrock.litreview.sources.literature import LiteratureSearcher

        searcher = None
        if args.searcher:
            searcher = LiteratureSearcher.from_settings(load_settings(check_endpoint=False))
        for path in args.questions:
            bq = BenchQuestion.load(path)
            say(f"[{bq.id}] building corpus")
            build_corpus(bq, (args.out or CORPORA) / bq.id, searcher=searcher, progress=say)
        return 0

    from bedrock.config import load_settings as load_bedrock_settings
    from bedrock.litreview.bench.run import evaluate
    from bedrock.litreview.llm import make_client
    from bedrock.litreview.service import standalone_scheduler

    settings = load_settings(model_set=args.models)
    judge_client = None
    if args.judge_file:
        from bedrock.litreview.bench.filejudge import FileClient

        judge_client = FileClient(args.judge_file, version=f"file:{args.judge_file.name}")
        say(f"judging from {args.judge_file} ({len(judge_client.answers)} answer(s) on file)")

    async def go() -> list[Path]:
        # The benchmark is a long job on its own machine, so it opens a backend
        # and a scheduler of its own - the same ones the app would give it, just
        # with no chat to wait behind. `evaluate` is blocking, so it runs on a
        # worker thread and its model calls come back to this loop.
        async with standalone_scheduler(load_bedrock_settings()) as scheduler:
            client = make_client(settings, scheduler=scheduler, loop=asyncio.get_running_loop())
            cards = []
            for path in args.questions:
                bq = BenchQuestion.load(path)
                _, out_dir = await asyncio.to_thread(
                    evaluate,
                    bq, settings=settings, client=client, skip_baselines=args.skip_baselines,
                    corpus_dir=(args.corpora / bq.id) if args.corpora else None,
                    skip_distractors=args.skip_distractors, skip_perturb=args.skip_perturb,
                    from_run=args.from_run, judge_client=judge_client, progress=say,
                )
                cards.append(out_dir / "scorecard.md")
            return cards

    for card in asyncio.run(go()):
        print(card)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
