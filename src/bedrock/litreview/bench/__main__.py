"""python -m bedrock.litreview.bench build-corpus|run|determinism-payloads|determinism|replay-control ..."""

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
        "--legacy-cache-read", action="store_true",
        help="also replay call-cache entries written before generation contexts were part of the key "
             "(they do not record seed, thinking policy or server)",
    )
    run.add_argument(
        "--judge-file",
        type=Path,
        help="judge with answers from this JSONL file instead of the local model; questions with no "
        "answer are written to <file>.pending.jsonl",
    )

    pay = sub.add_parser("determinism-payloads",
                         help="render pinned stage requests from frozen corpora for the determinism experiment (no model)")
    pay.add_argument("questions", nargs="+", type=Path)
    pay.add_argument("--corpora", type=Path, help="corpora root (default: evals/litreview/corpora)")
    pay.add_argument("--out", type=Path, required=True)

    det = sub.add_parser("determinism", help="send pinned requests repeatedly and record whether replies repeat")
    det.add_argument("payloads", type=Path)
    det.add_argument("--mode", choices=("fresh", "replay"), required=True,
                     help="fresh: no cache, every call must reach the server; replay: read an existing call cache")
    det.add_argument("--models", help="model set from defaults/models.yaml")
    det.add_argument("--repeats", type=int, default=20)
    det.add_argument("--condition", required=True, help="a name for this cell, e.g. same-process-serial")
    det.add_argument("--out", type=Path, required=True, help="experiment directory; a condition is never overwritten")
    det.add_argument("--manifest", type=Path, help="deployment manifest (JSON) to pin beside the results")
    det.add_argument("--cache", type=Path, help="replay mode: the call-cache directory to read")

    rc = sub.add_parser("replay-control",
                        help="record one pipeline run's model replies, then render twice from them and compare bytes")
    rc.add_argument("question", type=Path)
    rc.add_argument("--models", help="model set from defaults/models.yaml")
    rc.add_argument("--corpora", type=Path, help="corpora root (default: evals/litreview/corpora)")
    rc.add_argument("--work", type=Path, required=True)

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

    if args.command == "determinism-payloads":
        import json

        from bedrock.litreview.bench.determinism import build_payloads
        from bedrock.litreview.bench.run import CORPORA
        from bedrock.litreview.sources.corpus import read_papers

        sources = []
        for path in args.questions:
            bq = BenchQuestion.load(path)
            sources.append((bq.id, bq.question, read_papers((args.corpora or CORPORA) / bq.id / "papers.jsonl")))
        payloads = build_payloads(sources)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in payloads), encoding="utf-8")
        say(f"{len(payloads)} payloads written")
        print(args.out)
        return 0

    from bedrock.config import load_settings as load_bedrock_settings
    from bedrock.litreview.bench.run import evaluate
    from bedrock.litreview.llm import make_client
    from bedrock.litreview.service import standalone_scheduler

    settings = load_settings(model_set=args.models)

    if args.command == "determinism":
        return _determinism(args, settings, say)
    if args.command == "replay-control":
        return _replay_control(args, settings, say)
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
            client = make_client(settings, scheduler=scheduler, loop=asyncio.get_running_loop(),
                                 legacy_cache=args.legacy_cache_read)
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


def _determinism(args, settings, say) -> int:
    """One experiment cell, through the same scheduler every review uses."""
    import copy
    import json

    from bedrock.config import load_settings as load_bedrock_settings
    from bedrock.litreview.bench.determinism import FreshModeUnavailable, load_payloads, run_cell
    from bedrock.litreview.llm import make_client
    from bedrock.litreview.service import standalone_scheduler

    payloads = load_payloads(args.payloads)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    if args.mode == "replay" and not args.cache:
        say("determinism: replay mode needs --cache DIR")
        return 1
    models = {"relevance": settings.model_for("relevance"), "extract": settings.model_for("extract")}

    async def go():
        async with standalone_scheduler(load_bedrock_settings()) as scheduler:
            loop = asyncio.get_running_loop()

            def client_for(rep: int):
                if args.mode == "fresh":
                    # A new client with no cache at all, every repetition.
                    return make_client(settings, scheduler=scheduler, loop=loop, cache=False)
                replay = copy.copy(settings)
                replay.cache_dir = args.cache
                return make_client(replay, scheduler=scheduler, loop=loop)

            return await asyncio.to_thread(
                run_cell, payloads, client_for, models=models, repeats=args.repeats, mode=args.mode,
                condition=args.condition, out_dir=args.out, manifest=manifest,
            )

    try:
        result = asyncio.run(go())
    except (FreshModeUnavailable, FileExistsError) as exc:
        say(f"determinism: {exc}")
        return 1
    print(json.dumps(result.row(), indent=2, sort_keys=True))
    return 0 if result.repeatable else 1


def _replay_control(args, settings, say) -> int:
    """Record one run's replies from the model, then render twice from them."""
    import copy
    import json

    from bedrock.config import load_settings as load_bedrock_settings
    from bedrock.litreview.bench.determinism import RecordingClient, replay_control
    from bedrock.litreview.bench.run import CORPORA
    from bedrock.litreview.db import Database
    from bedrock.litreview.llm import make_client
    from bedrock.litreview.runner import answer_question
    from bedrock.litreview.service import standalone_scheduler
    from bedrock.litreview.sources.corpus import read_papers

    bq = BenchQuestion.load(args.question)
    papers = read_papers((args.corpora or CORPORA) / bq.id / "papers.jsonl")

    def settings_for(d):
        s = copy.copy(settings)
        s.cache_dir, s.out_dir, s.db_path = d / "cache", d / "out", d / "db.sqlite3"
        return s

    async def record():
        async with standalone_scheduler(load_bedrock_settings()) as scheduler:
            live = make_client(settings, scheduler=scheduler, loop=asyncio.get_running_loop(), cache=False)
            recorder = RecordingClient(live)
            d = args.work / "recorded"
            await asyncio.to_thread(answer_question, bq.question, settings=settings_for(d), client=recorder,
                                    papers=papers, db=Database(d / "db.sqlite3"), out_dir=d / "out")
            return recorder.replies, recorder.model_version

    replies, version = asyncio.run(record())
    result = replay_control(bq.question, papers, replies, settings_for=settings_for, version=version,
                            work_dir=args.work)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
