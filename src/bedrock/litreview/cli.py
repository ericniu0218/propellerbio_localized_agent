"""Command line.

    bedrock-litreview doctor                # can this machine run a review? what to change
    bedrock-litreview search "If I want to prevent psoriasis, what should I target?" --out papers.jsonl
    bedrock-litreview patents "If I want to prevent psoriasis, what should I target?" --out patents.jsonl
    bedrock-litreview ask "If I want to prevent psoriasis, what should I target?" --corpus papers.jsonl --patents patents.jsonl
    bedrock-litreview ask "..." --search    # search and answer in one go (needs the network)
    bedrock-litreview ask "..." --no-model  # rules + code only; no written sections
    bedrock-litreview runs                  # the most recent runs
    bedrock-litreview verify-run RUN_ID     # re-prove a stored report
    bedrock-litreview forget RUN_ID         # delete one run's report and working notes
    bedrock-litreview rate REPORT_ID 4 --rater jd --comment "missed IL-36"

The same reviews run inside the terminal app as `/review`; this is the shell
way in, for a script, a cron job or a machine with no app running. Both go
through `service.run_review`, so both get one pipeline and one scheduler.

`search` and `patents` are the only commands that go to the internet, and `ask`
does so only with `--search` (never for patents — pass `--patents FILE`,
always a separate, explicit step; a patents file has nothing frozen between
runs the way `--corpus` does). On an air-gapped machine, run `search` and
`patents` on a machine that is allowed out and carry the files across.
`patents` needs a free EPO OPS account (developers.epo.org) and
`EPO_OPS_KEY`/`EPO_OPS_SECRET`; see `sources/epo_ops.py`.

Only `ask` with a model opens a connection to the model server. `search`,
`patents`, `verify-run`, `rate`, `runs` and `ask --no-model` never build a
backend at all.

The benchmark has its own entry point: `python -m bedrock.litreview.bench --help`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
from pathlib import Path

from bedrock.litreview.config import ConfigError, load_settings
from bedrock.litreview.sources.corpus import CorpusSearcher
from bedrock.litreview.sources.literature import SOURCES, LiteratureSearcher, SearchFailed
from bedrock.litreview.stages.s0_question import QuestionError
from bedrock.litreview.stages.s1_collect import NoPapers, NothingFound


def _say(msg: str) -> None:
    print(msg, file=sys.stderr)


def _db_settings():
    """Settings for the commands that only read the database.

    `require_models=False` because `runs`, `verify-run` and `rate` need
    `db_path` and nothing about models: a site whose `models.yaml` override
    defines only its own set must still be able to list and re-prove its past
    runs, rather than being told its machine profile's default set is unknown.
    """
    return load_settings(check_endpoint=False, require_models=False)


def _searcher_overrides(args) -> dict:
    overrides = {}
    if args.sources:
        overrides["sources"] = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    if args.no_full_text:
        overrides["full_text_max"] = 0
    return overrides


def _ask(args) -> int:
    """`ask`, which is the one command that may use a model.

    With a model this needs an event loop, because every call goes through
    bedrock's scheduler: `standalone_scheduler` opens a backend for this process
    alone, and `service.run_review` runs the pipeline on a worker thread behind
    it. Without a model there is nothing to talk to, so no backend is opened.
    """
    from bedrock.litreview.llm import ModelServerUnavailable, ReviewInterrupted
    from bedrock.litreview.service import ReviewRequest, run_review, standalone_scheduler

    try:
        settings = load_settings(model_set=args.models, check_endpoint=not args.no_model)
    except ConfigError as exc:
        _say(f"ask: {exc}")
        return 1

    patents = None
    if args.patents:
        from bedrock.litreview.sources.patent_corpus import read_patents

        try:
            patents = read_patents(args.patents)
        except (OSError, ValueError) as exc:
            _say(f"ask: {exc}")
            return 1

    request = ReviewRequest(
        question=args.question,
        corpus=args.corpus,
        search=args.search,
        refresh=args.refresh,
        resume=args.resume,
        use_model=not args.no_model,
        model_set=args.models,
        sources=_searcher_overrides(args).get("sources"),
        full_text=not args.no_full_text,
        out_dir=args.out,
        patents=patents,
    )

    async def go():
        from bedrock.config import load_settings as load_bedrock_settings

        def review(scheduler):
            return run_review(
                request, settings=settings, scheduler=scheduler,
                progress=_say, cancel=threading.Event(),
                # Preflight's non-blocking findings - the disk warning, mainly.
                # `timeout_floor_applied` because `standalone_scheduler` below
                # has already raised this process's timeout to 600, so warning
                # about the configured 120 would be a false alarm here.
                warn=lambda message: _say(f"ask: {message}"),
                timeout_floor_applied=True,
            )

        # A run without a model never reaches a server, so none is opened: the
        # machine allowed to search is usually not the machine with the GPU.
        if args.no_model:
            return await review(None)
        async with standalone_scheduler(load_bedrock_settings()) as scheduler:
            return await review(scheduler)

    try:
        result = asyncio.run(go())
    except NothingFound as exc:
        _say(f"ask: {exc}")
        return 1
    except NoPapers:
        _say("ask: no papers for this question yet — pass --corpus FILE, or --search")
        return 1
    except ReviewInterrupted as exc:
        _say(f"ask: {exc}")
        _say(f"ask: resume it with `bedrock-litreview ask {args.question!r} --resume {_unfinished_run_id(settings)}`")
        if isinstance(exc, ModelServerUnavailable):
            _say("ask: on a slow machine, raise `request_timeout` in ~/.config/bedrock/config.toml.")
        return 1
    except (ConfigError, QuestionError, SearchFailed) as exc:
        _say(f"ask: {exc}")
        return 1
    except (OSError, ValueError) as exc:
        _say(f"ask: {exc}")
        return 1

    print(result.paths["html"])
    if result.report.integrity:
        _say(f"ask: the report failed its integrity check ({len(result.report.integrity)} violation(s)); "
             "it opens with a warning")
        return 1
    return 0


def _patents(args) -> int:
    """`patents QUESTION --out FILE`: search EPO OPS, write JSON lines.

    Never opens a model connection — this is a `search`-shaped command, not an
    `ask`-shaped one. The records it writes are read back only by `ask
    --patents FILE`, or `/review --patents FILE` in the app, into the
    generated "Competitor landscape" section; nothing here reaches the
    model-facing pipeline (`sources/patents_base.py`).
    """
    from bedrock.litreview.knowledge import load_knowledge
    from bedrock.litreview.sources.epo_ops import EpoOps, PatentSearchFailed
    from bedrock.litreview.sources.patent_corpus import write_patents
    from bedrock.litreview.sources.patents_base import PatentSearchRequest
    from bedrock.litreview.stages.s0_question import parse_question

    settings = load_settings(check_endpoint=False)
    if not settings.epo_ops_key or not settings.epo_ops_secret:
        _say("patents: set EPO_OPS_KEY and EPO_OPS_SECRET (free account at developers.epo.org)")
        return 1
    try:
        spec = parse_question(args.question, None, model=None, knowledge=load_knowledge())
        searcher = EpoOps(
            settings.epo_ops_key, settings.epo_ops_secret,
            claims_max=args.claims_max if args.claims_max is not None
            else int(settings.knob("patents.claims_max", 40)),
        )
        request = PatentSearchRequest(
            question=spec.text, spec=spec,
            exact_terms={e.canonical: e.exact_terms for e in spec.entities},
            max_results=args.max_results,
        )
        records = searcher.search(request)
    except (QuestionError, PatentSearchFailed, ValueError) as exc:
        _say(f"patents: {exc}")
        return 1
    write_patents(records, args.out)
    print(args.out)
    return 0


def _forget(args) -> int:
    """`forget RUN_ID`: delete one run's output, after saying what will go.

    Confirmed by default, unlike `/digest forget`, because there is no trash to
    recover from and a report can be hours of a machine's work. `--yes` is for
    scripts. Naming the question first matters: run ids are hashes, and the way
    this goes wrong is deleting a neighbour of the intended one.
    """
    from bedrock.litreview.api import forget_run
    from bedrock.litreview.db import Database

    settings = _db_settings()
    db = Database(settings.db_path)
    run = db.get_run(args.run_id)
    if run is None:
        _say(f"forget: no run {args.run_id!r}. `bedrock-litreview runs` lists them.")
        return 1

    if not args.yes:
        question = db.get_question(run.question_id)
        report_id = db.report_id_for(args.run_id)
        _say(f"Run {args.run_id}, started {run.started}")
        _say(f"  question: {question.text if question else '(not stored)'}")
        _say(f"  report:   {report_id or 'none (the run did not finish)'}")
        _say("This deletes the run, its stage artifacts and its report files. The question and "
             "its papers stay, so the same question can be asked again. It cannot be undone.")
        try:
            if input("Delete it? [y/N] ").strip().lower() not in {"y", "yes"}:
                _say("forget: left alone.")
                return 1
        except EOFError:  # piped input with no answer is not a yes
            _say("forget: no answer, so nothing was deleted. Use --yes in a script.")
            return 1

    result = forget_run(args.run_id, db=db, settings=settings)
    if result is None:  # deleted between the look-up and the delete
        _say(f"forget: no run {args.run_id!r}.")
        return 1

    deleted = result.deleted
    print(f"Deleted run {deleted.run_id} and {deleted.artifacts} stage artifact(s).")
    if deleted.report_deleted:
        print(f"Deleted report {deleted.report_id} and {len(result.files_removed)} file(s).")
        if result.report_dir is not None and not result.files_removed:
            # The database row went, but the rendered copy was somewhere else.
            print(f"No files were found in {result.report_dir}. A report written with "
                  "--out elsewhere has to be deleted by hand.")
        if deleted.ratings:
            print(f"Deleted {deleted.ratings} rating(s) of that report.")
    elif deleted.report_shared_with:
        print(f"Kept report {deleted.report_id}: run(s) {', '.join(deleted.report_shared_with)} "
              "produced the identical report and still point at it.")
    else:
        print("The run wrote no report, so there was none to delete.")
    return 0


def _doctor(args) -> int:
    """`doctor`: every setup check at once, and what was read to decide them.

    The first-run banner (`bedrock.doctor`) runs only while there is no config
    file, which is the one moment a machine is guaranteed *not* to be set up
    yet. This is the re-runnable version: run it after changing a setting, after
    IT changes the server, or when a review refuses to start and the message is
    about something you cannot see.

    Every check runs even when an earlier one failed, because a machine that is
    new to this usually has more than one thing to fix and one-error-per-attempt
    is a slow way to find that out. Exit status 1 means at least one finding
    would stop a review.
    """
    from bedrock.litreview.config import config_file
    from bedrock.litreview.service import Finding, config_findings, findings, standalone_scheduler

    def show(items) -> None:
        for finding in items:
            first, *rest = finding.detail.splitlines() or [""]
            print(f"[{finding.mark}] {finding.title}: {first}")
            for line in rest:
                print(f"       {line}")

    try:
        settings = load_settings(model_set=args.models)
    except ConfigError as exc:
        # A bad [litreview] table, an unknown model set, or a model endpoint
        # that is not on the local network. All three stop every review, and all
        # three are exactly what someone runs this to find out.
        show([Finding(title="configuration", ok=False, blocking=True, detail=str(exc))])
        return 1

    print("Literature review — setup check\n")
    for label, value in (
        ("model set", settings.model_set),
        ("models.yaml", config_file("models.yaml")),
        ("pipeline.yaml", config_file("pipeline.yaml")),
        ("report.yaml", config_file("report.yaml")),
        ("model server", settings.base_url),
        ("machine profile", settings.profile_name),
        ("data directory", settings.data_dir),
        ("database", settings.db_path),
        ("reports", settings.out_dir),
    ):
        print(f"  {label + ':':<17}{value}")
    print()

    if args.no_server:
        found = config_findings(settings)
    else:
        async def go():
            from bedrock.config import load_settings as load_bedrock_settings

            async with standalone_scheduler(load_bedrock_settings()) as scheduler:
                return await findings(settings, scheduler.backend)

        try:
            found = asyncio.run(go())
        except OSError as exc:
            found = config_findings(settings) + [
                Finding(title="model server", ok=False, blocking=True,
                        detail=f"could not be reached at all: {exc}")
            ]

    show(found)
    blocking = [f for f in found if f.blocking]
    warnings = [f for f in found if not f.ok and not f.blocking]
    print()
    if blocking:
        print(f"{len(blocking)} thing(s) would stop a review. Fix those and run this again.")
        return 1
    if warnings:
        print(f"A review can run. {len(warnings)} thing(s) above are worth knowing first.")
        return 0
    print("Ready. `bedrock-litreview ask \"<question>\" --corpus <papers.jsonl>` runs one.")
    return 0


def _unfinished_run_id(settings) -> str:
    """The most recent run that has no report, so the message can name it."""
    from bedrock.litreview.db import Database

    try:
        for run in Database(settings.db_path).recent_runs(20):
            if run.finished is None:
                return run.id
    except Exception:  # noqa: BLE001 - a hint must not replace the real error
        pass
    return "<run-id>"



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bedrock-litreview")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="search the literature and save papers as JSON lines (needs network)")
    search.add_argument("question")
    search.add_argument("--out", type=Path, required=True, help="where to write the papers (.jsonl)")
    search.add_argument("--max-results", type=int, default=200)

    patents = sub.add_parser(
        "patents", help="search EPO OPS and save patent records as JSON lines (needs network + EPO_OPS_KEY/SECRET)")
    patents.add_argument("question")
    patents.add_argument("--out", type=Path, required=True, help="where to write the records (.jsonl)")
    patents.add_argument("--max-results", type=int, default=100)
    patents.add_argument("--claims-max", type=int, default=None,
                         help="cap on the (per-document) claim-1 calls; default from defaults/pipeline.yaml")

    ask = sub.add_parser("ask", help="answer a question with a cited report")
    ask.add_argument("question")
    papers_from = ask.add_mutually_exclusive_group()
    papers_from.add_argument("--corpus", type=Path, help="papers as JSON lines (the searcher's handoff format)")
    papers_from.add_argument("--search", action="store_true", help="search the literature now (needs network)")
    ask.add_argument("--patents", type=Path, help="patent records as JSON lines (from `bedrock-litreview patents`)")
    ask.add_argument("--models", help="model set from defaults/models.yaml (office, laptop, medgemma)")
    ask.add_argument("--out", type=Path, help="output directory (default: the reports folder in the review data directory)")
    ask.add_argument("--no-model", action="store_true", help="run without any model")
    ask.add_argument("--refresh", action="store_true", help="re-collect papers instead of reusing the frozen set")
    ask.add_argument("--resume", help="resume a run id from the stage it reached")

    for p in (search, ask):
        p.add_argument("--sources", help=f"comma-separated subset of {','.join(SOURCES)} (default: defaults/pipeline.yaml)")
        p.add_argument("--no-full-text", action="store_true", help="abstracts only; skip open-access full text")

    verify = sub.add_parser("verify-run", help="re-check a stored run's report against the papers it pinned")
    verify.add_argument("run_id", nargs="?", help="the run to check")
    verify.add_argument("--all", action="store_true", help="check every finished run in the database")
    verify.add_argument("--db", help="database file (default: the configured one; the benchmark uses bench.sqlite3 in the same directory)")

    runs = sub.add_parser("runs", help="list the most recent runs")
    runs.add_argument("--limit", type=int, default=20)

    forget = sub.add_parser("forget", help="delete one run's report and working notes (cannot be undone)")
    forget.add_argument("run_id")
    forget.add_argument("--yes", action="store_true", dest="yes",
                        help="do not ask for confirmation (for scripts)")

    doctor = sub.add_parser("doctor", help="check this machine can run a review, and say what to change")
    doctor.add_argument("--models", help="check a particular model set instead of this machine's default")
    doctor.add_argument("--no-server", action="store_true", dest="no_server",
                        help="skip the checks that need the model server running")

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

    try:
        return _run(args)
    except ConfigError as exc:
        # A bad config file or a missing model set is something to fix, not a
        # crash to report: say what is wrong and exit non-zero.
        _say(f"{args.command}: {exc}")
        return 1


def _run(args) -> int:
    """Everything after the arguments are read. Split out so `main` has exactly
    one place that turns a `ConfigError` into a message."""
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

    if args.command == "patents":
        return _patents(args)

    if args.command == "ask":
        return _ask(args)

    if args.command == "doctor":
        return _doctor(args)

    if args.command == "runs":
        from bedrock.litreview.db import Database

        db = Database(_db_settings().db_path)
        for run in db.recent_runs(args.limit):
            report_id = db.report_id_for(run.id)
            question = db.get_question(run.question_id)
            state = report_id or f"unfinished (stage {run.stage_reached})"
            print(f"{run.id}  {run.started}  {state}  {question.text if question else ''}")
        return 0

    if args.command == "verify-run":
        from bedrock.litreview.db import Database
        from bedrock.litreview.report.invariants import verify_run

        db = Database(args.db or _db_settings().db_path)
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

    if args.command == "forget":
        return _forget(args)

    if args.command == "rate":
        from bedrock.litreview.api import rate_report

        rating = rate_report(args.report_id, args.score, rater=args.rater, comments=args.comment)
        print(rating.model_dump_json())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
