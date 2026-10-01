"""Command line.

    bedrock-litreview doctor                # can this machine run a review? what to change
    bedrock-litreview search "If I want to prevent psoriasis, what should I target?" --out papers.jsonl
    bedrock-litreview patents "If I want to prevent psoriasis, what should I target?" --out patents.jsonl
    bedrock-litreview ask "If I want to prevent psoriasis, what should I target?" --corpus papers.jsonl --patents patents.jsonl
    bedrock-litreview ask "..." --search    # search and answer in one go (needs the network)
    bedrock-litreview ask "..." --no-model  # rules + code only; no written sections
    bedrock-litreview ask "..." --search --deck   # also write report.pptx, the target-proposal deck
    bedrock-litreview runs                  # the most recent runs
    bedrock-litreview verify-run RUN_ID     # re-prove a stored report, and say if today's rules differ
    bedrock-litreview verify-run --all --require-same-rules   # CI: also fail on changed rules
    bedrock-litreview forget RUN_ID         # delete one run's report and working notes
    bedrock-litreview rate REPORT_ID 4 --rater jd --comment "missed IL-36"
    bedrock-litreview integrity plan --run RUN_ID      # which identifiers a refresh would send
    bedrock-litreview integrity refresh --run RUN_ID --yes   # retractions/concerns (needs the network)
    bedrock-litreview integrity import bundle.jsonl    # the same, carried in from a machine allowed out

The same reviews run inside the terminal app as `/review`; this is the shell
way in, for a script, a cron job or a machine with no app running. Both go
through `service.run_review`, so both get one pipeline and one scheduler.

`search`, `patents` and `integrity refresh --yes` go to the internet. `ask`
does with `--search`, and - unless retraction checks are set to manual - to ask
PubMed and Crossref about its papers' identifiers before reading them. It never
goes out for patents: pass `--patents FILE`, always a separate, explicit step,
because a patents file has nothing frozen between runs the way `--corpus` does.
On an air-gapped machine, run `search` and `patents` on one that is allowed out
and carry the files across. `patents` needs a free EPO OPS account
(developers.epo.org) and `EPO_OPS_KEY`/`EPO_OPS_SECRET`; see
`sources/epo_ops.py`.

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
import time
from pathlib import Path

from bedrock.litreview.config import ConfigError, load_settings
from bedrock.litreview.progress import throttled
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
    from bedrock.litreview.journal import ReviewNotRecorded
    from bedrock.litreview.service import ReviewRequest, local_history, run_review, standalone_scheduler

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
        deck=args.deck,
    )

    async def go():
        from bedrock.config import load_settings as load_bedrock_settings

        bedrock_settings = load_bedrock_settings()
        history, actor_id = local_history(bedrock_settings)

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
                # A shell has no status bar to overwrite, and every line it
                # prints is permanent, so the per-item ticks are throttled to
                # a handful per stage rather than one per paper.
                tick=throttled(_say, now=time.monotonic),
                history=history, actor_id=actor_id,
            )

        # A run without a model never reaches a server, so none is opened: the
        # machine allowed to search is usually not the machine with the GPU.
        if args.no_model:
            return await review(None)
        # The review's own endpoint: bedrock's `api_base`, or in agent-pack
        # mode the agent server's inference API (`config.resolve_endpoint`).
        async with standalone_scheduler(bedrock_settings, review_settings=settings) as scheduler:
            return await review(scheduler)

    try:
        result = asyncio.run(go())
    except ReviewNotRecorded as exc:
        _say(f"ask: {exc}")
        return 1
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
    if args.deck:
        _write_deck(result, settings)
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


def _write_deck(result, settings) -> None:
    """Write report.pptx beside the other report files.

    The deck is drawn by code from the saved report and the figures the run
    prepared, so this never reaches the network. A missing python-pptx is said
    plainly and does not change the exit code: the report itself is finished.
    """
    import os
    from datetime import datetime

    from bedrock.litreview.report.slides import to_pptx
    from bedrock.litreview.service import deck_figures

    now = datetime.now()
    try:
        deck = to_pptx(
            result.report, f"review-{result.report_id}", figures=deck_figures(settings, result.run.id),
            author=os.environ.get("BEDROCK_USER", ""), date=f"{now.day} {now:%B %Y}",
        )
    except ImportError:
        _say('ask: the PowerPoint needs python-pptx: pip install -e ".[server]"')
        return
    path = result.paths["html"].with_name("report.pptx")
    path.write_bytes(deck)
    _say(f"ask: wrote {path}")


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



def _integrity_ids(args, db) -> list[str]:
    """The identifiers a command is about, normalised; a malformed one is an error."""
    from bedrock.litreview.publication_integrity import normalize_id, paper_ids
    from bedrock.litreview.runner import load_result

    if args.ids:
        out = []
        for raw in args.ids:
            kind, _, value = raw.partition(":")
            norm = normalize_id(kind.lower(), value) if value else None
            if norm is None:
                raise ValueError(f"not a PMID:, PMCID: or DOI: identifier: {raw!r}")
            out.append(norm)
        return list(dict.fromkeys(out))
    result = load_result(db, args.run)
    cited = {ref.paper_id for ref in result.report.references}
    papers = [p for p in result.papers if args.all_papers or p.id in cited]
    # At most a PMID and a DOI per paper - each one sent says what is being
    # read - and never a PMCID, which no validated lookup uses.
    return list(dict.fromkeys(i for p in papers for i in paper_ids(p) if not i.startswith("PMCID:")))


def _integrity(args, *, refresher=None, clock=None) -> int:
    """`integrity plan|refresh|import|status`. Only `refresh --yes` uses the network."""
    import hashlib
    import json
    from datetime import datetime, timedelta, timezone

    from bedrock.litreview.db import Database
    from bedrock.litreview.publication_integrity import BundleError, derive_state, observation_from_bundle
    from bedrock.litreview.sources.integrity import IntegrityRefresher, RefreshRefused, plan

    settings = _db_settings()
    db = Database(args.db or settings.db_path)
    command = args.integrity_command

    if command in ("include", "exclude"):
        from bedrock.litreview.publication_integrity import decide_reinstated

        try:
            print(decide_reinstated(db, args.paper, include=command == "include", by=args.by, note=args.note,
                                    now=clock() if clock else datetime.now(timezone.utc)))
        except ValueError as exc:
            _say(f"integrity {command}: {exc}")
            return 1
        return 0

    if command == "import":
        data = args.bundle.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        imported = 0
        for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obs, raw = observation_from_bundle(json.loads(line), bundle_sha256=digest, line=n)
            except (BundleError, ValueError) as exc:
                _say(f"integrity import: refused, nothing after line {n - 1} recorded: {exc}")
                return 1
            db.add_integrity_observation(obs, raw)
            imported += 1
        print(f"imported {imported} observation(s) from {args.bundle.name} (sha256 {digest[:12]}); "
              "the bundle's origin is not authenticated")
        return 0

    try:
        ids = _integrity_ids(args, db)
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        requests = plan(ids, providers)
    except (ValueError, RefreshRefused) as exc:
        _say(f"integrity {command}: {exc}")
        return 1

    if command == "status":
        now = clock() if clock else datetime.now(timezone.utc)
        ttl = timedelta(days=int(settings.knob("integrity.stale_after_days", 7)))
        for i in ids:
            state = derive_state([i], db.integrity_observations([i]), now=now, ttl=ttl)
            shown = ", ".join(state.statuses) or "no notice recorded"
            print(f"{i}: {shown} [{state.freshness}]")
        return 0

    from bedrock.litreview.sources.integrity import checkable

    sendable, skipped = checkable(ids)
    if command in ("plan", "refresh") and skipped:
        print(f"{len(skipped)} identifier(s) cannot be checked (a PMCID alone has no validated lookup): "
              + " ".join(skipped))
    if command == "plan" or (command == "refresh" and not args.yes):
        print(f"{len(sendable)} identifier(s) would be sent, as {len(requests)} request(s):")
        for provider, batch in requests:
            host = {"europepmc": "www.ebi.ac.uk", "pubmed": "eutils.ncbi.nlm.nih.gov",
                    "crossref": "api.crossref.org"}[provider]
            print(f"  {provider} ({host}): {' '.join(batch)}")
        print("Identifiers say what is being read. Nothing else is sent: no question, report or finding.")
        if command == "refresh":
            _say("integrity refresh: nothing was sent; add --yes to send these requests")
            return 2
        return 0

    refresher = refresher or IntegrityRefresher(api_key=settings.ncbi_api_key, email=settings.contact_email)
    bundle = args.bundle_out.open("w", encoding="utf-8") if args.bundle_out else None
    failed = 0
    try:
        for provider, batch in requests:
            obs, raw, line = refresher.fetch(provider, batch)
            if not args.no_store:
                db.add_integrity_observation(obs, raw)
            if bundle is not None and line:
                bundle.write(json.dumps(line, sort_keys=True) + "\n")
            found = sorted({e.status for e in obs.events})
            print(f"{provider}: {obs.outcome} for {len(batch)} id(s)" + (f"; notices: {', '.join(found)}" if found else ""))
            failed += obs.outcome == "error"
    finally:
        refresher.close()
        if bundle is not None:
            bundle.close()
    return 1 if failed else 0


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
    ask.add_argument("--deck", action="store_true",
                     help="also write report.pptx, the target-proposal deck (with --search, paper figures are fetched)")

    for p in (search, ask):
        p.add_argument("--sources", help=f"comma-separated subset of {','.join(SOURCES)} (default: defaults/pipeline.yaml)")
        p.add_argument("--no-full-text", action="store_true", help="abstracts only; skip open-access full text")

    aliases = sub.add_parser(
        "aliases", help="propose names for the knowledge table from HGNC and MeSH downloads; a person approves them")
    asub = aliases.add_subparsers(dest="aliases_command", required=True)
    propose = asub.add_parser("propose", help="write proposed names to a file for a person to read (no network)")
    propose.add_argument("--hgnc", type=Path, help="HGNC complete set or custom download (tab-separated)")
    propose.add_argument("--mesh", type=Path, help="MeSH descriptors, NLM's XML download (descYYYY.xml)")
    propose.add_argument("--symbol", action="append", default=[],
                         help="an HGNC symbol the table does not know yet, to propose a new target entry (repeatable)")
    propose.add_argument("--out", type=Path, required=True, help="where to write the proposals (.yaml)")
    propose.add_argument("--table", type=Path,
                         help="the table to propose for (default: the package's knowledge/entities.yaml)")
    approve_cmd = asub.add_parser("approve", help="write the names marked `approve: true` into the table")
    approve_cmd.add_argument("proposals", type=Path)
    approve_cmd.add_argument("--table", type=Path, help="the table to edit (default: the package's knowledge/entities.yaml)")

    verify = sub.add_parser("verify-run", help="re-check a stored run's report against the papers it pinned")
    verify.add_argument("run_id", nargs="?", help="the run to check")
    verify.add_argument("--all", action="store_true", help="check every finished run in the database")
    verify.add_argument("--db", help="database file (default: the configured one; the benchmark uses bench.sqlite3 in the same directory)")
    verify.add_argument(
        "--require-same-rules", action="store_true",
        help="also fail when the config, knowledge, prompts or code differ from what the run recorded, "
             "or cannot be compared (for CI; by default a difference is reported, not failed)",
    )
    verify.add_argument("--details", action="store_true", help="list every environment difference, not just the summary")

    integrity = sub.add_parser(
        "integrity", help="publication notices (retractions, concerns, corrections): plan, refresh, import, status")
    isub = integrity.add_subparsers(dest="integrity_command", required=True)
    for name, text in (("plan", "show the identifiers a refresh would send, and to whom (sends nothing)"),
                       ("refresh", "ask Europe PMC, PubMed and Crossref about these papers (needs the network and --yes)"),
                       ("status", "what this machine already knows about these papers (no network)")):
        cmd = isub.add_parser(name, help=text)
        target = cmd.add_mutually_exclusive_group(required=True)
        target.add_argument("--run", help="the papers a stored run cites")
        target.add_argument("--ids", nargs="+", help="identifiers: PMID:123, PMCID:PMC123, DOI:10.x/y")
        cmd.add_argument("--all-papers", action="store_true", help="with --run: every paper it pinned, not only those cited")
        cmd.add_argument("--providers", default="europepmc,pubmed,crossref",
                         help="comma-separated: europepmc, pubmed, crossref")
        cmd.add_argument("--db", help="database file (default: the configured one)")
        if name == "refresh":
            cmd.add_argument("--yes", action="store_true", help="send the requests (without it, nothing is sent)")
            cmd.add_argument("--bundle-out", type=Path, help="also write the raw responses to this offline bundle")
            cmd.add_argument("--no-store", action="store_true",
                             help="do not record in the database (on the machine that is allowed out)")
    for name, text in (("include", "add back a paper that was retracted and later reinstated (your decision)"),
                       ("exclude", "leave a reinstated paper out again")):
        cmd = isub.add_parser(name, help=text)
        cmd.add_argument("paper", help="PMID:123, PMID123, PMC123 or a DOI")
        cmd.add_argument("--by", required=True, help="who is deciding (recorded with the decision)")
        cmd.add_argument("--note", help="why")
        cmd.add_argument("--db", help="database file (default: the configured one)")
    imp = isub.add_parser("import", help="record an offline bundle carried in from a machine allowed out")
    imp.add_argument("bundle", type=Path)
    imp.add_argument("--db", help="database file (default: the configured one)")

    sub.add_parser("doctor", help="check this machine can run a review: server, models, window, structured "
                                  "output (sends no prompt)")

    runs = sub.add_parser("runs", help="list the most recent runs")
    runs.add_argument("--limit", type=int, default=20)

    forget = sub.add_parser("forget", help="delete one run's report and working notes (cannot be undone)")
    forget.add_argument("run_id")
    forget.add_argument("--yes", action="store_true", dest="yes",
                        help="do not ask for confirmation (for scripts)")

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


def _doctor() -> int:
    """Print every readiness check; exit 0 only when a review can run."""
    from bedrock.litreview.readiness import operational, review_readiness

    checks = asyncio.run(review_readiness())
    for check in checks:
        print(check.line())
    return 0 if operational(checks) else 1


def _aliases(args) -> int:
    """`aliases propose` and `aliases approve` (`knowledge/aliases.py`). Files
    only: nothing here reaches the network or a model."""
    from bedrock.litreview.knowledge import TABLE, Knowledge
    from bedrock.litreview.knowledge import aliases as al

    try:
        if args.aliases_command == "propose":
            if not args.hgnc and not args.mesh:
                _say("aliases propose: give --hgnc, --mesh or both")
                return 1
            if args.symbol and not args.hgnc:
                _say("aliases propose: --symbol needs --hgnc")
                return 1
            # The table `approve --table` will edit, so what is proposed and
            # what is refused are read against the same names.
            kb = Knowledge.load(args.table or TABLE)
            proposals, sources = [], []
            if args.hgnc:
                proposals += al.propose_targets(kb, al.read_hgnc(args.hgnc), args.symbol)
                sources.append({"kind": "hgnc", "file": args.hgnc.name, "sha256": al.sha256(args.hgnc)})
            if args.mesh:
                proposals += al.propose_diseases(kb, al.read_mesh(args.mesh))
                sources.append({"kind": "mesh", "file": args.mesh.name, "sha256": al.sha256(args.mesh)})
            al.write_proposals(proposals, sources, args.out)
            named = sum(len(p.names) for p in proposals)
            refused = sum(len(p.refused) for p in proposals)
            print(f"{args.out}: {named} names proposed for {len(proposals)} entities, {refused} refused. "
                  "Nothing is used until a person marks names `approve: true` and runs `aliases approve`.")
            return 0
        changes = al.approve(args.proposals, args.table or TABLE)
    except (al.AliasError, OSError) as exc:
        _say(f"aliases: {exc}")
        return 1
    if not changes:
        print("no names are marked `approve: true`; the table is unchanged")
        return 0
    for line in changes:
        print(line)
    print(f"wrote {args.table or TABLE}. Review the diff and commit it; searches for these entities now use "
          "the new names, so their frozen paper sets are collected again the next time a search is asked for.")
    return 0


def _run(args) -> int:
    """Everything after the arguments are read. Split out so `main` has exactly
    one place that turns a `ConfigError` into a message."""
    if args.command == "doctor":
        return _doctor()
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

    if args.command == "aliases":
        return _aliases(args)

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
        from bedrock.litreview.report.invariants import assess_run

        db = Database(args.db or _db_settings().db_path)
        if args.all:
            # Only runs that finished a report; an interrupted run has nothing to verify.
            run_ids = db.finished_run_ids()
        else:
            run_ids = [args.run_id]
        failed = 0
        for run_id in run_ids:
            try:
                assessment = assess_run(db, run_id)
            except Exception as exc:  # a missing pinned paper or an unreadable artifact is a failure too
                print(f"{run_id}: 1 violation(s)")
                print(f"  - cannot be verified: {type(exc).__name__}: {exc}")
                failed += 1
                continue
            violations = assessment.violations
            print(f"{run_id}: {'OK' if not violations else f'{len(violations)} violation(s)'}")
            for v in violations:
                print(f"  - {v}")
            # The environment is a disclosure beside the verdict, never one of
            # its violations: an intact report checked under changed rules is
            # still intact, and is said to be checked under changed rules.
            print(f"  {assessment.summary()}")
            env = assessment.environment
            shown = env.differences if args.details else [
                d for d in env.differences if d.kind in ("changed", "source_changed", "missing_now")]
            for d in shown:
                print(f"    ~ {d.line()}")
            if args.details:
                for note in env.notes:
                    print(f"    · {note}")
            # Notices recorded on this machine about the papers cited: advice
            # for a person, never a violation, never fetched here.
            for advisory in assessment.advisories:
                print(f"  ! {advisory}")
            failed += bool(assessment.exit_code(require_same_rules=args.require_same_rules))
        return 1 if failed else 0

    if args.command == "forget":
        return _forget(args)

    if args.command == "integrity":
        return _integrity(args)

    if args.command == "rate":
        from bedrock.litreview.api import rate_report

        rating = rate_report(args.report_id, args.score, rater=args.rater, comments=args.comment)
        print(rating.model_dump_json())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
