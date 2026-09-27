"""Pipeline orchestrator — a question in, a cited report out. Resumable.

Python owns the control flow. Each stage is called in order, writes its output
to the database before the next begins, and a run that crashed at Stage 6 is
resumed from Stage 6 with `resume_run_id` — Stages 0-5 are read back, not
redone. (Model calls are cached as well, so even a fresh run replays them.)

    0 question   read the question: entities, exact terms, type, goal
    1 collect    papers from the searcher, ingested; frozen per question, and
                 pinned per run: the run stores a manifest of (paper id,
                 content hash), and a resumed or reopened run reads exactly
                 those versions, never the question's current set
    2 gate       exact-entity gate
    3 rank       relevance ranking of every collected paper
    4 plan       which information, from which papers
    5 extract    verified findings from the planned papers
    6 score      grouped evidence, targets, conflicts
    7 write      the report, section by section, under the citation guard
    8 render     JSON, Markdown, HTML

The model never decides what happens next.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import TypeAdapter

from bedrock.litreview.config import PACKAGE_DIR, Settings, config_file, load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.facets import facets_for
from bedrock.litreview.knowledge import TABLE as KNOWLEDGE_TABLE
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import load_prompt, make_client, version_of
from bedrock.litreview.models import (
    EvidencePlan,
    EvidenceSummary,
    EvidenceUnit,
    GateResult,
    Paper,
    QuestionSpec,
    RankedPaper,
    Report,
    Run,
    stable_id,
)
from bedrock.litreview.sources.base import PaperSearcher
from bedrock.litreview.sources.corpus import file_sha256
from bedrock.litreview.stages.s0_question import QuestionError, parse_question
from bedrock.litreview.stages.s1_collect import NothingFound, collect
from bedrock.litreview.stages.s2_gate import gate_papers
from bedrock.litreview.stages.s3_rank import rank_papers
from bedrock.litreview.stages.s4_plan import build_plan
from bedrock.litreview.stages.s5_extract import extract_plan
from bedrock.litreview.stages.s6_score import summarize
from bedrock.litreview.stages.s7_write import WriteContext, write_report
from bedrock.litreview.report.invariants import check_report
from bedrock.litreview.stages.s8_render import render, to_json

log = logging.getLogger(__name__)

_GATES = TypeAdapter(dict[str, GateResult])
_RANKED = TypeAdapter(list[RankedPaper])
_UNITS = TypeAdapter(list[EvidenceUnit])


@dataclass
class RunResult:
    run: Run
    spec: QuestionSpec
    papers: list[Paper]
    gates: dict[str, GateResult]
    ranked: list[RankedPaper]
    plan: EvidencePlan
    summary: EvidenceSummary
    report: Report
    report_id: str
    paths: dict[str, Path] = field(default_factory=dict)
    extract_stats: dict[str, int] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def input_files() -> list[Path]:
    """The config and knowledge files a run reads. Hashed onto the Run, so a
    past report can be matched to the tables and knobs it was written under.

    `config_file` returns the override in ~/.config/bedrock/litreview when there
    is one, so what is hashed is the file this run actually read, not the
    packaged default it replaced."""
    files = [
        config_file("models.yaml"),
        config_file("pipeline.yaml"),
        config_file("report.yaml"),
        KNOWLEDGE_TABLE,
    ]
    files += sorted(p for p in KNOWLEDGE_TABLE.parent.glob("*.yaml") if p != KNOWLEDGE_TABLE)
    return files


def _rel(path: Path) -> str:
    """A packaged file as `defaults/models.yaml`; anything else in full.

    An override lives outside the package, so it is recorded as the absolute
    path it was read from - which is the useful thing to know about it."""
    try:
        return path.resolve().relative_to(PACKAGE_DIR).as_posix()
    except ValueError:
        return str(path)


def _record_inputs(run: Run) -> None:
    """Hash the config and knowledge files as this invocation reads them.

    The first hash of each file is kept: it is what the run's earliest stages
    read. On a resume, a file that has changed since is recorded beside it as
    `resumed:<path>`, and a warning says so — the run was built under both."""
    current = {_rel(p): file_sha256(p) for p in input_files() if p.exists()}
    changed = sorted(k for k, v in current.items() if k in run.input_hashes and run.input_hashes[k] != v)
    if changed:
        log.warning("run %s: changed since its earlier stages ran: %s", run.id, ", ".join(changed))
    for k, v in current.items():
        if k not in run.input_hashes:
            run.input_hashes[k] = v
        elif run.input_hashes[k] != v:
            run.input_hashes[f"resumed:{k}"] = v


def code_version() -> str | None:
    """`git describe --always --dirty` of the package, else its installed
    version, else None. Recorded on the Run only — never in the report, whose
    bytes must not depend on it. Never raises."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=PACKAGE_DIR, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 — provenance is best effort; a run must not fail on it
        pass
    try:
        return importlib.metadata.version("bedrock-private-ai")
    except Exception:  # noqa: BLE001
        return None


def answer_question(
    question: str,
    *,
    settings: Settings | None = None,
    client=None,
    use_model: bool = True,
    searcher: PaperSearcher | None = None,
    papers: list[Paper] | None = None,
    db: Database | None = None,
    refresh_papers: bool = False,
    approved_terms: dict[str, list[str]] | None = None,
    out_dir: Path | None = None,
    resume_run_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    settings = settings or load_settings()
    say = progress or (lambda msg: log.info(msg))
    if client is None and use_model:
        client = make_client(settings)
    if not use_model:
        client = None
    db = db or Database(settings.db_path)
    kb = load_knowledge()
    m = settings.model_for

    run = db.get_run(resume_run_id) if resume_run_id else None
    if resume_run_id and run is None:
        # Starting a fresh run instead would look like a resume and silently
        # redo every stage; a mistyped id should say so.
        raise QuestionError(f"there is no run {resume_run_id!r} to resume; `bedrock-litreview runs` lists them")

    def load(name: str):
        return db.get_artifact(run.id, name) if run else None

    def reached(stage: int) -> bool:
        return run is not None and run.stage_reached >= stage

    # 0 — the question
    if reached(0):
        spec = QuestionSpec.model_validate_json(load("spec"))
    else:
        spec = parse_question(
            question, client, model=m("question") if client else None, knowledge=kb, approved_terms=approved_terms
        )
    say(f"question: {spec.question_type}, goal {spec.goal}, about "
        + ", ".join(f"{e.canonical} ({e.kind})" for e in spec.required_entities))

    if run is None:
        # Two runs of one question in the same second must not share an id:
        # the second would overwrite the first's artifacts, its paper
        # manifest included. The run id never reaches the report.
        started = _now()
        n = 0
        while db.get_run(run_id := stable_id("run", spec.id, started, *([n] if n else []), length=12)):
            n += 1
        run = Run(id=run_id, question_id=spec.id, started=started)
    run.model_versions = {job: version_of(client, m(job)) for job in ("question", "relevance", "extract", "write")} if client else {}
    run.prompt_versions = {name: load_prompt(name).version for name in ("question", "relevance", "extract", "write_section")}
    _record_inputs(run)
    run.code_version = code_version()
    db.save_question(spec)

    def done(stage: int, name: str | None = None, body: str | None = None) -> None:
        if name is not None and body is not None:
            db.put_artifact(run.id, name, body)
        run.stage_reached = max(run.stage_reached, stage)
        db.save_run(run)

    done(0, "spec", spec.model_dump_json())

    # 1 — collect (frozen per question unless asked to refresh). A resumed
    # run reads the exact paper versions it pinned, never the question's
    # current set: its gates, ranking and plan were built from those.
    pinned = db.get_artifact(run.id, "papers_manifest") if resume_run_id else None
    if pinned is not None:
        collected = db.get_paper_versions(json.loads(pinned))
        manifest_body = pinned
        ignored = " (papers given now are ignored)" if papers is not None or refresh_papers else ""
        say(f"papers: resuming on the {len(collected)} papers this run pinned{ignored}")
    else:
        if resume_run_id and reached(2):
            log.warning("run %s pinned no papers; resuming on the question's current set, "
                        "which may have changed since its later stages were built", run.id)
        frozen = db.get_papers(spec.id)
        if papers is None and frozen and not refresh_papers:
            collected = frozen
            manifest = db.save_paper_versions(frozen)
            say(f"papers: reusing the frozen set of {len(collected)} for this question")
        else:
            collected = collect(spec, searcher=searcher, papers=papers)
            if not collected:
                # A report over no papers says nothing and looks like an answer.
                raise NothingFound("no papers were found for this question: the search returned nothing, "
                               "or the corpus file is empty")
            manifest = db.save_papers(spec.id, collected)
            corpus_path = getattr(searcher, "path", None) if papers is None else None
            if corpus_path is not None:
                run.input_hashes[f"corpus:{corpus_path}"] = getattr(searcher, "sha256", None) or file_sha256(
                    Path(corpus_path)
                )
            say(f"papers: collected {len(collected)}")
        manifest_body = json.dumps([[pid, sha] for pid, sha in manifest])
    run.papers_sha256 = hashlib.sha256(manifest_body.encode("utf-8")).hexdigest()
    by_id = {p.id: p for p in collected}
    done(1, "papers_manifest", manifest_body)

    # 2 — gate
    if reached(2):
        gates = _GATES.validate_json(load("gates"))
    else:
        gates = gate_papers(collected, spec)
        done(2, "gates", _GATES.dump_json(gates).decode())
    passing = sum(1 for g in gates.values() if g.status == "pass")
    say(f"gate: {passing} of {len(collected)} concern the exact entity")

    # 3 — rank
    facets = facets_for(spec)
    if reached(3):
        ranked = _RANKED.validate_json(load("ranking"))
    else:
        ranked = rank_papers(
            collected, gates, spec, facets, client,
            model=m("relevance") if client else None,
            max_graded=int(settings.knob("rank.max_graded", 40)),
        )
        done(3, "ranking", _RANKED.dump_json(ranked).decode())
    say(f"rank: {sum(1 for r in ranked if r.grade is not None)} papers graded")

    # 4 — plan
    if reached(4):
        plan = EvidencePlan.model_validate_json(load("plan"))
    else:
        plan = build_plan(
            spec, facets, ranked, by_id,
            max_papers=int(settings.knob("plan.max_papers", 12)),
            min_grade=int(settings.knob("plan.min_grade", 2)),
            sections=tuple(settings.knob("extract.sections", ["abstract", "results"])),
            discussion_when_no_results=bool(settings.knob("extract.discussion_when_no_results", True)),
        )
        done(4, "plan", plan.model_dump_json())
    say(f"plan: read {len(plan.jobs)} papers; facets "
        + ", ".join(f"{fp.facet.id}={fp.status}" for fp in plan.facets))

    # 5 — extract
    if reached(5):
        units = _UNITS.validate_json(load("evidence"))
        extract_stats = json.loads(load("extract_stats") or "{}")
    elif client is None:
        units, extract_stats = [], {}
        done(5, "evidence", "[]")
    else:
        units, extract_stats = extract_plan(
            plan, by_id, spec, facets, client,
            model=m("extract"),
            chunk_chars=int(settings.knob("extract.chunk_chars", 6000)),
            max_chunks=int(settings.knob("extract.max_chunks_per_paper", 6)),
            methods_chars=int(settings.knob("extract.methods_context_chars", 2500)),
            knowledge=kb,
            progress=say,
        )
        db.put_artifact(run.id, "extract_stats", json.dumps(extract_stats, sort_keys=True))
        done(5, "evidence", _UNITS.dump_json(units).decode())
    say(f"extract: {len(units)} verified findings")

    # 6 — score
    summary = summarize(units, spec, facets, ranked, plan)
    done(6, "summary", summary.model_dump_json())

    # 7 — write
    ctx = WriteContext(
        spec=spec, facets=facets, plan=plan, summary=summary, ranked=ranked, papers=by_id, gates=gates,
        client=client, model=m("write") if client else None, knowledge=kb, extract_stats=extract_stats,
        max_targets=int(settings.knob("write.max_targets", 6)),
        units_per_target=int(settings.knob("write.units_per_target", 5)),
        max_units_per_section=int(settings.knob("write.max_units_per_section", 14)),
    )
    report = write_report(ctx, model_versions=run.model_versions, prompt_versions=run.prompt_versions)
    # The integrity check: everything the report shows as the paper's words is
    # the paper's words, on the papers this run pinned. A violation is a bug;
    # it is written on the report, which then opens with a warning.
    report.integrity = check_report(report, collected, spec, kb)
    if report.integrity:
        log.error("integrity check failed with %d violation(s): %s", len(report.integrity), report.integrity[:3])
        say(f"INTEGRITY CHECK FAILED: {len(report.integrity)} violation(s); see the top of the report")
    body = to_json(report)
    report_id = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    db.save_report(report_id, run.id, spec.id, body)
    done(7, "report", body)
    say(f"write: {report.stats.get('sentences_kept', 0)} sentences kept, "
        f"{report.stats.get('sentences_dropped', 0)} removed by the citation check, "
        f"{report.stats.get('sentences_repeated', 0)} as repeats")

    # 8 — render. The directory is named by content, so an identical re-run
    # lands on the same files.
    target_dir = Path(out_dir or settings.out_dir) / spec.id / report_id
    paths = render(report, target_dir)
    run.finished = _now()
    run.report_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    run_json = target_dir / "run.json"
    run_json.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    run_json.chmod(0o600)  # as the three report files are; see `s8_render.render`
    done(8)
    say(f"report: {paths['html']}")

    return RunResult(
        run=run, spec=spec, papers=collected, gates=gates, ranked=ranked, plan=plan, summary=summary,
        report=report, report_id=report_id, paths=paths, extract_stats=extract_stats,
    )


def load_result(db: Database, run_id: str, papers: list[Paper] | None = None) -> RunResult:
    """Re-open a finished run exactly as it was written, from its stored
    artifacts — nothing is recomputed. For scoring an earlier report against
    a later one after the code has changed (`python -m bedrock.litreview.bench run
    --from-run`)."""
    run = db.get_run(run_id)
    if run is None or run.stage_reached < 7:
        raise ValueError(f"run {run_id!r} has no finished report")

    def load(name: str) -> str:
        body = db.get_artifact(run_id, name)
        if body is None:
            raise ValueError(f"run {run_id!r} is missing its {name!r} artifact")
        return body

    spec = QuestionSpec.model_validate_json(load("spec"))
    if papers is not None:
        collected = papers
    elif (pinned := db.get_artifact(run_id, "papers_manifest")) is not None:
        collected = db.get_paper_versions(json.loads(pinned))
    else:
        log.warning("run %s is unpinned (written before paper manifests); using the question's current "
                    "papers, which may have changed since", run_id)
        collected = db.get_papers(spec.id)
    report_body = load("report")
    return RunResult(
        run=run,
        spec=spec,
        papers=collected,
        gates=_GATES.validate_json(load("gates")),
        ranked=_RANKED.validate_json(load("ranking")),
        plan=EvidencePlan.model_validate_json(load("plan")),
        summary=EvidenceSummary.model_validate_json(load("summary")),
        report=Report.model_validate_json(report_body),
        report_id=hashlib.sha256(report_body.encode("utf-8")).hexdigest()[:16],
        extract_stats=json.loads(db.get_artifact(run_id, "extract_stats") or "{}"),
    )

