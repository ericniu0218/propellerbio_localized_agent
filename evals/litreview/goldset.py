"""The gold set's shape, the checks that refuse a half-built one, and what the
pipeline reads of a paper.

Shared by `goldset_sheet.py`, which picks the papers and writes the labelling
sheets and an empty template, and `run_extract.py`, which scores extraction
against the filled-in set. The labels come from a scientist: the gold set
exists to check the model, so nothing here fills one in.

One entry per paper, one JSON object per line:

    {"paper_id": "PMID:123", "corpus": "corpora/psoriasis-prevention-targets",
     "question": "If I want to prevent psoriasis, what should I target?",
     "kind": "animal", "kind_confirmed": true, "holdout": false,
     "none_expected": false,
     "findings": [{"target": "IL-23", "direction": "block", "disease": "psoriasis",
                   "evidence_type": "in_vivo", "species": "mouse", "outcome": "improved",
                   "n": 12, "effect_detail": "ear thickness down 60%",
                   "quote": "Anti-IL-23 antibody ... reduced ear thickness by 60% ..."}]}

A line whose object has a `_comment` key is a note, and is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bedrock.litreview.config import EVALS_DIR, Settings
from bedrock.litreview.extract.verify import verify_quote
from bedrock.litreview.facets import facets_for
from bedrock.litreview.ingest import ingest
from bedrock.litreview.models import Direction, EvidenceType, ExtractionJob, Outcome, Paper, QuestionSpec
from bedrock.litreview.sources.corpus import read_papers
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import methods_context, plan_chunks

Kind = Literal["human_clinical", "animal", "in_vitro", "negative_result", "review", "mentions_not_about"]
QUOTAS: tuple[tuple[str, int], ...] = (
    ("human_clinical", 5), ("animal", 5), ("in_vitro", 4), ("negative_result", 3), ("review", 2),
    ("mentions_not_about", 1),
)
"""The mix in evals/litreview/README.md, "How to build it"."""
KIND_TITLES = {
    "human_clinical": "Human clinical trial",
    "animal": "Mouse / animal study",
    "in_vitro": "In vitro / cell study",
    "negative_result": "Negative or no-effect result",
    "review": "Review article",
    "mentions_not_about": "Mentions the subject but is not about it",
}
PROPOSED = ("negative_result", "mentions_not_about")
"""Kinds code can only propose; a person confirms them while reading."""
HOLDOUT = 5
PLACEHOLDER = "EXAMPLE-"
Species = Literal["human", "mouse", "rat", "cyno", "cell_line", "other"]
"""`EvidenceUnit.species`, which code normalises the model's words to."""


class GoldFinding(BaseModel):
    """One finding a person decided should be extracted, in the fields the
    scorer compares. `quote` is the paper's sentence, copied exactly."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1)
    direction: Direction
    disease: str = Field(min_length=1)
    evidence_type: EvidenceType
    species: Species
    outcome: Outcome
    n: int | None = Field(default=None, ge=1)
    effect_detail: str = ""
    quote: str = Field(min_length=1)


class GoldPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(min_length=1)
    corpus: str | None = None
    """The frozen corpus directory, relative to evals/litreview/ (or absolute)."""
    question: str | None = None
    """The question the paper was picked for; extraction is asked it."""
    doi: str | None = None
    title: str | None = None
    sheet: str | None = None
    kind: Kind
    kind_confirmed: bool = True
    """False for a kind code only proposed, until a person confirms it."""
    kind_proposed_because: str | None = None
    holdout: bool = False
    none_expected: bool = False
    """True when the paper should produce no finding at all - a review, a
    paper that is not about the subject. Required for an empty `findings`."""
    findings: list[GoldFinding] = Field(default_factory=list)


class GoldsetError(ValueError):
    """Every problem in a gold set, each naming the line, the paper and the field."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


def corpus_dir(corpus: str) -> Path:
    path = Path(corpus)
    return path if path.is_absolute() else EVALS_DIR / path


def corpus_name(path: Path) -> str:
    """How an entry names its corpus: relative to evals/litreview/ when inside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(EVALS_DIR.resolve()))
    except ValueError:
        return str(path)


def read_corpus(corpus: str) -> dict[str, Paper]:
    """A frozen corpus, prepared the way the pipeline prepares papers."""
    return {p.id: p for p in ingest(read_papers(corpus_dir(corpus) / "papers.jsonl"))}


def _loc(loc: tuple) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def load_goldset(path: Path) -> list[GoldPaper]:
    """Read and check a gold set. Raises `GoldsetError` listing every problem
    - never scores a set that is half template."""
    path = Path(path)
    if not path.exists():
        raise GoldsetError([f"No gold set at {path}. Build it first: see evals/litreview/README.md, "
                            "\"Labelling the gold set\"."])
    problems: list[str] = []
    entries: list[GoldPaper] = []
    corpora: dict[str, dict[str, Paper] | None] = {}
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: not JSON ({exc})")
            continue
        if "_comment" in row:
            continue
        pid = row.get("paper_id", "?")
        where = f"line {number}, paper {pid}"
        if str(pid).startswith(PLACEHOLDER):
            problems.append(f"{where}: a placeholder from the template, not a labelled paper; delete it")
            continue
        try:
            entry = GoldPaper.model_validate(row)
        except ValidationError as exc:
            problems += [f"{where}, field {_loc(e['loc'])}: {e['msg']}" for e in exc.errors()]
            continue
        if entry.paper_id in seen:
            problems.append(f"{where}: listed twice")
        seen.add(entry.paper_id)
        if not entry.kind_confirmed:
            problems.append(f"{where}, field kind_confirmed: code proposed kind {entry.kind!r} "
                            f"({entry.kind_proposed_because or 'no reason recorded'}); confirm or correct it "
                            "while reading, then set kind_confirmed to true")
        if entry.kind == "review" and entry.findings:
            problems.append(f"{where}, field findings: a review article must expect zero findings - a "
                            "review citing a trial is not a trial")
        if entry.none_expected and entry.findings:
            problems.append(f"{where}, field none_expected: true, but findings are listed")
        if not entry.findings and not entry.none_expected:
            problems.append(f"{where}, field findings: empty. If the paper really has nothing to extract, "
                            "say so with \"none_expected\": true; otherwise it is not labelled yet")
        if entry.corpus is None:
            problems.append(f"{where}, field corpus: missing; the scorer reads the paper from its corpus")
        else:
            if entry.corpus not in corpora:
                try:
                    corpora[entry.corpus] = read_corpus(entry.corpus)
                except FileNotFoundError:
                    corpora[entry.corpus] = None
            papers = corpora[entry.corpus]
            if papers is None:
                problems.append(f"{where}, field corpus: no corpus at {corpus_dir(entry.corpus)}")
            elif entry.paper_id not in papers:
                problems.append(f"{where}, field paper_id: not in {entry.corpus}")
            else:
                texts = [t for t in (papers[entry.paper_id].title, *papers[entry.paper_id].sections.values()) if t]
                for i, f in enumerate(entry.findings):
                    if not any(verify_quote(f.quote, t) != "failed" for t in texts):
                        problems.append(f"{where}, field findings[{i}].quote: not in the paper's text; copy "
                                        "the sentence exactly")
        if entry.question is None:
            problems.append(f"{where}, field question: missing; extraction is asked the question the paper "
                            "was picked for")
        entries.append(entry)
    if problems:
        raise GoldsetError(problems)
    return entries


# --- what the pipeline reads of a paper -----------------------------------------------------------


def read_question(question: str) -> QuestionSpec:
    """The question read by code alone. The sheet and the scorer both use this,
    so the parts of a long paper they pick are the same; the model's reading
    can add a name the knowledge table lacks, which the gold set does not test."""
    return parse_question(question)


def extraction_job(paper: Paper, spec: QuestionSpec, settings: Settings) -> ExtractionJob:
    """The job Stage 4 gives Stage 5 for this paper (`s4_plan.build_plan`),
    with every facet, because a gold-set paper is not ranked or tagged."""
    wanted = [s for s in settings.knob("extract.sections", ["abstract", "results"]) if paper.sections.get(s)]
    if (settings.knob("extract.discussion_when_no_results", True) and "results" not in paper.sections
            and paper.sections.get("discussion")):
        wanted.append("discussion")
    if not wanted and paper.abstract:
        wanted = ["abstract"]
    return ExtractionJob(paper_id=paper.id, sections=wanted, facets=[f.id for f in facets_for(spec)],
                         abstract_only=not paper.has_full_text)


def read_by_pipeline(paper: Paper, spec: QuestionSpec, settings: Settings) -> tuple[list[tuple[str, int, str]], str]:
    """The chunks Stage 5 sends the model for this paper, and the methods
    context it sends beside each. Mirrors `s5_extract.extract_paper`."""
    job = extraction_job(paper, spec, settings)
    query = f"{spec.text} {' '.join(t for e in spec.entities for t in e.exact_terms)}"
    chunks = plan_chunks(paper, job, query, chunk_chars=int(settings.knob("extract.chunk_chars", 6000)),
                         max_chunks=int(settings.knob("extract.max_chunks_per_paper", 6)))
    methods = methods_context(paper, int(settings.knob("extract.methods_context_chars", 2500)))
    return chunks, methods
