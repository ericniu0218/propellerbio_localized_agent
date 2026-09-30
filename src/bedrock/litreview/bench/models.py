"""Shapes used by the benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

SupportLabel = Literal["supported", "partially_supported", "unsupported", "contradicted"]
GateStatus = Literal["pass", "fail", "not_measured"]
CoverageLabel = Literal["covered", "partial", "not_covered"]

SUPPORT_SCORE = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0, "contradicted": 0.0}
COVERAGE_SCORE = {"covered": 1.0, "partial": 0.5, "not_covered": 0.0}


class CorpusSpec(BaseModel):
    queries: list[str]
    max_per_query: int = 40
    near_miss_queries: list[str] = Field(default_factory=list)
    near_miss_max: int = 6
    full_text_max: int = 60


class ReferenceSpec(BaseModel):
    reviews_query: str | None = None
    reviews_max: int = 5
    open_targets_disease: str | None = None
    open_targets_min_phase: float = 3


class BenchQuestion(BaseModel):
    """One benchmark question: what to ask, how its sample corpus was found,
    and which external references grade the answer."""

    id: str
    question: str
    corpus: CorpusSpec
    references: ReferenceSpec = Field(default_factory=ReferenceSpec)
    notes: str | None = None

    @classmethod
    def load(cls, path: Path) -> "BenchQuestion":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# --- judge outputs (these are JSON schemas for guided decoding) ----------------


class SupportAnswers(BaseModel):
    """What the judge model returns about one claim: where the evidence says
    it, and what it cannot find. Code decides the verdict
    (`judge.verdict_from`) — it verifies the sentence, checks every name and
    number in the claim against the evidence, and compares which way the
    claim and the sentence say the effect went.

    Two earlier designs failed on MedGemma 4B, in opposite directions: asked
    for a verdict it rejected claims its own reasoning said were stated;
    asked "is the main point stated?" it said yes to almost everything,
    including claims with the effect reversed.
    """

    reasoning: str = Field(max_length=300, description="One short sentence")
    evidence_sentence: str = Field(
        max_length=700,
        description="The one sentence from the evidence that states the claim, copied word for word; "
        "empty if no sentence does",
    )
    missing: list[str] = Field(
        max_length=6,
        description="Details copied from the claim that the evidence does not state; empty if none",
    )


class CoverageAnswers(BaseModel):
    """What the judge model returns about one reference fact and one answer:
    the answer's sentence that states it. Code decides the coverage."""

    reasoning: str = Field(max_length=300, description="One short sentence")
    answer_sentence: str = Field(
        max_length=700,
        description="The sentence from the answer that states the fact, copied word for word; empty if none",
    )


class SupportVerdict(BaseModel):
    reasoning: str
    verdict: SupportLabel
    unsupported_part: str | None = None


class NuggetOut(BaseModel):
    fact: str
    quote: str
    importance: Literal["vital", "okay"]


class NuggetList(BaseModel):
    nuggets: list[NuggetOut] = Field(default_factory=list)


class CoverageVerdict(BaseModel):
    reasoning: str
    verdict: CoverageLabel


# --- results ------------------------------------------------------------------


class Claim(BaseModel):
    id: str
    system: str
    section: str = ""
    text: str
    citations: list[str] = Field(default_factory=list)
    """Pipeline: short evidence ids. Baselines: paper ids."""


class ClaimResult(BaseModel):
    claim: Claim
    cited_support: SupportLabel | None = None
    cited_reasoning: str | None = None
    corpus_support: SupportLabel | None = None
    numbers_ok: bool | None = None
    near_misses: list[str] = Field(default_factory=list)


class Nugget(BaseModel):
    id: str
    fact: str
    quote: str
    importance: str
    source: str
    reachable: bool | None = None
    coverage: dict[str, CoverageLabel] = Field(default_factory=dict)


JUDGED_SCORES = ("cited_support", "corpus_grounded", "nugget_recall", "vital_nugget_recall")
"""The scores a judge's verdicts decide. The rest are counted in code."""

OVERALL_PARTS = ("cited_support", "corpus_grounded", "nugget_recall")
"""The judged scores `run.overall` is built from."""


class SystemScore(BaseModel):
    system: str
    claims: int = 0
    cited_support: float | None = None
    corpus_grounded: float | None = None
    numeric_fidelity: float | None = None
    near_miss_claims: int = 0
    nugget_recall: float | None = None
    vital_nugget_recall: float | None = None
    target_recall: float | None = None
    quantitative_supported_claims: int = 0
    distinct_sources_cited: int = 0
    overall: float | None = None
    judged: dict[str, tuple[int, int]] = Field(default_factory=dict)
    """Judged score -> (items it rests on, items it should rest on). A claim
    or review fact with no verdict - no answer on file, a failed judge call -
    is left out of its score, so 0.93 over 30 of 42 claims is not 0.93 over
    42 (OPEN_ISSUES #28). Empty on scorecards written before 30 September 2026."""

    def judged_fraction(self, score: str) -> float | None:
        pair = self.judged.get(score)
        return pair[0] / pair[1] if pair and pair[1] else None

    def overall_judged_fraction(self) -> float | None:
        """The lowest judged fraction among the scores `overall` is built from:
        the overall is only as judged as its least-judged part."""
        parts = [f for f in (self.judged_fraction(s) for s in OVERALL_PARTS) if f is not None]
        return min(parts) if parts else None


class Gate(BaseModel):
    """One pipeline check: it passed, it failed, or it was not measured.

    Two states were not enough. A gate stored as a bool, computed as
    "value is None or value == 0", passed when its input was never built:
    obesity's "no distractor leaked" PASSed on 26 September with no
    distractor made at all, and "every gate passes" was repeated from that
    run as a measurement (OPEN_ISSUES #27). Now a gate with no input says so,
    and why, and is left out of both counts.
    """

    status: GateStatus
    reason: str | None = None
    """Why the gate was not measured: which input was missing."""

    @model_validator(mode="before")
    @classmethod
    def _from_bool(cls, data):
        # Scorecards written before 30 September 2026 stored a bool - and
        # counted a gate with no input as passed, which cannot be undone here.
        if isinstance(data, bool):
            return {"status": "pass" if data else "fail"}
        return data

    def __bool__(self) -> bool:
        # `if gate:` is the bug this class exists to end: a gate that was
        # not measured would read as passed.
        raise TypeError("a gate is pass, fail or not measured; read its .status")


class Scorecard(BaseModel):
    question_id: str
    question: str
    report_id: str
    created: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    context_windows: dict[str, int] | None = None
    """Model -> the context window `service.preflight` established for this
    run's own model calls. None when the run called no model (a stored report
    re-scored from a judge file), and on every scorecard written before the
    benchmark ran preflight."""
    report_context_windows: dict[str, int] | None = None
    """The windows the scored report was written at: this run's own, or, for a
    stored report (`--from-run`), what the run that wrote it recorded. None
    means not recorded, which is every report written before preflight ran."""
    report_context_source: str | None = None
    """Which run recorded `report_context_windows`: "this run", or the
    `created` stamp of the history row it was read from."""
    raters: dict[str, int] = Field(default_factory=dict)
    """Who supplied the verdicts: rater -> answers used. A model judge is its
    model version; a judge file names a judge per answer (`judge` in
    answers.jsonl: a model id or `scientist:<name>`, "unknown" on rows written
    before it was recorded). `code` counts verdicts settled in code (a claim
    verbatim in its evidence, or with none). Empty on scorecards written
    before 30 September 2026 (OPEN_ISSUES #25)."""
    systems: list[SystemScore]
    pipeline: dict[str, float | int | str | None] = Field(default_factory=dict)
    gates: dict[str, Gate] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def gates_passed(self) -> int:
        return sum(1 for g in self.gates.values() if g.status == "pass")

    @property
    def gates_measured(self) -> int:
        """The gates that measured something: the denominator of "passed"."""
        return sum(1 for g in self.gates.values() if g.status != "not_measured")

    @property
    def gates_not_measured(self) -> dict[str, str]:
        """Gate -> why it was not measured."""
        return {name: g.reason or "" for name, g in self.gates.items() if g.status == "not_measured"}
