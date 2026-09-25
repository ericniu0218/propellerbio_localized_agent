"""Shapes used by the benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

SupportLabel = Literal["supported", "partially_supported", "unsupported", "contradicted"]
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


class Scorecard(BaseModel):
    question_id: str
    question: str
    report_id: str
    created: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    systems: list[SystemScore]
    pipeline: dict[str, float | int | str | None] = Field(default_factory=dict)
    gates: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
