"""The data model. Read this file first.

A scientist asks a question — "if I want to prevent psoriasis, what should I
target?" — and gets back a cited report. Everything in between is a view of a
table of `EvidenceUnit`s: the ranking decides which papers are worth reading,
the plan decides which of them to read for what, extraction fills the table,
and the report, the benchmark and the chat all read from it.

The shapes, in the order a question flows through them:

    QuestionSpec      what was asked, with the exact entity names that count
    Paper             what the searcher handed us
    GateResult        does this paper concern the *exact* entity asked about
    RankedPaper       how relevant it is to the question, and why
    EvidencePlan      which information we need, from which papers
    EvidenceUnit      one verified finding
    EvidenceSummary   the table, grouped (targets, conflicts, facets)
    Report            the answer, sentence by sentence, each with citations

Change this file only by agreement — every stage and the review UI read it.

Rules enforced elsewhere in code, not by convention:

1. Every EvidenceUnit carries a verbatim `quote`, and `extract/verify.py`
   string-matches it against the source text. No match, the unit is dropped.
   The quote stored is the paper's own text at a recorded position, never the
   model's copy of it, and every other field the model wrote is located in the
   text it was shown or blanked (`provenance`).

2. No model ever scores paper quality. `PaperSignals` holds objective facts
   looked up in code; a human decides what they mean. A model *does* grade
   relevance to the question (`RankedPaper.grade`) — that is a different
   question, and its justification is a verified quote like everything else.

3. The exact entity is exact. A question about RSV is answered from RSV papers,
   not human metapneumovirus papers that "look close enough". `QuestionEntity`
   carries the names that count and the near-misses that must not.

4. Every sentence in a report cites evidence, and code checks that the numbers
   and names in it appear in what it cites. A sentence that fails is dropped.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_serializer

# --- enums, as literal types so the JSON schema constrains generation -------

Direction = Literal["block", "activate", "unclear"]
Modality = Literal[
    "antibody", "peptide", "siRNA", "small_molecule", "vaccine", "genetic", "other"
]
EvidenceType = Literal["in_vitro", "in_vivo", "human_clinical", "human_genetic"]
Outcome = Literal["improved", "no_effect", "worsened", "mixed", "association"]
"""`association` is an observation without an intervention — "IL-17A is raised
in lesional skin". It is evidence about mechanism, never about efficacy."""
Setting = Literal["prevention", "treatment", "unclear"]
Section = Literal["abstract", "results", "discussion", "methods", "other"]
QuoteStatus = Literal["exact", "fuzzy", "failed"]
ReviewStatus = Literal["unreviewed", "confirmed", "corrected", "rejected"]

QuestionType = Literal[
    "target_identification", "intervention_efficacy", "mechanism", "safety", "other"
]
Goal = Literal["prevent", "treat", "understand", "other"]
EntityKind = Literal["disease", "pathogen", "drug", "target", "species", "other"]
GateStatus = Literal["pass", "near_miss", "absent", "review", "retracted"]
StudyKind = Literal["primary_research", "review", "other"]
# How the question's subject figures in a paper: what it studies, a risk factor
# or comorbidity of something else it studies, or a passing mention.
SubjectRole = Literal["studied", "exposure", "passing"]


def stable_id(*parts: object, length: int = 16) -> str:
    """Deterministic id from content. `uuid4()` would make two runs of the same
    question produce different ids, and different ids mean different bytes in
    the report — which breaks the one acceptance test the client can run."""
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


# --- the question ------------------------------------------------------------


class QuestionEntity(BaseModel):
    """One thing the question is about, and exactly which names count as it.

    `text` is copied verbatim from the question and verified against it, the
    same way quotes are verified against papers. `exact_terms` and
    `near_miss_terms` come from the knowledge table in code, never from the
    model: the model does not get to decide that "close enough" is the same.
    """

    text: str
    kind: EntityKind
    canonical: str
    exact_terms: list[str]
    near_miss_terms: list[str] = Field(default_factory=list)
    near_miss_patterns: list[str] = Field(default_factory=list)
    required: bool = True
    """A required entity must appear in a paper for the paper to count at all."""
    resolved_from: Literal["knowledge_table", "verbatim", "approved"] = "verbatim"


class QuestionSpec(BaseModel):
    """What was asked. Frozen once a run starts."""

    id: str
    text: str
    question_type: QuestionType
    goal: Goal
    entities: list[QuestionEntity]
    population: str | None = None
    parsed_by: str = "rules"
    prompt_version: str | None = None

    @property
    def required_entities(self) -> list[QuestionEntity]:
        return [e for e in self.entities if e.required]

    def first(self, *kinds: str) -> QuestionEntity | None:
        for kind in kinds:
            for entity in self.entities:
                if entity.kind == kind:
                    return entity
        return None

    @property
    def subject(self) -> QuestionEntity | None:
        """The thing the answer is about: the disease or pathogen, else the drug."""
        return self.first("disease", "pathogen", "drug", "target", "other")


# --- models' raw output ------------------------------------------------------


class ParsedEntity(BaseModel):
    text: str = Field(description="Copied exactly from the question. Never reworded.")
    kind: EntityKind


class ParsedQuestion(BaseModel):
    """What the question-reading model returns. Code resolves the rest."""

    question_type: QuestionType
    goal: Goal
    entities: list[ParsedEntity] = Field(default_factory=list)
    population: str | None = Field(
        default=None, description="Who the question is about, copied exactly, e.g. 'infants'"
    )


class RelevanceJudgement(BaseModel):
    """What the relevance model returns for one paper's title and abstract.

    The model answers narrow questions; code turns the answers into a grade
    (`stages/s3_rank.grade_from`). Asked for a 0-3 grade directly, a small
    model gives nearly every on-topic paper a 3, and asked which information
    needs a paper supplies it ticks all of them in the order listed. Asked
    "is the disease what this paper studies, or a risk factor for something
    else?" and "which molecule did the paper's own experiments change?", it
    can be right, and the second answer can be checked against the abstract.

    Field order is generation order under guided decoding: the evidence
    sentence and the reason come before the decisions they support.
    """

    quote: str = Field(
        max_length=700, description="The sentence from the title or abstract that best shows what the paper found"
    )
    reason: str = Field(max_length=300, description="One short line: what the paper's own work did and found")
    about_exact_entity: bool = Field(
        description="True only if the paper studies the exact entity asked about"
    )
    subject_role: SubjectRole
    study_kind: StudyKind
    tested: str = Field(
        max_length=120,
        description="The molecule, pathway, cell type or drug the paper's own experiments changed, "
        "copied from the abstract; empty if none"
    )


class ExtractedFinding(BaseModel):
    """One finding, exactly as the model returns it — raw, unnormalised.

    Deliberately separate from `EvidenceUnit`. The model's job is to copy what
    the paper says; turning "C57BL/6 mice" into "mouse" is code's job
    (`extract/normalize.py`). Asking the model to do both adds a second way to
    be wrong, and a non-deterministic one.

    This class is compiled into the JSON schema handed to guided decoding, so
    the model is physically unable to return a bad shape. `facet` is narrowed
    to the plan's facet ids at call time.
    """

    quote: str = Field(max_length=700, description="The verbatim sentence this came from. Never paraphrase.")
    target: str = Field(max_length=120, description="The molecule, exactly as the paper writes it")
    change: str = Field(
        default="",
        max_length=120,
        description="The words from the quote saying what was done to the target, e.g. 'deficiency', "
        "'knockout', 'blocking', 'overexpression'; empty if nothing was done to it",
    )
    direction: Direction
    modality: Modality
    intervention: str | None = Field(
        default=None, max_length=160, description="The drug, antibody or agent used, as written; null if none"
    )
    disease: str = Field(max_length=160, description="The disease or model, as the paper writes it")
    evidence_type: EvidenceType
    species: str = Field(max_length=120, description="As written, e.g. 'C57BL/6 mice', 'healthy volunteers'")
    effect: str = Field(
        default="",
        max_length=120,
        description="The words from the quote saying what happened to the disease, e.g. 'aggravated', "
        "'reduced', 'did not significantly reduce'; empty if the sentence does not say",
    )
    outcome: Outcome
    setting: Setting
    effect_detail: str = Field(max_length=300, description="The concrete result with numbers where given")
    n: int | None = Field(default=None, description="Subjects/animals, null if not stated")
    combo_with: str | None = Field(
        default=None, max_length=120, description="Second target, if the finding is about a combination"
    )
    target_is_primary: bool = Field(
        description="False if the paper is really about something else and only mentions this"
    )
    facet: str


class ExtractionResult(BaseModel):
    """The whole reply for one chunk of one section. An empty list is a correct answer."""

    # Bounded so a small model that starts repeating itself stops inside the
    # token budget instead of being cut off mid-reply (the MedGemma run lost
    # two results chunks that way). Rule 8 of the prompt says the same.
    findings: list[ExtractedFinding] = Field(default_factory=list, max_length=8)
    is_review: bool = Field(
        default=False,
        description="True if this text summarises other people's work rather than reporting its own",
    )


class DraftSentence(BaseModel):
    # Citations first: the model commits to which evidence it is using before
    # it writes the sentence, rather than writing and then finding support.
    citations: list[str] = Field(description="Evidence ids, e.g. E3. At least one.", max_length=6)
    text: str = Field(max_length=450)


class DraftSection(BaseModel):
    """What the writing model returns for one report section.

    Bounded, so a model that starts repeating itself stops inside its token
    budget: the MedGemma run lost a whole section to a reply that was cut off.
    """

    sentences: list[DraftSentence] = Field(default_factory=list, max_length=10)


# --- papers ------------------------------------------------------------------


class PaperSignals(BaseModel):
    """Objective facts, looked up in code. No model involvement, ever."""

    journal: str | None = None
    year: int | None = None
    citation_count: int | None = None
    is_preprint: bool = False
    is_retracted: bool = False
    open_access: bool = False
    publication_types: list[str] = Field(default_factory=list)


TextSource = Literal[
    "europepmc_xml", "grobid_pdf", "drop_folder", "flat_text", "abstract_only", "none"
]


class Paper(BaseModel):
    """A paper the searcher found. `sections` may be empty — that is normal
    and must be visible everywhere, since abstract-only evidence is weaker.

    The searcher may hand us sections already split, or raw full text in
    `full_text` with its `full_text_format`; ingestion splits the latter.
    """

    id: str
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    published: date | None = None
    url: str | None = None

    signals: PaperSignals = Field(default_factory=PaperSignals)

    text_source: TextSource = "none"
    sections: dict[str, str] = Field(default_factory=dict)
    """Section name -> text. Empty when we only have an abstract."""
    full_text: str | None = None
    full_text_format: Literal["jats", "tei", "text"] | None = None

    found_by_queries: list[str] = Field(default_factory=list)
    """Which query produced this hit. Needed to explain coverage."""
    source: str | None = None
    """Which searcher handed it over."""

    is_review: bool = False
    synthetic: bool = False
    """True only for benchmark distractors. Never set by a searcher."""

    merged_from: list[dict[str, object]] = Field(default_factory=list)
    """The duplicate records folded into this one: which source, which ids and
    which flags each carried. A review or retraction flag on any copy is kept
    on the merged paper; this list is how that can be checked."""
    signal_conflicts: list[str] = Field(default_factory=list)
    """Where the merged copies disagreed (year, journal, preprint status)."""

    @property
    def has_full_text(self) -> bool:
        return any(name in self.sections for name in ("results", "discussion", "methods"))

    def head_text(self) -> str:
        """Title and abstract: what relevance is judged on."""
        return f"{self.title}\n\n{self.abstract or ''}".strip()

    def all_text(self) -> str:
        parts = [self.title, self.abstract or ""]
        parts.extend(text for name, text in self.sections.items() if name != "abstract")
        return "\n\n".join(p for p in parts if p)


class PatentRecord(BaseModel):
    """Patents are a separate, lower-confidence class. They feed the competitor
    landscape section only, never the biological evidence sections.

    We deliberately store claim 1 and nothing else from the body — patents are
    long, legally worded, and hard for a small model to read correctly.
    """

    id: str
    title: str
    abstract: str | None = None
    assignee: str | None = None
    priority_date: date | None = None
    claim_1: str | None = None
    jurisdiction: str | None = None


# --- gate, rank, plan --------------------------------------------------------


class GateResult(BaseModel):
    """Does this paper concern the exact entity asked about? Pure code."""

    paper_id: str
    status: GateStatus
    reason: str
    exact_hits: dict[str, int] = Field(default_factory=dict)
    near_miss_hits: dict[str, list[str]] = Field(default_factory=dict)


class RankedPaper(BaseModel):
    """One paper's place in the ranking for one question, with its reasons."""

    paper_id: str
    rank: int
    gate: GateStatus
    gate_reason: str = ""
    grade: int | None = None
    """0-3 relevance to the question, computed in code from the model's answers
    (`s3_rank.grade_from`). None if not graded."""
    about_exact_entity: bool | None = None
    subject_role: SubjectRole | None = None
    study_kind: StudyKind | None = None
    tested: str | None = None
    """What the paper's own experiments changed, as the model copied it — kept
    only if code finds it in the title or abstract."""
    facets: list[str] = Field(default_factory=list)
    """The information needs this paper can supply. Assigned by code from the
    abstract's study-design words (`facets.paper_facets`), not by the model."""
    quote: str | None = None
    quote_status: QuoteStatus | None = None
    reason: str | None = None
    lexical_score: float = 0.0
    exact_mentions: int = 0
    evidence_hints: list[str] = Field(default_factory=list)
    """Study-design words found in the abstract by code: 'human_clinical' etc."""
    has_full_text: bool = False
    graded_by: str | None = None

    @property
    def usable(self) -> bool:
        return self.gate == "pass" and self.study_kind != "review"


class Facet(BaseModel):
    """One piece of information the answer needs. Templates live in code
    (`litreview/facets.py`), keyed by question type — the model never plans."""

    id: str
    title: str
    need: str
    """What information is needed, in words the extraction prompt can use."""
    fields: list[str] = Field(default_factory=list)
    min_sources: int = 1
    max_sources: int = 4


class FacetPlan(BaseModel):
    facet: Facet
    paper_ids: list[str] = Field(default_factory=list)
    status: Literal["covered", "thin", "gap"] = "gap"
    shortfall: int = 0


class ExtractionJob(BaseModel):
    paper_id: str
    sections: list[str]
    facets: list[str]
    abstract_only: bool = False


class EvidencePlan(BaseModel):
    """Which information the report needs, from how many papers, and which.

    Built by code from the ranking. A facet the corpus cannot support is a
    `gap`, and the report says so rather than filling it from anywhere else.
    """

    question_id: str
    facets: list[FacetPlan]
    jobs: list[ExtractionJob]
    best_source_id: str | None = None

    @property
    def paper_ids(self) -> list[str]:
        return [job.paper_id for job in self.jobs]


# --- evidence ----------------------------------------------------------------


class EvidenceUnit(BaseModel):
    """A verified, normalised finding. The unit of everything downstream."""

    id: str
    paper_id: str
    question_id: str

    # normalised by extract/normalize.py and the knowledge table, so an
    # extracted "IL-23p19" and "interleukin-23" land in the same row.
    target: str
    target_raw: str
    direction: Direction
    modality: Modality
    intervention: str | None = None
    disease: str
    disease_raw: str
    evidence_type: EvidenceType
    species: str  # canonical: human | mouse | rat | cyno | cell_line | other
    species_raw: str
    outcome: Outcome
    setting: Setting = "unclear"
    effect_detail: str
    n: int | None
    combo_with: str | None
    target_is_primary: bool
    facet: str | None = None

    # provenance — how this unit came to exist, and how far to trust it
    quote: str
    """The paper's own words at `quote_start:quote_end` of `section` — never
    the model's copy, which is kept in `quote_model` when it differed."""
    quote_status: QuoteStatus
    section: Section
    quote_start: int | None = None
    quote_end: int | None = None
    """Character offsets into `Paper.sections[section]`. None only for units
    written before quotes were anchored."""
    quote_model: str | None = None
    """What the model returned, when it differed from the paper by typography
    or a harmless word. Audit only: never displayed, never cited."""
    provenance: dict[str, str] = Field(default_factory=dict)
    """Where each model-written field was found in the text the model was
    shown: field -> 'quote' | 'chunk' | 'methods' | 'none'. A field found
    nowhere is blanked; a target found nowhere drops the unit. The target,
    agent and combination partner record where the reader is shown them
    instead: 'quote' | 'context' | 'qualifier' | 'naming' (units stored
    before that say 'chunk' or 'methods'), and 'naming' -> 'chunk' |
    'methods' says where the naming sentence came from."""
    context_quote: str | None = None
    """The paper's own sentence that supplied the subjects or sample size,
    when that was not the quote itself (usually a Methods sentence)."""
    context_section: str | None = None
    context_start: int | None = None
    context_end: int | None = None
    qualifier_quote: str | None = None
    """The paper's sentence right after the quote, when it qualifies it
    ("Although this result did not reach statistical significance ...").
    Same section as the quote, at `qualifier_start:qualifier_end`. Found by
    `extract/verify.py` `qualifier_after`, never by the model."""
    qualifier_start: int | None = None
    qualifier_end: int | None = None
    naming_quote: str | None = None
    """The paper's sentence that names the target or agent, when the quote,
    the context sentence and the qualifier do not ("HDAC6 inhibitors, such as
    tubastatin A, reduce food intake ..." beside a quote about tubastatin A).
    From the passage the model read, else the Methods; in `naming_section` at
    `naming_start:naming_end`. Found by `extract/provenance.py`, never by the
    model. It shows where a name comes from; it is not evidence the quote's
    result is about that name (`report/guard.py`, check 9)."""
    naming_section: str | None = None
    naming_start: int | None = None
    naming_end: int | None = None
    methods_subjects: str | None = None
    """Set when the quote is the authors' conclusion and names people ("...
    in patients with asthma"), but the paper's Methods name no human
    subjects, only animals or cells: what they do name, "mice and cells".
    The finding's subjects and evidence type then come from the Methods, not
    the quote, the Methods sentence is its context sentence, and the citation
    guard lets no sentence resting only on such findings say "patients"
    (`report/guard.py`, check 4). Found by `extract/provenance.py`
    `methods_subjects`, never by the model."""
    from_discussion: bool = False
    """Authors overstate in the Discussion. Kept, but flagged."""
    entity_match: Literal["exact", "paper_context"] = "paper_context"
    corrected: list[str] = Field(default_factory=list)
    """Fields code overruled because the model's own copied words, or the
    quote's own words, said otherwise, e.g. 'outcome: improved -> worsened
    ("aggravated")' or 'evidence_type: in_vivo -> in_vitro ("HBECs")'."""
    """`exact`: the finding itself names the asked-about entity. `paper_context`:
    it does not, but the paper passed the gate, so the context supplies it."""

    extracted_by_model: str
    prompt_version: str

    # human review
    review_status: ReviewStatus = "unreviewed"
    reviewed_by: str | None = None
    corrected_fields: list[str] = Field(default_factory=list)
    """Which fields a human changed. This is the safety net, and after months
    of use it is also the only realistic path to fine-tuning anything."""

    @model_serializer(mode="wrap")
    def _omit_absent_optional(self, handler):
        """Leave the qualifier, naming-sentence and Methods-subjects fields
        out of the JSON when there is none.

        They were added after reports were already stored, and a stored run
        is verified by re-serialising its report and comparing the bytes
        (`report/invariants.py` `check_pinned`). Writing `"qualifier_quote":
        null` into every old unit would make every old run fail that check."""
        data = handler(self)
        if isinstance(data, dict):
            for group in (("qualifier_quote", "qualifier_start", "qualifier_end"),
                          ("naming_quote", "naming_section", "naming_start", "naming_end"),
                          ("methods_subjects",)):
                if data.get(group[0]) is None:
                    for name in group:
                        data.pop(name, None)
        return data


class TargetSummary(BaseModel):
    """Everything the evidence table says about one target. Pure code."""

    target: str
    names_seen: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    unit_ids: list[str] = Field(default_factory=list)
    by_evidence_type: dict[str, int] = Field(default_factory=dict)
    by_outcome: dict[str, int] = Field(default_factory=dict)
    prevention_units: int = 0
    interventions: list[str] = Field(default_factory=list)
    conflict: bool = False


class Conflict(BaseModel):
    """Papers disagreeing. A finding, not an error — surfaced loudly."""

    target: str
    direction: str
    supporting_papers: list[str]
    opposing_papers: list[str]
    unit_ids: list[str]


class EvidenceSummary(BaseModel):
    question_id: str
    units: list[EvidenceUnit]
    short_ids: dict[str, str] = Field(default_factory=dict)
    """unit id -> "E1". Short ids are what the writing model cites."""
    targets: list[TargetSummary] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    facet_units: dict[str, list[str]] = Field(default_factory=dict)

    def by_short_id(self) -> dict[str, EvidenceUnit]:
        by_id = {u.id: u for u in self.units}
        return {short: by_id[uid] for uid, short in self.short_ids.items()}


# --- the report --------------------------------------------------------------


class Sentence(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    """Short evidence ids ("E3"). The renderer maps them to reference numbers."""
    kept: bool = True
    drop_reason: str | None = None


class ReportSection(BaseModel):
    id: str
    title: str
    kind: Literal["written", "generated"]
    """`written` by the model under the citation guard; `generated` by code."""
    sentences: list[Sentence] = Field(default_factory=list)
    body_markdown: str | None = None
    subsections: list["ReportSection"] = Field(default_factory=list)
    note: str | None = None


class Reference(BaseModel):
    number: int
    paper_id: str
    citation: str
    url: str | None = None
    signals: PaperSignals = Field(default_factory=PaperSignals)
    text_source: str = "none"
    evidence_ids: list[str] = Field(default_factory=list)


class Report(BaseModel):
    """The answer. Contains nothing time- or run-dependent, so the same
    question over the same papers produces the same bytes."""

    question_id: str
    question: str
    interpretation: dict[str, object] = Field(default_factory=dict)
    sections: list[ReportSection]
    references: list[Reference] = Field(default_factory=list)
    evidence: list[EvidenceUnit] = Field(default_factory=list)
    short_ids: dict[str, str] = Field(default_factory=dict)
    best_source_id: str | None = None
    stats: dict[str, int] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    integrity: list[str] = Field(default_factory=list)
    """Violations found by `report/invariants.py`. Empty on a sound report; any
    entry is a bug, and the rendered report opens with a warning banner."""

    def kept_sentences(self) -> list[tuple[str, Sentence]]:
        out: list[tuple[str, Sentence]] = []

        def walk(section: ReportSection) -> None:
            for s in section.sentences:
                if s.kept:
                    out.append((section.id, s))
            for sub in section.subsections:
                walk(sub)

        for section in self.sections:
            walk(section)
        return out


# --- runs and feedback -------------------------------------------------------


class Run(BaseModel):
    """One execution of the pipeline. Reports are immutable and carry a run id,
    so "which version did you read?" always has an answer."""

    id: str
    question_id: str
    started: str
    finished: str | None = None
    stage_reached: int = -1
    model_versions: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    paper_set_frozen: bool = True
    """A re-run reuses the papers already gathered. Going back out to the
    sources for new ones is a separate, explicit action."""
    report_sha256: str | None = None
    papers_sha256: str | None = None
    """Hash of the run's paper manifest (paper id + content hash, in order).
    The manifest itself is the run's `papers_manifest` artifact."""
    input_hashes: dict[str, str] = Field(default_factory=dict)
    """Content hashes of the config and knowledge files the run read, and of
    the corpus file if one was given."""
    code_version: str | None = None


class Rating(BaseModel):
    """A scientist's verdict on a report. Stored now so that, once there are
    enough, they can be compared against the automatic benchmark."""

    report_id: str
    rater: str
    score: int = Field(ge=1, le=5)
    comments: str | None = None
    flagged_sentences: list[str] = Field(default_factory=list)
    created: str


ReportSection.model_rebuild()
