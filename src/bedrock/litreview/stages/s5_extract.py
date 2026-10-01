"""Stage 5 — turn the planned papers into EvidenceUnits.

The order below is the whole design, and it is deliberate:

    model extracts  ->  code verifies quotes  ->  code checks the quote is
                        this paper's own result, not someone else's
                    ->  code checks the quote reports a result at all, not
                        an aim, a method or a prediction
                    ->  code checks the entity
                    ->  code checks direction and outcome against the
                        words the model copied  ->  code normalises
                        (a target through the paper's own abbreviations;
                        an outcome measure is flagged, never a target row),
                        and sets the evidence type from the quote's own
                        words  ->  code dedupes

The model does one narrow thing — copy findings out of one chunk of one
section — and every subsequent step is deterministic code that can be read,
tested and trusted. The model never normalises, never scores quality, and
never sees a whole paper.

Right after the quote verifies, before anything else: does this paper's own
sentence actually report this paper's own result? A quote can be word-for-word
real and still be Introduction or Discussion prose citing earlier work — a
citation bracket Europe PMC stripped down to "[]" or "[12]", an author-year
citation, a "Previous studies have shown..." lead-in. `extract/verify.py`
`other_work_reason` catches this; see it for why each pattern is safe against
a confidence interval, a concentration or an abbreviation that merely looks
like one. This runs before the entity check on purpose: a borrowed result can
still name the right entity, and would otherwise sail through.

Then: does the sentence report a result at all? "In this study, we
investigated the role of the AhR in ..." is the paper's own sentence about its
own work, and names the right target, but it is an aim, not a finding - and a
report cited it for "AhR regulates ...". Aims, methods, predictions ("is
therefore predicted to benefit patients") and open questions ("remains
unclear") are dropped by `extract/verify.py` `not_a_result_reason`, which
keeps a quote that states its aim and then its result.

Not a check, but found by code at the end: when the paper's very next
sentence qualifies the quote ("Although this result did not reach statistical
significance ..."), it is stored beside it as `qualifier_quote`, so the reader,
the writer and the citation guard all see it (`extract/verify.py`
`qualifier_after`).

Also by code, while provenance is traced: a target or agent the quote does
not name, found elsewhere in the passage, is shown to the reader too - in the
context sentence, the qualifier, or else the passage's sentence that names it,
stored as `naming_quote` (`extract/provenance.py`). A target that cannot be
shown drops the finding (`target_not_shown`).

And one check of who was studied that reads past the passage: when the quote
is the authors' conclusion and names people ("... in patients with asthma"),
but the paper's Methods name no human subjects, only animals or cells, the
finding's subjects and evidence type come from the Methods, the Methods
sentence is shown as its context, and the unit records `methods_subjects`
(`extract/verify.py` `conclusion_reason`, `extract/provenance.py`
`methods_subjects`; counted as `people_not_in_methods`). A paper with no
Methods text is left alone.

The labels are the model's, checked by code (issue #9). The evidence type
comes from the quote's own words about what was studied before the species:
"HBECs were exposed to CRE" is in_vitro though the context sentence names
mice, and a Western blot of "mast cells from individuals with obesity" is
human_clinical, not human_genetic, because nothing in it is genetic
(`settle_evidence_type`, `extract/normalize.py` `system_in_quote`). An
`improved` or `worsened` finding with nothing done to its target is an
`association` (`reconcile`). Every change is written on the unit
(`corrected`) and counted as `<field>_corrected`.

Which sections we send, and why (set by the plan, from config/pipeline.yaml):

  abstract    extract from it. Clean sentences, high signal.
  results     extract from it. The primary data lives here. Long results
              sections are split on paragraph boundaries; when there are more
              pieces than the budget, code keeps the ones that share the most
              vocabulary with the question.
  methods     NOT extracted from. Passed as *context*, so species and sample
              size are available where the outcome is. Only the paragraphs
              that mention subjects, animals or sample sizes are sent.
  discussion  only when a paper has no results section; flagged
              `from_discussion`, because authors overstate there.

After the model, the entity check: a finding whose own words name a
near-miss (hMPV in an RSV question) or a different known disease is dropped,
even though the paper passed the gate. A psoriasis paper can report an atopic
dermatitis cohort too, and that finding does not answer a psoriasis question.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from pydantic import ValidationError

from bedrock.litreview.extract.dedupe import dedupe_within_paper
from bedrock.litreview.extract.provenance import (
    METHODS,
    NONE,
    SUBJECT_WORDS,
    MethodsSubjects,
    Provenance,
    methods_subjects,
    species_in,
    trace,
)
from bedrock.litreview.extract.normalize import (
    classify_direction,
    classify_outcome,
    classify_setting,
    extract_n,
    genetic_words,
    infer_evidence_type,
    names_intervention,
    normalize_species,
    system_in_quote,
    target_changed_in,
)
from bedrock.litreview.extract.verify import normalize as normalize_text
from bedrock.litreview.extract.verify import (
    Located,
    conclusion_reason,
    locate_quote,
    not_a_result_reason,
    other_work_reason,
    qualifier_after,
    region_of,
    strip_leading_label,
    verify_quote,
)
from bedrock.litreview.facets import subject_label
from bedrock.litreview.knowledge import Knowledge, TermMatcher, load_knowledge
from bedrock.litreview.knowledge.abbreviations import paper_abbreviations, resolve_target
from bedrock.litreview.lexical import BM25
from bedrock.litreview.llm import LLMError, ask_json, load_prompt, strict_schema, version_of
from bedrock.litreview.models import (
    EvidencePlan,
    EvidenceUnit,
    ExtractedFinding,
    ExtractionJob,
    ExtractionResult,
    Facet,
    Paper,
    QuestionSpec,
    stable_id,
)

log = logging.getLogger(__name__)

PROMPT = "extract"
CONTEXT_SECTION = "methods"

_SUBJECT_WORDS = SUBJECT_WORDS


# --- text preparation ---------------------------------------------------------


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries into pieces of at most ~max_chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        # One enormous paragraph (flat PDF text): fall back to sentences.
        buf = ""
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", para):
            if buf and len(buf) + len(sentence) + 1 > max_chars:
                pieces.append(buf)
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            pieces.append(buf)

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > max_chars:
            chunks.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n\n{piece}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def methods_context(paper: Paper, max_chars: int) -> str:
    """The Methods paragraphs that say who or what was studied, and how many."""
    text = paper.sections.get(CONTEXT_SECTION, "")
    if not text:
        return "(no methods section available)"
    picked, total = [], 0
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para and _SUBJECT_WORDS.search(para):
            if total + len(para) > max_chars:
                para = para[: max(0, max_chars - total)]
            picked.append(para)
            total += len(para)
            if total >= max_chars:
                break
    return "\n\n".join(picked) or "(no methods paragraphs describing subjects)"


def plan_chunks(paper: Paper, job: ExtractionJob, query: str, *, chunk_chars: int, max_chunks: int) -> list[tuple[str, int, str]]:
    """(section, chunk index, text) to send, within the per-paper budget."""
    candidates: list[tuple[str, int, str]] = []
    for section in job.sections:
        for i, chunk in enumerate(chunk_text(paper.sections.get(section, ""), chunk_chars)):
            candidates.append((section, i, chunk))
    if len(candidates) <= max_chunks:
        return candidates
    # Keep the abstract, then the chunks that best match the question.
    keep = [c for c in candidates if c[0] == "abstract"][:1]
    rest = [c for c in candidates if c not in keep]
    bm25 = BM25({f"{s}:{i}": t for s, i, t in rest})
    best = {doc for doc, _ in bm25.rank(query)[: max_chunks - len(keep)]}
    keep += [c for c in rest if f"{c[0]}:{c[1]}" in best]
    return keep


# --- the model call -----------------------------------------------------------


def extraction_schema(facets: list[Facet]) -> dict:
    return strict_schema(ExtractionResult.model_json_schema(), enums={"facet": [f.id for f in facets]})


def render_prompt(
    spec: QuestionSpec, facets: list[Facet], job: ExtractionJob, section: str, text: str, methods: str
) -> str:
    subject = spec.subject
    near = subject.near_miss_terms if subject else []
    return load_prompt(PROMPT).render(
        question=spec.text,
        subject=subject_label(spec),
        exact_terms=", ".join(f'"{t}"' for t in (subject.exact_terms if subject else [])),
        near_miss_line=(
            "These are different things and do NOT count: " + ", ".join(f'"{t}"' for t in near[:12]) + "."
            if near else ""
        ),
        job_facets=", ".join(job.facets),
        facets="\n".join(f"- `{f.id}` — {f.title}: {f.need}" for f in facets),
        methods_text=methods,
        section_name=section,
        section_text=text,
    )


def _extract_chunk(prompt: str, schema: dict, client, *, model: str) -> ExtractionResult:
    """One model call. Guided decoding makes a malformed reply impossible."""
    data = ask_json(client, prompt, model=model, schema=schema, max_tokens=4096)
    return ExtractionResult.model_validate(data)


# --- the checks after the model ---------------------------------------------------


def _locate_in_chunk(quote: str, chunk: str, section_text: str) -> Located:
    """The quote, found in the part of the section this chunk came from, with
    offsets into the whole section."""
    region = region_of(chunk, section_text)
    if region is None:
        # Cannot place the chunk (should not happen: chunks are the section's
        # own paragraphs). Verify against what the model saw, then place it.
        if verify_quote(quote, chunk) == "failed":
            return locate_quote(quote, chunk, widen=False)
        return locate_quote(quote, section_text)
    lo, hi = region
    found = locate_quote(quote, section_text[lo:hi])
    if found.status == "failed":
        return found
    start, end = found.start + lo, found.end + lo
    return Located(found.status, start, end, section_text[start:end])


def entity_check(finding: ExtractedFinding, spec: QuestionSpec, kb: Knowledge) -> tuple[str | None, str]:
    """(drop reason or None, 'exact' | 'paper_context').

    The finding's own field for the entity is read first — `disease` for a
    disease or pathogen, `intervention` for a drug, `target` for a target. If
    that field names a near-miss or a different known disease, the finding is
    about something else, whatever its quote mentions in passing. Only when
    the field is neutral ("skin inflammation") does the quote decide.
    """
    match = "paper_context"
    # A disease the question itself names is never "another" disease: in
    # "does tirzepatide reduce liver fat in adults with obesity?" a finding
    # filed under "obesity with MASLD" is about the question's own population,
    # and dropping it as off-topic threw away every result with numbers in it
    # (the paper's Results; only its abstract's p-values survived).
    asked = {e.canonical for e in spec.entities}
    for entity in spec.required_entities:
        matcher = TermMatcher(entity.exact_terms, entity.near_miss_terms, entity.near_miss_patterns)
        if entity.kind == "disease":
            primary = [finding.disease]
        elif entity.kind == "pathogen":
            primary = [finding.disease, finding.target]
        elif entity.kind == "drug":
            primary = [finding.intervention or ""]
        else:
            primary = [finding.target]
        exact, near = matcher.summary(" | ".join(primary))
        if exact:
            match = "exact"
            continue
        if near:
            return f"near_miss:{near[0]}", match
        if entity.kind in ("disease", "pathogen"):
            others = {
                e.canonical for e, _ in kb.scan(finding.disease, ("disease", "pathogen"))
                if e.canonical != entity.canonical and e.canonical not in asked
            }
            if others:
                return f"other_disease:{sorted(others)[0]}", match
        exact, near = matcher.summary(" | ".join([finding.quote, finding.target, finding.intervention or ""]))
        if exact:
            match = "exact"
        elif near:
            return f"near_miss:{near[0]}", match
    return None, match


def something_done(finding: ExtractedFinding) -> bool:
    """Did the quote report an intervention, even one whose direction on the
    target code cannot read? An agent the paper names (`intervention`, as
    provenance kept it); a block or activate word beside the target's name in
    the quote itself ("PGGT1B deficiency"; `extract/normalize.py`
    `target_changed_in`); or words the model copied from the quote for what
    was done to the target that name an intervention ("treatment with", "LOF
    variants"; `names_intervention`)."""
    if (finding.intervention or "").strip():
        return True
    if target_changed_in(finding.quote, finding.target):
        return True
    change = finding.change.strip()
    return bool(change) and normalize_text(change) in normalize_text(finding.quote) and names_intervention(change)


def reconcile(finding: ExtractedFinding) -> tuple[str, str, str, list[str]]:
    """(direction, outcome, setting, corrections): the model's labels, overruled
    where the words it copied from its own quote unambiguously say otherwise.

    The copied words must be in the quote, or they are ignored. An
    observation with nothing done to a target stays an `association` whatever
    its verbs: "IL-17A was increased in lesions" did not worsen anything. And
    it becomes one when the model called it `improved` or `worsened`: with the
    direction still `unclear` and nothing done (`something_done`), there is no
    intervention for the outcome to be the outcome of.
    """
    quote = normalize_text(finding.quote)
    direction, outcome, notes = finding.direction, finding.outcome, []
    change = finding.change.strip()
    if change and normalize_text(change) in quote:
        said = classify_direction(change)
        if said and said != direction:
            notes.append(f'direction: {direction} -> {said} ("{change}")')
            direction = said
    effect = finding.effect.strip()
    if effect and normalize_text(effect) in quote and outcome != "mixed":
        said = classify_outcome(effect)
        if said and said != outcome and not (outcome == "association" and direction == "unclear"):
            notes.append(f'outcome: {outcome} -> {said} ("{effect}")')
            outcome = said
    if direction == "unclear" and outcome in ("improved", "worsened") and not something_done(finding):
        # Nothing was done to the target, so nothing it did improved or
        # worsened anything: "similar increases in hypothalamic Sik3 mRNA
        # levels in db/db mice" was filed `worsened`, and a worsened finding
        # reads as a negative result (the "negative" section, the "against"
        # count of Stage 6). The model's own label is overruled here too, not
        # only a label code read from its copied words. Measured on the 593
        # stored qwen3:8b findings: 12 change, every one an observation (Sik3
        # mRNA in db/db and HFD-fed mice, ROS in DIO mice, a correlation with
        # body weight); the drugs, diets and LOF variants beside them keep
        # their outcome.
        notes.append(f"outcome: {outcome} -> association (nothing done to the target)")
        outcome = "association"
    setting = finding.setting
    said_setting = classify_setting(f"{finding.quote} {finding.effect_detail} {finding.intervention or ''}", setting)
    if said_setting and said_setting != setting:
        notes.append(f"setting: {setting} -> {said_setting}")
        setting = said_setting
    return direction, outcome, setting, notes


_HUMAN_WORDS = re.compile(r"\b(?:patients?|participants?|people|persons?|adults?|children|women|men|subjects?|"
                          r"volunteers?|humans?|individuals?|cohort)\b", re.I)


def paper_design_check(evidence_type: str, species_raw: str, quote: str, paper: Paper) -> tuple[str, str | None]:
    """A human label on a finding from a paper that reports only animal or cell
    work, whose quote names no people, is the paper's design instead.

    A mouse study's "TZP treatment was associated with reduced abundance of
    ... proteins" was filed human_clinical (tirzepatide/liver-fat, E28): the
    quote names no species, and nothing overruled the model's label. The
    paper's own publication type and wording (`study_design`) decide, and
    only ever weaken a human label, never create one."""
    from bedrock.litreview.study_design import grounded_labels, design_signals

    if not evidence_type.startswith("human") or species_raw or _HUMAN_WORDS.search(quote):
        return evidence_type, None
    labels = grounded_labels(design_signals(paper))
    if not labels or labels - {"animal_interventional", "in_vitro"}:
        return evidence_type, None
    settled = "in_vivo" if "animal_interventional" in labels else "in_vitro"
    return settled, f"evidence_type: {evidence_type} -> {settled} (the paper reports only animal or cell work)"


def settle_evidence_type(
    stated: str,
    *,
    species: str,
    species_raw: str,
    model_species: str,
    quote: str,
    context: str,
    unbacked: MethodsSubjects | None = None,
) -> tuple[str, str | None]:
    """(evidence type, correction note or None): the model's label, checked
    against the paper's words, strongest evidence last.

    1. The normalised species (`infer_evidence_type`): a mouse is never human
       evidence. When the paper states no subjects, the model's own species
       may still *weaken* a human label - a model that says "mice" and
       "human_clinical" in one breath has not found human evidence - but
       never strengthen one.
    2. The quote's own words about what was studied
       (`extract/normalize.py` `system_in_quote`): cells in a dish are
       in_vitro even when the context sentence names mice, and people's
       tissue is human_clinical unless genetic-study words say otherwise.
       These are the paper's words about the very result, so they may
       strengthen a label as well as weaken it. When the quote does not say,
       a human_genetic label still needs a genetic word in the quote or the
       context sentence, or it is human_clinical.
    3. The Methods, when the quote is a conclusion naming people the Methods
       never studied (issue #4, `extract/provenance.py` `methods_subjects`).

    Every change is noted on the unit (`EvidenceUnit.corrected`), counted as
    `evidence_type_corrected`, so a rising rate shows up in the run's stats.
    """
    evidence_type = infer_evidence_type(species, stated)
    why = f"species {species}"
    if not species_raw:
        weakened = infer_evidence_type(normalize_species(model_species), stated)
        if stated.startswith("human") and not weakened.startswith("human"):
            evidence_type, why = weakened, f"model's species {normalize_species(model_species)}"
    said = system_in_quote(quote, evidence_type, context)
    if said is not None:
        evidence_type, why = said[0], f'"{said[1]}"'
    elif evidence_type == "human_genetic" and not (genetic_words(quote) or genetic_words(context)):
        evidence_type, why = "human_clinical", "no genetic words"
    if unbacked is not None:
        evidence_type, why = unbacked.evidence_type, f"methods: {unbacked.label}"
    if evidence_type == stated:
        return stated, None
    return evidence_type, f"evidence_type: {stated} -> {evidence_type} ({why})"


def settle_facet(facet: str, *, evidence_type: str, direction: str, outcome: str, setting: str,
                 facet_ids: list[str]) -> str:
    """Which information need a finding supplies, checked against its labels.

    The model files findings; code refuses filings the labels contradict. The
    MedGemma run filed mouse results under "Human clinical evidence", and the
    report's clinical section duly presented them as human data.
    """
    def first(*ids: str) -> str:
        return next((i for i in ids if i in facet_ids), facet)

    fallback = first("causal", "preclinical", "mechanism", "direct") if direction != "unclear" \
        else first("mechanism", "direct")
    if facet == "clinical" and evidence_type != "human_clinical":
        return fallback
    if facet == "prevention" and setting != "prevention":
        return fallback
    if facet == "negative" and outcome not in ("no_effect", "worsened", "mixed"):
        return fallback
    if facet in ("causal", "preclinical") and direction == "unclear":
        return first("mechanism", "direct")
    return facet


def _to_unit(
    finding: ExtractedFinding,
    *,
    status: str,
    match: str,
    paper: Paper,
    spec: QuestionSpec,
    section: str,
    chunk: int,
    model_version: str,
    prompt_version: str,
    kb: Knowledge,
    facet_ids: list[str] | None = None,
    located: Located,
    model_quote: str,
    prov: Provenance,
    qualifier: Located | None = None,
    unbacked: MethodsSubjects | None = None,
) -> EvidenceUnit:
    # Fields the model wrote, as far as the text it was shown supports them
    # (`extract/provenance.py`). Labels are read from these too: a result
    # summary provenance rejected must not set the setting either.
    v = dict(prov.values)
    where = dict(prov.where)
    context, context_section = prov.context, prov.context_section
    if unbacked is not None:
        # A conclusion names people the Methods never studied (issue #4):
        # who was studied comes from the Methods. Their sentence becomes the
        # context sentence unless there already is one, and the subjects are
        # shown only if the context sentence names them; else "not stated".
        if context is None and unbacked.context is not None:
            context, context_section = unbacked.context, unbacked.context_section
            where["context"] = METHODS
        if context is not None and unbacked.species_raw and species_in(unbacked.species_raw, context.text):
            v["species_raw"], where["species"] = unbacked.species_raw, where.get("context", METHODS)
        else:
            v["species_raw"] = ""
            if "species" in where:
                where["species"] = NONE
    species = normalize_species(v["species_raw"] or (unbacked.species_raw if unbacked else ""))
    direction, outcome, setting, corrected = reconcile(
        finding.model_copy(update={"effect_detail": v["effect_detail"], "intervention": v["intervention"]})
    )
    evidence_type, note = settle_evidence_type(
        finding.evidence_type, species=species, species_raw=v["species_raw"], model_species=finding.species,
        quote=finding.quote, context=context.text if context else "", unbacked=unbacked,
    )
    if note:
        corrected.append(note)
    evidence_type, note = paper_design_check(evidence_type, v["species_raw"], finding.quote, paper)
    if note:
        corrected.append(note)
    facet = settle_facet(
        finding.facet, evidence_type=evidence_type, direction=direction, outcome=outcome,
        setting=setting, facet_ids=facet_ids or [finding.facet],
    )
    subject = spec.subject
    if subject is not None and subject.kind in ("disease", "pathogen") and (
        match == "exact" or not kb.scan(v["disease_raw"], ("disease", "pathogen"))
    ):
        disease = subject.canonical
    else:
        disease = kb.normalize(v["disease_raw"], "disease")
    return EvidenceUnit(
        # The model's own quote is part of the id: two findings copied from
        # different halves of one sentence widen to the same paper sentence,
        # and must not share an id.
        id=stable_id(spec.id, paper.id, section, chunk, normalize_text(finding.quote), finding.target, outcome,
                     normalize_text(model_quote) if model_quote else ""),
        paper_id=paper.id,
        question_id=spec.id,
        target=kb.normalize(finding.target, "target"),
        target_raw=finding.target,
        direction=direction,  # type: ignore[arg-type]
        modality=finding.modality,
        intervention=v["intervention"],
        disease=disease,
        disease_raw=v["disease_raw"],
        evidence_type=evidence_type,  # type: ignore[arg-type]
        species=species,
        species_raw=v["species_raw"],
        outcome=outcome,  # type: ignore[arg-type]
        setting=setting,  # type: ignore[arg-type]
        effect_detail=v["effect_detail"],
        n=v["n"],
        combo_with=kb.normalize(v["combo_with"], "target") if v["combo_with"] else None,
        target_is_primary=finding.target_is_primary,
        facet=facet,
        quote=finding.quote,
        quote_status=status,  # type: ignore[arg-type]
        section=section,  # type: ignore[arg-type]
        quote_start=located.start,
        quote_end=located.end,
        quote_model=model_quote if model_quote.strip() != finding.quote else None,
        provenance=where,
        context_quote=context.text if context else None,
        context_section=context_section,
        context_start=context.start if context else None,
        context_end=context.end if context else None,
        qualifier_quote=qualifier.text if qualifier else None,
        qualifier_start=qualifier.start if qualifier else None,
        qualifier_end=qualifier.end if qualifier else None,
        naming_quote=prov.naming.text if prov.naming else None,
        naming_section=prov.naming_section if prov.naming else None,
        naming_start=prov.naming.start if prov.naming else None,
        naming_end=prov.naming.end if prov.naming else None,
        methods_subjects=unbacked.label if unbacked else None,
        from_discussion=(section == "discussion"),
        entity_match=match,  # type: ignore[arg-type]
        corrected=corrected,
        extracted_by_model=model_version,
        prompt_version=prompt_version,
    )


def extract_paper(
    paper: Paper,
    job: ExtractionJob,
    spec: QuestionSpec,
    facets: list[Facet],
    client,
    *,
    model: str,
    chunk_chars: int = 6000,
    max_chunks: int = 6,
    methods_chars: int = 2500,
    knowledge: Knowledge | None = None,
    stats: Counter | None = None,
    outcomes: dict | None = None,
    on_chunk=None,
) -> list[EvidenceUnit]:
    """Extract every verified finding from one paper.

    A chunk that fails is logged and skipped. One bad chunk must never take
    down a paper, and one bad paper must never take down a report.

    `outcomes`, when given, is filled with what happened to each chunk - sent,
    failed, how many findings it proposed and why each rejected one was
    rejected - and which units the paper kept. Bookkeeping only: it changes
    no call and no result, and feeds the run's coverage ledger (`coverage.py`).
    """
    kb = knowledge or load_knowledge()
    stats = stats if stats is not None else Counter()
    chunk_log: list[dict] = []
    if outcomes is not None:
        outcomes.clear()
        outcomes.update({"chunks": chunk_log, "review_discarded": False, "unit_ids": [], "duplicates": 0})
    prompt_version = load_prompt(PROMPT).version
    model_version = version_of(client, model)
    schema = extraction_schema(facets)
    methods = methods_context(paper, methods_chars)
    query = f"{spec.text} {' '.join(t for e in spec.entities for t in e.exact_terms)}"
    species_filter = [normalize_species(e.text) for e in spec.entities if e.kind == "species"]
    # The paper's own "long form (ABBR)" definitions, read once per paper, so
    # its "AHR" is judged by what it says AHR means (`knowledge/abbreviations.py`).
    abbreviations = paper_abbreviations(paper)

    units: list[EvidenceUnit] = []
    planned = plan_chunks(paper, job, query, chunk_chars=chunk_chars, max_chunks=max_chunks)
    if outcomes is not None:
        outcomes["planned"] = [f"{section}:{i}" for section, i, _ in planned]
    for done, (section, chunk_i, text) in enumerate(planned, 1):
        entry = {"id": f"{section}:{chunk_i}", "status": "failed", "findings": 0, "rejected": {}}
        chunk_log.append(entry)
        before = Counter(stats)
        prompt = render_prompt(spec, facets, job, section, text, methods)
        try:
            result = _extract_chunk(prompt, schema, client, model=model)
        except (LLMError, ValidationError) as exc:
            log.warning("extraction failed: paper=%s section=%s chunk=%d: %s", paper.id, section, chunk_i, exc)
            stats["chunk_failed"] += 1
            # Reported before `continue`: a chunk whose call failed is a chunk
            # this paper will not spend time on again, so the caller's count
            # must not stall on it.
            if on_chunk is not None:
                on_chunk(done, len(planned))
            continue
        if on_chunk is not None:
            on_chunk(done, len(planned))
        stats["chunks"] += 1
        entry["status"] = "completed"
        if result.is_review:
            stats["review_chunks"] += 1
            entry["status"] = "review"
            if section == "abstract":
                # The paper describes itself as a review: a review citing a
                # trial is not a trial. Nothing from it counts.
                log.info("paper=%s reads as a review; discarding its findings", paper.id)
                if outcomes is not None:
                    outcomes["review_discarded"] = True
                return []
            continue

        for finding in result.findings:
            stats["findings"] += 1
            # Looked for only in the passage the model read — located in the
            # section so its offsets are the section's — never elsewhere in
            # the section, where the same words may carry other qualifiers.
            located = _locate_in_chunk(finding.quote, text, paper.sections.get(section, ""))
            if located.status == "failed":
                # The model produced a sentence that is not in the paper, or
                # changed one that is. This is the fabrication guard doing its
                # job; the finding does not exist.
                stats["quote_failed"] += 1
                if located.reason.startswith("meaning changed"):
                    stats["quote_meaning_changed"] += 1
                log.debug("unverified quote discarded (%s): paper=%s quote=%r",
                          located.reason, paper.id, finding.quote[:80])
                continue
            if section == "abstract":
                # Europe PMC's structured abstracts keep their own headings
                # ("Key Results:", "Background and Purpose:") glued onto the
                # sentence text. Verification is against the paper's real
                # characters, so this only trims the label off the front and
                # slides `start` forward by the same count - `end` is
                # untouched, so the paper[start:end] == text anchor
                # `report/invariants.py` checks still holds.
                stripped_text, stripped_start = strip_leading_label(located.text, located.start)
                if stripped_text != located.text:
                    located = Located(located.status, stripped_start, located.end, stripped_text, located.reason)
            model_quote = finding.quote
            # From here on every check reads the paper's own sentence.
            finding = finding.model_copy(update={"quote": located.text})
            status = located.status
            other_work = other_work_reason(finding.quote, section)
            if other_work:
                # A true sentence about someone *else's* result: a citation
                # bracket Europe PMC stripped down to "[]"/"[12]", an
                # author-year citation, a "Previous studies have shown..."
                # lead-in, or a paragraph the parser could only file as
                # Introduction/Background prose. It verified because the
                # paper really wrote it - it is dropped because the paper
                # did not run it.
                stats[other_work.split(":", 1)[0]] += 1
                log.debug("finding dropped (%s): paper=%s quote=%r", other_work, paper.id, finding.quote[:80])
                continue
            not_a_result = not_a_result_reason(finding.quote)
            if not_a_result:
                # The paper's own words about its own work, but not a result:
                # what the study set out to do ("In this study, we
                # investigated ..."), how ("To assess X, mice were fed ..."),
                # what it predicts or hopes for ("is therefore predicted to
                # benefit patients"), or what is still unknown. A quote that
                # states its aim and then its result is kept; see
                # `not_a_result_reason` for how the two are told apart.
                stats[not_a_result.split(":", 1)[0]] += 1
                log.debug("finding dropped (%s): paper=%s quote=%r", not_a_result, paper.id, finding.quote[:80])
                continue
            drop, match = entity_check(finding, spec, kb)
            if drop:
                stats[drop.split(":", 1)[0]] += 1
                log.debug("finding dropped (%s): paper=%s", drop, paper.id)
                continue
            # The paper's next sentence, when it qualifies this one ("Although
            # this result did not reach statistical significance ..."). Code
            # finds it, not the model; `extract/verify.py` says what counts.
            # Found before tracing, because a name it states is shown by it.
            section_text = paper.sections.get(section, "")
            qualifier = (qualifier_after(section_text, located.start, located.end)
                         if section_text[located.start:located.end] == located.text else None)
            if qualifier is not None and section == "abstract":
                # The qualifying sentence is shown to the reader exactly like
                # the quote, so it gets the same label strip - e.g. a quote
                # ending a Results part whose qualifying next sentence opens
                # "Conclusion and Implications: ...".
                stripped_text, stripped_start = strip_leading_label(qualifier.text, qualifier.start)
                if stripped_text != qualifier.text:
                    qualifier = Located(qualifier.status, stripped_start, qualifier.end, stripped_text,
                                        qualifier.reason)
            # Every other field the model wrote must be in the text it read,
            # and every name it gave must be in text the reader is shown.
            prov = trace(
                quote=finding.quote, chunk=text, methods=methods, section=section,
                section_text=paper.sections.get(section, ""),
                methods_section_text=paper.sections.get(CONTEXT_SECTION, ""),
                target=finding.target, intervention=finding.intervention, combo_with=finding.combo_with,
                disease_raw=finding.disease, species_raw=finding.species,
                n=finding.n if finding.n is not None else extract_n(finding.quote),
                effect_detail=finding.effect_detail, kb=kb,
                qualifier=qualifier.text if qualifier else "", quote_start=located.start,
            )
            if prov.drop:
                stats[prov.drop] += 1
                log.debug("finding dropped (%s): paper=%s target=%r", prov.drop, paper.id, finding.target)
                continue
            for name, where in prov.where.items():
                if where == NONE:
                    stats[f"{name}_unlocated"] += 1
            if prov.naming is not None:
                stats["naming_sentence"] += 1
            # The authors' conclusion names people; do the Methods back it?
            # Read from the paper's own Methods, not the model's species.
            unbacked = None
            if section_text[located.start:located.end] == located.text \
                    and conclusion_reason(section_text, located.start, located.end, section):
                unbacked = methods_subjects(
                    finding.quote, methods_section_text=paper.sections.get(CONTEXT_SECTION, ""),
                    abstract=paper.sections.get("abstract", ""),
                )
                if unbacked is not None:
                    stats["people_not_in_methods"] += 1
            unit = _to_unit(
                finding, status=status, match=match, paper=paper, spec=spec, section=section,
                chunk=chunk_i, model_version=model_version, prompt_version=prompt_version, kb=kb,
                facet_ids=[f.id for f in facets], located=located, model_quote=model_quote, prov=prov,
                qualifier=qualifier, unbacked=unbacked,
            )
            # The target, read through the paper's own definitions: "AHR" the
            # paper defines as airway hyperresponsiveness is not the AhR
            # receptor, and a readout like "MMDs" is kept but flagged, so
            # Stage 6 never gives it a target row.
            resolved = resolve_target(finding.target, kb, abbreviations)
            unit = unit.model_copy(update={"target": resolved.name, "outcome_measure": resolved.outcome_measure})
            # A species the paper states decides. When it states none, the
            # model's own species may still rule a finding out (it said
            # "patients" in a mouse question), never in; unknown is kept and
            # shown as "not stated".
            species_seen = unit.species if unit.species_raw else normalize_species(finding.species)
            if species_filter and (unit.species_raw or species_seen != "other") and species_seen not in species_filter:
                stats["species_mismatch"] += 1
                continue
            # Once per field: an outcome read from the copied words and then
            # found to be an association is one corrected outcome, not two.
            for field in dict.fromkeys(note.split(":", 1)[0] for note in unit.corrected):
                stats[f"{field}_corrected"] += 1
            if unit.outcome_measure:
                stats["target_outcome_measure"] += 1
            units.append(unit)
        if outcomes is not None:
            delta = Counter(stats)
            delta.subtract(before)
            entry["findings"] = delta.pop("findings", 0)
            entry["rejected"] = {k: v for k, v in sorted(delta.items())
                                 if v > 0 and k not in ("chunks",) and not k.endswith("_corrected")
                                 and not k.endswith("_unlocated")}

    kept = dedupe_within_paper(units)
    stats["duplicates"] += len(units) - len(kept)
    stats["kept"] += len(kept)
    if outcomes is not None:
        outcomes["unit_ids"] = [u.id for u in kept]
        outcomes["duplicates"] = len(units) - len(kept)
    return kept


def extract_plan(
    plan: EvidencePlan,
    papers: dict[str, Paper],
    spec: QuestionSpec,
    facets: list[Facet],
    client,
    *,
    model: str,
    progress=None,
    tick=None,
    traces: dict[str, dict] | None = None,
    **kwargs,
) -> tuple[list[EvidenceUnit], dict[str, int]]:
    """Extract across the whole plan; returns the units and the reject tally.

    The tally matters as much as the units. A run where 30% of findings fail
    quote verification is telling you something broke upstream, and it
    should be loud.

    `progress` stays one line per finished paper. `tick` is finer, because a
    paper is up to `max_chunks` model calls at roughly two minutes each: it
    reports the chunk within the paper, so the longest silence in the run is
    one call rather than one paper. Display only; see `progress.py`.
    """
    from bedrock.litreview.progress import safe_tick

    tick = safe_tick(tick)
    stats: Counter = Counter()
    all_units: list[EvidenceUnit] = []
    for i, job in enumerate(plan.jobs, 1):
        outcomes = None if traces is None else traces.setdefault(job.paper_id, {})
        tick("extract", i - 1, len(plan.jobs), f"reading paper {i}")
        units = extract_paper(
            papers[job.paper_id], job, spec, facets, client, model=model, stats=stats, outcomes=outcomes,
            on_chunk=lambda done, total, _i=i: tick("extract", _i - 1, len(plan.jobs), f"paper {_i} chunk {done}/{total}"),
            **kwargs,
        )
        all_units.extend(units)
        tick("extract", i, len(plan.jobs))
        if progress:
            progress(f"extracted {i}/{len(plan.jobs)} papers, {len(all_units)} findings")
    if stats["findings"] and stats["quote_failed"] / stats["findings"] > 0.3:
        log.warning(
            "%d of %d findings failed quote verification — check parsing before prompts",
            stats["quote_failed"], stats["findings"],
        )
    return all_units, dict(stats)
