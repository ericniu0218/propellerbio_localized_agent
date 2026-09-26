"""Stage 7 — write the report: ONE model call per section, each seeing only its
own slice of evidence. Never the whole report — that is the long, vague task
where a 30B model falls over, and where "it just stops working" comes from.

Two kinds of section (layout in config/report.yaml):

  written     one model call, from one slice of evidence, returning sentences
              that each carry the evidence ids they rely on. Every sentence
              then passes the citation guard (report/guard.py) or is dropped.
              A sentence that repeats one already kept earlier in the report
              is dropped too, even though it came from a different section's
              model call — see `already_kept` in `write_section_text`.
  generated   written by code: the most relevant source, the gaps, the
              evidence plan, the relevance ranking, the evidence table, the
              method and the excluded papers. These are facts about the run,
              and a model has nothing to add to them.

The Report contains nothing time- or run-dependent, so the same question
over the same papers produces the same bytes.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from pydantic import ValidationError

from bedrock.litreview.extract.provenance import text_supported
from bedrock.litreview.facets import subject_label
from bedrock.litreview.knowledge import Knowledge, load_knowledge
from bedrock.litreview.llm import LLMError, ask_json, load_prompt, strict_schema
from bedrock.litreview.models import (
    DraftSection,
    EvidencePlan,
    EvidenceSummary,
    EvidenceUnit,
    Facet,
    GateResult,
    Paper,
    QuestionSpec,
    RankedPaper,
    Reference,
    Report,
    ReportSection,
    Sentence,
)
from bedrock.litreview.report.guard import CitationGuard
from bedrock.litreview.report.layout import SectionSpec, load_layout
from bedrock.litreview.stages.s6_score import TARGET_ORDER_RULE, best_units

log = logging.getLogger(__name__)

PROMPT = "write_section"


# --- labels -------------------------------------------------------------------


def short_authors(paper: Paper) -> str:
    if not paper.authors:
        return "Anonymous"
    first = paper.authors[0].split(",")[0].strip()
    return f"{first} et al." if len(paper.authors) > 1 else first


def paper_label(paper: Paper) -> str:
    year = paper.signals.year or (paper.published.year if paper.published else "")
    journal = paper.signals.journal or ""
    return f"{short_authors(paper)} {year} {journal}".strip()


def citation_text(paper: Paper) -> str:
    authors = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
    year = paper.signals.year or (paper.published.year if paper.published else None)
    bits = [authors or "Anonymous", paper.title.rstrip(".") + "."]
    if paper.signals.journal:
        bits.append(f"*{paper.signals.journal}*")
    if year:
        bits.append(str(year))
    ids = []
    if paper.doi:
        ids.append(f"doi:{paper.doi}")
    if paper.pmid:
        ids.append(f"PMID {paper.pmid}")
    if paper.pmcid:
        ids.append(paper.pmcid)
    return " ".join(bits) + (f" ({'; '.join(ids)})" if ids else "")


def paper_url(paper: Paper) -> str | None:
    if paper.url:
        return paper.url
    if paper.doi:
        return f"https://doi.org/{paper.doi}"
    if paper.pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/"
    return None


def cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ").strip()


# --- context ------------------------------------------------------------------


@dataclass
class WriteContext:
    spec: QuestionSpec
    facets: list[Facet]
    plan: EvidencePlan
    summary: EvidenceSummary
    ranked: list[RankedPaper]
    papers: dict[str, Paper]
    gates: dict[str, GateResult]
    client: object = None
    model: str | None = None
    knowledge: Knowledge = field(default_factory=load_knowledge)
    extract_stats: dict[str, int] = field(default_factory=dict)
    max_targets: int = 6
    units_per_target: int = 5
    max_units_per_section: int = 14
    sentence_stats: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.rank_of = {r.paper_id: r.rank for r in self.ranked}
        self.by_unit_id = {u.id: u for u in self.summary.units}
        self.labels = {pid: paper_label(p) for pid, p in self.papers.items()}
        self.vocabulary = sorted(
            {v for u in self.summary.units for v in (u.target_raw, u.intervention or "", u.target) if v}
        )


_WHO = {"human_clinical": "people", "human_genetic": "human genetics", "in_vivo": "animals", "in_vitro": "cells"}


def _evidence_lines(units: list[EvidenceUnit], ctx: WriteContext) -> str:
    """What the writing model sees: the paper's own words, and who was studied
    where the paper says so — not the direction and outcome labels. MedGemma
    4B turned labels into sentences ("Blocking participants improved
    psoriasis"); the labels are for code (target rows, conflicts), the quote
    is for the writer.

    No model-written text is offered as a source of wording: the old "Result"
    line was the extraction model's summary, and sentences built from it
    carried details no quote contained. The context sentence, when there is
    one, is the paper's own Methods or Results sentence naming the subjects."""
    lines = []
    for u in units:
        sid = ctx.summary.short_ids[u.id]
        n = f", n={u.n}" if u.n is not None else ""
        drug = f" · agent: {u.intervention}" if u.intervention else ""
        who = (f"studied in {_WHO.get(u.evidence_type, u.evidence_type)} ({u.species_raw}{n})"
               if u.species_raw else "who was studied: not stated in the quoted text")
        context = f'\n    Context: "{u.context_quote}"' if u.context_quote else ""
        lines.append(
            f"[{sid}] about: {u.target_raw or u.target}{drug} · {who}\n"
            f"    Paper: {ctx.labels.get(u.paper_id, u.paper_id)}\n"
            f'    Quote: "{u.quote}"{context}'
        )
    return "\n".join(lines)


def write_section_text(
    ctx: WriteContext,
    spec: SectionSpec,
    units: list[EvidenceUnit],
    subsection: str = "",
    already_kept: list[set[str]] | None = None,
) -> tuple[list[Sentence], str | None]:
    """One model call, then the guard. Returns sentences and an optional note.

    `already_kept` is the content words of every sentence already kept
    earlier in this same report (from an earlier section, or an earlier
    per-target subsection). Each writing call only ever sees its own slice of
    evidence, so the same finding can be the best evidence for more than one
    section — the "answer" section and a target's own subsection, say — and
    the model then drafts the same sentence twice. `already_kept` is threaded
    through every call by `write_report`, one shared list, so a sentence a
    later section drafts is checked against it with the exact same repeat
    test used within one call, and dropped the same way: silently, and
    without ever becoming a shown or counted "dropped" sentence — a repeat
    states nothing new, it is not a claim that failed a check.
    """
    if not units:
        return [], "No evidence for this section in the papers read."
    if ctx.client is None or not ctx.model:
        return [], "Not written: no model available."
    short = {ctx.summary.short_ids[u.id]: u for u in units}
    prompt = load_prompt(PROMPT).render(
        question=ctx.spec.text,
        subject=subject_label(ctx.spec),
        section_title=spec.title,
        subsection=f" — {subsection}" if subsection else "",
        instructions=spec.instructions,
        evidence=_evidence_lines(units, ctx),
        sentences=spec.sentences,
    )
    schema = strict_schema(DraftSection.model_json_schema(), enums={"citations": list(short)})
    try:
        draft = DraftSection.model_validate(
            ask_json(ctx.client, prompt, model=ctx.model, schema=schema, max_tokens=1500)
        )
    except (LLMError, ValidationError) as exc:
        log.warning("writing failed for section %s: %s", spec.id, exc)
        return [], "Not written: the model call failed."

    guard = CitationGuard(
        ctx.spec, ctx.knowledge, short, paper_labels=ctx.labels, vocabulary=ctx.vocabulary, findings=spec.findings
    )
    sentences = []
    # Seeded with the report's earlier kept sentences, so a repeat of one of
    # those is caught by the very same check that catches a repeat within
    # this call — `said` just starts non-empty instead of the check running
    # twice with two thresholds that could drift apart.
    said: list[set[str]] = list(already_kept) if already_kept is not None else []
    drafted = 0  # sentences the model actually wrote something for (not blank)
    repeated = 0  # ... of those, how many were dropped as a repeat
    for draft_sentence in draft.sentences:
        text = draft_sentence.text.strip()
        if not text:
            continue
        drafted += 1
        citations = list(dict.fromkeys(draft_sentence.citations))
        if len(text) < MIN_SENTENCE_CHARS:
            ctx.sentence_stats["too_short"] += 1
            continue
        words = _content_words(text)
        if any(_overlap(words, earlier) >= REPEAT_OVERLAP for earlier in said):
            # Small models loop: the MedGemma run wrote one sentence four times
            # in a row. A repeat adds nothing and reads as extra evidence. The
            # same is true of a sentence this report already kept elsewhere —
            # `said` may hold those too, so this one check covers both.
            ctx.sentence_stats["repeated"] += 1
            repeated += 1
            continue
        said.append(words)
        verdict = guard.check(text, citations)
        if text[-1] not in ".!?":
            # The model often returns a copied clause with no full stop, and
            # several of those run together are unreadable. Punctuation is not
            # a claim, so this is safe after the guard has seen the sentence.
            text += "."
        ctx.sentence_stats["kept" if verdict.kept else "dropped"] += 1
        sentences.append(Sentence(text=text, citations=citations, kept=verdict.kept, drop_reason=verdict.reason))
        if verdict.kept and already_kept is not None:
            # Recorded for the sections written after this one, not for this
            # call's own remaining sentences — those are already in `said`.
            already_kept.append(words)
    note = None
    if sentences and not any(s.kept for s in sentences):
        note = "Every drafted sentence failed the citation check and was removed."
    elif not sentences and drafted and repeated == drafted:
        # Nothing too short, nothing that ever reached the guard — every
        # sentence the model wrote was a repeat, and (because `said` starts
        # from `already_kept`) that can only happen when it repeated content
        # already kept earlier in this report. Left silent, this is the
        # "heading with nothing under it" a scientist has no way to read: a
        # gap in the papers looks identical to a section whose only content
        # was already said. Say which one this is.
        note = "Everything drafted here is already stated in an earlier section."
    return sentences, note


REPEAT_OVERLAP = 0.8
# A "sentence" this short states nothing. MedGemma emitted a bare "A." twice,
# which then counted as two unsupported claims.
MIN_SENTENCE_CHARS = 25
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def _overlap(a: set[str], b: set[str]) -> float:
    """Share of the shorter sentence's words that the other one also has."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _slice(ctx: WriteContext, source: str) -> list[EvidenceUnit]:
    units = ctx.summary.units
    limit = ctx.max_units_per_section
    if source == "all":
        return best_units(units, ctx.rank_of, limit)
    if source == "negative":
        conflict_ids = {uid for c in ctx.summary.conflicts for uid in c.unit_ids}
        chosen = [u for u in units if u.outcome in ("no_effect", "worsened", "mixed") or u.id in conflict_ids]
        return best_units(chosen, ctx.rank_of, limit)
    if source == "prevention":
        # By setting, not by facet: a prophylaxis finding is prevention
        # evidence whichever information need it was filed under, and it
        # belongs in the clinical section too.
        return best_units([u for u in units if u.setting == "prevention"], ctx.rank_of, limit)
    if source.startswith("facet:"):
        facet_id = source.split(":", 1)[1]
        return best_units([u for u in units if u.facet == facet_id], ctx.rank_of, limit)
    if source == "top_targets":
        chosen: list[EvidenceUnit] = []
        for t in ctx.summary.targets[: ctx.max_targets]:
            chosen += best_units([ctx.by_unit_id[i] for i in t.unit_ids], ctx.rank_of, 2)
        return chosen[: max(limit, 2 * ctx.max_targets)]
    raise ValueError(f"unknown section source {source!r}")


# --- generated sections ---------------------------------------------------------


def _best_source(ctx: WriteContext, refs: dict[str, int]) -> str:
    pid = ctx.plan.best_source_id
    if not pid:
        subject = subject_label(ctx.spec)
        return f"None of the collected papers passed the relevance threshold for {subject}."
    paper = ctx.papers[pid]
    row = next(r for r in ctx.ranked if r.paper_id == pid)
    ref = f" [{refs[pid]}]" if pid in refs else ""
    lines = [f"**{paper.title}**{ref}", "", f"{citation_text(paper)}", ""]
    design = ", ".join(row.evidence_hints) or "design not stated in abstract"
    text = "full text" if paper.has_full_text else "abstract only"
    grade = f"{row.grade}/3" if row.grade is not None else "not graded"
    lines.append(f"Relevance grade {grade} · {design} · read from {text}")
    if row.tested:
        lines.append(f"Its own experiments changed: **{row.tested}** (name checked against the abstract)")
    if row.quote and row.quote_status in ("exact", "fuzzy"):
        lines += ["", f"> {row.quote}", ""]
    # The ranking model's one-line reason is its own prose: shown only when
    # it restates the title and abstract, like any other model-written field.
    if row.reason and text_supported(row.reason, paper.head_text(), ctx.knowledge):
        lines.append(f"Why it ranks first: {row.reason}")
    own = [u for u in ctx.summary.units if u.paper_id == pid]
    if own:
        ids = ", ".join(ctx.summary.short_ids[u.id] for u in own[:8])
        lines += ["", f"It supplied {len(own)} verified finding(s) to this report ({ids})."]
    return "\n".join(lines)


def _gaps(ctx: WriteContext) -> str:
    subject = subject_label(ctx.spec)
    bullets = []
    units_by_facet = Counter(u.facet for u in ctx.summary.units)
    for fp in ctx.plan.facets:
        f = fp.facet
        if fp.status == "gap":
            bullets.append(f"No paper that was read supplied **{f.title.lower()}** ({f.need}).")
        elif fp.status == "thin":
            bullets.append(
                f"Only {len(fp.paper_ids)} paper(s) supplied **{f.title.lower()}**; {f.min_sources} were wanted."
            )
        elif units_by_facet.get(f.id, 0) == 0:
            bullets.append(f"Papers were read for **{f.title.lower()}**, but no verified finding was extracted.")
    jobs = ctx.plan.jobs
    abstract_only = sum(1 for j in jobs if j.abstract_only)
    if abstract_only:
        bullets.append(f"{abstract_only} of {len(jobs)} papers read were available as abstract only.")
    bullets.append("Figures and tables are not read: results reported only in a figure are not in this report.")
    near = [g for g in ctx.gates.values() if g.status == "near_miss"]
    if near:
        examples = sorted({h for g in near for hs in g.near_miss_hits.values() for h in hs})[:4]
        bullets.append(
            f"{len(near)} collected paper(s) were excluded as being about something close to, but not, "
            f"{subject} ({', '.join(examples)})."
        )
    dropped = ctx.sentence_stats.get("dropped", 0)
    if dropped:
        bullets.append(
            f"{dropped} drafted sentence(s) were removed because they stated something their cited evidence does not."
        )
    if ctx.spec.population:
        bullets.append(
            f"The question concerns {ctx.spec.population}. Findings were not filtered by population; "
            "check who was studied in the evidence table."
        )
    preprints = sum(1 for j in jobs if ctx.papers[j.paper_id].signals.is_preprint)
    if preprints:
        bullets.append(f"{preprints} paper(s) read are preprints, not yet peer reviewed.")
    if ctx.spec.question_type == "target_identification":
        bullets.append(TARGET_ORDER_RULE)
    return "\n".join(f"- {b}" for b in bullets)


def _plan(ctx: WriteContext) -> str:
    units_by_facet = Counter(u.facet for u in ctx.summary.units)
    rows = ["| Information needed | Papers wanted | Papers read that can supply it | Findings | Status |", "|---|---|---|---|---|"]
    for fp in ctx.plan.facets:
        f = fp.facet
        rows.append(
            f"| **{cell(f.title)}** — {cell(f.need)} | {f.min_sources}–{f.max_sources} | "
            f"{len(fp.paper_ids)} | {units_by_facet.get(f.id, 0)} | {fp.status} |"
        )
    jobs = ctx.plan.jobs
    full = sum(1 for j in jobs if not j.abstract_only)
    rows += ["", f"Papers read: **{len(jobs)}** ({full} in full text, {len(jobs) - full} abstract only)."]
    return "\n".join(rows)


def _ranking(ctx: WriteContext, refs: dict[str, int], limit: int = 20) -> str:
    subject = subject_label(ctx.spec)
    rows = [
        "Relevance grade (0–3) is a written rule applied to the model's answers about each abstract: "
        f"**3** the paper's own experiments changed a named molecule, pathway or drug and measured the "
        f"effect on {subject}; **2** it studies {subject} with its own data but changed nothing; **1** "
        f"{subject} is only a risk factor or a passing mention, or the paper is a review; **0** it is not "
        "about the exact entity. The model's sentence and the name it gave are checked against the "
        "abstract. The grade is **not** a judgement of quality: journal, citations and preprint status "
        "are shown as looked-up facts. *Supplies* is assigned by code from study-design words.",
        "",
        "| # | Paper | Grade | Changed | Supplies | Design | Text | Journal, year | Cited by | Ref |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    usable = [r for r in ctx.ranked if r.usable][:limit]
    for r in usable:
        p = ctx.papers[r.paper_id]
        flags = " (preprint)" if p.signals.is_preprint else ""
        rows.append(
            f"| {r.rank} | {cell(p.title[:110])}{flags} | {r.grade if r.grade is not None else '–'} | "
            f"{cell(r.tested or '')} | {cell(', '.join(r.facets))} | {cell(', '.join(r.evidence_hints))} | "
            f"{'full' if r.has_full_text else 'abstract'} | {cell(p.signals.journal or '')}, "
            f"{p.signals.year or ''} | {p.signals.citation_count if p.signals.citation_count is not None else ''} | "
            f"{refs.get(r.paper_id, '')} |"
        )
    total_usable = sum(1 for r in ctx.ranked if r.usable)
    if total_usable > limit:
        rows.append(f"\n{total_usable - limit} further papers passed the gate; see the JSON report for all.")
    return "\n".join(rows)


def _evidence_table(ctx: WriteContext, refs: dict[str, int]) -> str:
    rows = [
        "| Id | Target | Agent | Evidence | Subjects | Outcome | Setting | Result | Ref |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for u in ctx.summary.units:
        n = f" (n={u.n})" if u.n is not None else ""
        rows.append(
            f"| {ctx.summary.short_ids[u.id]} | {cell(u.target)} | {cell(u.intervention or '')} | "
            f"{u.evidence_type}{' · discussion' if u.from_discussion else ''} | "
            f"{cell(u.species_raw) if u.species_raw else 'not stated'}{n} | "
            f"{u.outcome} | {u.setting} | {cell(u.effect_detail)} | {refs.get(u.paper_id, '')} |"
        )
    if not ctx.summary.units:
        rows.append("| – | no verified findings | | | | | | | |")
    return "\n".join(rows)


def _method(ctx: WriteContext) -> str:
    g = Counter(x.status for x in ctx.gates.values())
    graded = sum(1 for r in ctx.ranked if r.grade is not None)
    s = ctx.extract_stats
    lines = [
        f"- **Collected:** {len(ctx.papers)} papers from the searcher.",
        f"- **Exact-entity gate:** {g.get('pass', 0)} passed; {g.get('near_miss', 0)} about a near-miss; "
        f"{g.get('absent', 0)} not about the subject; {g.get('review', 0)} reviews; {g.get('retracted', 0)} retracted.",
        f"- **Relevance:** {graded} papers graded by the model from title and abstract; the rest ranked lexically.",
        f"- **Read:** {len(ctx.plan.jobs)} papers, chosen by the evidence plan.",
        f"- **Extracted:** {s.get('findings', 0)} findings proposed; {s.get('quote_failed', 0)} discarded because "
        f"their quote is not in the paper ({s.get('quote_meaning_changed', 0)} of them because the model's copy "
        f"changed a number, a negation or another word that carries meaning); "
        f"{s.get('near_miss', 0) + s.get('other_disease', 0)} discarded as about another entity; "
        f"{s.get('other_work', 0)} discarded as someone else's result, not this paper's own; "
        f"{s.get('target_unlocated', 0)} discarded because the text the model read never names their target; "
        f"{s.get('duplicates', 0)} duplicates merged; **{len(ctx.summary.units)} verified findings kept**.",
        f"- **Fields the paper does not state:** blanked rather than shown — who was studied on "
        f"{s.get('species_unlocated', 0)} findings (shown as \"not stated\"), sample size on "
        f"{s.get('n_unlocated', 0)}, the result summary on {s.get('effect_detail_unlocated', 0)}, the agent on "
        f"{s.get('intervention_unlocated', 0)}.",
        f"- **Written:** {ctx.sentence_stats.get('kept', 0)} sentences kept; {ctx.sentence_stats.get('dropped', 0)} "
        f"removed by the citation check; {ctx.sentence_stats.get('repeated', 0)} removed as repeats; "
        f"{ctx.sentence_stats.get('too_short', 0)} too short to state anything.",
        "- Every sentence cites the findings it relies on. Every quote shown is the paper's own sentence, taken "
        "from a recorded position in the paper; a \"fuzzy\" match means the model's copy differed from it only in "
        "typography or a word that carries no meaning. Numbers, names and who was studied in each sentence were "
        "checked against the paper's text it cites.",
    ]
    return "\n".join(lines)


def _excluded(ctx: WriteContext, limit: int = 80) -> str:
    rows = ["| Paper | Why excluded |", "|---|---|"]
    order = {"near_miss": 0, "absent": 1, "review": 2, "retracted": 3}
    excluded = sorted(
        (g for g in ctx.gates.values() if g.status != "pass"),
        key=lambda g: (order.get(g.status, 9), ctx.papers[g.paper_id].title),
    )
    low = [r for r in ctx.ranked if r.gate == "pass" and r.study_kind == "review"]
    for gate in excluded[:limit]:
        rows.append(f"| {cell(ctx.papers[gate.paper_id].title[:120])} | {gate.status}: {cell(gate.reason)} |")
    for r in low:
        rows.append(f"| {cell(ctx.papers[r.paper_id].title[:120])} | review (per relevance check) |")
    if len(excluded) > limit:
        rows.append(f"| … | {len(excluded) - limit} more in the JSON report |")
    if len(rows) == 2:
        return "No collected paper was excluded."
    return "\n".join(rows)


# --- assembly ---------------------------------------------------------------------


def _collect_citations(section: ReportSection, out: list[str]) -> None:
    for s in section.sentences:
        if s.kept:
            out.extend(s.citations)
    for sub in section.subsections:
        _collect_citations(sub, out)


def build_references(ctx: WriteContext, sections: list[ReportSection]) -> dict[str, int]:
    """Number papers in order of first citation, then any other paper that
    contributed evidence, then the most relevant source if still missing."""
    by_short = ctx.summary.by_short_id()
    cited: list[str] = []
    for section in sections:
        _collect_citations(section, cited)
    order: list[str] = []
    for short in cited:
        pid = by_short[short].paper_id
        if pid not in order:
            order.append(pid)
    for u in ctx.summary.units:
        if u.paper_id not in order:
            order.append(u.paper_id)
    if ctx.plan.best_source_id and ctx.plan.best_source_id not in order:
        order.append(ctx.plan.best_source_id)
    return {pid: i for i, pid in enumerate(order, 1)}


def write_report(ctx: WriteContext, *, model_versions: dict[str, str], prompt_versions: dict[str, str]) -> Report:
    layout = load_layout(ctx.spec.question_type)
    sections: list[ReportSection] = []
    # Each written section, and each per-target subsection, sees only its own
    # slice of evidence and is written by its own model call — that is the
    # whole point of Stage 7 (see the module docstring). But the same finding
    # can be the best evidence for two different sections (the answer and
    # that target's own subsection, most often), and a model then drafts the
    # same sentence twice. One list, appended to as sections are written in
    # this order and read by every call after it, is what makes a report-wide
    # repeat check possible without the sections calling each other.
    already_kept: list[set[str]] = []

    for spec in layout:
        if spec.kind != "written":
            sections.append(ReportSection(id=spec.id, title=spec.title, kind="generated"))
            continue
        if spec.source == "per_target":
            section = ReportSection(id=spec.id, title=spec.title, kind="written")
            for t in ctx.summary.targets[: ctx.max_targets]:
                units = best_units([ctx.by_unit_id[i] for i in t.unit_ids], ctx.rank_of, ctx.units_per_target)
                sentences, note = write_section_text(ctx, spec, units, subsection=t.target, already_kept=already_kept)
                names = [n for n in t.names_seen if n != t.target][:3]
                title = t.target + (f" ({', '.join(names)})" if names else "")
                section.subsections.append(
                    ReportSection(
                        id=f"{spec.id}:{t.target}", title=title, kind="written",
                        sentences=sentences, note=_target_line(t) + (f" {note}" if note else ""),
                    )
                )
            if not section.subsections:
                section.note = "No target-level evidence was extracted from the papers read."
            sections.append(section)
            continue
        units = _slice(ctx, spec.source or "all")
        sentences, note = write_section_text(ctx, spec, units, already_kept=already_kept)
        sections.append(ReportSection(id=spec.id, title=spec.title, kind="written", sentences=sentences, note=note))

    refs = build_references(ctx, sections)
    generators = {
        "best_source": lambda: _best_source(ctx, refs),
        "gaps": lambda: _gaps(ctx),
        "plan": lambda: _plan(ctx),
        "ranking": lambda: _ranking(ctx, refs),
        "evidence_table": lambda: _evidence_table(ctx, refs),
        "method": lambda: _method(ctx),
        "excluded": lambda: _excluded(ctx),
    }
    for section in sections:
        if section.kind == "generated":
            section.body_markdown = generators[section.id]()

    references = []
    for pid, number in sorted(refs.items(), key=lambda kv: kv[1]):
        paper = ctx.papers[pid]
        references.append(
            Reference(
                number=number,
                paper_id=pid,
                citation=citation_text(paper),
                url=paper_url(paper),
                signals=paper.signals,
                text_source=paper.text_source,
                evidence_ids=[ctx.summary.short_ids[u.id] for u in ctx.summary.units if u.paper_id == pid],
            )
        )

    g = Counter(x.status for x in ctx.gates.values())
    stats = {
        "papers_collected": len(ctx.papers),
        "gate_pass": g.get("pass", 0),
        "gate_near_miss": g.get("near_miss", 0),
        "gate_absent": g.get("absent", 0),
        "gate_review": g.get("review", 0),
        "gate_retracted": g.get("retracted", 0),
        "papers_graded": sum(1 for r in ctx.ranked if r.grade is not None),
        "papers_read": len(ctx.plan.jobs),
        "papers_abstract_only": sum(1 for j in ctx.plan.jobs if j.abstract_only),
        "chunks_read": ctx.extract_stats.get("chunks", 0),
        "chunks_failed": ctx.extract_stats.get("chunk_failed", 0),
        "findings_proposed": ctx.extract_stats.get("findings", 0),
        "labels_corrected": ctx.extract_stats.get("direction_corrected", 0)
        + ctx.extract_stats.get("outcome_corrected", 0),
        "findings_quote_failed": ctx.extract_stats.get("quote_failed", 0),
        "findings_other_entity": ctx.extract_stats.get("near_miss", 0) + ctx.extract_stats.get("other_disease", 0),
        "findings_other_work": ctx.extract_stats.get("other_work", 0),
        "evidence_units": len(ctx.summary.units),
        "sentences_kept": ctx.sentence_stats.get("kept", 0),
        "sentences_dropped": ctx.sentence_stats.get("dropped", 0),
        "sentences_repeated": ctx.sentence_stats.get("repeated", 0),
    }
    interpretation = {
        "question_type": ctx.spec.question_type,
        "goal": ctx.spec.goal,
        "subject": subject_label(ctx.spec),
        "population": ctx.spec.population,
        "entities": [
            {
                "text": e.text,
                "kind": e.kind,
                "canonical": e.canonical,
                "required": e.required,
                "exact_terms": e.exact_terms,
                "near_misses_excluded": e.near_miss_terms + e.near_miss_patterns,
                "resolved_from": e.resolved_from,
            }
            for e in ctx.spec.entities
        ],
    }
    return Report(
        question_id=ctx.spec.id,
        question=ctx.spec.text,
        interpretation=interpretation,
        sections=sections,
        references=references,
        evidence=ctx.summary.units,
        short_ids=ctx.summary.short_ids,
        best_source_id=ctx.plan.best_source_id,
        stats=stats,
        model_versions=model_versions,
        prompt_versions=prompt_versions,
    )


def _target_line(t) -> str:
    types = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in t.by_evidence_type.items())
    outcomes = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in t.by_outcome.items())
    drugs = f" Agents: {', '.join(t.interventions[:5])}." if t.interventions else ""
    conflict = " **Papers disagree on this target.**" if t.conflict else ""
    return (
        f"{len(t.paper_ids)} paper(s). Evidence (papers): {types}. Outcomes (papers): {outcomes}.{drugs}{conflict}"
    )
