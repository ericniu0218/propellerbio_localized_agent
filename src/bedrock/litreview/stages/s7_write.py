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
              With `entailment_check` on (off by default), a sentence the
              guard kept is then read by the local model against its cited
              text, and dropped if it is not supported
              (report/entailment.py).
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
from datetime import date

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
    PatentRecord,
    QuestionSpec,
    RankedPaper,
    Reference,
    Report,
    ReportSection,
    Sentence,
)
from bedrock.litreview.report.entailment import report_stats
from bedrock.litreview.report import claims, synthesis
from bedrock.litreview.report.guard import CitationGuard, GuardResult
from bedrock.litreview.report.layout import APPENDIX, SectionSpec, load_layout
from bedrock.litreview.stages.s6_score import TARGET_ORDER_RULE
from bedrock.litreview.stages.s6_score import best_units
from bedrock.litreview.stages.s6_score import best_units as _best_units

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
    patents: list[PatentRecord] = field(default_factory=list)
    """Competitor-landscape input, from `bedrock-litreview patents` (EPO OPS).
    Read only by `_competitors` below, generated (code, no model call) for
    exactly the reason no other generated section takes one: nothing here is
    a biological claim, so there is nothing for the citation guard to check."""
    entailment: object = None
    """A `report.entailment.EntailmentCheck`, or None (the default): the
    optional model check run on sentences the guard kept. See that module."""
    demoted: frozenset[str] = frozenset()
    """Papers with a published expression of concern: their findings are kept
    and marked, ranked after all others, and never used in the Answer."""
    reinstated_waiting: dict[str, str] = field(default_factory=dict)
    """Retracted, later reinstated, and not (yet) added back by a person."""
    added_back: dict[str, dict] = field(default_factory=dict)
    """Retracted, later reinstated, and added back by a person: who and when."""
    integrity_checked: set[str] | None = None
    """Papers with at least one successful retraction lookup recorded when the
    run started; None when the run did not record it."""
    integrity_check_stopped: str | None = None
    """Why this run's automatic retraction check stopped early, if it did."""
    integrity_notes: dict[str, str] = field(default_factory=dict)
    """Paper id -> recorded publication notices (concern, correction...), from
    the run's integrity snapshot. Shown beside the paper wherever it is cited;
    never used to drop a finding - a person decides what a notice means."""
    section_inputs: dict[str, list[str]] = field(default_factory=dict)
    """Section id (`targets:IL23A` for a subsection) -> the unit ids it was
    given to write from. For the coverage ledger; never in the report."""
    designs: dict[str, dict] = field(default_factory=dict)
    """Paper id -> its study-design record (`study_design.design_signals`):
    what the paper says it is, for `report/synthesis.py`."""
    studies: dict = field(default_factory=dict)
    """Paper id -> `report.synthesis.Study`, filled by `write_report`."""
    refs: dict = field(default_factory=dict)
    """Paper id -> reference number, filled by `write_report`."""
    collection_note: str | None = None
    """Set when the papers were collected for a different reading of the
    question (`stages/s1_collect.stale_collection`): the first gap shown."""

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
    one, is the paper's own Methods or Results sentence naming the subjects.
    The qualifier, when there is one, is the paper's very next sentence and
    limits the quote ("Although this result did not reach statistical
    significance ..."). Its label says so, because the prompt file only
    explains Quote and Context, and the citation guard drops a sentence that
    leaves out a "not significant" it states. The naming sentence, when there
    is one, is the paper's sentence that names the target or agent the quote
    does not ("about:" would otherwise be a name the writer cannot see in any
    quote); its label says the quote's result stays with what the quote
    names, which the citation guard enforces (check 9). A finding whose
    quote is a conclusion naming people the paper's Methods never studied
    (`methods_subjects`) says so, since the quote's own "patients" would
    otherwise invite the writer to repeat it, and the guard drops that."""
    lines = []
    for u in units:
        sid = ctx.summary.short_ids[u.id]
        n = f", n={u.n}" if u.n is not None else ""
        drug = f" · agent: {u.intervention}" if u.intervention else ""
        who = (f"studied in {_WHO.get(u.evidence_type, u.evidence_type)} ({u.species_raw}{n})"
               if u.species_raw else "who was studied: not stated in the quoted text")
        context = f'\n    Context: "{u.context_quote}"' if u.context_quote else ""
        if u.qualifier_quote:
            context += ("\n    Qualifier (the paper's next sentence; it may limit the quote, e.g. \"not "
                        f'significant\" - keep any such limit when you use the quote): "{u.qualifier_quote}"')
        if u.naming_quote:
            context += ("\n    Names (the paper's sentence that names what the quote does not; the quote's result "
                        f'belongs to what the quote itself names - name that too): "{u.naming_quote}"')
        if u.methods_subjects:
            context += (f"\n    Subjects: the quote is the authors' conclusion; the paper's Methods studied only "
                        f"{u.methods_subjects}, not people - do not say patients or people for it")
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
    ctx.section_inputs[f"{spec.id}:{subsection}" if subsection else spec.id] = [u.id for u in units]
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
        if _garbled_start(text):
            # A small-model drafting glitch, the same family as the two
            # checks above: the writer duplicates and truncates its own
            # first word ("Hypoth, hypothalamic Sik3 mRNA levels were
            # significantly increased ..." - obesity S13, evals/litreview/
            # hand_audit/qwen_obesity_after.md). Not a guard failure - the
            # words are the paper's own, just mangled in the retelling - so
            # it is dropped the same silent way as `too_short` and
            # `repeated`, not surfaced as a citation-guard drop reason.
            ctx.sentence_stats["garbled"] += 1
            continue
        expansion = _invented_expansion(text, [short[c] for c in citations if c in short], ctx)
        if expansion:
            # The writer spelled out an abbreviation its evidence never
            # does: "compared to best medical standard (BMS)", where the
            # paper's BMS is bariatric metabolic surgery (tirzepatide/liver-fat
            # run, 29 September 2026). The citation guard checks numbers, names
            # and subjects, not expansions, so it is checked here, and the
            # sentence is shown as removed with the reason.
            ctx.sentence_stats["dropped"] += 1
            sentences.append(Sentence(text=text, citations=citations, kept=False, drop_reason=expansion))
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
        if verdict.kept and getattr(ctx, "entailment", None) is not None:
            # Optional and off by default (report/entailment.py): the local
            # model's second look at a sentence the guard kept. It can only
            # remove, and what it removes is stored as dropped like any other
            # sentence, so the pure-code re-check never needs the model.
            why = ctx.entailment.review(text, {c: short[c] for c in citations}, ctx.sentence_stats)
            if why:
                verdict = GuardResult(False, why)
        if verdict.kept:
            # A leftover connective from the source sentence the model
            # paraphrased ("However, weight loss was observed ..." lifting a
            # paper's own "However, X"), never something the writer meant
            # relative to a preceding sentence it cannot see - the prompt
            # asks for standalone report sentences, never connected
            # discourse. Safe to do only now, after the guard has judged the
            # model's real words - stripping first would let the guard judge
            # a sentence the model never actually wrote.
            text = _strip_leading_connective(text)
        if text[-1] not in ".!?":
            # The model often returns a copied clause with no full stop, and
            # several of those run together are unreadable. Punctuation is not
            # a claim, so this is safe after the guard has seen the sentence.
            text += "."
        ctx.sentence_stats["kept" if verdict.kept else "dropped"] += 1
        sentences.append(Sentence(text=text, citations=citations, kept=verdict.kept, drop_reason=verdict.reason))
    # The layout's maximum, applied after the guard so it counts only
    # sentences that survived it: a model asked for "3 to 5" that writes nine
    # keeps its first five. Nothing is ever added to reach the minimum.
    kept_so_far = 0
    for i, sentence in enumerate(sentences):
        if not sentence.kept:
            continue
        kept_so_far += 1
        if kept_so_far > spec.max_sentences:
            sentences[i] = sentence.model_copy(update={
                "kept": False, "drop_reason": f"over this section's limit of {spec.max_sentences} sentences"})
            ctx.sentence_stats["kept"] -= 1
            ctx.sentence_stats["over_limit"] += 1
    if already_kept is not None:
        # Recorded for the sections written after this one - only what is
        # still kept after the limit above, so a later section never drops a
        # sentence as a repeat of one the reader was not shown.
        already_kept.extend(_content_words(s.text) for s in sentences if s.kept)
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


def _invented_expansion(text: str, cited: list[EvidenceUnit], ctx: WriteContext) -> str | None:
    """Why a drafted sentence's "long form (ABBR)" is not the paper's, or None.

    Accepted when the long form is in the cited text itself, or is how a
    cited paper defines the abbreviation, or is its standard expansion
    (`knowledge/glossary.yaml`). A different expansion of an abbreviation the
    paper defines is refused outright: it changes what was compared."""
    from bedrock.litreview.knowledge.abbreviations import find_abbreviations, paper_abbreviations

    found = find_abbreviations(text)
    if not found:
        return None
    evidence = " ".join(" ".join(x for x in (u.quote, u.context_quote, u.qualifier_quote, u.naming_quote) if x)
                        for u in cited).lower()
    defined: dict[str, str] = {}
    for pid in dict.fromkeys(u.paper_id for u in cited):
        if pid in ctx.papers:
            for short_form, long_form in paper_abbreviations(ctx.papers[pid]).items():
                defined.setdefault(short_form, long_form)
    standard = synthesis._load_glossary()

    def same(a: str, b: str) -> bool:
        wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
        return bool(wa) and (wa <= wb or wb <= wa)

    for short_form, long_form in found.items():
        if long_form.lower() in evidence:
            continue
        if short_form in defined:
            if same(long_form, defined[short_form]):
                continue
            return (f"expands {short_form} as “{long_form}”; the cited paper defines it as "
                    f"“{defined[short_form]}”")
        if short_form in standard and same(long_form, standard[short_form].split(" (")[0]):
            continue
        return f"expands {short_form} as “{long_form}”, which neither the cited text nor the paper says"
    return None


REPEAT_OVERLAP = 0.8
# A "sentence" this short states nothing. MedGemma emitted a bare "A." twice,
# which then counted as two unsupported claims.
MIN_SENTENCE_CHARS = 25
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# A short alphabetic fragment, a comma, then a longer word straight after -
# only ever checked at the very start of the sentence. Deliberately not "any
# word is a prefix of the next word" (that fires constantly on ordinary
# English: "an animal", "over overexpressed"); anchoring to the opening of
# the sentence is what keeps this safe, because a real sentence essentially
# never opens "Word, longerword-starting-with-word ...".
_GARBLED_START = re.compile(r"^([a-zA-Z]{3,12}),\s+([a-zA-Z]+)")


def _garbled_start(text: str) -> bool:
    """A small model's drafting glitch: it duplicates and truncates its own
    first word ("Hypoth, hypothalamic Sik3 mRNA levels were significantly
    increased ..." - the real obesity S13 sentence in evals/litreview/
    hand_audit/qwen_obesity_after.md). Scoped tightly to that shape: the
    fragment must be a case-insensitive prefix of the word right after the
    comma, and strictly shorter than it."""
    m = _GARBLED_START.match(text)
    if not m:
        return False
    fragment, word = m.group(1), m.group(2)
    return len(word) > len(fragment) and word.lower().startswith(fragment.lower())


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


# Only "However," - deliberately not "Moreover,". See the docstring below:
# "however" is already in `extract/verify.py`'s `_FILLER`, so removing it
# cannot change `content_words()` and cannot change a guard verdict.
# "moreover" is not filtered, and removing it provably can.
_LEADING_CONNECTIVE = re.compile(r"^however,\s+", re.IGNORECASE)


def _strip_leading_connective(text: str) -> str:
    """Drop a leading "However, " (case-insensitive) from an already-kept
    sentence, and capitalize the letter now at the start.

    Real example, obesity S29 (evals/litreview/hand_audit/
    qwen_obesity_after.md): "However, fasting plasma levels of FFAs and
    β-hydroxybutyrate were higher and lower, respectively, in ob/ob HKO mice
    ..." - the model paraphrased the paper's own qualifier sentence,
    connective included. The writer is asked for standalone report
    sentences, never connected discourse, so a leading "However" is always a
    leftover from the source, never something the writer meant relative to a
    preceding sentence it cannot see.

    Why only "However," and not "Moreover," too: this runs *after* the
    citation guard, so the stored text differs from the text the guard
    judged - and `report/invariants.py` `check_sentences` re-runs that same
    guard over the stored text on every `verify-run`. The two must therefore
    always agree. They do for "however", which `extract/verify.py`'s
    `_FILLER` already excludes from `content_words()`, so removing it cannot
    move guard check 8's or check 10's word arithmetic. "moreover" is not
    filtered: removing it changes the denominator of check 8's overlap ratio
    and was measured flipping a real verdict, which would report a correctly
    written sentence as a violation forever. Making it safe means adding
    "moreover" to `_FILLER`, which moves those checks for every sentence in
    every report - an accuracy change that needs its own measurement, not a
    side effect of a cosmetic fix."""
    match = _LEADING_CONNECTIVE.match(text)
    if not match:
        return text
    rest = text[match.end():]
    return rest[:1].upper() + rest[1:] if rest else text


def _overlap(a: set[str], b: set[str]) -> float:
    """Share of the shorter sentence's words that the other one also has."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _slice(ctx: WriteContext, source: str) -> list[EvidenceUnit]:
    units = ctx.summary.units
    limit = ctx.max_units_per_section
    if ctx.spec.question_type != "target_identification":
        # Lead with findings that measure what the question asks about.
        prefer = synthesis.section_preference(ctx)

        def best_units(chosen, rank_of, n, demoted=frozenset()):  # noqa: F811 - the same rule, subject first
            return _best_units(chosen, rank_of, n, demoted, prefer=prefer)
    else:
        best_units = _best_units
    if source == "all":
        return best_units(units, ctx.rank_of, limit, ctx.demoted)
    if source == "negative":
        conflict_ids = {uid for c in ctx.summary.conflicts for uid in c.unit_ids}
        chosen = [u for u in units if u.outcome in ("no_effect", "worsened", "mixed") or u.id in conflict_ids]
        if ctx.spec.question_type != "target_identification":
            # Not the diet that induced the disease, not a comparison of
            # genotype subgroups, not two arms that both got the drug: none of
            # those is the requested treatment failing or doing harm.
            chosen = [u for u in chosen if synthesis.substantive(u)
                      and claims.attributable(u, ctx.spec, ctx.papers.get(u.paper_id))
                      and claims.contrast_of(u, ctx.spec, ctx.papers.get(u.paper_id)) not in ("subgroup", "co_intervention")]
        return best_units(chosen, ctx.rank_of, limit, ctx.demoted)
    if source == "prevention":
        # By setting, not by facet: a prophylaxis finding is prevention
        # evidence whichever information need it was filed under, and it
        # belongs in the clinical section too.
        return best_units([u for u in units if u.setting == "prevention"], ctx.rank_of, limit, ctx.demoted)
    if source.startswith("facet:"):
        facet_id = source.split(":", 1)[1]
        return best_units([u for u in units if u.facet == facet_id], ctx.rank_of, limit, ctx.demoted)
    if source == "top_targets":
        # The Answer recommends. Findings from a paper with a published
        # expression of concern are not used here; they stay in the target
        # sections and the table, marked (`_withheld_note`).
        chosen: list[EvidenceUnit] = []
        for t in ctx.summary.targets[: ctx.max_targets]:
            usable = [ctx.by_unit_id[i] for i in t.unit_ids if ctx.by_unit_id[i].paper_id not in ctx.demoted]
            chosen += best_units(usable, ctx.rank_of, 2)
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


def _realized(ctx: WriteContext) -> dict[str, set[str]]:
    """Facet -> the papers that gave at least one verified finding for it.

    What a need actually got, as opposed to the plan's `status`, which only
    says papers whose abstracts looked right were chosen to be read."""
    out: dict[str, set[str]] = {}
    for u in ctx.summary.units:
        # A quote that reports no result (an enrolment count, a baseline
        # table, an aim) does not cover a need, however verified it is.
        if synthesis.substantive(u):
            out.setdefault(u.facet or "other", set()).add(u.paper_id)
    return out


def _realized_status(ctx: WriteContext, facet: Facet, got: int) -> str:
    if ctx.client is None and not ctx.summary.units:
        return "not extracted"
    if got == 0:
        return "gap"
    return "thin" if got < facet.min_sources else "covered"


def _gaps(ctx: WriteContext, comparison_rows: list[synthesis.SynthesisRow] | None = None) -> str:
    subject = subject_label(ctx.spec)
    bullets = [f"**Incomplete search.** {ctx.collection_note}"] if ctx.collection_note else []
    realized = _realized(ctx)
    extracted = ctx.client is not None or bool(ctx.summary.units)
    for fp in ctx.plan.facets:
        f = fp.facet
        got, planned = len(realized.get(f.id, ())), len(fp.paper_ids)
        if not extracted:
            if planned == 0:
                bullets.append(f"No paper selected for reading could supply **{f.title.lower()}** ({f.need}).")
            continue
        if got == 0 and planned == 0:
            bullets.append(f"No paper that was read supplied **{f.title.lower()}** ({f.need}).")
        elif got == 0:
            bullets.append(
                f"{planned} paper(s) were read for **{f.title.lower()}**, but no verified finding was extracted from them."
            )
        elif got < f.min_sources:
            bullets.append(
                f"Only {got} paper(s) gave verified findings for **{f.title.lower()}**; {f.min_sources} were wanted."
            )
    if not extracted:
        bullets.append("No findings were extracted: this report was assembled without a model.")
    rows = comparison_rows if comparison_rows is not None else synthesis.synthesis_rows(ctx, ctx.studies)
    no_human = [row for row in rows if row.basis is not None and not row.tested_human]
    if no_human:
        measures = []
        for row in no_human:
            numbers = sorted({ctx.refs[u.paper_id] for u in row.units if u.paper_id in ctx.refs})
            refs = "".join(f"[{n}]" for n in numbers)
            measures.append(f"{cell(row.label)} {refs}".rstrip())
        noun = " targets" if ctx.spec.question_type == "target_identification" else ""
        bullets.append(f"No tested effect in humans among the papers read for{noun}: " + "; ".join(measures) + ".")
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
    for note in synthesis.statistics_notes(ctx, getattr(ctx, "refs", {}) or {}):
        bullets.append(note)
    population = synthesis.population_note(ctx, ctx.studies)
    if population:
        bullets.append(population)
    elif ctx.spec.population:
        bullets.append(
            f"The question concerns {ctx.spec.population}. Findings were not filtered by population; "
            "check who was studied in the evidence table."
        )
    if any(u.paper_id in ctx.demoted for u in ctx.summary.units):
        bullets.append(
            "Findings from papers with a published expression of concern are kept, marked ⚠, ranked after all "
            "other evidence and not used in the Answer. Each such reference quotes the notice and links to it."
        )
    if ctx.integrity_checked is not None and jobs:
        unchecked = [j.paper_id for j in jobs if j.paper_id not in ctx.integrity_checked]
        if unchecked:
            bullets.append(
                f"{len(unchecked)} of the {len(jobs)} papers read have never been checked for retractions or "
                "expressions of concern (the retraction-check setting is manual, or the services could not be "
                "reached). `/reviews settings` explains the choice."
            )
    if ctx.integrity_check_stopped:
        bullets.append(f"The automatic retraction check stopped early: {ctx.integrity_check_stopped}.")
    for pid, notices in sorted(ctx.reinstated_waiting.items()):
        paper = ctx.papers.get(pid)
        name = f"{paper_label(paper)} “{paper.title.rstrip('.')}”" if paper else pid
        bullets.append(
            f"{name} ({pid}) was retracted and later **reinstated** ({notices}). It is left out of this report "
            f"until a scientist reviews it and adds it back: `/reviews reinstate {pid}` in the app, or "
            f"`bedrock-litreview integrity include {pid} --by <name>`."
        )
    for pid, decision in sorted(ctx.added_back.items()):
        bullets.append(
            f"{pid} was retracted, later reinstated, and added back by {decision.get('decided_by')} on "
            f"{str(decision.get('decided_at', ''))[:10]}"
            + (f" (“{decision['note']}”)" if decision.get("note") else "") + "."
        )
    noticed = [pid for pid in ctx.plan.paper_ids if pid in ctx.integrity_notes]
    if noticed:
        bullets.append(
            f"{len(noticed)} paper(s) read have a recorded publication notice (expression of concern, correction "
            "or similar); their findings are kept and marked, for a person to judge."
        )
    preprints = sum(1 for j in jobs if ctx.papers[j.paper_id].signals.is_preprint)
    if preprints:
        bullets.append(f"{preprints} paper(s) read are preprints, not yet peer reviewed.")
    if ctx.spec.question_type == "target_identification":
        bullets.append(TARGET_ORDER_RULE)
    return "\n".join(f"- {b}" for b in bullets)


def _plan(ctx: WriteContext) -> str:
    units_by_facet = Counter(u.facet for u in ctx.summary.units)
    realized = _realized(ctx)
    rows = ["| Information needed | Papers wanted | Papers read that could supply it | Papers with verified findings "
            "| Findings | Status |", "|---|---|---|---|---|---|"]
    for fp in ctx.plan.facets:
        f = fp.facet
        got = len(realized.get(f.id, ()))
        rows.append(
            f"| **{cell(f.title)}** — {cell(f.need)} | {f.min_sources}–{f.max_sources} | "
            f"{len(fp.paper_ids)} | {got} | {units_by_facet.get(f.id, 0)} | {_realized_status(ctx, f, got)} |"
        )
    rows += ["", "Status counts papers that gave verified findings, not papers chosen to be read."]
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
        rows.append(f"\n{total_usable - limit} further papers passed the gate; all are listed in report.json, saved beside this report (in the Research tab: Download JSON).")
    return "\n".join(rows)


def _evidence_table(ctx: WriteContext, refs: dict[str, int]) -> str:
    rows = [
        "| Id | Target | Agent | Evidence | Subjects | Outcome | Setting | Result | Ref |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for u in ctx.summary.units:
        n = f" (n={u.n})" if u.n is not None else ""
        rows.append(
            f"| {ctx.summary.short_ids[u.id]} | {cell(u.target)}{' (measured)' if u.outcome_measure else ''} | "
            f"{cell(u.intervention or '')} | "
            f"{u.evidence_type}{' · discussion' if u.from_discussion else ''} | "
            f"{cell(u.species_raw) if u.species_raw else 'not stated'}{n} | "
            f"{u.outcome} | {u.setting} | {cell(u.effect_detail)} | {refs.get(u.paper_id, '')}"
            f"{' ⚠ concern' if u.paper_id in ctx.demoted else ''} |"
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
        f"{s.get('not_a_result', 0)} discarded because their quote states an aim, a method, a prediction or an "
        f"open question rather than a result; "
        f"{s.get('target_unlocated', 0)} discarded because the text the model read never names their target; "
        f"{s.get('duplicates', 0)} duplicates merged; **{len(ctx.summary.units)} verified findings kept**.",
        f"- **Fields the paper does not state:** blanked rather than shown — who was studied on "
        f"{s.get('species_unlocated', 0)} findings (shown as \"not stated\"), sample size on "
        f"{s.get('n_unlocated', 0)}, the result summary on {s.get('effect_detail_unlocated', 0)}, the agent on "
        f"{s.get('intervention_unlocated', 0)}.",
        f"- **Outcome measures:** {s.get('target_outcome_measure', 0)} findings kept as evidence but given no "
        f"target row, because what they name is a measurement (airway hyperresponsiveness, monthly migraine days, "
        f"BMI and the like), judged from the paper's own abbreviation definitions where it gives one.",
        f"- **Written:** {ctx.sentence_stats.get('kept', 0)} sentences kept; {ctx.sentence_stats.get('dropped', 0)} "
        f"removed by the citation check; {ctx.sentence_stats.get('repeated', 0)} removed as repeats; "
        f"{ctx.sentence_stats.get('too_short', 0)} too short to state anything; "
        f"{ctx.sentence_stats.get('garbled', 0)} removed as a corrupted, duplicated-fragment draft.",
        "- Every sentence cites the findings it relies on. Every quote shown is the paper's own sentence, taken "
        "from a recorded position in the paper; a \"fuzzy\" match means the model's copy differed from it only in "
        "typography or a word that carries no meaning. Numbers, names and who was studied in each sentence were "
        "checked against the paper's text it cites.",
    ]
    if getattr(ctx, "entailment", None) is not None:
        st = ctx.sentence_stats
        lines.insert(len(lines) - 1, (
            f"- **Entailment check (optional, not yet checked by a scientist):** the local model re-read "
            f"{st.get('entailment_checked', 0)} sentence(s) the citation check kept against the text they cite; "
            f"{st.get('entailment_dropped', 0)} removed as not supported (included in the removals above), "
            f"{st.get('entailment_partial', 0)} kept as only partly supported, "
            f"{st.get('entailment_failed', 0)} left unchecked because the call failed."
        ))
    return "\n".join(lines)


def _competitors(ctx: WriteContext, limit: int = 60) -> str:
    """Who else has filed or been granted a patent naming these entities —
    grouped by assignee, one row per family so EP/US/CN filings of the same
    invention are not counted three times.

    No model call, no citation guard: this is a list of what the patent
    office record says, not a claim about biology. `sources/patents_base.py`
    has the full reasoning for why patents never reach the evidence pipeline.
    """
    if not ctx.patents:
        return (
            "No patent search was run for this question. Run "
            "`bedrock-litreview patents \"<question>\" --out patents.jsonl` and pass "
            "`--patents patents.jsonl` to `ask` to add one."
        )

    by_assignee: dict[str, list[PatentRecord]] = {}
    for record in ctx.patents:
        by_assignee.setdefault(record.assignee or "Unknown / unassigned", []).append(record)

    # Distinct families, not raw records: three filings of one EP/US/CN
    # family must read as one family, or the count contradicts the very
    # table underneath it that just finished merging them into one row.
    families = {r.family_id or r.id for r in ctx.patents}
    lines = [f"{len(families)} patent famil{'y' if len(families) == 1 else 'ies'} found "
             f"({len(ctx.patents)} filing{'' if len(ctx.patents) == 1 else 's'}), grouped by assignee.", ""]
    lines += ["| Assignee | Earliest priority | Filings | Lead publication | Status |", "|---|---|---|---|---|"]
    rows = []
    for assignee, records in by_assignee.items():
        records = sorted(records, key=lambda r: r.priority_date or date.max)
        earliest = records[0].priority_date
        lead = min(records, key=lambda r: r.priority_date or date.max)
        granted = sum(1 for r in records if (r.kind_code or "").upper().startswith("B"))
        status = f"{granted} granted, {len(records) - granted} pending" if granted else f"{len(records)} pending/unknown"
        rows.append((
            earliest or date.max,
            f"| {cell(assignee)} | {earliest or '—'} | {len(records)} | "
            f"{cell(lead.title[:90])} ({lead.id}) | {status} |",
        ))
    rows.sort(key=lambda r: r[0])
    lines += [row for _, row in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"| … | | {len(rows) - limit} more assignees | | |")
    lines += ["", f"Source: EPO Open Patent Services. Claim 1 (never the full claim set or description) "
                   f"is recorded for {sum(1 for p in ctx.patents if p.claim_1)} of {len(ctx.patents)} families "
                   f"in the JSON report, for a human to read — not summarised by any model here."]
    return "\n".join(lines)


BACKGROUND_QUOTED = 12
BACKGROUND_LISTED = 40


def background_reviews(ctx: WriteContext) -> list[str]:
    """The review articles to show as background: excluded from the evidence
    as reviews (by the gate, or by the relevance check), naming the exact
    subject, most relevant first. Retracted ones never reach here - the gate
    marks them retracted, not review."""
    from bedrock.litreview.knowledge import TermMatcher

    matchers = [TermMatcher(e.exact_terms) for e in ctx.spec.required_entities]
    reviews = {g.paper_id for g in ctx.gates.values() if g.status == "review"}
    reviews |= {r.paper_id for r in ctx.ranked if r.gate == "pass" and r.study_kind == "review"}
    keep = [pid for pid in reviews
            if all(m.exact_count(ctx.papers[pid].head_text()) for m in matchers)]
    order = {r.paper_id: (-r.lexical_score, r.paper_id) for r in ctx.ranked}
    return sorted(keep, key=lambda pid: order.get(pid, (0.0, pid)))


def _background_quotes(ctx: WriteContext, paper: Paper, limit: int = 2) -> list[str]:
    """Up to two of the review's own abstract sentences that name the subject,
    the ones sharing most words with the question, in the review's order."""
    from bedrock.litreview.extract.verify import split_sentences
    from bedrock.litreview.knowledge import TermMatcher
    from bedrock.litreview.lexical import BM25
    from bedrock.litreview.stages.s3_rank import query_text

    subject = ctx.spec.subject
    matcher = TermMatcher(subject.exact_terms) if subject else None
    sentences = [x.strip() for x in split_sentences(paper.abstract or "") if x.strip()]
    named = {i: x for i, x in enumerate(sentences) if 30 <= len(x) <= 450 and (matcher is None or matcher.exact_count(x))}
    if not named:
        return []
    scores = dict(BM25({str(i): x for i, x in named.items()}).rank(query_text(ctx.spec)))
    best = sorted(named, key=lambda i: (-scores.get(str(i), 0.0), i))[:limit]
    return [named[i] for i in sorted(best)]


def _background(ctx: WriteContext) -> str:
    """Reviews found for the question, with links and their own words - shown
    so the scientist sees everything the search turned up, and counted as
    evidence nowhere: a review describing a trial is not a trial. No model is
    involved; the quotes are sentences copied from each review's abstract."""
    from bedrock.litreview.study_design import design_signals

    ids = background_reviews(ctx)
    if not ids:
        return "No review article about the exact subject was collected."
    blocks = [
        "Review articles found for this question, for context. **They are not evidence in this report**: "
        "nothing below is counted in the targets, the evidence table or the Answer, which rest only on the "
        "primary studies themselves. Quotes are each review's own words."
    ]
    names = {"systematic_review": "systematic review", "meta_analysis": "meta-analysis",
             "narrative_review": "review", "guideline": "guideline"}
    for pid in ids[:BACKGROUND_QUOTED]:
        paper = ctx.papers[pid]
        roles = [names[r] for r in design_signals(paper)["roles"] if r in names] or ["review"]
        url = paper_url(paper)
        head = f"**{paper_label(paper)}** — *{paper.title.rstrip('.')}* · {', '.join(roles)}"
        if url:
            head += f" · <{url}>"
        if pid in ctx.integrity_notes:
            head += f" · ⚠ publication notice: {ctx.integrity_notes[pid]}"
        quotes = _background_quotes(ctx, paper)
        blocks.append("\n".join([head, *(f"> {q}" for q in quotes)]))
    rest = ids[BACKGROUND_QUOTED:BACKGROUND_QUOTED + BACKGROUND_LISTED]
    if rest:
        blocks.append("Also collected:\n" + "\n".join(
            f"- {paper_label(ctx.papers[pid])} — {ctx.papers[pid].title.rstrip('.')}"
            + (f" <{paper_url(ctx.papers[pid])}>" if paper_url(ctx.papers[pid]) else "") for pid in rest))
    if len(ids) > BACKGROUND_QUOTED + BACKGROUND_LISTED:
        blocks.append(f"… and {len(ids) - BACKGROUND_QUOTED - BACKGROUND_LISTED} more.")
    return "\n\n".join(blocks)


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
        rows.append(f"| … | {len(excluded) - limit} more, listed in report.json saved beside this report |")
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


def write_report(ctx: WriteContext, *, model_versions: dict[str, str], prompt_versions: dict[str, str],
                 tick=None) -> Report:
    """`tick` reports each section as it is written: one model call per
    written section, plus one per target subsection, and nothing said between
    them until now. Display only; see `progress.py`."""
    from bedrock.litreview.progress import safe_tick

    tick = safe_tick(tick)
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

    # Only the written sections cost a model call; a generated one is code and
    # returns at once, so counting all of them would make the bar jump.
    written_total = sum(1 for s in layout if s.kind == "written")
    written_done = 0
    for spec in layout:
        if spec.id == "prevention" and ctx.spec.goal != "prevent":
            # A treatment question does not get a prevention section: a
            # disease induced in an animal model and then treated is not
            # prevention, and the section filled with exactly that.
            continue
        if spec.kind != "written":
            sections.append(ReportSection(id=spec.id, title=spec.title, kind="generated"))
            continue
        tick("write", written_done, written_total, spec.title)
        if spec.source == "per_target":
            section = ReportSection(id=spec.id, title=spec.title, kind="written")
            targets = ctx.summary.targets[: ctx.max_targets]
            for n, t in enumerate(targets, 1):
                units = best_units([ctx.by_unit_id[i] for i in t.unit_ids], ctx.rank_of, ctx.units_per_target,
                                   ctx.demoted)
                tick("write", written_done, written_total, f"{spec.title} {n}/{len(targets)}")
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
            written_done += 1
            tick("write", written_done, written_total)
            continue
        units = _slice(ctx, spec.source or "all")
        sentences, note = write_section_text(ctx, spec, units, already_kept=already_kept)
        if spec.source == "top_targets" and ctx.demoted:
            withheld = _withheld_note(ctx)
            if withheld and not units:
                note = withheld  # the evidence exists; it was kept out on purpose
            elif withheld:
                note = f"{note} {withheld}" if note else withheld
        sections.append(ReportSection(id=spec.id, title=spec.title, kind="written", sentences=sentences, note=note))
        written_done += 1
        tick("write", written_done, written_total)

    refs = build_references(ctx, sections)
    if ctx.integrity_notes:
        _flag_notices(ctx, sections, refs)
    studies = synthesis.profiles(ctx)
    _basis_notes(ctx, sections, refs, studies)
    ctx.studies = studies
    ctx.refs = refs
    _decisive_fallback(ctx, sections, refs, studies)
    comparison_rows = synthesis.synthesis_rows(ctx, studies)
    generators = {
        "glance": lambda: synthesis.glance(ctx, refs, studies),
        "synthesis": lambda: synthesis.render_synthesis(comparison_rows, refs),
        "disagreements": lambda: synthesis.disagreements(ctx, refs, studies, comparison_rows),
        "numbers": lambda: synthesis.key_results(ctx, refs, studies),
        "best_source": lambda: _best_source(ctx, refs),
        "background": lambda: _background(ctx),
        "gaps": lambda: _gaps(ctx, comparison_rows),
        "plan": lambda: _plan(ctx),
        "ranking": lambda: _ranking(ctx, refs),
        "evidence_table": lambda: _evidence_table(ctx, refs),
        "method": lambda: _method(ctx),
        "excluded": lambda: _excluded(ctx),
        "competitors": lambda: _competitors(ctx),
    }
    for section in sections:
        if section.kind == "generated" and section.id != "glossary":
            section.body_markdown = generators[section.id]()
    # Last: it expands the abbreviations the main text actually uses.
    main_text = "\n".join(
        [s.text for sec in sections if sec.id not in APPENDIX for _, s in _kept(sec)]
        + [sec.body_markdown or "" for sec in sections if sec.id in ("glance", "synthesis", "disagreements", "numbers")]
    )
    for section in sections:
        if section.id == "glossary":
            section.body_markdown = synthesis.glossary(ctx, main_text, refs)

    references = []
    for pid, number in sorted(refs.items(), key=lambda kv: kv[1]):
        paper = ctx.papers[pid]
        notice = ctx.integrity_notes.get(pid)
        if notice and pid in ctx.demoted:
            notice += (". Ranked after all other evidence and not used in the Answer; the scientist makes the "
                       "final call")
        references.append(
            Reference(
                number=number,
                paper_id=pid,
                citation=citation_text(paper) + (f" — **Publication notice:** {notice}." if notice else ""),
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
        + ctx.extract_stats.get("outcome_corrected", 0)
        + ctx.extract_stats.get("evidence_type_corrected", 0),
        "findings_quote_failed": ctx.extract_stats.get("quote_failed", 0),
        "findings_other_entity": ctx.extract_stats.get("near_miss", 0) + ctx.extract_stats.get("other_disease", 0),
        "findings_other_work": ctx.extract_stats.get("other_work", 0),
        "findings_not_a_result": ctx.extract_stats.get("not_a_result", 0),
        "evidence_units": len(ctx.summary.units),
        "sentences_kept": ctx.sentence_stats.get("kept", 0),
        "sentences_dropped": ctx.sentence_stats.get("dropped", 0),
        "sentences_repeated": ctx.sentence_stats.get("repeated", 0),
    }
    if getattr(ctx, "entailment", None) is not None:
        # Only when the check ran, so a report written with it off keeps the
        # exact bytes it had before the check existed.
        stats.update(report_stats(ctx.sentence_stats))
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


def _decisive_fallback(ctx: WriteContext, sections: list[ReportSection], refs: dict[str, int], studies) -> None:
    """When the strongest direct evidence for the question did not survive
    into the Answer, say so, and give it in the paper's own words.

    The guard can refuse the sentence written from the decisive finding (it
    did: a trial's "-8·09%" once read as a number missing from its own
    quote), and the Answer was then written from whatever else survived - two
    mouse studies, in one report. Decisive: a result the requested treatment
    is responsible for, about the question's subject, from a randomised trial,
    else from any human study. Nothing is generated: the fallback is the
    finding's verified quote, with its citation."""
    if ctx.spec.question_type == "target_identification":
        return
    answer = next((sec for sec in sections if sec.id == "answer"), None)
    if answer is None or not ctx.summary.units:
        return
    about = synthesis.subject_test(ctx)

    def decisive(u, tiers):
        s = studies.get(u.paper_id)
        paper = ctx.papers.get(u.paper_id)
        return (s is not None and s.tier in tiers and synthesis.substantive(u) and about(u)
                and claims.attributable(u, ctx.spec, paper)
                and claims.contrast_of(u, ctx.spec, paper) in ("between_arm", "within_arm", "unclear")
                and not claims.statistical_anomalies(u.quote))

    for tiers in (("rct",), synthesis.HUMAN_TIERS):
        best = [u for u in ctx.summary.units if decisive(u, tiers)]
        if best:
            break
    else:
        return
    by_short = {short: uid for uid, short in ctx.summary.short_ids.items()}
    cited_papers = {ctx.by_unit_id[by_short[c]].paper_id for s in answer.sentences if s.kept for c in s.citations
                    if c in by_short and by_short[c] in ctx.by_unit_id}
    if any(u.paper_id in cited_papers for u in best):
        return
    top = sorted(best, key=lambda u: (u.section != "results", ctx.rank_of.get(u.paper_id, 10**6), u.id))[0]
    line = (f"The most direct evidence could not be stated in a checked sentence; in the paper's own words: "
            f"“{top.quote}” {synthesis.study_label(studies[top.paper_id], refs)}.")
    answer.note = f"{answer.note} {line}" if answer.note else line


def _kept(section: ReportSection):
    for s in section.sentences:
        if s.kept:
            yield section.id, s
    for sub in section.subsections:
        yield from _kept(sub)


def _basis_notes(ctx: WriteContext, sections: list[ReportSection], refs: dict[str, int], studies) -> None:
    """Under every written section with kept sentences: which studies its
    sentences rest on - design, who and how many, full text or abstract -
    computed from the cited findings, never written by the model. The
    sentence text itself is untouched, so `report/invariants.py` re-checks
    exactly what the guard judged."""
    by_short = {short: uid for uid, short in ctx.summary.short_ids.items()}

    def walk(section: ReportSection) -> None:
        cited = [ctx.by_unit_id[by_short[c]] for s in section.sentences if s.kept for c in s.citations
                 if c in by_short and by_short[c] in ctx.by_unit_id]
        line = synthesis.basis_note(cited, studies, refs)
        if line:
            section.note = f"{section.note} {line}" if section.note else line
        for sub in section.subsections:
            walk(sub)

    for section in sections:
        if section.kind == "written":
            walk(section)


def _withheld_note(ctx: WriteContext) -> str | None:
    """Say which leading targets the Answer left out, and why, so nothing is
    hidden: their only evidence comes from papers with an expression of concern."""
    names = [t.target for t in ctx.summary.targets[: ctx.max_targets]
             if t.unit_ids and all(ctx.by_unit_id[i].paper_id in ctx.demoted for i in t.unit_ids)]
    if not names:
        return None
    return ("Not used in this answer: " + ", ".join(names) + " - the only evidence for "
            + ("it comes" if len(names) == 1 else "them comes")
            + " from paper(s) with a published expression of concern. It is shown, marked, under Candidate "
            "targets and in the evidence table for a scientist to judge.")


def _flag_notices(ctx: WriteContext, sections: list[ReportSection], refs: dict[str, int]) -> None:
    """Put a recorded concern or correction next to every written section that
    cites the paper, so no conclusion rests on it unqualified."""
    by_short = {short: uid for uid, short in ctx.summary.short_ids.items()}

    def walk(section: ReportSection) -> None:
        cited = {ctx.by_unit_id[by_short[c]].paper_id for s in section.sentences if s.kept
                 for c in s.citations if c in by_short and by_short[c] in ctx.by_unit_id}
        flagged = sorted((refs[pid], ctx.integrity_notes[pid]) for pid in cited
                         if pid in ctx.integrity_notes and pid in refs)
        if flagged:
            line = "Cites " + "; ".join(f"[{n}], which has a publication notice: {note}" for n, note in flagged) + "."
            section.note = f"{section.note} {line}" if section.note else line
        for sub in section.subsections:
            walk(sub)

    for section in sections:
        if section.kind == "written":
            walk(section)


def _target_line(t) -> str:
    types = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in t.by_evidence_type.items())
    outcomes = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in t.by_outcome.items())
    drugs = f" Agents: {', '.join(t.interventions[:5])}." if t.interventions else ""
    conflict = " **Papers disagree on this target.**" if t.conflict else ""
    return (
        f"{len(t.paper_ids)} paper(s). Evidence (papers): {types}. Outcomes (papers): {outcomes}.{drugs}{conflict}"
    )
