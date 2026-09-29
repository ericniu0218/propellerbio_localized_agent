"""Stage 3 — rank every collected paper by relevance to the question.

Three signals, each with its own job:

  gate         (Stage 2, code) does the paper concern the exact entity at all.
               Papers that fail are ranked below every paper that passes, and
               keep their reason.
  lexical      (code, BM25) how much of the question's vocabulary the title
               and abstract share. Decides which papers get a model grade
               when there are more than the budget allows, and breaks ties.
  grade        (0-3) how directly the paper's own results answer the
               question. The model answers narrow questions about the
               abstract — is the subject what the paper studies, what did its
               own experiments change — and code turns the answers into the
               grade (`grade_from`), after checking the model's quote and the
               name it gave against the abstract. An unverifiable quote caps
               the grade at 1: an unverifiable reason is no reason.

The model grades relevance, never quality. Journal, citations, preprint and
retraction status are shown next to the rank as looked-up facts.

The top-ranked usable paper is the "most relevant source" the report leads
with: the single paper that most directly answers the exact question.
"""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from bedrock.litreview.extract.verify import normalize as normalize_text
from bedrock.litreview.extract.verify import locate_quote
from bedrock.litreview.facets import agent_label, paper_facets, subject_label
from bedrock.litreview.knowledge import TermMatcher, clean
from bedrock.litreview.lexical import BM25
from bedrock.litreview.llm import LLMError, ask_json, load_prompt, strict_schema, version_of
from bedrock.litreview.models import Facet, GateResult, Paper, QuestionSpec, RankedPaper, RelevanceJudgement

log = logging.getLogger(__name__)

PROMPT = "relevance"
ABSTRACT_CHARS = 3500

_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("human_clinical", re.compile(
        r"\b(randomi[sz]ed|placebo|phase\s*(?:i{1,3}|[1-3])[ab]?\b|clinical trial|double-blind|"
        r"patients (?:were|with|received|treated)|participants|prospective cohort)", re.I)),
    ("human_genetic", re.compile(
        r"\b(GWAS|genome-wide|mendelian randomi[sz]ation|polymorphisms?|loss-of-function variants?|"
        r"risk alleles?|SNPs?)\b", re.I)),
    ("in_vivo", re.compile(
        r"\b(mice|mouse|murine|rats?|in vivo|knockout|transgenic|cynomolgus|macaques?|animal models?|"
        r"cotton rats?|calves|ferrets?)\b", re.I)),
    ("in_vitro", re.compile(r"\b(in vitro|cell lines?|cultured|organoids?|primary cells)\b", re.I)),
]


def evidence_hints(paper: Paper) -> list[str]:
    text = paper.head_text()
    return [name for name, pattern in _HINTS if pattern.search(text)]


# Words that separate papers answering this *kind* of question from papers
# that merely share its subject. Every paper past the gate already names the
# subject, so repeating its synonyms in the query only rewards papers that use
# unusual spellings of it ("psoriasis vulgaris") — it does not find answers.
_TYPE_WORDS = {
    "target_identification": "target targeting inhibition inhibitor blockade blocking knockout deficient "
                             "deficiency antagonist neutralizing antibody pathway mediates drives trial efficacy",
    "intervention_efficacy": "efficacy trial randomized placebo response outcome treated dose",
    "mechanism": "mechanism pathway signaling mediates drives regulates knockout",
    "safety": "safety adverse events tolerability risk toxicity",
    "other": "",
}
_GOAL_WORDS = {
    "prevent": "prevent prevention prophylaxis onset relapse recurrence protect",
    "treat": "treat treatment therapy remission",
}


def query_text(spec: QuestionSpec) -> str:
    """What BM25 looks for: the question's own words, the vocabulary of its
    type and goal, and the non-subject entities it names (a drug, a target)."""
    others = " ".join(
        t for e in spec.entities if e is not spec.subject for t in e.exact_terms[:3]
    )
    return " ".join(
        x for x in (spec.text, _TYPE_WORDS.get(spec.question_type, ""), _GOAL_WORDS.get(spec.goal, ""), others) if x
    )


# What "the paper's own experiments changed something" means, per question type.
_TESTED_RULES = {
    "intervention_efficacy": (
        "the name of {agent}, copied exactly as the abstract writes it, if the paper's own work gives "
        "{agent} and measures its effect on {subject}. Otherwise an empty string."
    ),
    "safety": (
        "the name of {agent}, copied exactly as the abstract writes it, if the paper's own work gives "
        "{agent} and reports its adverse effects. Otherwise an empty string."
    ),
}
_DEFAULT_TESTED_RULE = (
    "the one specific molecule, pathway, cell type, antigen or drug that the paper's own experiments "
    "blocked, activated, deleted, knocked down, neutralised, vaccinated against or gave — and then "
    "measured the effect on {subject}. Copy its name exactly as the abstract writes it (for example "
    '"IL-23" or "guselkumab"). Never the name of {subject} itself. An empty string if the paper only '
    "describes, measures or associates things without changing one."
)


def _agent_entity(spec: QuestionSpec):
    if spec.question_type in _TESTED_RULES:
        return spec.first("drug", "target")
    return None


def check_tested(tested: str, paper: Paper, spec: QuestionSpec) -> str | None:
    """The name the model says the paper changed, if code can stand behind it:
    it must be in the title or abstract, it must not be the subject itself,
    and for a question about one agent it must be that agent."""
    name = tested.strip().strip("\"'.,;: ")
    if len(name) < 2 or normalize_text(name) not in normalize_text(paper.head_text()):
        return None
    agent = _agent_entity(spec)
    if agent is not None:
        return name if TermMatcher(agent.exact_terms).exact_count(name) else None
    subject = spec.subject
    if subject is not None:
        cleaned = clean(name)
        rest = cleaned
        for m in TermMatcher(subject.exact_terms).matches(cleaned):
            rest = rest.replace(m.text, " ")
        if not re.search(r"[A-Za-z0-9]{2,}", rest.replace("infection", "").replace("disease", "")):
            return None
    return name


def grade_from(judgement: RelevanceJudgement, *, quote_ok: bool, tested: str | None) -> int:
    """The grade, from the model's answers. Written down so it is the same
    rule for every paper and every run:

      0  the paper is not about the exact entity asked about
      1  the entity is a risk factor or passing mention, the paper is a
         review, or the model's quote is not in the abstract
      2  the paper studies the entity with its own data, but changed nothing
      3  its own experiments changed a named molecule, pathway or drug and
         measured the effect — for a question about one agent, that agent
    """
    if not judgement.about_exact_entity:
        return 0
    if judgement.subject_role != "studied" or judgement.study_kind == "review" or not quote_ok:
        return 1
    return 3 if tested else 2


def relevance_request(paper: Paper, spec: QuestionSpec) -> tuple[str, dict]:
    """The prompt and schema one relevance grade sends. No model involved -
    the determinism benchmark pins these to replay against a server."""
    subject = spec.subject
    prompt = load_prompt(PROMPT)
    near = subject.near_miss_terms if subject else []
    label = subject_label(spec)
    rule = _TESTED_RULES.get(spec.question_type, _DEFAULT_TESTED_RULE)
    rendered = prompt.render(
        question=spec.text,
        subject=label,
        exact_terms=", ".join(f'"{t}"' for t in (subject.exact_terms if subject else [])),
        near_miss_line=(
            "These are different things and do NOT count: " + ", ".join(f'"{t}"' for t in near[:12]) + "."
            if near else ""
        ),
        title=paper.title,
        publication_types=", ".join(paper.signals.publication_types) or "not stated",
        abstract=(paper.abstract or "(no abstract)")[:ABSTRACT_CHARS],
        tested_rule=rule.format(subject=label, agent=agent_label(spec)),
    )
    return rendered, strict_schema(RelevanceJudgement.model_json_schema())


def grade_paper(
    paper: Paper, spec: QuestionSpec, client, *, model: str
) -> tuple[RelevanceJudgement | None, str]:
    """One model call. Returns the judgement (None on failure) and the quote status."""
    rendered, schema = relevance_request(paper, spec)
    try:
        data = ask_json(client, rendered, model=model, schema=schema, max_tokens=600)
        judgement = RelevanceJudgement.model_validate(data)
    except (LLMError, ValidationError) as exc:
        log.warning("relevance grading failed for paper=%s: %s", paper.id, exc)
        return None, "failed"
    located = locate_quote(judgement.quote, paper.head_text())
    if located.status != "failed":
        # Shown in the report as the paper's own words, so it is the paper's
        # text, whole sentence, not the model's copy of it.
        judgement = judgement.model_copy(update={"quote": located.text})
    return judgement, located.status


# --- which papers get a grade -----------------------------------------------------------

POLICIES = ("lexical", "evidence_aware")
DEFAULT_RESERVATIONS = {"human_interventional": 8, "human_genetic_observational": 4, "prevention": 4, "mechanistic": 8}
_LANES = (
    ("human_interventional", {"human_rct", "human_interventional_other"}),
    ("human_genetic_observational", {"human_genetic", "human_observational"}),
    ("prevention", None),
    ("mechanistic", {"animal_interventional", "in_vitro"}),
)


def evidence_aware_selection(
    passing: list[RankedPaper],
    papers: dict[str, Paper],
    spec: QuestionSpec,
    designs: dict[str, dict],
    *,
    max_graded: int,
    reservations: dict[str, int] | None = None,
    knowledge=None,
) -> tuple[list[str], dict[str, str]]:
    """Which gate-passing papers get a relevance grade, and why each one does or not.

    The lexical policy grades the top `max_graded` by BM25, which on a real
    corpus fills the budget with mechanism papers and leaves human trials and
    cohorts ungraded - and an ungraded paper is never read. This policy keeps
    the same budget and reserves part of it, lane by lane in a fixed order:

      human_interventional         a human trial, by publication type or the
                                   paper's own design wording
      human_genetic_observational  a human cohort, case-control or genetic study
      prevention                   (prevention questions only) prevention wording
      mechanistic                  an animal or cell experiment the paper reports

    then fills what is left by lexical score. Within a lane, papers that name
    an approved target or drug no chosen paper names yet go first, then by
    lexical score; ties break on paper id, so input order never matters. A
    lane with too few papers gives its slots back to the fill. Reviews,
    guidelines and notices take no slot at all: they are not primary evidence.

    Design labels decide what is *looked at*; they certify nothing. A human
    paper that grades 1 is still a grade-1 paper. The model is not involved.
    """
    from bedrock.litreview.facets import _FACET_WORDS
    from bedrock.litreview.knowledge import load_knowledge
    from bedrock.litreview.study_design import grounded_labels, is_nonprimary

    kb = knowledge or load_knowledge()
    reservations = dict(DEFAULT_RESERVATIONS if reservations is None else reservations)
    why: dict[str, str] = {}
    pool: list[RankedPaper] = []
    for row in sorted(passing, key=lambda r: (-r.lexical_score, r.paper_id)):
        design = designs.get(row.paper_id)
        if is_nonprimary(design):
            why[row.paper_id] = "excluded_nonprimary:" + ",".join(
                r for r in design["roles"] if r != "primary")
        else:
            pool.append(row)

    names = {
        row.paper_id: {e.canonical for e, _ in kb.scan(papers[row.paper_id].head_text(), ("target", "drug"))}
        for row in pool
    }
    chosen: list[str] = []
    covered: set[str] = set()

    def take(pid: str, reason: str) -> None:
        chosen.append(pid)
        covered.update(names[pid])
        why[pid] = reason

    for lane, labels in _LANES:
        quota = min(int(reservations.get(lane, 0)), max_graded - len(chosen))
        if quota <= 0:
            continue
        if lane == "prevention":
            if spec.goal != "prevent":
                continue
            fits = [r for r in pool if _FACET_WORDS["prevention"].search(papers[r.paper_id].head_text())]
        else:
            fits = [r for r in pool if grounded_labels(designs.get(r.paper_id)) & labels]
        fits = [r for r in fits if r.paper_id not in why]
        taken = 0
        for new_names_only in (True, False):
            for r in fits:
                if taken >= quota:
                    break
                if r.paper_id in why or (new_names_only and not (names[r.paper_id] - covered)):
                    continue
                take(r.paper_id, f"reserved:{lane}")
                taken += 1
    for r in pool:
        if len(chosen) >= max_graded:
            break
        if r.paper_id not in why:
            take(r.paper_id, "lexical_fill")
    for r in pool:
        why.setdefault(r.paper_id, f"below_budget_{max_graded}")
    return chosen, why


def rank_papers(
    papers: list[Paper],
    gates: dict[str, GateResult],
    spec: QuestionSpec,
    facets: list[Facet],
    client=None,
    *,
    model: str | None = None,
    max_graded: int = 40,
    selection: dict[str, str] | None = None,
    policy: str = "lexical",
    designs: dict[str, dict] | None = None,
    reservations: dict[str, int] | None = None,
    knowledge=None,
) -> list[RankedPaper]:
    """`selection`, when given, is filled with why each gate-passing paper was
    or was not sent for a grade, for the coverage ledger. It changes nothing.

    `policy` picks which papers get a grade within `max_graded`: `lexical`
    (the default: top BM25) or `evidence_aware` (reserved lanes by study
    design, then BM25; `evidence_aware_selection`). It never changes how a
    grade is given."""
    if policy not in POLICIES:
        raise ValueError(f"unknown rank.selection {policy!r}; expected one of {', '.join(POLICIES)}")
    by_id = {p.id: p for p in papers}
    bm25 = BM25({p.id: f"{p.title} {p.title} {p.abstract or ''}" for p in papers})
    query = query_text(spec)
    lexical = {doc_id: score for doc_id, score in bm25.rank(query)}

    rows: dict[str, RankedPaper] = {}
    for p in papers:
        gate = gates[p.id]
        rows[p.id] = RankedPaper(
            paper_id=p.id,
            rank=0,
            gate=gate.status,
            gate_reason=gate.reason,
            lexical_score=lexical.get(p.id, 0.0),
            exact_mentions=sum(gate.exact_hits.values()),
            evidence_hints=evidence_hints(p),
            has_full_text=p.has_full_text,
            study_kind="review" if gate.status == "review" else None,
        )

    passing = sorted(
        (r for r in rows.values() if r.gate == "pass"), key=lambda r: (-r.lexical_score, r.paper_id)
    )
    graded_ids: set[str] = set()
    if policy == "evidence_aware":
        if designs is None:
            from bedrock.litreview.study_design import designs_for

            designs = designs_for([by_id[r.paper_id] for r in passing])
        chosen_ids, reasons = evidence_aware_selection(
            passing, by_id, spec, designs, max_graded=max_graded, reservations=reservations, knowledge=knowledge)
        rows_by_id = {r.paper_id: r for r in passing}
        to_grade = [rows_by_id[pid] for pid in chosen_ids]
    else:
        to_grade = passing[:max_graded]
        reasons = {
            row.paper_id: f"lexical_top_{max_graded}" if i < max_graded else f"below_lexical_cutoff_{max_graded}"
            for i, row in enumerate(passing)
        }
    if selection is not None:
        selection.update(reasons)
    if client is not None and model:
        graded_by = version_of(client, model)
        for row in to_grade:
            paper = by_id[row.paper_id]
            judgement, status = grade_paper(paper, spec, client, model=model)
            if judgement is None:
                continue
            tested = check_tested(judgement.tested, paper, spec) if judgement.tested else None
            row.grade = grade_from(judgement, quote_ok=status != "failed", tested=tested)
            row.reason = judgement.reason if status != "failed" else f"[quote did not verify] {judgement.reason}"
            row.about_exact_entity = judgement.about_exact_entity
            row.subject_role = judgement.subject_role
            row.study_kind = judgement.study_kind
            row.tested = tested
            row.quote = judgement.quote
            row.quote_status = status  # type: ignore[assignment]
            row.graded_by = graded_by
            graded_ids.add(row.paper_id)
    for row in rows.values():
        if row.gate == "pass":
            tested_flag = (row.tested is not None) if row.paper_id in graded_ids else None
            row.facets = paper_facets(by_id[row.paper_id], facets, tested=tested_flag)

    def order(r: RankedPaper) -> tuple:
        gate_rank = {"pass": 0, "near_miss": 2, "absent": 3, "review": 4, "retracted": 5}[r.gate]
        if r.gate == "pass" and r.study_kind == "review":
            gate_rank = 1
        graded = 0 if r.grade is not None else 1
        return (gate_rank, graded, -(r.grade or 0), -len(r.facets), -r.lexical_score, r.paper_id)

    ranked = sorted(rows.values(), key=order)
    for i, row in enumerate(ranked, 1):
        row.rank = i
    return ranked
