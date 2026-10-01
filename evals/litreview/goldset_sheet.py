"""Labelling sheets for the gold set: 20 papers picked by quota, each shown the
way the pipeline reads it, and an empty template to fill in.

    python -m evals.litreview.goldset_sheet corpora-searcher/obesity-treatment-targets
    python -m evals.litreview.goldset_sheet corpora/psoriasis-prevention-targets --seed 1 --out DIR

The corpus is a frozen corpus directory (relative to evals/litreview/, or a
path). Its question comes from the question file with the same id in
evals/litreview/questions/, or --question. Writes to --out (default
evals/litreview/goldset/<corpus id>/):

    index.md                 the papers, why each was picked, which to confirm, which are held back
    sheets/NN-<paper>.md     one sheet per paper
    goldset.template.jsonl   one entry per paper with empty findings, for the labeller

**How papers are picked.** The quotas in evals/litreview/README.md: 5 human
clinical trials, 5 animal studies, 4 in vitro studies, 3 negative or
no-effect results, 2 reviews, 1 paper that mentions the subject but is not
about it. Design and review categories come from what the paper says it is
(`study_design.py`: publication types and the paper's own design wording,
never a model). The last two categories code can only *propose*: a
null-result cue in the abstract (`negation._NULL`: "did not", "no
significant", ...), or the exact-entity gate's own reason ("mentions
psoriasis only in passing"). Those entries say `"kind_confirmed": false`,
and `run_extract` refuses to score until a person confirms or corrects them
while reading. The scarcest category is filled first, so a paper that fits
two goes where it is needed; within a category the order is seeded, so the
same corpus and seed always give the same papers. A corpus short of a
category says so in the index rather than filling it with something else.

**Five are held back**, one each from trials, animal, in vitro, negative and
review papers (seeded), so the holdout is not five of one kind.

**The sheet shows the paper and nothing the pipeline produced**: the title,
its identifiers, the chunks Stage 5 would send the model, and the methods
context sent beside them. No category, no gate reason, no design label, no
grade, no finding. A labeller who sees the model's answer anchors on it, and
the gold set then measures agreement with the model instead of the paper.
Why each paper was picked is in the index and the template.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from bedrock.litreview.config import EVALS_DIR, Settings, load_settings
from bedrock.litreview.knowledge import TermMatcher
from bedrock.litreview.models import Paper, QuestionSpec
from bedrock.litreview.negation import _NULL
from bedrock.litreview.stages.s2_gate import gate_papers
from bedrock.litreview.study_design import HUMAN, designs_for, grounded_labels, is_nonprimary
from evals.litreview.goldset import (
    HOLDOUT,
    KIND_TITLES,
    PROPOSED,
    QUOTAS,
    corpus_dir,
    corpus_name,
    read_by_pipeline,
    read_corpus,
    read_question,
)

TRIALS = ("human_rct", "human_interventional_other")
REVIEW_ROLES = ("systematic_review", "meta_analysis", "narrative_review")
# The cues that usually mean the paper's own result was null, proposed before
# the rest of `_NULL` ("was not", "without", "but not"), which fire on many
# positive abstracts too.
STRONG_NULL = re.compile(r"\b(no significant|not significantly|did not|failed to|no effect|no difference|"
                         r"not associated|had no)\b", re.I)


@dataclass(frozen=True)
class Candidate:
    paper: Paper
    kind: str
    because: str
    tier: int = 0
    """Lower is picked first: a review before an editorial, a strong
    null-result cue before a weak one, a paper the gate let through on one
    mention before one it excluded."""


def candidates(papers: list[Paper], spec: QuestionSpec) -> dict[str, list[Candidate]]:
    """Every paper that could fill each category, with the reason code has."""
    out: dict[str, list[Candidate]] = {kind: [] for kind, _ in QUOTAS}
    designs = designs_for(papers)
    gates = gate_papers(papers, spec)
    subject = spec.subject
    matcher = TermMatcher(subject.exact_terms, subject.near_miss_terms, subject.near_miss_patterns) if subject else None
    for p in sorted(papers, key=lambda p: p.id):
        record = designs[p.id]
        labels = grounded_labels(record)
        if p.is_review or is_nonprimary(record):
            # A review proper before a guideline, editorial or notice: all must
            # yield nothing, but the review is the case a model gets wrong.
            reviewish = bool(set(record["roles"]) & set(REVIEW_ROLES)) or p.is_review
            out["review"].append(Candidate(
                p, "review", f"roles {', '.join(record['roles'])}" if is_nonprimary(record)
                else "flagged as a review by its source", 0 if reviewish else 1))
            continue
        human, animal, cells = labels & set(HUMAN), "animal_interventional" in labels, "in_vitro" in labels
        if labels & set(TRIALS):
            out["human_clinical"].append(Candidate(p, "human_clinical", ", ".join(sorted(labels & set(TRIALS)))))
        elif animal and not human:
            out["animal"].append(Candidate(p, "animal", "animal_interventional"))
        elif cells and not human and not animal:
            out["in_vitro"].append(Candidate(p, "in_vitro", "in_vitro"))
        cue = STRONG_NULL.search(p.abstract or "") or _NULL.search(p.abstract or "")
        if cue:
            out["negative_result"].append(Candidate(
                p, "negative_result", f"its abstract says {cue.group(0).strip()!r}",
                0 if STRONG_NULL.match(cue.group(0)) else 1))
        gate = gates[p.id]
        if matcher is not None:
            head, _ = matcher.summary(p.head_text())
            in_title, _ = matcher.summary(p.title)
            if gate.status == "pass" and head == 1 and not in_title:
                out["mentions_not_about"].append(Candidate(
                    p, "mentions_not_about", f"the gate passed it on one mention in its abstract ({gate.reason})", 0))
            elif gate.status == "absent" and "only in passing" in gate.reason:
                out["mentions_not_about"].append(Candidate(p, "mentions_not_about", f"gate: {gate.reason}", 1))
            elif gate.status == "near_miss" and gate.reason.startswith("mainly about"):
                out["mentions_not_about"].append(Candidate(p, "mentions_not_about", f"gate: {gate.reason}", 2))
    return out


def pick(pool: dict[str, list[Candidate]], seed: int) -> tuple[list[Candidate], dict[str, int]]:
    """The papers, by quota, and each category's shortfall."""
    rng = random.Random(seed)
    chosen: dict[str, Candidate] = {}
    shortfall: dict[str, int] = {}
    order = [kind for kind, _ in QUOTAS]
    quota = dict(QUOTAS)
    # The scarcest category first, so a paper that fits two goes where it is needed.
    for kind in sorted(order, key=lambda k: (len(pool[k]) / quota[k], order.index(k))):
        tiers: dict[int, list[Candidate]] = {}
        for c in pool[kind]:
            if c.paper.id not in chosen:
                tiers.setdefault(c.tier, []).append(c)
        ranked = []
        for tier in sorted(tiers):
            group = list(tiers[tier])
            rng.shuffle(group)
            ranked += group
        for c in ranked[: quota[kind]]:
            chosen[c.paper.id] = c
        if len(ranked) < quota[kind]:
            shortfall[kind] = quota[kind] - len(ranked)
    picked = sorted(chosen.values(), key=lambda c: (order.index(c.kind), c.paper.id))
    return picked, shortfall


def holdout(picked: list[Candidate], seed: int) -> set[str]:
    """One held back from each of the first five categories (seeded)."""
    rng = random.Random(f"holdout-{seed}")
    out: set[str] = set()
    for kind, _ in QUOTAS:
        group = [c.paper.id for c in picked if c.kind == kind]
        if group and len(out) < HOLDOUT:
            out.add(rng.choice(group))
    rest = sorted(c.paper.id for c in picked if c.paper.id not in out)
    while len(out) < min(HOLDOUT, len(picked)) and rest:
        out.add(rest.pop(rng.randrange(len(rest))))
    return out


def _slug(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", paper_id).strip("-")


def render_sheet(n: int, paper: Paper, spec: QuestionSpec, settings: Settings, held_back: bool) -> str:
    """The paper as Stage 5 reads it, and nothing the pipeline produced."""
    chunks, methods = read_by_pipeline(paper, spec, settings)
    ids = " · ".join(x for x in (paper.id, f"doi:{paper.doi}" if paper.doi else "",
                                 paper.signals.journal or "", str(paper.signals.year or "")) if x)
    lines = [f"# Gold-set sheet {n:02d}: {paper.title}", "", ids, ""]
    if held_back:
        lines += ["> **Held back.** Label it like the others. Whoever tunes prompts must not open this sheet "
                  "or its entry until the end.", ""]
    lines += [f"Question: {spec.text}", "",
              f"Write every finding the text below reports into this paper's entry in the template "
              f"(`\"sheet\": \"sheets/{n:02d}-{_slug(paper.id)}.md\"`). How: evals/litreview/README.md, "
              "\"Labelling the gold set\".", ""]
    by_section: dict[str, list[tuple[int, str]]] = {}
    for section, i, text in chunks:
        by_section.setdefault(section, []).append((i, text))
    for section, parts in by_section.items():
        lines += [f"## {section.capitalize()}", ""]
        expected = list(range(max(i for i, _ in parts) + 1))
        if [i for i, _ in parts] != expected:
            lines += ["*The pipeline reads only some parts of this section; only those are shown.*", ""]
        lines += ["\n\n[...]\n\n".join(text for _, text in parts), ""]
    lines += ["## Methods, as sent beside every part above (for who was studied, and how many)", "", methods, ""]
    return "\n".join(lines)


def write_sheets(corpus: str | Path, question: str, out_dir: Path, *, seed: int = 0,
                 settings: Settings | None = None) -> Path:
    settings = settings or load_settings(check_endpoint=False)
    name = corpus_name(Path(corpus)) if Path(corpus).is_absolute() else str(corpus)
    papers = read_corpus(name)
    spec = read_question(question)
    picked, shortfall = pick(candidates(list(papers.values()), spec), seed)
    held = holdout(picked, seed)

    out_dir = Path(out_dir)
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    entries = [{"_comment": (
        "One entry per paper. Read its sheet, then fill `findings` (see evals/litreview/README.md, "
        "'Labelling the gold set'). An entry with no finding needs \"none_expected\": true. Confirm or correct "
        "every kind with \"kind_confirmed\": false. Delete this line when done.")}]
    index = [f"# Gold set: {name}", "", f"Question: {question}", "",
             f"Seed {seed}. {len(picked)} papers, {len(held)} held back. Sheets show the paper only; why each "
             "paper was picked is here and in the template.", "",
             "| # | Sheet | Kind | Why | Confirm when reading | Held back |", "|---|---|---|---|---|---|"]
    for n, c in enumerate(picked, 1):
        sheet = f"sheets/{n:02d}-{_slug(c.paper.id)}.md"
        (out_dir / sheet).write_text(render_sheet(n, c.paper, spec, settings, c.paper.id in held), encoding="utf-8")
        proposed = c.kind in PROPOSED
        entries.append({
            "paper_id": c.paper.id, "corpus": name, "question": question, "doi": c.paper.doi,
            "title": c.paper.title, "sheet": sheet, "kind": c.kind, "kind_confirmed": not proposed,
            "kind_proposed_because": c.because, "holdout": c.paper.id in held, "none_expected": False,
            "findings": [],
        })
        index.append(f"| {n} | [{sheet}]({sheet}) | {KIND_TITLES[c.kind]} | {c.because} | "
                     f"{'**confirm when reading**' if proposed else ''} | {'yes' if c.paper.id in held else ''} |")
    if shortfall:
        index += ["", "This corpus is short of: " + "; ".join(
            f"{KIND_TITLES[k].lower()} ({v} fewer than the quota)" for k, v in shortfall.items())
            + ". Pick those from another corpus rather than filling them with something else."]
    (out_dir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    template = out_dir / "goldset.template.jsonl"
    template.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    return template


def question_for(corpus: str | Path) -> str:
    """The question whose id is the corpus directory's name."""
    qid = Path(corpus).name
    for path in sorted((EVALS_DIR / "questions").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("id") == qid:
            return data["question"]
    raise SystemExit(f"no question file in evals/litreview/questions/ has id {qid!r}; pass --question")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.litreview.goldset_sheet")
    parser.add_argument("corpus", help="frozen corpus directory, relative to evals/litreview/ or a path")
    parser.add_argument("--question", help="the question text (default: from the question file with the corpus id)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="default: evals/litreview/goldset/<corpus id>/")
    args = parser.parse_args(argv)
    if not (corpus_dir(args.corpus) / "papers.jsonl").exists():
        print(f"goldset_sheet: no papers.jsonl in {corpus_dir(args.corpus)}", file=sys.stderr)
        return 1
    question = args.question or question_for(args.corpus)
    out = args.out or EVALS_DIR / "goldset" / Path(args.corpus).name
    print(write_sheets(args.corpus, question, out, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
