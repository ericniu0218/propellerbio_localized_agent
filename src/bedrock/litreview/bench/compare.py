"""Put two benchmark runs side by side, or refuse to and say why.

    python -m bedrock.litreview.bench compare A B [--force]

A and B are `scorecard.json` files, or the run directories that hold them
(`evals/litreview/results/<question>/<time>/`). The output is B minus A for
every system, score and gate.

Before this, before-and-after numbers were read off two scorecards by eye,
and three things made most of those readings wrong without anything saying
so (OPEN_ISSUES #25, #28):

- **Unjudged claims.** A claim with no answer on file is left out of its
  score. The 27 September migraine run judged 30, then 40, of 42 claims, and
  its 70.0 against an earlier 80.0 was mostly the missing claims.
- **Different raters.** Answers on file were written by more than one judge,
  and the earlier ones were never re-checked. A score that moves because the
  rater changed is not a change in the pipeline.
- **Different questions.** Scores for two questions are not a before and after.

So compare refuses - prints why and exits non-zero - when the questions
differ, when any score's judged fraction differs by more than
`COMPARABLE_WITHIN`, when either run does not record its judged fractions or
raters, when either run used answers whose judge is unknown, or when the two
rater mixes differ. `--force` prints the comparison anyway, with the reasons
at the top.

What compare cannot remove is one rater's own noise: the same asthma report
scored 69.4, 69.2 and 69.2 across three passes of one judge. A difference
inside that spread is not a change, whatever compare prints.
"""

from __future__ import annotations

from pathlib import Path

from bedrock.litreview.bench.history import window_text
from bedrock.litreview.bench.models import JUDGED_SCORES, Gate, Scorecard, SystemScore

COMPARABLE_WITHIN = 0.05
"""How far apart two runs' judged fractions, and each rater's share of the
verdicts, may be before compare refuses: 5 percentage points.

A starting value, not a measured one. It is loose enough that two runs of
the same report through the same answers compare, and tight enough to have
refused the 27 September migraine comparison (30 of 42 judged against 42 of
42). Tighten it once a rater's own spread is measured on judged fractions."""

CODE = "code"
UNKNOWN = "unknown"

SYSTEM_NAMES = {"pipeline": "This pipeline", "naive_rag": "Chatbot + pasted abstracts",
                "closed_book": "Chatbot, no papers"}
SCORES = (
    ("Overall (0–100)", "overall"),
    ("Claims", "claims"),
    ("Accuracy: claims supported by their citations", "cited_support"),
    ("Groundedness: claims supported by the collected papers", "corpus_grounded"),
    ("Numbers present in the evidence", "numeric_fidelity"),
    ("Claims mentioning a near-miss entity", "near_miss_claims"),
    ("Completeness: review facts covered (reachable)", "nugget_recall"),
    ("Completeness: vital review facts covered", "vital_nugget_recall"),
    ("Completeness: reference targets named (reachable)", "target_recall"),
    ("Detail: supported claims with numbers", "quantitative_supported_claims"),
    ("Distinct sources cited", "distinct_sources_cited"),
)


def load_scorecard(path: Path) -> tuple[Scorecard, Path]:
    """A scorecard from its JSON, its Markdown's directory, or the run directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "scorecard.json"
    elif path.suffix == ".md":
        path = path.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"no scorecard at {path}")
    return Scorecard.model_validate_json(path.read_text(encoding="utf-8")), path


def rater_mix(card: Scorecard) -> dict[str, float]:
    """Each judge's share of the verdicts a judge supplied. Code-settled
    verdicts are left out: code gives the same verdict every time, so a
    different number of them is a different report, not a different rater."""
    judges = {name: n for name, n in card.raters.items() if name != CODE and n}
    total = sum(judges.values())
    return {name: n / total for name, n in sorted(judges.items())} if total else {}


def _pct(pair: tuple[int, int]) -> str:
    return f"{pair[0]}/{pair[1]} ({pair[0] / pair[1]:.0%})"


def _judged_pairs(card: Scorecard) -> dict[tuple[str, str], tuple[int, int]]:
    pairs = {(s.system, name): pair for s in card.systems for name, pair in s.judged.items() if pair[1]}
    total, judged = card.pipeline.get("perturbations"), card.pipeline.get("perturbations_judged")
    if isinstance(total, int) and isinstance(judged, int) and total:
        pairs[("pipeline", "perturbations")] = (judged, total)
    return pairs


def refusals(a: Scorecard, b: Scorecard) -> list[str]:
    """Why A and B cannot be compared; empty when they can."""
    out: list[str] = []
    if a.question_id != b.question_id:
        out.append(f"A answers {a.question_id!r} and B answers {b.question_id!r}: scores for different "
                   "questions are not a before and after.")

    for label, card in (("A", a), ("B", b)):
        if not any(s.judged for s in card.systems):
            out.append(f"{label} does not record how many claims each score rests on (scored before 30 September "
                       "2026), so its scores may rest on unjudged claims.")
    pa, pb = _judged_pairs(a), _judged_pairs(b)
    for key in [k for k in pa if k in pb]:
        fa, fb = pa[key][0] / pa[key][1], pb[key][0] / pb[key][1]
        if abs(fa - fb) > COMPARABLE_WITHIN + 1e-9:
            system, score = key
            out.append(f"{SYSTEM_NAMES.get(system, system)}, {score}: A judged {_pct(pa[key])}, B {_pct(pb[key])} - "
                       f"more than {COMPARABLE_WITHIN:.0%} apart, so the difference could be the unjudged part.")

    for label, card in (("A", a), ("B", b)):
        if not card.raters:
            out.append(f"{label} does not record who judged it (scored before 30 September 2026).")
        elif card.raters.get(UNKNOWN):
            out.append(f"{card.raters[UNKNOWN]} of {label}'s verdicts came from answers with no recorded judge "
                       "(\"unknown\"), which were written by more than one rater, so its raters cannot be shown "
                       "to match. Re-judge those claims with one named judge.")
    ma, mb = rater_mix(a), rater_mix(b)
    if a.raters and b.raters and not _same_mix(ma, mb):
        out.append(f"The raters differ: A was judged by {_mix_text(ma)}, B by {_mix_text(mb)}. A score that "
                   "moves because the rater changed is not a change in the pipeline.")
    return out


def _same_mix(a: dict[str, float], b: dict[str, float]) -> bool:
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= COMPARABLE_WITHIN + 1e-9 for k in a)


def _mix_text(mix: dict[str, float]) -> str:
    return ", ".join(f"{name} {share:.0%}" for name, share in mix.items()) or "no judge (code only)"


def _value(score: SystemScore | None, attr: str):
    return None if score is None else getattr(score, attr)


def _fmt(x, attr: str = "") -> str:
    if x is None:
        return "–"
    if attr == "overall":
        return f"{x:.1f}"
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def _delta(a, b, attr: str) -> str:
    if a is None or b is None:
        return "–"
    d = b - a
    if attr == "overall":
        return f"{d:+.1f}"
    return f"{d:+.2f}" if isinstance(d, float) else f"{d:+d}"


def _judged_cell(score: SystemScore | None, attr: str) -> str:
    if score is None:
        return "–"
    if attr == "overall":
        f = score.overall_judged_fraction()
        return "–" if f is None else f"{f:.0%}"
    pair = score.judged.get(attr)
    return f"{pair[0]}/{pair[1]}" if pair else "–"


def _gate_text(gate: Gate | None) -> str:
    if gate is None:
        return "–"
    return {"pass": "PASS", "fail": "**FAIL**"}.get(gate.status, f"not measured: {gate.reason}")


def render_comparison(a: Scorecard, b: Scorecard, path_a: Path, path_b: Path, refused: list[str]) -> str:
    rows = ["# Benchmark comparison", ""]
    if refused:
        rows += ["**Printed anyway (--force). These runs are not comparable:**", ""]
        rows += [f"- {r}" for r in refused] + [""]
    rows += [
        f"A: `{path_a}` — question `{a.question_id}`, report `{a.report_id}`, {a.created}  ",
        f"B: `{path_b}` — question `{b.question_id}`, report `{b.report_id}`, {b.created}",
        "",
        f"Context window the report was written at: A {window_text(a.report_context_windows)}; "
        f"B {window_text(b.report_context_windows)}  ",
        f"Raters: A {_mix_text(rater_mix(a)) if a.raters else 'not recorded'}; "
        f"B {_mix_text(rater_mix(b)) if b.raters else 'not recorded'}",
    ]
    systems_a, systems_b = {s.system: s for s in a.systems}, {s.system: s for s in b.systems}
    for system in dict.fromkeys([*systems_a, *systems_b]):
        sa, sb = systems_a.get(system), systems_b.get(system)
        rows += ["", f"## {SYSTEM_NAMES.get(system, system)}", "",
                 "| Score | A | B | B − A | Judged A | Judged B |", "|---|---|---|---|---|---|"]
        for label, attr in SCORES:
            va, vb = _value(sa, attr), _value(sb, attr)
            judged = attr == "overall" or attr in JUDGED_SCORES
            rows.append(f"| {label} | {_fmt(va, attr)} | {_fmt(vb, attr)} | {_delta(va, vb, attr)} | "
                        f"{_judged_cell(sa, attr) if judged else ''} | {_judged_cell(sb, attr) if judged else ''} |")
    rows += ["", "## Pipeline checks", "", "| Check | A | B |", "|---|---|---|"]
    for name in dict.fromkeys([*a.gates, *b.gates]):
        rows.append(f"| {name} | {_gate_text(a.gates.get(name))} | {_gate_text(b.gates.get(name))} |")
    rows += ["", "A difference inside one rater's own spread is not a change: the same asthma report scored 69.4, "
             "69.2 and 69.2 across three passes of one judge (OPEN_ISSUES #25)."]
    return "\n".join(rows) + "\n"


def compare(path_a: Path, path_b: Path, *, force: bool = False, out=print, err=print) -> int:
    """Print the comparison, or refuse. Returns the exit code."""
    a, path_a = load_scorecard(path_a)
    b, path_b = load_scorecard(path_b)
    refused = refusals(a, b)
    if refused and not force:
        err("compare: refused - these runs are not comparable:")
        for reason in refused:
            err(f"  - {reason}")
        err("compare: --force prints the comparison anyway, with these reasons at the top.")
        return 1
    out(render_comparison(a, b, path_a, path_b, refused), end="")
    return 0
