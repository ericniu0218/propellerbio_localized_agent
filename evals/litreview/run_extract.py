"""The extraction scoreboard. Run this constantly.

    python -m evals.litreview.run_extract

Without a number, "the extraction looks pretty good" is the best anyone can
say, and three weeks later a scientist finds a mouse study reported as a human
trial and nobody can tell whether that is rare or everywhere.

The per-field table is the part that tells you what to *do*. An overall score
of 0.84 is not actionable. "evidence_type 0.78" is — it means the model is
confusing in vitro with in vivo, which is a prompt fix you can make in ten
minutes and re-measure immediately.

Two rules for using this:

Fix patterns, not instances. One paper failing oddly is noise. Five failing the
same way is a prompt change.

Never tune against the whole gold set. Five of the twenty papers are held back
(`holdout: true`) and must not be looked at until the end, or you will write a
prompt that is excellent at fifteen specific papers and mediocre at everything
else.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

GOLDSET = Path(__file__).parent / "goldset.jsonl"

# The fields we score. `quote` is excluded because it is verified mechanically
# rather than compared, and `effect_detail` because free text will not match
# exactly and scoring it as a string would be meaningless.
SCORED_FIELDS = ("target", "direction", "disease", "evidence_type", "species", "outcome", "n")

# What makes two findings "the same finding" when pairing predicted against
# expected. Deliberately loose: we are asking "did it find this result", not
# "did it word it identically".
MATCH_KEY = ("target", "direction", "disease", "evidence_type", "outcome")


def load_goldset(path: Path = GOLDSET) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"No gold set at {path}.\n"
            "Build it first — see evals/litreview/README.md. Everything else in this\n"
            "stage is guesswork until it exists."
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _key(finding: dict) -> tuple:
    return tuple(str(finding.get(f, "")).lower() for f in MATCH_KEY)


def score(expected: list[dict], predicted: list[dict]) -> dict:
    """Precision, recall, and per-field accuracy over matched pairs."""
    exp_by_key = {_key(f): f for f in expected}
    pred_by_key = {_key(f): f for f in predicted}

    matched = set(exp_by_key) & set(pred_by_key)
    precision = len(matched) / len(pred_by_key) if pred_by_key else 0.0
    recall = len(matched) / len(exp_by_key) if exp_by_key else 0.0

    field_hits: dict[str, list[int]] = defaultdict(list)
    for key in matched:
        for field in SCORED_FIELDS:
            field_hits[field].append(
                1 if str(exp_by_key[key].get(field)) == str(pred_by_key[key].get(field)) else 0
            )

    return {
        "precision": precision,
        "recall": recall,
        "matched": len(matched),
        "expected": len(exp_by_key),
        "predicted": len(pred_by_key),
        "missed": [dict(zip(MATCH_KEY, k)) for k in set(exp_by_key) - set(pred_by_key)],
        "spurious": [dict(zip(MATCH_KEY, k)) for k in set(pred_by_key) - set(exp_by_key)],
        "fields": {f: (sum(v) / len(v) if v else 0.0) for f, v in field_hits.items()},
    }


def report(results: dict, quote_validity: float) -> None:
    print()
    print(f"  Precision       {results['precision']:.2f}   "
          f"({results['matched']} of {results['predicted']} extracted were right)")
    print(f"  Recall          {results['recall']:.2f}   "
          f"({results['matched']} of {results['expected']} expected were found)")
    print(f"  Quote validity  {quote_validity:.2f}   " +
          ("" if quote_validity >= 1.0 else "<-- MUST be 1.00, investigate before anything else"))
    print()
    print("  Per-field accuracy (on matched findings)")
    worst = min(results["fields"], key=results["fields"].get) if results["fields"] else None
    for field, accuracy in sorted(results["fields"].items(), key=lambda kv: kv[1]):
        marker = "  <-- worst, fix this first" if field == worst else ""
        print(f"    {field:<16} {accuracy:.2f}{marker}")
    print()

    if results["missed"]:
        print(f"  Missed ({len(results['missed'])}) — recall failures, usually a prompt rule:")
        for item in results["missed"][:5]:
            print(f"    {item}")
    if results["spurious"]:
        print(f"  Spurious ({len(results['spurious'])}) — precision failures, usually missing a skip rule:")
        for item in results["spurious"][:5]:
            print(f"    {item}")
    print()


def main() -> None:
    gold = load_goldset()
    holdout = [p for p in gold if p.get("holdout")]
    tuning = [p for p in gold if not p.get("holdout")]
    print(f"Gold set: {len(tuning)} tuning papers, {len(holdout)} held back")

    # TODO(person-2): load each paper's sections, run extract_paper() against
    # the real client, collect predictions, then score against paper["findings"].
    raise SystemExit(
        "person-2: wire this to stages.s5_extract.extract_paper and report on\n"
        "the tuning set. Score the holdout only when you think you are done."
    )


if __name__ == "__main__":
    main()
