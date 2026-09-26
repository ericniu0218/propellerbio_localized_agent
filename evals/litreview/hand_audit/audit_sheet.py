"""Hand-audit sheet: sampled report sentences beside the full paper text.

The in-pipeline checks and the judge see only the quote and its context
sentence. This sheet shows the paragraph around the quote and the abstract,
so a human (or the orchestrator) can check a sentence against what the paper
says in context: subgroup-only results, a claim the next sentence qualifies,
the wrong paper cited.

    .venv/bin/python .litreview-audit/audit_sheet.py REPORT_JSON CORPUS_JSONL N SEED OUT_MD
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bedrock.litreview.models import Report  # noqa: E402
from bedrock.litreview.sources.corpus import read_papers  # noqa: E402

report = Report.model_validate_json(Path(sys.argv[1]).read_text())
papers = {p.id: p for p in read_papers(Path(sys.argv[2]))}
n, seed, out = int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])

by_id = {u.id: u for u in report.evidence}
by_short = {short: by_id[uid] for uid, short in report.short_ids.items() if uid in by_id}
kept = list(report.kept_sentences())
sample = sorted(random.Random(seed).sample(range(len(kept)), min(n, len(kept))))

lines = [f"# Audit sheet: {Path(sys.argv[1]).parent.name}",
         f"{len(kept)} kept sentences, {len(sample)} sampled (seed {seed}).", ""]
for k in sample:
    section_id, s = kept[k]
    lines += [f"## S{k} [{section_id}]", "", f"> {s.text}", "", f"cites: {', '.join(s.citations)}", ""]
    for c in s.citations:
        u = by_short.get(c)
        if u is None:
            lines += [f"- {c}: NOT IN EVIDENCE", ""]
            continue
        p = papers.get(u.paper_id)
        title = p.title if p else "?"
        lines += [f"### {c} = {u.paper_id} ({u.section}) — {title}",
                  f"target={u.target_raw!r} outcome={u.outcome} direction={u.direction} "
                  f"evidence={u.evidence_type} species={u.species_raw!r} n={u.n} setting={u.setting}",
                  f"quote: {u.quote}"]
        if u.context_quote:
            lines.append(f"context: {u.context_quote}")
        if u.qualifier_quote:
            lines.append(f"qualifier: {u.qualifier_quote}")
        if p is not None:
            text = p.sections.get(u.section, "")
            a, b = (u.quote_start or 0), (u.quote_end or 0)
            lo, hi = max(0, a - 900), min(len(text), b + 900)
            lines += ["", "paper, around the quote:", "```",
                      text[lo:a] + " <<<" + text[a:b] + ">>> " + text[b:hi], "```"]
            if u.section != "abstract":
                lines += ["abstract:", "```", (p.abstract or "")[:2500], "```"]
        lines.append("")
out.write_text("\n".join(lines), encoding="utf-8")
print(out, len(sample))
