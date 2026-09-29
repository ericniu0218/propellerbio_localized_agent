"""Stage 8 — render the Report: JSON, Markdown and self-contained HTML.

Because the report is structured data, every format is a renderer, not a
rewrite — the future PPTX deck included. The exact format the client wants
will arrive later; it changes config/report.yaml and this file, nothing else.

The HTML carries no external assets (fonts, scripts, stylesheets): it has to
open on an air-gapped machine. Hovering a citation shows the exact sentence
it rests on, which is the review screen's core promise — click a claim, see
the sentence it came from.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from bedrock.litreview.config import private_dir
from bedrock.litreview.models import EvidenceUnit, Report, ReportSection, Sentence


def _ref_map(report: Report) -> tuple[dict[str, int], dict[str, str]]:
    """short id -> reference number, and short id -> quote."""
    by_paper = {r.paper_id: r.number for r in report.references}
    short_to_unit = {short: uid for uid, short in report.short_ids.items()}
    units = {u.id: u for u in report.evidence}
    refs, quotes = {}, {}
    for short, uid in short_to_unit.items():
        unit = units.get(uid)
        if unit is not None and unit.paper_id in by_paper:
            refs[short] = by_paper[unit.paper_id]
            quotes[short] = unit.quote
    return refs, quotes


def _numbers(sentence: Sentence, refs: dict[str, int]) -> list[int]:
    return list(dict.fromkeys(refs[c] for c in sentence.citations if c in refs))


def _attach(text: str, marker: str) -> str:
    text = text.rstrip()
    if text.endswith((".", "!", "?")):
        return f"{text[:-1]} {marker}{text[-1]}"
    return f"{text} {marker}"


def _named(unit: EvidenceUnit) -> str:
    """The names a unit's naming sentence is shown for ("HDAC6"), from its
    provenance: the target, agent or partner the quote itself does not name."""
    values = {"target": unit.target_raw or unit.target, "intervention": unit.intervention,
              "combo_with": unit.combo_with}
    names = [v for k, v in values.items() if v and unit.provenance.get(k) == "naming"]
    return ", ".join(dict.fromkeys(names)) or "the finding's target"


def _interpretation_md(report: Report) -> list[str]:
    i = report.interpretation
    lines = [f"**How the question was read.** Type: {str(i.get('question_type', '')).replace('_', ' ')} · "
             f"Goal: {i.get('goal', '')}" + (f" · Population: {i['population']}" if i.get("population") else "")]
    for e in i.get("entities", []):
        terms = ", ".join(f'"{t}"' for t in e["exact_terms"][:10])
        near = e["near_misses_excluded"]
        line = f"- **{e['canonical']}** ({e['kind']}{', required' if e['required'] else ''}) — counted as: {terms}"
        if near:
            line += " · excluded as near-misses: " + ", ".join(f'"{t}"' for t in near[:10])
        lines.append(line)
    return lines


_INTEGRITY_NOTE = (
    "Something shown below as a paper's words, or a field beside it, did not match the paper it came from. "
    "That is a fault in this software, not in the literature: do not rely on this report until it is fixed."
)


def _integrity_md(report: Report) -> list[str]:
    """The warning a report opens with when `report/invariants.py` found a violation."""
    if not report.integrity:
        return []
    return ["> **Warning: this report failed its integrity check.** " + _INTEGRITY_NOTE, ">",
            *[f"> - {v}" for v in report.integrity], ""]


def to_markdown(report: Report) -> str:
    refs, quotes = _ref_map(report)
    s = report.stats
    subject = report.interpretation.get("subject", "the subject")
    out = [
        f"# {report.question}",
        "",
        *_integrity_md(report),
        f"> Compiled from {s.get('papers_read', 0)} papers read in detail, chosen from the "
        f"{s.get('gate_pass', 0)} of {s.get('papers_collected', 0)} collected papers that concern "
        f"{subject} exactly. Every sentence cites its sources, every quote shown is the paper's own "
        "sentence, and sentences whose numbers, names or subjects were not in the paper text they cite "
        "were removed.",
        "",
        *_interpretation_md(report),
        "",
    ]

    def emit(section: ReportSection, level: int) -> None:
        out.append(f"{'#' * level} {section.title}")
        out.append("")
        if section.note and section.kind == "written":
            out.extend([f"*{section.note}*", ""])
        kept = [x for x in section.sentences if x.kept]
        if kept:
            parts = []
            for sentence in kept:
                nums = _numbers(sentence, refs)
                marker = f"[{', '.join(map(str, nums))}]" if nums else ""
                parts.append(_attach(sentence.text, marker) if marker else sentence.text)
            out.extend([" ".join(parts), ""])
        if section.body_markdown:
            out.extend([section.body_markdown, ""])
        for sub in section.subsections:
            emit(sub, level + 1)

    for section in report.sections:
        emit(section, 2)

    out += ["## References", ""]
    for r in report.references:
        flags = []
        if r.signals.is_retracted:
            flags.append("**RETRACTED**")
        if r.signals.is_preprint:
            flags.append("preprint")
        if r.signals.citation_count is not None:
            flags.append(f"cited by {r.signals.citation_count}")
        flags.append("full text" if r.text_source not in ("abstract_only", "none") else "abstract only")
        link = f" <{r.url}>" if r.url else ""
        ev = f" · evidence {', '.join(r.evidence_ids)}" if r.evidence_ids else ""
        out.append(f"{r.number}. {r.citation}{link} — {'; '.join(flags)}{ev}")
    out.append("")

    if report.evidence:
        out += ["## Evidence quotes", "", "The exact sentence each finding was taken from.", ""]
        by_uid = {u.id: u for u in report.evidence}
        for uid, short in sorted(report.short_ids.items(), key=lambda kv: int(kv[1][1:])):
            unit = by_uid[uid]
            out.append(f"- **{short}** [{refs.get(short, '?')}] ({unit.section}, {unit.quote_status} match) \"{unit.quote}\"")
            if unit.context_quote:
                out.append(f"  - context ({unit.context_section}): \"{unit.context_quote}\"")
            if unit.qualifier_quote:
                out.append(f"  - qualifier (next sentence): \"{unit.qualifier_quote}\"")
            if unit.naming_quote:
                out.append(f"  - names {_named(unit)} ({unit.naming_section}): \"{unit.naming_quote}\"")
            if unit.methods_subjects:
                out.append(f"  - subjects: the quote is the authors' conclusion; the paper's methods studied only "
                           f"{unit.methods_subjects}")
        out.append("")

    dropped = [(sec.title, x) for sec in _walk(report.sections) for x in sec.sentences if not x.kept]
    if dropped:
        out += ["## Sentences removed by the citation check", ""]
        for title, x in dropped:
            out.append(f"- ({title}) \"{x.text}\" — {x.drop_reason}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _walk(sections: list[ReportSection]):
    for section in sections:
        yield section
        yield from _walk(section.subsections)


# --- HTML -----------------------------------------------------------------------

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])")
_LINK = re.compile(r"&lt;(https?://[^\s&]+)&gt;")


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    return _LINK.sub(r'<a href="\1">\1</a>', text)


def md_to_html(md: str) -> str:
    """Just enough Markdown for the generated sections: paragraphs, bullets,
    quotes and tables. No dependency, because the target machine is offline."""
    blocks, out = re.split(r"\n\s*\n", md.strip()), []
    for block in blocks:
        lines = block.splitlines()
        if all(line.lstrip().startswith("|") for line in lines):
            rows = [[c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]] for line in lines]
            rows = [r for r in rows if not all(set(c) <= {"-", ":"} and c for c in r)]
            head, *body = rows
            out.append("<div class=\"table\"><table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>" + "".join(
                           "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table></div>")
        elif all(line.startswith("- ") for line in lines):
            out.append("<ul>" + "".join(f"<li>{_inline(line[2:])}</li>" for line in lines) + "</ul>")
        elif all(line.startswith(">") for line in lines):
            out.append("<blockquote>" + _inline(" ".join(line.lstrip("> ") for line in lines)) + "</blockquote>")
        else:
            parts = []
            for line in lines:
                if line.startswith("> "):
                    parts.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
                elif line.startswith("- "):
                    parts.append(f"<ul><li>{_inline(line[2:])}</li></ul>")
                else:
                    parts.append(_inline(line))
            out.append("<p>" + "<br>".join(parts) + "</p>")
    return "\n".join(out)


_CSS = """
:root { --bg:#fbfbf8; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc; --accent:#1f5f8b; --card:#ffffff; --warn:#8b3a1f; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ecebe6; --muted:#a3a29b; --line:#34332f; --accent:#7fb3d9; --card:#1f1f1d; --warn:#e0906f; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 80px; }
h1 { font-size: 1.7rem; line-height:1.25; margin: 0 0 12px; }
h2 { font-size: 1.25rem; margin: 36px 0 8px; padding-top: 12px; border-top: 1px solid var(--line); }
h3 { font-size: 1.05rem; margin: 20px 0 4px; }
.integrity { border: 2px solid #b00020; background: #fdecee; color: #5c0011; padding: 8px 12px; margin: 8px 0 16px; }
.lede { color: var(--muted); border-left: 3px solid var(--accent); padding: 4px 12px; margin: 8px 0 16px; }
.note { color: var(--muted); font-style: italic; }
a { color: var(--accent); }
a.cite { text-decoration: none; font-size: .85em; vertical-align: super; }
.table { overflow-x: auto; margin: 8px 0; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; background: var(--card); }
th, td { border: 1px solid var(--line); padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: var(--line); }
blockquote { margin: 8px 0; padding: 6px 12px; border-left: 3px solid var(--line); color: var(--muted); }
ol.refs li { margin-bottom: 6px; font-size: .9rem; }
.flag { color: var(--warn); font-weight: 600; }
details { margin: 8px 0; } summary { cursor: pointer; color: var(--muted); }
.quotes li { font-size: .88rem; margin-bottom: 4px; }
"""


def to_html(report: Report) -> str:
    refs, quotes = _ref_map(report)
    s = report.stats
    subject = report.interpretation.get("subject", "the subject")
    body: list[str] = [f"<h1>{html.escape(report.question)}</h1>"]
    if report.integrity:
        body.append('<div class="integrity"><strong>Warning: this report failed its integrity check.</strong> '
                    + html.escape(_INTEGRITY_NOTE) + "<ul>"
                    + "".join(f"<li>{html.escape(v)}</li>" for v in report.integrity) + "</ul></div>")
    body.append(
        f"<p class=\"lede\">Compiled from {s.get('papers_read', 0)} papers read in detail, chosen from the "
        f"{s.get('gate_pass', 0)} of {s.get('papers_collected', 0)} collected papers that concern "
        f"{html.escape(str(subject))} exactly. Every sentence cites its sources — hover a citation to see "
        "the exact sentence it rests on.</p>"
    )
    body.append(md_to_html("\n".join(_interpretation_md(report))))

    def cite(sentence: Sentence) -> str:
        links = []
        seen = set()
        for c in sentence.citations:
            num = refs.get(c)
            if num is None or num in seen:
                continue
            seen.add(num)
            title = html.escape(f"{c}: {quotes.get(c, '')}", quote=True)
            links.append(f'<a class="cite" href="#ref-{num}" title="{title}">[{num}]</a>')
        return "".join(links)

    def emit(section: ReportSection, level: int) -> None:
        tag = f"h{min(level, 4)}"
        body.append(f'<{tag} id="{html.escape(section.id)}">{html.escape(section.title)}</{tag}>')
        if section.note and section.kind == "written":
            body.append(f'<p class="note">{_inline(section.note)}</p>')
        kept = [x for x in section.sentences if x.kept]
        if kept:
            body.append("<p>" + " ".join(f"{html.escape(x.text)}{cite(x)}" for x in kept) + "</p>")
        if section.body_markdown:
            content = md_to_html(section.body_markdown)
            if section.id in ("excluded", "evidence_table", "ranking"):
                content = f"<details><summary>Show</summary>{content}</details>" if section.id == "excluded" else content
            body.append(content)
        for sub in section.subsections:
            emit(sub, level + 1)

    for section in report.sections:
        emit(section, 2)

    body.append("<h2 id=\"references\">References</h2><ol class=\"refs\">")
    for r in report.references:
        flags = []
        if r.signals.is_retracted:
            flags.append('<span class="flag">RETRACTED</span>')
        if r.signals.is_preprint:
            flags.append('<span class="flag">preprint</span>')
        if r.signals.citation_count is not None:
            flags.append(f"cited by {r.signals.citation_count}")
        flags.append("full text" if r.text_source not in ("abstract_only", "none") else "abstract only")
        link = f' <a href="{html.escape(r.url)}">{html.escape(r.url)}</a>' if r.url else ""
        ev = f" · evidence {', '.join(r.evidence_ids)}" if r.evidence_ids else ""
        body.append(f'<li id="ref-{r.number}">{_inline(r.citation)}{link} — {"; ".join(flags)}{ev}</li>')
    body.append("</ol>")

    if report.evidence:
        by_uid = {u.id: u for u in report.evidence}
        body.append("<h2>Evidence quotes</h2><details open><summary>The exact sentence each finding was taken from"
                    "</summary><ul class=\"quotes\">")
        for uid, short in sorted(report.short_ids.items(), key=lambda kv: int(kv[1][1:])):
            u = by_uid[uid]
            num = refs.get(short, "?")
            body.append(f'<li id="{short}"><strong>{short}</strong> <a href="#ref-{num}">[{num}]</a> '
                        f'({u.section}, {u.quote_status} match) “{html.escape(u.quote)}”'
                        + (f'<br><small>context ({html.escape(u.context_section or "")}): '
                           f'“{html.escape(u.context_quote)}”</small>' if u.context_quote else "")
                        + (f'<br><small>qualifier (next sentence): “{html.escape(u.qualifier_quote)}”</small>'
                           if u.qualifier_quote else "")
                        + (f'<br><small>names {html.escape(_named(u))} ({html.escape(u.naming_section or "")}): '
                           f'“{html.escape(u.naming_quote)}”</small>' if u.naming_quote else "")
                        + (f"<br><small>subjects: the quote is the authors' conclusion; the paper's methods "
                           f"studied only {html.escape(u.methods_subjects)}</small>" if u.methods_subjects else "")
                        + "</li>")
        body.append("</ul></details>")

    dropped = [(sec.title, x) for sec in _walk(report.sections) for x in sec.sentences if not x.kept]
    if dropped:
        body.append("<h2>Sentences removed by the citation check</h2><details><summary>Show</summary><ul>")
        for title, x in dropped:
            body.append(f"<li>({html.escape(title)}) “{html.escape(x.text)}” — {html.escape(x.drop_reason or '')}</li>")
        body.append("</ul></details>")

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(report.question[:80])}</title><style>{_CSS}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>\n"
    )


def to_json(report: Report) -> str:
    return report.model_dump_json(indent=2) + "\n"


def render(report: Report, out_dir: Path) -> dict[str, Path]:
    """Write the three files, readable only by this account.

    A report quotes unpublished papers and answers a question that is itself
    confidential — which target a company is considering is the strategy. The
    default reports directory is already 0700 (`config.private_dir`, applied by
    `service.run_review`), but `--out` and `/review --out` may point anywhere,
    including a shared drive or /tmp, so the permissions go on the files
    themselves rather than resting on whichever parent they happen to land in.
    """
    out_dir = private_dir(Path(out_dir))
    paths = {
        "json": out_dir / "report.json",
        "md": out_dir / "report.md",
        "html": out_dir / "report.html",
    }
    bodies = {"json": to_json(report), "md": to_markdown(report), "html": to_html(report)}
    for name, path in paths.items():
        path.write_text(bodies[name], encoding="utf-8")
        path.chmod(0o600)
    return paths
