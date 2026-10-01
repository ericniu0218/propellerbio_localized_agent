"""The report as a PDF. Code only - no model.

Another renderer of the same Report (stages/s8_render.py renders JSON, Markdown
and HTML; report/slides.py the deck): it lays out the Markdown the report
already renders, so the PDF holds exactly what the Markdown holds - the same
sentences with the same [n] citations, the same tables, every quote and every
reference with its link - and cannot say anything the report does not.

The Markdown it reads is the report's own (`to_markdown`), so only what that
renderer writes needs handling: headings, paragraphs, "- " lists (nested two
deep), numbered lines, "> " quotes, pipe tables, "---", the <sub> evidence
trail, **bold**, *italic* and <https://...> links.

The font ships with the package (DejaVu Sans, report/fonts/, free to
redistribute - see LICENSE-DejaVu.txt) because the server is air-gapped and a
report is full of characters a standard PDF font does not have: ≥ ± · − η² β.
A character the font still lacks is shown as "[?]" rather than as an empty
box, and `missing_glyphs` says which, so a test can hold the line.

Needs reportlab (`pip install -e ".[server]"`), imported only when a PDF is made.
"""

from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

from bedrock.litreview.models import Report

FONTS = Path(__file__).resolve().parent / "fonts"
FONT, FONT_BOLD, FONT_ITALIC = "DejaVuSans", "DejaVuSans-Bold", "DejaVuSans-Oblique"
PDF_VERSION = "pdf.v1"


@lru_cache(maxsize=1)
def _register() -> frozenset[int]:
    """Register the bundled font once; return the code points it can draw."""
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for name, file in ((FONT, "DejaVuSans.ttf"), (FONT_BOLD, "DejaVuSans-Bold.ttf"),
                       (FONT_ITALIC, "DejaVuSans-Oblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
    addMapping(FONT, 0, 0, FONT)
    addMapping(FONT, 1, 0, FONT_BOLD)
    addMapping(FONT, 0, 1, FONT_ITALIC)
    addMapping(FONT, 1, 1, FONT_BOLD)
    face = pdfmetrics.getFont(FONT).face
    return frozenset(face.charToGlyph)


def missing_glyphs(text: str) -> set[str]:
    """Characters in `text` the bundled font cannot draw (whitespace aside)."""
    drawable = _register()
    return {c for c in text if not c.isspace() and ord(c) not in drawable}


# --- inline Markdown -> reportlab's paragraph markup ---------------------------------------------------

_AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])")
_CODE = re.compile(r"`([^`]+)`")


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str) -> str:
    """One line of the report's Markdown as paragraph markup."""
    links: list[str] = []

    def hold(m: re.Match) -> str:
        links.append(m.group(1))
        return f"\x00{len(links) - 1}\x00"

    text = _AUTOLINK.sub(hold, _safe(text))
    text = re.sub(r"</?sub>", "", text).replace("\\|", "|")
    text = _esc(text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    text = _CODE.sub(r"\1", text)
    for i, url in enumerate(links):
        shown = _esc(url)
        text = text.replace(f"\x00{i}\x00", f'<link href="{shown}" color="#1f5f8b">{shown}</link>')
    return text


def _safe(text: str) -> str:
    """Characters the font cannot draw, made visible instead of an empty box."""
    missing = missing_glyphs(text)
    return "".join("[?]" if c in missing else c for c in text) if missing else text


# --- blocks -------------------------------------------------------------------------------------------------


def _cells(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|") and not inner.endswith("\\|"):
        inner = inner[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", inner)]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?", line.strip()))


def to_pdf(report: Report) -> bytes:
    """The whole report - reading view, references and Appendix - as a PDF."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from bedrock.litreview.stages.s8_render import to_markdown

    _register()
    ink, muted, line_colour, accent = (colors.HexColor("#1d1d1b"), colors.HexColor("#5b5b56"),
                                       colors.HexColor("#d9d8d2"), colors.HexColor("#1f5f8b"))
    base = ParagraphStyle("body", fontName=FONT, fontSize=9.5, leading=13.2, textColor=ink, alignment=TA_LEFT,
                          spaceAfter=5)
    styles = {
        1: ParagraphStyle("h1", parent=base, fontName=FONT_BOLD, fontSize=17, leading=21, spaceBefore=4,
                          spaceAfter=8, textColor=accent),
        2: ParagraphStyle("h2", parent=base, fontName=FONT_BOLD, fontSize=13.5, leading=17, spaceBefore=12,
                          spaceAfter=5, textColor=accent),
        3: ParagraphStyle("h3", parent=base, fontName=FONT_BOLD, fontSize=11.5, leading=15, spaceBefore=8,
                          spaceAfter=4),
        4: ParagraphStyle("h4", parent=base, fontName=FONT_BOLD, fontSize=10.5, leading=14, spaceBefore=6,
                          spaceAfter=3),
    }
    small = ParagraphStyle("small", parent=base, fontSize=7.8, leading=10, textColor=muted)
    quote = ParagraphStyle("quote", parent=base, leftIndent=14, textColor=muted, fontSize=9)
    cell = ParagraphStyle("cell", parent=base, fontSize=7.6, leading=9.6, spaceAfter=0)
    head_cell = ParagraphStyle("head", parent=cell, fontName=FONT_BOLD)

    page_w, _page_h = LETTER
    margin = 0.7 * inch
    width = page_w - 2 * margin

    def bullet(text: str, level: int, mark: str = "•") -> Paragraph:
        indent = 12 + 14 * level
        return Paragraph(inline(text), ParagraphStyle(f"li{level}", parent=base, leftIndent=indent,
                                                       bulletIndent=indent - 10, spaceAfter=2.5),
                         bulletText=mark)

    def table(rows: list[list[str]]) -> Table:
        n = max(len(r) for r in rows)
        rows = [r + [""] * (n - len(r)) for r in rows]
        # Column widths from how much each column holds, with a floor, so a
        # short column (a reference number) is narrow and a quote column wide.
        weight = [max(6, min(60, max(len(r[i]) for r in rows))) for i in range(n)]
        total = sum(weight)
        widths = [width * w / total for w in weight]
        data = [[Paragraph(inline(c), head_cell if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, line_colour),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efe9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        return t

    story: list = []
    paragraph: list[str] = []
    quoted: list[str] = []
    rows: list[list[str]] = []

    def flush() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), base))
            paragraph.clear()
        if quoted:
            story.append(Paragraph(inline(" ".join(q for q in quoted if q)), quote))
            quoted.clear()
        if rows:
            story.append(table(rows))
            story.append(Spacer(1, 6))
            rows.clear()

    for raw in to_markdown(report).splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("|"):
            if paragraph or quoted:
                flush()
            if not _is_separator(stripped):
                rows.append(_cells(stripped))
            continue
        if rows:
            flush()
        if not stripped:
            flush()
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            flush()
            story.append(Paragraph(inline(heading.group(2)), styles[len(heading.group(1))]))
            continue
        if stripped == "---":
            flush()
            story.append(HRFlowable(width="100%", thickness=0.5, color=line_colour, spaceBefore=4, spaceAfter=6))
            continue
        if stripped.startswith(">"):
            if paragraph:
                flush()
            quoted.append(stripped.lstrip(">").strip())
            continue
        if stripped.startswith("<sub>"):
            flush()
            story.append(Paragraph(inline(stripped), small))
            continue
        item = re.match(r"^( *)- (.*)$", line)
        if item:
            flush()
            story.append(bullet(item.group(2), min(len(item.group(1)) // 2, 3)))
            continue
        numbered = re.match(r"^( *)(\d+)\. (.*)$", line)
        if numbered:
            flush()
            story.append(bullet(numbered.group(3), min(len(numbered.group(1)) // 2, 3), f"{numbered.group(2)}."))
            continue
        if quoted:
            flush()
        paragraph.append(stripped)
    flush()

    title = report.question

    def furniture(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(muted)
        shown = title if len(title) <= 110 else title[:107] + "..."
        canvas.drawString(margin, 0.45 * inch, _safe(shown))
        canvas.drawRightString(page_w - margin, 0.45 * inch, f"Page {doc.page}")
        canvas.restoreState()

    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=LETTER, leftMargin=margin, rightMargin=margin, topMargin=margin,
                            bottomMargin=0.8 * inch, title=title[:250], author="Bedrock literature review",
                            subject="Literature review report", creator=PDF_VERSION, invariant=1)
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)
    return out.getvalue()
