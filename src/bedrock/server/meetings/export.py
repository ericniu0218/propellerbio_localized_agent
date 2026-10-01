"""A meeting as a file somebody can send to whoever missed it.

Three shapes. Plain text for pasting into anything; WEBVTT for anything that
plays subtitles or ingests transcripts; Word for the people who will only open
Word. All three carry the same content: the notes first, then every line of
the transcript with its timestamp and speaker, and — when lines were spoken in
Chinese and the notes model translated them — the English under each one.

The English is a courtesy. The original line is the record: it is what the
citations quote and what retrieval searches, so it always comes first and is
never replaced.
"""

from __future__ import annotations

import io
import re
from datetime import datetime

from bedrock.meetings.transcript import Cue, parse_vtt

FORMATS = {"txt": "text/plain; charset=utf-8", "vtt": "text/vtt; charset=utf-8",
           "md": "text/markdown; charset=utf-8",
           "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


def _lines(meeting: dict, english: bool) -> list[tuple[Cue, str | None]]:
    """Every cue, paired with its English line when there is one."""
    cues = parse_vtt(meeting.get("transcript") or "WEBVTT\n")
    translated = parse_vtt(meeting.get("transcriptEnglish") or "WEBVTT\n") if english else []
    by_start = {(c.start_ms, c.speaker): c.text for c in translated}
    out = []
    for cue in cues:
        en = by_start.get((cue.start_ms, cue.speaker))
        out.append((cue, en if en and en != cue.text else None))
    return out


def _when(meeting: dict) -> str:
    raw = meeting.get("recordedAt") or ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone().strftime("%d %b %Y, %H:%M")
    except ValueError:
        return raw


def _header(meeting: dict) -> list[str]:
    facts = [_when(meeting)]
    ms = meeting.get("durationMs") or 0
    if ms:
        facts.append(f"{round(ms / 1000)} s" if ms < 60_000 else f"{round(ms / 60000)} min")
    count = meeting.get("speakerCount") or 0
    if count:
        facts.append("1 speaker" if count == 1 else f"{count} speakers")
    return [meeting.get("title") or "Meeting", " · ".join(f for f in facts if f)]


def _notes(meeting: dict) -> list[tuple[str, list[str]]]:
    sections = [("Summary", [meeting["summary"]] if meeting.get("summary") else [])]
    for label, key in (("Decisions", "decisions"), ("Action items", "actions"), ("Findings", "findings")):
        sections.append((label, list(meeting.get(key) or [])))
    return [(label, items) for label, items in sections if items]


def as_text(meeting: dict, english: bool = False) -> str:
    title, facts = _header(meeting)
    out = [title, facts, ""]
    for note in meeting.get("caveats") or []:
        out.append(f"Note: {note}")
    if meeting.get("caveats"):
        out.append("")
    for label, items in _notes(meeting):
        out += [label.upper(), *(f"- {i}" if label != "Summary" else i for i in items), ""]
    out += ["TRANSCRIPT", ""]
    for cue, en in _lines(meeting, english):
        out.append(cue.render())
        if en:
            out.append(f"    EN: {en}")
    return "\n".join(out).rstrip() + "\n"


def as_markdown(meeting: dict, english: bool = False) -> str:
    title, facts = _header(meeting)
    out = [f"# {title}", "", f"*{facts}*", ""]
    for note in meeting.get("caveats") or []:
        out.append(f"> {note}")
    if meeting.get("caveats"):
        out.append("")
    for label, items in _notes(meeting):
        out += [f"## {label}", "", *(f"- {i}" if label != "Summary" else i for i in items), ""]
    out += ["## Transcript", ""]
    for cue, en in _lines(meeting, english):
        who = f"**{cue.speaker}:** " if cue.speaker else ""
        out.append(f"`{cue.timestamp}` {who}{cue.text}  ")
        if en:
            out.append(f"&nbsp;&nbsp;&nbsp;&nbsp;*{en}*  ")
    return "\n".join(out).rstrip() + "\n"


def as_vtt(meeting: dict, english: bool = False) -> str:
    """The stored WEBVTT; with ``english``, the translated one where it exists."""
    if english and meeting.get("transcriptEnglish"):
        return meeting["transcriptEnglish"]
    return meeting.get("transcript") or "WEBVTT\n"


def as_docx(meeting: dict, english: bool = False) -> bytes:
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Word export needs python-docx: pip install python-docx") from exc

    title, facts = _header(meeting)
    doc = Document()
    doc.add_heading(title, level=1)
    doc.add_paragraph(facts)
    for note in meeting.get("caveats") or []:
        if not (note or "").strip():
            continue
        doc.add_paragraph().add_run(note).italic = True
    for label, items in _notes(meeting):
        doc.add_heading(label, level=2)
        for item in items:
            doc.add_paragraph(item, style="List Bullet" if label != "Summary" else None)
    doc.add_heading("Transcript", level=2)
    for cue, en in _lines(meeting, english):
        p = doc.add_paragraph()
        stamp = p.add_run(f"[{cue.timestamp}] ")
        stamp.font.size = Pt(8)
        stamp.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        if cue.speaker:
            p.add_run(f"{cue.speaker}: ").bold = True
        p.add_run(cue.text)
        if en:
            q = doc.add_paragraph()
            q.paragraph_format.left_indent = Pt(24)
            q.add_run(en).italic = True
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def filename(meeting: dict, fmt: str) -> str:
    """The real name, which may be Chinese; ``content_disposition`` deals
    with the header's Latin-1 limit."""
    raw = meeting.get("title") or "meeting"
    stem = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", raw).strip("-")[:80] or "meeting"
    return f"{stem}.{fmt}"


def content_disposition(meeting: dict, fmt: str) -> str:
    """HTTP headers are Latin-1, so a Chinese title in a bare ``filename=``
    raised UnicodeEncodeError and the download was a 500. RFC 5987: an
    ASCII fallback in ``filename=`` and the real name in ``filename*=``."""
    from urllib.parse import quote

    name = filename(meeting, fmt)
    ascii_stem = re.sub(r"[^A-Za-z0-9\-]+", "-", name[: -len(fmt) - 1]).strip("-") or "meeting"
    return f'attachment; filename="{ascii_stem}.{fmt}"; filename*=UTF-8\'\'{quote(name)}'


def render(meeting: dict, fmt: str, english: bool = False) -> tuple[bytes, str]:
    """(body, media type) for a format in ``FORMATS``."""
    if fmt == "txt":
        body: bytes | str = as_text(meeting, english)
    elif fmt == "md":
        body = as_markdown(meeting, english)
    elif fmt == "vtt":
        body = as_vtt(meeting, english)
    elif fmt == "docx":
        body = as_docx(meeting, english)
    else:
        raise ValueError(fmt)
    return (body.encode("utf-8") if isinstance(body, str) else body), FORMATS[fmt]
