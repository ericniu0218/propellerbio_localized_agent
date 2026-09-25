"""Split a paper into named sections.

Extraction quality depends on this more than on the prompt. A model given one
clean Results section does well; a model given twelve pages of undifferentiated
PDF text does badly, and no amount of prompt tuning recovers it.

Three input shapes, in descending order of how much we like them:

  Europe PMC JATS XML   sections already labelled. Nothing to guess.
  GROBID TEI            <div type="results"> and friends. Nearly as good.
  flat text             headings found by regex. Lossy, and the reason we try
                        the other two first.

Everything funnels into the same canonical names, because Stage 5 keys off them:
abstract, methods, results, discussion, other.
"""

from __future__ import annotations

import re

CANONICAL = ("abstract", "methods", "results", "discussion", "other")

# Journals name these sections a dozen ways. Longest-matching wins, so
# "materials and methods" is not caught by the "results" pattern inside it.
_HEADING_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*abstract\b", re.I), "abstract"),
    (re.compile(r"^\s*summary\b", re.I), "abstract"),
    (re.compile(r"^\s*(materials?\s+and\s+methods?|methods?|experimental\s+procedures?|study\s+design)\b", re.I), "methods"),
    (re.compile(r"^\s*(results?\s+and\s+discussion)\b", re.I), "results"),
    (re.compile(r"^\s*results?\b", re.I), "results"),
    (re.compile(r"^\s*(discussion|conclusions?|interpretation)\b", re.I), "discussion"),
    (re.compile(r"^\s*(introduction|background)\b", re.I), "other"),
    (re.compile(r"^\s*(references?|bibliography|acknowledge?ments?|funding|author\s+contributions|supplementary)\b", re.I), "other"),
]

# JATS <sec sec-type="..."> values, when the publisher bothered to set them.
_JATS_SEC_TYPE = {
    "intro": "other",
    "introduction": "other",
    "materials|methods": "methods",
    "methods": "methods",
    "materials": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusions": "discussion",
    "supplementary-material": "other",
}


def canonical_section(heading: str) -> str:
    """Map any heading onto one of CANONICAL. Unknown headings become 'other'."""
    if not heading:
        return "other"
    for pattern, name in _HEADING_MAP:
        if pattern.match(heading):
            return name
    return "other"


def _classify(heading: str) -> str | None:
    """Like canonical_section, but None for a heading no pattern recognises —
    so a descriptive subsection title ("IL-23 drives epidermal hyperplasia")
    can inherit its parent's section instead of being thrown into 'other'."""
    for pattern, name in _HEADING_MAP:
        if pattern.match(heading or ""):
            return name
    return None


def _sec_type(value: str | None) -> str | None:
    value = (value or "").lower()
    if not value:
        return None
    if "result" in value:
        return "results"
    if "method" in value or "material" in value:
        return "methods"
    if "discussion" in value or "conclusion" in value:
        return "discussion"
    if "intro" in value or "supplement" in value:
        return "other"
    return None


def _parse_xml(xml_text: str):
    from lxml import etree

    try:
        root = etree.fromstring(xml_text.encode("utf-8"), parser=etree.XMLParser(recover=True, huge_tree=True))
    except (etree.XMLSyntaxError, ValueError):
        return None
    if root is None:
        return None
    # Drop namespaces: JATS and TEI both arrive with and without them, and
    # nothing here needs them.
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _text(el) -> str:
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


class _Collector:
    def __init__(self) -> None:
        self.parts: dict[str, list[str]] = {}
        self.unlabelled: list[str] = []

    def add(self, name: str | None, text: str) -> None:
        if not text:
            return
        if name is None:
            self.unlabelled.append(text)
        else:
            self.parts.setdefault(name, []).append(text)

    def result(self) -> dict[str, str]:
        # Some journals put results under descriptive headings with no
        # "Results" parent at all. If nothing was labelled results, those
        # unlabelled sections are the results; otherwise they are 'other'.
        if self.unlabelled:
            self.parts.setdefault("results" if "results" not in self.parts else "other", []).extend(
                self.unlabelled
            )
        return {name: "\n\n".join(chunks).strip() for name, chunks in self.parts.items() if "".join(chunks).strip()}


def from_jats(xml_text: str) -> dict[str, str]:
    """Europe PMC full-text XML -> {section: text}.

    The happy path. Always try this before touching a PDF: Europe PMC serves
    full text free and already structured for the open-access subset, which
    skips PDF parsing entirely and is both faster and more accurate.

    <xref>, <table-wrap> and <fig> are stripped: figure captions read like
    results sentences and produce findings that cite a caption rather than the
    data. Reference markers go too, keeping the sentence around them.
    """
    from lxml import etree

    root = _parse_xml(xml_text)
    if root is None:
        return {}
    etree.strip_elements(
        root, "xref", "table-wrap", "table-wrap-group", "fig", "fig-group", "disp-formula",
        "supplementary-material", "graphic", "media", "ref-list", with_tail=False,
    )
    out = _Collector()

    for abstract in root.iter("abstract"):
        if (abstract.get("abstract-type") or "").lower() in {"graphical", "teaser", "toc"}:
            continue
        paras = [_text(p) for p in abstract.iter("p")] or [_text(abstract)]
        out.add("abstract", "\n\n".join(t for t in paras if t))
        break

    body = next(root.iter("body"), None)
    if body is None:
        return out.result()

    def walk(sec, inherited: str | None) -> None:
        title_el = sec.find("title")
        title = _text(title_el) if title_el is not None else ""
        name = _sec_type(sec.get("sec-type")) or _classify(title) or inherited
        for child in sec:
            tag = child.tag if isinstance(child.tag, str) else ""
            if tag == "title" or tag == "label":
                continue
            if tag == "sec":
                walk(child, name)
            else:
                out.add(name, _text(child))

    for child in body:
        tag = child.tag if isinstance(child.tag, str) else ""
        if tag == "sec":
            walk(child, None)
        elif tag:
            out.add("other", _text(child))
    return out.result()


def from_grobid_tei(tei_text: str) -> dict[str, str]:
    """GROBID TEI -> {section: text}.

    GROBID flattens subsections into a sequence of <div>s, each with a <head>.
    A head no pattern recognises is a subsection of whatever came before it,
    so it inherits that section.

    Known limitation to surface to the client rather than discover late:
    GROBID does not read figures. In biology a great deal of the result is a
    picture of a graph, and none of it reaches us. Tables need a separate
    extractor too.
    """
    from lxml import etree

    root = _parse_xml(tei_text)
    if root is None:
        return {}
    etree.strip_elements(root, "ref", "figure", "formula", with_tail=False)
    out = _Collector()

    for abstract in root.iter("abstract"):
        paras = [_text(p) for p in abstract.iter("p")] or [_text(abstract)]
        out.add("abstract", "\n\n".join(t for t in paras if t))
        break

    body = next(root.iter("body"), None)
    if body is None:
        return out.result()
    current: str | None = None
    for div in body.iter("div"):
        head = div.find("head")
        heading = _text(head) if head is not None else ""
        named = _classify(heading)
        if named is not None:
            current = named
        for p in div.findall("p"):
            out.add(current, _text(p))
    return out.result()


def from_flat_text(text: str) -> dict[str, str]:
    """Last resort: find headings in plain text.

    Works on well-behaved papers and fails quietly on badly parsed ones, which
    is why it is third choice. When this path is taken, extraction accuracy
    drops measurably — worth logging the ratio so the effect is visible.
    """
    if not text:
        return {}

    sections: dict[str, list[str]] = {}
    current = "other"
    for line in text.splitlines():
        stripped = line.strip()
        # A heading is short, on its own line, and matches a known pattern.
        # The length cap keeps a sentence that merely begins with "Results
        # showed that..." from being mistaken for a heading.
        if stripped and len(stripped) < 60:
            mapped = canonical_section(stripped)
            if mapped != "other" or _looks_like_heading(stripped):
                current = mapped
                continue
        sections.setdefault(current, []).append(line)

    return {name: "\n".join(lines).strip() for name, lines in sections.items() if "".join(lines).strip()}


def _looks_like_heading(line: str) -> bool:
    """Numbered or title-case short lines: '3. Results', 'RESULTS'."""
    return bool(re.match(r"^\d+\.?\s+[A-Z]", line) or (line.isupper() and len(line.split()) <= 5))


def split(paper_text: str | None = None, *, jats: str | None = None, tei: str | None = None) -> dict[str, str]:
    """Front door. Hands off to the best available parser."""
    if jats:
        return from_jats(jats)
    if tei:
        return from_grobid_tei(tei)
    if paper_text:
        return from_flat_text(paper_text)
    return {}
