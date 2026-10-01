"""Which paper figures a deck may show, decided by code from the paper's own JATS.

A scientist presenting a target proposal wants the figure behind a result, not
only the sentence. But a figure is the one thing in the deck nobody can verify
by string-matching, so this module chooses with a rule short enough to check by
eye, and with no model and no keyword guessing:

  A finding's figure is the first figure the paper itself cites (an
  `<xref ref-type="fig">`) in the paragraph its verified quote came from. If
  that paragraph cites none, the paragraph of the finding's context sentence.
  Nothing else.

So a figure is shown only because the authors pointed at it from the very
sentence the report quotes. A figure is also shown only when the paper carries
an open-access licence element: no licence recorded, no figure, and the licence
and source are always printed under it (`FigureAsset.licence`).

Finding the paragraph is the subtle part. Quotes are verified against the text
`parsing/sections.py` builds, and that text has every `<xref>` and `<fig>`
stripped out (a reference marker is not part of a sentence). The stored quote
therefore never contains "(Figure 2)". To find its paragraph in the raw JATS we
strip each `<p>` exactly the way `from_jats` does, with the same helpers, and
compare whitespace-normalised text. The xrefs are collected before stripping.

This file only reads text it is given: no network, no model. The text is the
paper's JATS in `Paper.full_text`, which ingestion throws away once it has split
the article into sections - so a stored run's papers carry none, and the caller
(`sources/figures.py`, `ArticleTexts`) re-reads it from Europe PMC and fills it
in on a copy. The article may have changed upstream since the review read it;
then the verified quote is simply not found in it and no figure is chosen.
Downloading the image files is `sources/figures.py` too; placing them on slides
is `report/slides.py`.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from bedrock.litreview.models import EvidenceUnit, Paper, Report
from bedrock.litreview.parsing.sections import _parse_xml, _text
from bedrock.litreview.report.deck_model import FigureAsset

MAX_PAPERS = 8
MAX_PER_PAPER = 2
MAX_FIGURES = 10
"""A deck is a talk, not a gallery: at most this many figures, from at most
this many papers, so the download (the only network step) stays small."""

# Everything `from_jats` removes before the text is split into sections. Quotes
# were verified against text with these gone, so a paragraph must lose them too
# before a quote is looked for in it. Keep in step with `parsing/sections.py`.
_STRIPPED = (
    "xref", "table-wrap", "table-wrap-group", "fig", "fig-group", "disp-formula",
    "supplementary-material", "graphic", "media", "ref-list",
)


def _words(text: str) -> str:
    return " ".join(text.split())


def _href(el) -> str:
    """The `xlink:href` of an element. `_parse_xml` drops namespaces from tag
    names but not from attributes, and a JATS file with an undeclared `xlink`
    prefix keeps the prefix, so match on the local name."""
    for key, value in el.attrib.items():
        if key.split("}")[-1].split(":")[-1] == "href":
            return (value or "").strip()
    return ""


def _plain(el) -> str:
    """Text of an element exactly as written, with whitespace tidied. Unlike
    `sections._text` this does not put a space between inline pieces, so
    "IL-<italic>23</italic>" stays "IL-23" in a caption."""
    return _words("".join(el.itertext()))


# --- figures -----------------------------------------------------------------


@dataclass
class JatsFigure:
    """One `<fig>` of a paper, in the paper's own words."""

    fig_id: str
    label: str
    """The paper's label, "Figure 2". The id (`F2`) when it has none: never a
    number made up here."""
    caption: str
    hrefs: list[str] = field(default_factory=list)
    """Each `<graphic xlink:href>`, as written: with or without a file
    extension. Europe PMC's file bundle is matched against these later."""


def _caption(fig) -> str:
    """Title then paragraphs, as plain text. A title with no closing
    punctuation gets a full stop before the paragraphs that follow, so the two
    do not run together; no word is added or changed."""
    caption = fig.find("caption")
    if caption is None:
        return ""
    parts: list[str] = []
    for child in caption:
        text = _plain(child)
        if not text:
            continue
        if parts and parts[-1][-1] not in ".?!:;":
            parts[-1] += "."
        parts.append(text)
    if not parts:  # caption text with no title/p wrapper
        return _plain(caption)
    return " ".join(parts)


class _Jats:
    """One paper's JATS, parsed once. Building it costs a parse; asking it
    about each finding's quote afterwards is cheap."""

    def __init__(self, jats: str) -> None:
        self.root = _parse_xml(jats or "")
        self._figures: dict[str, JatsFigure] | None = None
        self._paragraphs: list[tuple[object, str, list[str]]] | None = None

    # figures

    def figures(self) -> dict[str, JatsFigure]:
        """fig id -> figure, in the order the paper prints them."""
        if self._figures is None:
            found: dict[str, JatsFigure] = {}
            if self.root is not None:
                for fig in self.root.iter("fig"):
                    fig_id = (fig.get("id") or "").strip()
                    if not fig_id or fig_id in found:
                        continue
                    label_el = fig.find("label")
                    label = _plain(label_el).rstrip(".:").strip() if label_el is not None else ""
                    hrefs = [h for h in (_href(g) for g in fig.iter("graphic")) if h]
                    found[fig_id] = JatsFigure(fig_id, label or fig_id, _caption(fig), list(dict.fromkeys(hrefs)))
            self._figures = found
        return self._figures

    # the paragraphs a quote can come from

    def _paragraphs_with_figures(self) -> list[tuple[object, str, list[str]]]:
        """Every `<p>` that cites a figure: (element, stripped text, cited fig ids).

        Only these can give a quote a figure, so only these are stripped and
        compared. A figure id that is not a `<fig>` of the paper (a supplement,
        a table) is ignored."""
        if self._paragraphs is None:
            from lxml import etree

            known = self.figures()
            out: list[tuple[object, str, list[str]]] = []
            if self.root is not None:
                for p in self.root.iter("p"):
                    cited: list[str] = []
                    for xref in p.iter("xref"):
                        if (xref.get("ref-type") or "").lower() != "fig":
                            continue
                        # rid is a space-separated list of ids ("F2 F3").
                        cited += [r for r in (xref.get("rid") or "").split() if r in known and r not in cited]
                    if not cited:
                        continue
                    clone = copy.deepcopy(p)
                    etree.strip_elements(clone, *_STRIPPED, with_tail=False)
                    out.append((p, _words(_text(clone)), cited))
            self._paragraphs = out
        return self._paragraphs

    def figure_cited_with(self, quote: str) -> str | None:
        """The fig id cited in the paragraph holding `quote`, else None.

        The first cited figure that has an image to show wins. When the same
        sentence turns up in two paragraphs (an abstract repeating a results
        sentence) the first paragraph that cites a figure is used, and when a
        paragraph sits inside another (a list item) only the inner one counts,
        so a sentence is never given a figure cited by its neighbour."""
        needle = _words(quote or "")
        if not needle:
            return None
        known = self.figures()
        hits = [(p, cited) for p, text, cited in self._paragraphs_with_figures() if needle in text]
        for p, cited in hits:
            if any(other is not p and p in other.iterancestors() for other, _ in hits):
                continue
            for fig_id in cited:
                if known[fig_id].hrefs:
                    return fig_id
        return None

    # licence

    def licence(self) -> "PaperLicence | None":
        if self.root is None:
            return None
        meta = next(self.root.iter("article-meta"), None)
        if meta is None:
            return None
        el = next(meta.iter("license"), None)
        if el is None:
            return None
        url = _href(el)
        for ref in el.iter("license_ref"):  # NISO <ali:license_ref>, the newer way to give the URL
            url = url or _plain(ref)
        text = _plain(el)
        name = _licence_name(url, text)
        if not name:
            return None
        return PaperLicence(name=name, url=url or None)


# --- licence -----------------------------------------------------------------


@dataclass
class PaperLicence:
    name: str
    """"CC BY 4.0", "CC BY-NC 4.0", "CC0", or the licence text's first sentence."""
    url: str | None = None


_CC_URL = re.compile(r"creativecommons\.org/licenses/([a-z]+(?:-[a-z]+)*)/(\d(?:\.\d)?)(?:/([a-z]{2,3}))?(?![a-z0-9])", re.I)
_CC_WORDS = re.compile(
    r"Creative\s+Commons\s+(Attribution(?:[-\s]*Non[-\s]?Commercial)?(?:[-\s]*No[-\s]?Deriv(?:ative)?s?)?"
    r"(?:[-\s]*Share[-\s]?Alike)?)(?:\s+(\d(?:\.\d)?))?",
    re.I,
)
_CC_SHORT = re.compile(r"\bCC[\s-]BY(?:[\s-](?:NC|ND|SA))*(?:\s+\d(?:\.\d)?)?\b", re.I)


def _licence_name(url: str, text: str) -> str:
    """Short name from the licence URL, then from its words, else the text's
    first sentence, else the URL. "" when the element holds nothing at all."""
    for source in (url, text):
        if re.search(r"creativecommons\.org/publicdomain/zero", source, re.I):
            return "CC0"
        match = _CC_URL.search(source)
        if match:
            kind, version, jurisdiction = match.groups()
            return " ".join(part for part in ("CC", kind.upper(), version, (jurisdiction or "").upper()) if part)
    if re.search(r"\bCC0\b|Creative\s+Commons\s+(?:Zero|CC0)", text, re.I):
        return "CC0"
    short = _CC_SHORT.search(text)
    if short:
        pieces = re.findall(r"[A-Za-z0-9.]+", short.group(0).upper())  # CC, BY, NC, 4.0
        version = pieces[-1] if pieces[-1][0].isdigit() else ""
        return " ".join(part for part in ("CC", "-".join(x for x in pieces[1:] if x != version), version) if part)
    words = _CC_WORDS.search(text)
    if words:
        return _words_to_cc(*words.groups())
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].strip() if text.strip() else ""
    return first or url


def _words_to_cc(terms: str, version: str | None) -> str:
    """"Attribution-NonCommercial", "4.0" -> "CC BY-NC 4.0"."""
    t = re.sub(r"[-\s]", "", terms.lower())
    code = "BY" + ("-NC" if "noncommercial" in t else "") + ("-ND" if "noderiv" in t else "") + ("-SA" if "sharealike" in t else "")
    return f"CC {code}" + (f" {version}" if version else "")


# --- public functions --------------------------------------------------------


def paper_figures(jats: str) -> list[JatsFigure]:
    """Every figure of a paper's raw JATS: id, the paper's own label, caption
    as plain text and the image hrefs. Empty for text that is not JATS."""
    return list(_Jats(jats).figures().values())


def paper_licence(jats: str) -> PaperLicence | None:
    """The paper's licence, from the `<license>` in its front matter: a short
    name and the URL. None when the JATS has no `<license>`, which is how a
    paper stays out of the deck: a figure is only used under a stated licence."""
    return _Jats(jats).licence()


def figure_cited_with(jats: str, quote: str) -> str | None:
    """The id of the figure the paper cites in the paragraph that contains
    `quote`, or None. See the module docstring for how the paragraph is found."""
    return _Jats(jats).figure_cited_with(quote)


def _unit_figure(parsed: _Jats, unit: EvidenceUnit) -> str | None:
    return parsed.figure_cited_with(unit.quote) or (
        parsed.figure_cited_with(unit.context_quote) if unit.context_quote else None
    )


def choose_figures(report: Report, papers: Mapping[str, Paper]) -> list[FigureAsset]:
    """The figures a deck for this report may show, chosen by the rule above.

    Walks the report's kept sentences in order and, for each finding they cite,
    takes the figure its quote's paragraph cites (else its context sentence's).
    A paper counts only when `papers` gives it with JATS full text (a caller
    that re-reads articles on demand can hand over a mapping whose `get` does
    that; this function asks for a paper only when it could still add a
    figure, so such a mapping reads as few articles as the caps allow), a PMC
    id (the file bundle is fetched by it) and a licence. A figure appears once; `evidence_ids`
    lists every finding ("E3") that led to it. At most 8 papers, 2 figures per
    paper and 10 figures overall: the first ones found in report order, so the
    same report and papers always give the same list. `path` is None: nothing
    is downloaded here.
    """
    unit_by_short = {short: unit for unit in report.evidence if (short := report.short_ids.get(unit.id))}
    parsed: dict[str, tuple[_Jats, PaperLicence] | None] = {}
    chosen: dict[tuple[str, str], FigureAsset] = {}
    per_paper: dict[str, int] = {}

    def usable(paper_id: str) -> tuple[_Jats, PaperLicence] | None:
        if paper_id not in parsed:
            paper = papers.get(paper_id)
            result = None
            if paper and paper.pmcid and paper.full_text and paper.full_text_format == "jats":
                doc = _Jats(paper.full_text)
                licence = doc.licence()
                if licence is not None:
                    result = (doc, licence)
            parsed[paper_id] = result
        return parsed[paper_id]

    for _section_id, sentence in report.kept_sentences():
        for short in sentence.citations:
            unit = unit_by_short.get(short)
            if unit is None:
                continue
            if unit.paper_id not in parsed and (len(chosen) >= MAX_FIGURES or len(per_paper) >= MAX_PAPERS):
                continue  # the caps are full: a paper not seen yet cannot add a figure, so do not even ask for it
            doc = usable(unit.paper_id)
            if doc is None:
                continue
            jats, licence = doc
            fig_id = _unit_figure(jats, unit)
            if fig_id is None:
                continue
            key = (unit.paper_id, fig_id)
            if key not in chosen:
                if (
                    len(chosen) >= MAX_FIGURES
                    or per_paper.get(unit.paper_id, 0) >= MAX_PER_PAPER
                    or (unit.paper_id not in per_paper and len(per_paper) >= MAX_PAPERS)
                ):
                    continue
                figure = jats.figures()[fig_id]
                chosen[key] = FigureAsset(
                    paper_id=unit.paper_id,
                    pmcid=papers[unit.paper_id].pmcid or "",
                    fig_id=fig_id,
                    label=figure.label,
                    caption=figure.caption,
                    href=figure.hrefs[0],
                    licence=licence.name,
                    licence_url=licence.url,
                )
                per_paper[unit.paper_id] = per_paper.get(unit.paper_id, 0) + 1
            if short not in chosen[key].evidence_ids:
                chosen[key].evidence_ids.append(short)
    return list(chosen.values())
