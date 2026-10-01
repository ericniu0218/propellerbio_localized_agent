"""The target-proposal deck as data, shared by the code that decides what a
slide says (`report/slides.py`) and the code that draws it
(`report/deck_render.py`). No python-pptx here, so the deck's content can be
built and tested without it.

The deck follows Propeller's target-proposal template (single target or
bispecific) and is drawn in ArcDeck's slide format. Every statement on a slide
is a sentence the report kept under the citation guard, a line the report's
own code generated, a count code made from verified findings, or a paper's own
words (a quote or a figure caption). The model is never asked anything here.

Text is measured with a conservative width model (an average character half
the font size wide), so the renderer can split a slide before its text could
overflow: text that does not fit continues on a "(cont.)" slide, never shrunk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

DECK_VERSION = "deck.v1"
FIGURES_ARTIFACT = "deck_figures.v1"
"""Run artifact (`Database.put_artifact`) listing the figures chosen, and
where downloaded, for a run's deck: a JSON list of `FigureAsset.to_dict()`."""

# Widescreen 13.33 x 7.5 in, ArcDeck's canvas.
WIDTH_IN, HEIGHT_IN = 13.333, 7.5
BODY_WIDTH_IN = 12.1
_LINE = 1.2
_SPACE_AFTER = {0: 10, 1: 6}


def lines_needed(text: str, size: int, width_in: float = BODY_WIDTH_IN) -> int:
    per_line = max(1, int((width_in - 0.2) * 72 / (size * 0.5)))
    lines = 0
    for part in text.split("\n"):
        count, line = 1, 0
        for word in part.split(" "):
            # A word (a URL) wider than a line breaks across lines. An empty
            # word - a double space, or a sub-bullet's indent - adds nothing.
            count += max(len(word) - 1, 0) // per_line
            if line and line + 1 + len(word) > per_line:
                count, line = count + 1, len(word) % per_line
            else:
                line += (1 if line else 0) + len(word)
        lines += count
    return lines


def text_height_pt(text: str, size: int, level: int = 0, width_in: float = BODY_WIDTH_IN) -> float:
    return lines_needed(("    " * level) + text, size, width_in - 0.4 * level) * size * _LINE + _SPACE_AFTER[min(level, 1)]


Layout = Literal["title", "content", "table", "matrix", "references"]
VisualType = Literal["stats_callout", "bar_chart", "doughnut_chart", "comparison", "stacked_list", "callout_cards"]
Tone = Literal["positive", "negative", "neutral"]


@dataclass
class Para:
    text: str
    level: int = 0
    bold: bool = False
    link: str | None = None
    muted: bool = False
    refs: list[int] = field(default_factory=list)
    """Reference numbers this paragraph cites, so a page's footer lists only
    the references on that page."""


@dataclass
class VisualItem:
    label: str
    sublabel: str = ""
    value: float | None = None
    """Charts and stats boxes: the number, counted by code."""
    tone: Tone | None = None


@dataclass
class Visual:
    """An ArcDeck visual element. Every label and number is made by code from
    the report: a count, a canonical name, or a line the report generated."""

    type: VisualType
    items: list[VisualItem]
    caption: str = ""
    """What the visual counts, e.g. "Verified findings in people, by outcome"."""


@dataclass
class FigureAsset:
    """One figure from a cited open-access paper, chosen by code (the figure
    the paper itself cites in the paragraph a verified quote came from)."""

    paper_id: str
    pmcid: str
    fig_id: str
    label: str
    """The paper's own label, "Figure 2"."""
    caption: str
    """The paper's own caption, as plain text."""
    href: str
    """The <graphic xlink:href> in the paper's JATS."""
    licence: str
    """Short name ("CC BY 4.0") or the licence statement's first sentence."""
    licence_url: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    """Short ids ("E3") of the findings whose quote paragraph cites this figure."""
    path: str | None = None
    """Local image file once downloaded; None means not downloaded, not shown."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FigureAsset":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class RefMark:
    number: int
    short: str
    """"Smith 2020" - from the paper record, never model-written."""
    url: str | None = None


@dataclass
class Slide:
    title: str
    layout: Layout = "content"
    paras: list[Para] = field(default_factory=list)
    visual: Visual | None = None
    figure: FigureAsset | None = None
    figure_ref: int | None = None
    """Reference number of the paper the figure comes from."""
    table: list[list[str]] | None = None
    """Header row first. Cells may end with "[n]" reference markers."""
    kicker: str = ""
    """One line under the title: what kind of evidence the slide holds."""
    subtitle: str = ""
    """Title slide only: "Author | date" and the question."""
    refs: list[RefMark] = field(default_factory=list)
    """Every reference the slide cites; the renderer prints those on each page."""
    notes: list[str] = field(default_factory=list)
    warning: str = ""
    by_rule: bool = False
    """Assembled by a written rule from the report, not by the report's own
    sentences: the renderer marks it "Assembled by rule from the report -
    review before presenting"."""
    size: int = 16
