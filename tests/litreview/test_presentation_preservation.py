"""Presentation changes must not lose information. The report was reorganised
(reading view first, audit trail in an Appendix, navigation, evidence trails);
these check that every piece of the stored report still reaches the reader in
both the Markdown and the HTML: kept sentences, section notes (evidence basis,
publication notices), every line code generated (synthesis, key results,
limitations, method), every evidence quote, every reference and its link.

Run over the end-to-end pipeline report and a dense one: long titles, many
citations, a failed integrity check, a publication notice, a removed sentence,
missing metadata. Scientific changes made on purpose are not presentation and
are tested in test_claim_reliability.py.
"""

from __future__ import annotations

import html
import re

import pytest

from bedrock.litreview.models import PaperSignals, Reference, Report, ReportSection, Sentence
from bedrock.litreview.stages.s8_render import to_html, to_markdown
from tests.litreview.test_guard_cache_config import unit
from tests.litreview.test_synthesis import _report


def _flat(text: str) -> str:
    return " ".join(text.split())


def _html_text(page: str) -> str:
    page = re.sub(r"</?(?:strong|em|b|i|a|sub|sup|code|span)\b[^>]*>", "", page)  # inline: no word break
    return _flat(html.unescape(re.sub(r"<[^>]+>", " ", page)))


def _md_inline(md: str) -> str:
    """A generated line as the HTML shows it: emphasis and link brackets gone."""
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    md = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"\1", md)
    return _flat(re.sub(r"<(https?://[^>\s]+)>", r"\1", md))


def _pieces(report: Report):
    def walk(section):
        for s in section.sentences:
            if s.kept:
                yield "sentence", s.text.rstrip(".!? ")
        for sub in section.subsections:
            yield from walk(sub)

    for section in report.sections:
        yield from walk(section)
        # A written section with nothing kept says only "No evidence for this
        # section in the papers read": left out of the reading view on purpose
        # (the Limitations section and the Appendix's evidence plan name it).
        if section.note and not section.note.startswith("No evidence for this section"):
            yield "note", section.note
        for line in (section.body_markdown or "").splitlines():
            if line.startswith("- ") and "|" not in line:
                yield "generated", line[2:]
    for u in report.evidence:
        yield "quote", u.quote
    for r in report.references:
        yield "reference", r.citation
        if r.url:
            yield "link", r.url


def _dense() -> Report:
    long_title = "A very long title " * 12
    units = [unit(id=f"u{i}", paper_id=f"p{i}", quote=f"Liver fat content fell by {i}·5% (95% CI −{i}·9 to −{i}·1) "
                  f"versus placebo at week {12 * i} in {100 + i} adults; p<0·001.") for i in range(1, 13)]
    shorts = {u.id: f"E{i}" for i, u in enumerate(units, 1)}
    sentences = [Sentence(text=f"Liver fat content fell by {i}.5% versus placebo at week {12 * i}.",
                          citations=[f"E{i}", f"E{i % 12 + 1}"]) for i in range(1, 13)]
    sentences.append(Sentence(text="Tirzepatide cured everything.", citations=["E1"], kept=False,
                              drop_reason="not in the cited evidence"))
    return Report(
        question_id="q", question="Does tirzepatide reduce liver fat in adults with obesity?",
        sections=[
            ReportSection(id="answer", title="Answer", kind="written", sentences=sentences[:3],
                          note="Evidence basis: [1] randomised controlled trial. Cites [2], which has a publication "
                               "notice: Expression of concern (PMID:1) <https://pubmed.ncbi.nlm.nih.gov/1/>"),
            ReportSection(id="synthesis", title="How the studies line up", kind="generated",
                          body_markdown="- **Liver fat, measured** — improved in 2 studies [1][2]; no significant "
                                        "effect in 1 study [3]. Basis: **Studies disagree** (independent studies of "
                                        "the same kind report different directions)."),
            ReportSection(id="negative", title="Negative and conflicting results", kind="written",
                          sentences=sentences[3:]),
            ReportSection(id="gaps", title="Limitations, gaps and open questions", kind="generated",
                          body_markdown="- E3 [3]: interval printed as 0.58 to 0.42: the lower bound is above the "
                                        "upper.\n- 2 paper(s) read are preprints, not yet peer reviewed."),
        ],
        references=[Reference(number=i, paper_id=f"p{i}", citation=f"Author{i} A. {long_title}{i}. *J* 2025",
                              url=None if i == 5 else f"https://europepmc.org/article/MED/{i}",
                              signals=PaperSignals(is_preprint=i == 4), evidence_ids=[f"E{i}"])
                    for i in range(1, 13)],
        evidence=units, short_ids=shorts, integrity=["E7: quote not found in paper p7"])


@pytest.fixture(params=["pipeline", "dense"])
def report(request, tmp_path):
    return _report(tmp_path) if request.param == "pipeline" else _dense()


def test_every_piece_of_the_report_reaches_the_markdown_and_the_html(report):
    md, page = _flat(to_markdown(report)), _html_text(to_html(report))
    missing = []
    for kind, text in _pieces(report):
        if _flat(text) not in md and _md_inline(text) not in md:
            missing.append(("md", kind, text[:80]))
        if _md_inline(text) not in page and _flat(text) not in page:
            missing.append(("html", kind, text[:80]))
    assert not missing, missing


def test_removed_sentences_stay_out_of_the_reading_view_and_integrity_failures_are_loud():
    report = _dense()
    md = to_markdown(report)
    assert "Tirzepatide cured everything" not in md.split("\n# Appendix\n")[0]
    assert "failed its integrity check" in md and "failed its integrity check" in to_html(report)
