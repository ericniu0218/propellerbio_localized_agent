"""The PowerPoint deck (report/slides.py): a condensed renderer of the stored
report, never a second piece of research. What these check is what makes it
safe to hand a deck to someone who never opens the report: every statement is
one the report kept, citations and quotes travel with it, nothing is cut off,
and the report is left exactly as it was."""

from __future__ import annotations

import io
import re

import pytest

pptx = pytest.importorskip("pptx")

from bedrock.litreview.models import Reference, Report, ReportSection, Sentence  # noqa: E402
from bedrock.litreview.report import slides  # noqa: E402
from tests.litreview.test_guard_cache_config import unit  # noqa: E402
from tests.litreview.test_synthesis import _report  # noqa: E402


def _deck_text(data: bytes) -> list[list[str]]:
    deck = pptx.Presentation(io.BytesIO(data))
    out = []
    for slide in deck.slides:
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                texts += [p.text for p in shape.text_frame.paragraphs if p.text]
            if shape.has_table:
                texts += [c.text for row in shape.table.rows for c in row.cells]
        out.append(texts)
    return out


def _dense(n_sentences=24, n_refs=30, integrity=()):
    """A report with far more than a slide holds: long sentences, many
    references and long quotes, to exercise splitting."""
    units, short = [], {}
    for i in range(1, n_refs + 1):
        u = unit(id=f"u{i}", paper_id=f"p{i}", quote=f"In study {i}, tirzepatide reduced liver fat content by {i}.5% versus "
                 "placebo at week 52 (p < 0.001), " + "with a consistent effect across prespecified analyses " * 3
                 + "and no new safety signal.", target="liver fat", intervention="tirzepatide")
        units.append(u)
        short[u.id] = f"E{i}"
    long = ("Tirzepatide reduced liver fat content measured by MRI-PDFF compared with placebo at week 52 in adults "
            "with obesity and type 2 diabetes, with a between-group difference that remained after adjustment")
    written = [Sentence(text=f"{long} (sentence {k}).", citations=[f"E{k % n_refs + 1}"]) for k in range(n_sentences)]
    sections = [
        ReportSection(id="answer", title="Answer", kind="written", sentences=written[:4],
                      note="Evidence basis: [1] randomised controlled trial, patients (n=190), full text read."),
        ReportSection(id="glance", title="Evidence at a glance", kind="generated",
                      body_markdown="This report rests on **30 publications**.\n\n| Ref | x |\n|---|---|\n| [1] | y |"),
        ReportSection(id="synthesis", title="How the studies line up", kind="generated", body_markdown="\n".join(
            f"- **Measure {k}** — 3 independent studies (3 randomised controlled trials): improved in 3 studies "
            f"[1][2][3]. Basis: **Randomised comparisons: 3 trials** (no other human study)." for k in range(12))
            + "\n- 4 further measures are in the evidence table (Appendix)."),
        ReportSection(id="clinical", title="Clinical efficacy", kind="written", sentences=written[4:],
                      note="Evidence basis: [2] observational study."),
        ReportSection(id="safety", title="Safety", kind="written", sentences=[],
                      note="No evidence for this section in the papers read."),
        ReportSection(id="gaps", title="Limitations, gaps and open questions", kind="generated",
                      body_markdown="\n".join(f"- Limitation {k}: " + "a long explanation " * 12 for k in range(10))),
    ]
    refs = [Reference(number=i, paper_id=f"p{i}", citation=f"Author{i} A et al. A long title about liver fat {i}. "
                      f"*Journal* 2025 (PMID {1000 + i})", url=f"https://europepmc.org/article/MED/{1000 + i}",
                      evidence_ids=[f"E{i}"]) for i in range(1, n_refs + 1)]
    return Report(question_id="q", question="Does tirzepatide reduce liver fat in adults with obesity?",
                  interpretation={"question_type": "intervention_efficacy", "goal": "treat",
                                  "subject": "hepatic steatosis", "population": "adults with obesity"},
                  sections=sections, references=refs, evidence=units, short_ids=short,
                  stats={"papers_collected": 177, "gate_pass": 77, "papers_read": 16, "papers_abstract_only": 9},
                  integrity=list(integrity))


def test_every_kept_sentence_is_on_a_slide_with_its_references(tmp_path):
    report = _report(tmp_path)
    text = "\n".join(t for s in _deck_text(slides.to_pptx(report)) for t in s)
    refs, _ = slides._ref_map(report)
    for section in report.sections:
        if section.kind != "written":
            continue
        for s in section.sentences:
            marker = "".join(f"[{n}]" for n in slides._numbers(s, refs))
            assert (f"{s.text.rstrip()} {marker}".rstrip() in text) is s.kept, s.text


def test_the_deck_says_nothing_the_report_does_not():
    """Each paragraph on a findings slide is a kept sentence, the report's own
    note for that section, or one of the deck's two fixed statements."""
    report = _dense()
    refs, _ = slides._ref_map(report)
    allowed = {slides.ABSENCE, "No verified safety finding in the papers read."}
    for section in report.sections:
        allowed |= {f"{s.text} " + "".join(f"[{n}]" for n in slides._numbers(s, refs)) for s in section.sentences}
        allowed |= {slides.plain(section.note)} if section.note else set()
        allowed |= set(slides._bullets(section.body_markdown))
    for slide in slides.outline(report):
        if slide.title.startswith(("Answer", "Clinical efficacy", "Safety", "Limitations")):
            for p in slide.paras:
                assert p.text in allowed or p.text == "Key limitations", p.text


def test_the_title_slide_says_the_deck_is_condensed_and_safety_says_absence_is_not_safety():
    outline = slides.outline(_dense())
    assert outline[0].paras[0].text == slides.CONDENSED
    safety = next(s for s in outline if s.title == "Safety")
    assert [p.text for p in safety.paras] == ["No verified safety finding in the papers read.", slides.ABSENCE]


def test_long_content_continues_on_further_slides_and_every_slide_fits():
    report = _dense()
    outline = slides.outline(report)
    titles = [s.title for s in outline]
    for name in ("Clinical efficacy", "How the studies line up", "References", "Appendix: evidence quotes"):
        assert f"{name} (cont.)" in titles, name
    for s in outline:
        if s.table:
            widths = slides._COLUMN_WIDTHS[3]
            used = sum(slides.row_height_pt(r, widths) for r in s.table)
            used += sum(slides.text_height_pt(p.text, 12) for p in s.paras)
        else:
            used = sum(slides.text_height_pt(p.text, s.size, p.level) for p in s.paras)
        assert used <= slides.body_pt(bool(s.kicker), bool(s.warning)) or len(s.paras) == 1, s.title
    # nothing lost in the split: every reference and every table row is somewhere
    text = "\n".join(p.text for s in outline for p in s.paras)
    assert all(f"[{r.number}] " in text for r in report.references)
    assert sum(len(s.table) - 1 for s in outline if s.table) == 12


def test_references_are_hyperlinked_and_every_cited_quote_is_in_the_appendix():
    report = _dense()
    deck = pptx.Presentation(io.BytesIO(slides.to_pptx(report)))
    links = {r.hyperlink.address for slide in deck.slides for shape in slide.shapes if shape.has_text_frame
             for p in shape.text_frame.paragraphs for r in p.runs if r.hyperlink.address}
    assert {r.url for r in report.references} <= links
    appendix = "\n".join(p.text for s in slides.outline(report) if s.title.startswith("Appendix") for p in s.paras)
    by_short = {v: k for k, v in report.short_ids.items()}
    quotes = {u.id: u.quote for u in report.evidence}
    cited = {c for sec in report.sections for x in sec.sentences if x.kept for c in x.citations}
    assert cited
    for short in cited:
        assert f"{short} [" in appendix and " ".join(quotes[by_short[short]].split()) in appendix, short


def test_the_deck_is_deterministic_leaves_the_report_unchanged_and_names_no_local_paths(tmp_path):
    report = _report(tmp_path)
    before = report.model_dump_json()
    a, b = slides.to_pptx(report, "review-x"), slides.to_pptx(report, "review-x")
    assert report.model_dump_json() == before
    assert _deck_text(a) == _deck_text(b)
    import zipfile
    xml = "".join(zipfile.ZipFile(io.BytesIO(a)).read(n).decode("utf-8", "replace")
                  for n in zipfile.ZipFile(io.BytesIO(a)).namelist() if n.endswith(".xml"))
    assert "/Users/" not in xml and "/home/" not in xml and str(tmp_path) not in xml


def test_a_report_that_failed_its_integrity_check_says_so_on_the_deck():
    outline = slides.outline(_dense(integrity=["E3: quote not found in paper p3"]))
    assert outline[0].warning and next(s for s in outline if s.title == "Answer").warning
    assert not slides.outline(_dense())[0].warning


def test_preclinical_slides_are_labelled_not_human(tmp_path):
    outline = slides.outline(_report(tmp_path))
    for s in outline:
        if s.title.startswith(("Preclinical", "Mechanism")):
            assert s.kicker == slides._PRECLINICAL_NOTE


def test_the_answer_slide_leads_with_the_limitations_that_change_its_reading():
    gaps = ["5 paper(s) were read for negative and conflicting results, but no verified finding was extracted.",
            "9 of 16 papers read were available as abstract only.",
            "Figures and tables are not read.",
            "E38 [3]: interval printed as 0.58 to 0.42: the lower bound is above the upper.",
            "The question is about adults with obesity. 7 of the 13 independent studies name obesity."]
    assert slides.key_limitations(gaps) == [gaps[4], gaps[3], gaps[1]]
    report = _dense()
    report.sections[-1].body_markdown = "\n".join(f"- {g}" for g in gaps)
    answer = next(s for s in slides.outline(report) if s.title == "Answer")
    assert gaps[4] in [p.text for p in answer.paras]


def test_missing_human_effects_are_key_limitations_on_the_answer_and_limitations_slides():
    report = _dense()
    bullet = 'No tested effect in humans among the papers read for: Liver fat [1]; Insulin resistance [2].'
    report.sections[-1].body_markdown = f'- {bullet}\n- Figures and tables are not read.'
    assert slides.key_limitations([bullet]) == [bullet]
    outline = slides.outline(report)
    for title in ('Answer', 'Limitations and open questions'):
        assert bullet in [p.text for s in outline if s.title == title for p in s.paras]
