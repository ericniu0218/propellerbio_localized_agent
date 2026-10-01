"""Which paper figures the target-proposal deck may show (report/figures.py).

The rule these pin down: a finding's figure is the one the paper itself cites in
the paragraph its verified quote came from (else its context sentence's), and
only for a paper with JATS full text, a PMC id and a licence. Quotes were
verified against text with every <xref> stripped out, so most of these tests
are about finding the right paragraph in the raw JATS anyway.
"""

from __future__ import annotations

from bedrock.litreview.models import Paper, Report, ReportSection, Sentence
from bedrock.litreview.report import figures
from tests.litreview.test_guard_cache_config import unit

CC_BY = "https://creativecommons.org/licenses/by/4.0/"


def jats(body: str, *, license_xml: str | None = None, figs: str | None = None) -> str:
    """A small JATS article. `license_xml` is a whole <license> element."""
    permissions = f"<permissions><copyright-statement>(c) 2024</copyright-statement>{license_xml}</permissions>" if license_xml else ""
    figs = figs if figs is not None else (
        '<fig id="F1"><label>Figure 1</label><caption><title>Weight loss</title>'
        '<p>Mice were weighed daily.</p></caption><graphic xlink:href="nihms-1-f1"/></fig>'
        '<fig id="F2"><label>Figure 2.</label><caption><title>Food <italic>intake</italic></title></caption>'
        '<graphic xlink:href="pone.0001-g002.jpg"/><graphic xlink:href="pone.0001-g002.gif"/></fig>'
    )
    return (
        '<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:ali="http://www.niso.org/schemas/ali/1.0/">'
        f"<front><article-meta>{permissions}</article-meta></front>"
        f"<body><sec><title>Results</title>{body}{figs}</sec></body></article>"
    )


def lic(url: str = CC_BY, text: str = "This is an open access article.") -> str:
    return f'<license xlink:href="{url}"><license-p>{text}</license-p></license>'


PARA_F1 = (
    '<p>Mice given the antibody lost weight (<xref ref-type="fig" rid="F1">Figure 1</xref>A). '
    'Food intake fell by 20%<xref ref-type="fig" rid="F2">Figure 2</xref>.</p>'
)
PARA_NONE = "<p>The antibody was well tolerated in every group.</p>"
QUOTE_F1 = "Mice given the antibody lost weight"
QUOTE_NONE = "The antibody was well tolerated in every group."


# --- figures -----------------------------------------------------------------


def test_figures_have_the_papers_own_label_caption_and_hrefs_as_written():
    found = {f.fig_id: f for f in figures.paper_figures(jats(PARA_F1))}
    assert list(found) == ["F1", "F2"]
    assert found["F1"].label == "Figure 1"
    assert found["F1"].caption == "Weight loss. Mice were weighed daily."
    assert found["F1"].hrefs == ["nihms-1-f1"]  # no extension: kept as written
    assert found["F2"].label == "Figure 2"  # trailing full stop tidied
    assert found["F2"].caption == "Food intake"  # inline markup does not split a word
    assert found["F2"].hrefs == ["pone.0001-g002.jpg", "pone.0001-g002.gif"]


def test_a_figure_without_a_label_shows_its_own_id_and_text_that_is_not_jats_has_none():
    xml = jats(PARA_F1, figs='<fig id="F9"><graphic xlink:href="g9"/></fig>')
    [fig] = figures.paper_figures(xml)
    assert (fig.label, fig.caption) == ("F9", "")
    assert figures.paper_figures("not xml at all") == []
    assert figures.paper_figures("") == []


# --- licence -----------------------------------------------------------------


def test_licence_short_name_from_the_url():
    got = figures.paper_licence(jats(PARA_F1, license_xml=lic()))
    assert (got.name, got.url) == ("CC BY 4.0", CC_BY)


def test_non_commercial_licence_is_named_as_such():
    url = "http://creativecommons.org/licenses/by-nc/3.0/"
    got = figures.paper_licence(jats(PARA_F1, license_xml=lic(url)))
    assert got.name == "CC BY-NC 3.0"
    nd = figures.paper_licence(jats(PARA_F1, license_xml=lic("https://creativecommons.org/licenses/by-nc-nd/4.0/")))
    assert nd.name == "CC BY-NC-ND 4.0"


def test_cc0_from_the_public_domain_url():
    got = figures.paper_licence(jats(PARA_F1, license_xml=lic("https://creativecommons.org/publicdomain/zero/1.0/")))
    assert got.name == "CC0"


def test_licence_from_text_alone_and_from_a_niso_licence_ref():
    text = "This article is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License."
    got = figures.paper_licence(jats(PARA_F1, license_xml=f"<license><license-p>{text}</license-p></license>"))
    assert (got.name, got.url) == ("CC BY-NC 4.0", None)
    ref = f"<license><ali:license_ref>{CC_BY}</ali:license_ref><license-p>Open access.</license-p></license>"
    got = figures.paper_licence(jats(PARA_F1, license_xml=ref))
    assert (got.name, got.url) == ("CC BY 4.0", CC_BY)


def test_an_unrecognised_licence_is_its_first_sentence():
    text = "Reuse is allowed with credit to the society. Contact us for more."
    got = figures.paper_licence(jats(PARA_F1, license_xml=f"<license><license-p>{text}</license-p></license>"))
    assert got.name == "Reuse is allowed with credit to the society."


def test_no_licence_element_is_none():
    assert figures.paper_licence(jats(PARA_F1)) is None
    assert figures.paper_licence("<<<") is None


# --- the figure cited with a quote -------------------------------------------


def test_the_figure_cited_in_the_quotes_own_paragraph():
    xml = jats(PARA_NONE + PARA_F1)
    assert figures.figure_cited_with(xml, QUOTE_F1) == "F1"  # the first one cited, not the second
    assert figures.figure_cited_with(xml, QUOTE_NONE) is None  # a neighbour's figure is never borrowed


def test_whitespace_differences_do_not_matter():
    body = (
        '<p>Mice given the\n        antibody  lost weight (<xref ref-type="fig" rid="F2">Figure 2</xref>), '
        "and food intake fell.</p>"
    )
    assert figures.figure_cited_with(jats(body), "Mice given the antibody lost weight") == "F2"
    assert figures.figure_cited_with(jats(body), "Mice given the   antibody\nlost weight") == "F2"


def test_a_cited_id_list_and_ids_that_are_not_figures():
    body = (
        '<p>Mice lost weight (<xref ref-type="supplementary-material" rid="S1">Table S1</xref>, '
        '<xref ref-type="fig" rid="F9 F2">Figures</xref>).</p>'
    )
    assert figures.figure_cited_with(jats(body), "Mice lost weight") == "F2"  # F9 is not a figure of this paper


def test_a_figure_with_no_image_is_passed_over_for_the_next_one_cited():
    figs = '<fig id="F1"><label>Figure 1</label></fig><fig id="F2"><graphic xlink:href="g2"/></fig>'
    body = '<p>Mice lost weight (<xref ref-type="fig" rid="F1">1</xref>, <xref ref-type="fig" rid="F2">2</xref>).</p>'
    assert figures.figure_cited_with(jats(body, figs=figs), "Mice lost weight") == "F2"


# --- choose_figures ----------------------------------------------------------


def make_paper(pid="P1", body=PARA_F1 + PARA_NONE, *, licence=True, pmcid="PMC100", fmt="jats", **kw) -> Paper:
    return Paper(
        id=pid, title=f"Paper {pid}", pmcid=pmcid, full_text_format=fmt,
        full_text=jats(body, license_xml=lic() if licence else None), **kw,
    )


def make_report(cites, sentences_kept=True, units=None) -> Report:
    """`cites` is a list of (unit id, paper id, quote, context quote); each becomes
    its own kept sentence citing "E<n>"."""
    evidence, short, sentences = [], {}, []
    for n, (uid, pid, quote, context) in enumerate(cites, 1):
        evidence.append(unit(id=uid, paper_id=pid, quote=quote, context_quote=context))
        short[uid] = f"E{n}"
        sentences.append(Sentence(text=f"Finding {n}.", citations=[f"E{n}"], kept=sentences_kept))
    sections = [ReportSection(id="s", title="S", kind="written", sentences=sentences)]
    return Report(question_id="q", question="q?", sections=sections, evidence=evidence, short_ids=short)


def test_a_cited_figure_is_chosen_with_its_licence_source_and_findings():
    report = make_report([("u1", "P1", QUOTE_F1, None)])
    [fig] = figures.choose_figures(report, {"P1": make_paper()})
    assert (fig.paper_id, fig.pmcid, fig.fig_id, fig.label) == ("P1", "PMC100", "F1", "Figure 1")
    assert fig.caption == "Weight loss. Mice were weighed daily."
    assert fig.href == "nihms-1-f1"
    assert (fig.licence, fig.licence_url) == ("CC BY 4.0", CC_BY)
    assert fig.evidence_ids == ["E1"]
    assert fig.path is None


def test_a_figure_appears_once_and_collects_every_finding_that_led_to_it():
    report = make_report([("u1", "P1", QUOTE_F1, None), ("u2", "P1", "Food intake fell by 20%", None)])
    [fig] = figures.choose_figures(report, {"P1": make_paper()})
    assert fig.fig_id == "F1" and fig.evidence_ids == ["E1", "E2"]


def test_the_context_sentences_paragraph_is_used_only_when_the_quotes_has_no_figure():
    report = make_report([("u1", "P1", QUOTE_NONE, QUOTE_F1)])
    [fig] = figures.choose_figures(report, {"P1": make_paper()})
    assert fig.fig_id == "F1"
    # A quote that already has a figure never looks at its context sentence.
    other = '<p>Context sentence about dosing (<xref ref-type="fig" rid="F2">2</xref>).</p>'
    paper = make_paper(body=PARA_F1 + other)
    report = make_report([("u1", "P1", QUOTE_F1, "Context sentence about dosing")])
    assert [f.fig_id for f in figures.choose_figures(report, {"P1": paper})] == ["F1"]
    report = make_report([("u1", "P1", "Nothing like this anywhere", "Context sentence about dosing")])
    assert [f.fig_id for f in figures.choose_figures(report, {"P1": paper})] == ["F2"]


def test_a_quote_with_no_figure_anywhere_gives_none():
    report = make_report([("u1", "P1", QUOTE_NONE, None)])
    assert figures.choose_figures(report, {"P1": make_paper()}) == []


def test_a_paper_without_a_licence_is_not_used():
    report = make_report([("u1", "P1", QUOTE_F1, None)])
    assert figures.choose_figures(report, {"P1": make_paper(licence=False)}) == []


def test_a_non_commercial_licence_is_used_and_named():
    paper = make_paper()
    paper.full_text = jats(PARA_F1, license_xml=lic("https://creativecommons.org/licenses/by-nc/4.0/"))
    [fig] = figures.choose_figures(make_report([("u1", "P1", QUOTE_F1, None)]), {"P1": paper})
    assert fig.licence == "CC BY-NC 4.0"


def test_papers_without_jats_full_text_or_a_pmcid_are_skipped():
    report = make_report([("u1", "P1", QUOTE_F1, None), ("u2", "P2", QUOTE_F1, None), ("u3", "P3", QUOTE_F1, None),
                          ("u4", "P4", QUOTE_F1, None)])
    papers = {
        "P1": make_paper("P1", fmt="text"),
        "P2": make_paper("P2", pmcid=None),
        "P3": make_paper("P3", fmt=None),
        # P4 is not in the mapping at all
    }
    assert figures.choose_figures(report, papers) == []


def test_dropped_sentences_and_unknown_short_ids_are_ignored():
    report = make_report([("u1", "P1", QUOTE_F1, None)], sentences_kept=False)
    assert figures.choose_figures(report, {"P1": make_paper()}) == []
    report = make_report([("u1", "P1", QUOTE_F1, None)])
    report.sections[0].sentences[0].citations = ["E99"]
    assert figures.choose_figures(report, {"P1": make_paper()}) == []


def test_at_most_two_figures_per_paper_in_the_order_the_report_cites_them():
    figs = "".join(f'<fig id="F{i}"><label>Figure {i}</label><graphic xlink:href="g{i}"/></fig>' for i in range(1, 5))
    body = "".join(f'<p>Result number {i} held (<xref ref-type="fig" rid="F{i}">{i}</xref>).</p>' for i in range(1, 5))
    paper = Paper(id="P1", title="t", pmcid="PMC1", full_text_format="jats", full_text=jats(body, license_xml=lic(), figs=figs))
    # The report cites results 3, 1, 4, 2 in that order: the first two found are kept.
    report = make_report([(f"u{i}", "P1", f"Result number {i} held", None) for i in (3, 1, 4, 2)])
    assert [f.fig_id for f in figures.choose_figures(report, {"P1": paper})] == ["F3", "F1"]


def test_at_most_eight_papers_and_ten_figures():
    figs = "".join(f'<fig id="F{i}"><label>Figure {i}</label><graphic xlink:href="g{i}"/></fig>' for i in (1, 2))
    body = "".join(f'<p>Result number {i} held (<xref ref-type="fig" rid="F{i}">{i}</xref>).</p>' for i in (1, 2))

    def paper(pid):
        return Paper(id=pid, title="t", pmcid=f"PMC{pid[1:]}", full_text_format="jats",
                     full_text=jats(body, license_xml=lic(), figs=figs))

    papers = {f"P{n}": paper(f"P{n}") for n in range(1, 8)}
    # Seven papers, two figures each: 14 candidates, capped at 10 figures (so five papers).
    cites = [(f"u{n}{i}", f"P{n}", f"Result number {i} held", None) for n in range(1, 8) for i in (1, 2)]
    got = figures.choose_figures(make_report(cites), papers)
    assert len(got) == figures.MAX_FIGURES == 10
    assert [f.paper_id for f in got] == [f"P{n}" for n in range(1, 6) for _ in (1, 2)]

    # Twelve papers, one figure each: capped at 8 papers.
    papers = {f"P{n}": paper(f"P{n}") for n in range(1, 13)}
    cites = [(f"u{n}", f"P{n}", "Result number 1 held", None) for n in range(1, 13)]
    got = figures.choose_figures(make_report(cites), papers)
    assert [f.paper_id for f in got] == [f"P{n}" for n in range(1, 9)]


def test_a_finding_for_a_figure_already_chosen_is_recorded_even_when_the_caps_are_full():
    papers = {"P1": make_paper("P1")}
    cites = [("u1", "P1", QUOTE_F1, None)]
    report = make_report(cites)
    # A later finding from another paper cannot get in, but one citing P1's F1 still joins its evidence ids.
    report.evidence.append(unit(id="u2", paper_id="P1", quote="Food intake fell by 20%"))
    report.short_ids["u2"] = "E2"
    report.sections[0].sentences.append(Sentence(text="More.", citations=["E2"]))
    [fig] = figures.choose_figures(report, papers)
    assert fig.evidence_ids == ["E1", "E2"]


def test_the_same_report_and_papers_always_give_the_same_figures():
    report = make_report([("u1", "P1", QUOTE_F1, None), ("u2", "P2", QUOTE_F1, None)])
    papers = {"P2": make_paper("P2", pmcid="PMC2"), "P1": make_paper("P1")}
    first = [f.to_dict() for f in figures.choose_figures(report, papers)]
    again = [f.to_dict() for f in figures.choose_figures(report, dict(reversed(list(papers.items()))))]
    assert first == again
    assert [f["paper_id"] for f in first] == ["P1", "P2"]
