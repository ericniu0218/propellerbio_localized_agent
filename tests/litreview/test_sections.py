"""Section splitting. Extraction quality depends on this more than on the prompt."""

from bedrock.litreview.parsing.sections import from_flat_text, from_grobid_tei, from_jats

JATS = """<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <abstract><p>IL-23 drives psoriasis in mice.</p></abstract>
    <abstract abstract-type="graphical"><p>Graphical abstract text.</p></abstract>
  </article-meta></front>
  <body>
    <sec sec-type="intro"><title>Introduction</title><p>Psoriasis is common [<xref ref-type="bibr">1</xref>].</p></sec>
    <sec sec-type="materials|methods"><title>Materials and methods</title><p>C57BL/6 mice (n=10) were used.</p></sec>
    <sec><title>Results</title>
      <sec><title>IL-23 blockade reduces thickening</title>
        <p>Blockade reduced ear thickness by 60%<xref ref-type="bibr">2</xref>.</p>
        <fig><caption><p>Figure 1. Ear thickness rose.</p></caption></fig>
      </sec>
      <table-wrap><table><tr><td>99</td></tr></table></table-wrap>
    </sec>
    <sec><title>Discussion</title><p>Our data suggest IL-23 is central.</p></sec>
  </body>
</article>"""


def test_jats_sections_are_named_and_figures_stripped():
    sections = from_jats(JATS)
    assert sections["abstract"] == "IL-23 drives psoriasis in mice."
    assert "n=10" in sections["methods"]
    assert "reduced ear thickness by 60%." in sections["results"], "the xref is gone, the sentence is whole"
    assert "Figure 1" not in sections["results"], "captions must never become findings"
    assert "99" not in sections["results"]
    assert sections["discussion"] == "Our data suggest IL-23 is central."
    assert "Graphical" not in sections["abstract"]


def test_jats_without_a_results_heading_treats_descriptive_sections_as_results():
    xml = """<article><body>
      <sec><title>Introduction</title><p>Background.</p></sec>
      <sec><title>IL-17A is dispensable for disease onset</title><p>Knockout mice developed disease.</p></sec>
      <sec><title>Methods</title><p>Mice were bred.</p></sec></body></article>"""
    sections = from_jats(xml)
    assert sections["results"] == "Knockout mice developed disease."


def test_grobid_subsections_inherit_their_section():
    tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><profileDesc>
      <abstract><p>Abstract text here.</p></abstract></profileDesc></teiHeader>
      <text><body>
        <div><head>Results</head><p>First result.</p></div>
        <div><head>TNF blockade has no effect</head><p>Second result <ref type="bibr">[3]</ref>.</p></div>
        <div><head>Discussion</head><p>We discuss.</p></div>
      </body></text></TEI>"""
    sections = from_grobid_tei(tei)
    assert sections["abstract"] == "Abstract text here."
    assert sections["results"] == "First result.\n\nSecond result ."
    assert sections["discussion"] == "We discuss."


def test_flat_text_headings_are_found():
    text = "Abstract\nShort abstract.\n\nRESULTS\nThe drug worked.\n\nDiscussion\nIt matters."
    sections = from_flat_text(text)
    assert sections["results"] == "The drug worked."
    assert sections["discussion"] == "It matters."


def test_broken_xml_returns_nothing_rather_than_crashing():
    assert from_jats("") == {}
