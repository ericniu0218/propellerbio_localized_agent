"""Section splitting. Extraction quality depends on this more than on the prompt."""

import pytest

from bedrock.litreview.parsing.sections import from_flat_text, from_grobid_tei, from_jats
from bedrock.litreview.report.layout import sentence_bounds

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


def test_jats_with_no_recognisable_headings_at_all_is_filed_as_unknown_not_results():
    # No <title> anywhere - not even Methods or Discussion matched, unlike the
    # test above where "Methods" is recognised. This is the real shape of two
    # asthma papers in the corpus (PMID41858165, PMID41596081): every heading
    # there carries a numeral ("1. Introduction", "3. Results") that the
    # patterns don't expect, so nothing but boilerplate (Acknowledgments,
    # Author Contributions) is ever recognised in the whole body.
    xml = """<article><body>
      <sec><p>Asthma affects millions and its causes remain unclear.</p></sec>
      <sec><p>Mice were sensitised with allergen and monitored for six weeks.</p></sec>
      <sec><p>Treated mice showed a 40% reduction in airway inflammation.</p></sec>
      <sec><p>These findings suggest a new therapeutic avenue worth pursuing.</p></sec>
      </body></article>"""
    sections = from_jats(xml)
    assert "results" not in sections
    assert "40% reduction in airway inflammation" in sections["unknown"]
    assert "causes remain unclear" in sections["unknown"], "the unrecognised 'Introduction' text is in here too"


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


# --- the layout's sentence limits ----------------------------------------------------------------


@pytest.mark.parametrize("value, bounds", [("3 to 5", (3, 5)), ("1-4", (1, 4)), (" 4 ", (4, 4))])
def test_sentence_limits_are_parsed_into_numbers(value, bounds):
    assert sentence_bounds(value) == bounds


@pytest.mark.parametrize("value", ["a few", "5 to 3", "0 to 2", "3 to 50", ""])
def test_a_malformed_sentence_limit_is_a_config_error(value):
    with pytest.raises(ValueError, match="report.yaml"):
        sentence_bounds(value, "answer")


def test_a_section_keeps_at_most_its_maximum_and_is_never_padded(tmp_path, monkeypatch):
    """Nine sentences that all pass the guard, for an Answer allowed five:
    the first five are kept, the rest are marked, and none is invented."""
    import json as _json

    from bedrock.litreview.config import load_settings
    from bedrock.litreview.db import Database
    from bedrock.litreview.llm import FakeLLMClient
    from bedrock.litreview.runner import answer_question
    from bedrock.litreview.stages import s7_write
    from tests.litreview.fixtures import QUESTION, corpus, responder

    class Verdict:
        kept, reason = True, None

    class PassEverything:
        def __init__(self, *args, **kwargs):
            pass

        def check(self, text, citations):
            return Verdict()

    monkeypatch.setattr(s7_write, "CitationGuard", PassEverything)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]

    def chatty(prompt, schema):
        if (schema or {}).get("title") == "DraftSection":
            ids = schema["properties"]["sentences"]["items"]["properties"]["citations"]["items"]["enum"]
            return _json.dumps({"sentences": [
                {"citations": [ids[0]], "text": f"Distinct sentence {w} about {w}{w} and {w} findings."} for w in words]})
        return responder(prompt, schema)

    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "c", tmp_path / "o", tmp_path / "db"
    r = answer_question(QUESTION, settings=s, client=FakeLLMClient(responder=chatty), db=Database(s.db_path),
                        papers=corpus())
    answer = next(sec for sec in r.report.sections if sec.id == "answer")
    kept = [x for x in answer.sentences if x.kept]
    capped = [x for x in answer.sentences if x.drop_reason and "limit of 5" in x.drop_reason]
    assert len(kept) == 5 and len(capped) == 4 and len(answer.sentences) == 9
    assert [x.text for x in kept] == [x.text for x in answer.sentences[:5]]
    assert r.report.stats["sentences_kept"] == sum(len([x for x in sec.sentences if x.kept]) for sec in
                                                   [*r.report.sections, *[sub for s2 in r.report.sections
                                                                          for sub in s2.subsections]])
