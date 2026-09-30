"""Issue #23: the tools that make labelling the gold set fast and hard to get wrong.

The labels come from a scientist; nothing here writes one. These tests cover
the sheet generator (which papers, seeded; what a sheet may show), the checks
that refuse an unfinished gold set, and `run_extract` end to end on a
two-paper gold set with the fake model.
"""

from __future__ import annotations

import json
import re

import pytest

from bedrock.litreview.config import load_settings
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import Paper, PaperSignals
from bedrock.litreview.sources.corpus import write_papers
from bedrock.litreview.stages.s5_extract import extract_paper
from evals.litreview import run_extract
from evals.litreview.goldset import (
    KIND_TITLES,
    QUOTAS,
    GoldsetError,
    extraction_job,
    load_goldset,
    read_by_pipeline,
    read_corpus,
    read_question,
)
from evals.litreview.goldset_sheet import write_sheets
from tests.litreview.fixtures import QUESTION, corpus, fake_client

EXTRA = [
    Paper(id="V1", title="IL-23 signalling in cultured keratinocytes from psoriasis lesions",
          abstract="Keratinocytes were isolated from psoriasis lesions and cultured in vitro. IL-23 increased "
                   "IL-17A expression in the cultured cells."),
    Paper(id="A2", title="TYK2 inhibition in a mouse model of psoriasis",
          abstract="C57BL/6 mice were treated with a TYK2 inhibitor before imiquimod. TYK2 inhibition did not "
                   "reduce psoriasis-like ear thickness."),
    Paper(id="T2", title="Deucravacitinib in plaque psoriasis",
          abstract="Patients with psoriasis were randomly assigned to deucravacitinib or placebo. PASI-75 was "
                   "reached by 58% versus 13%.",
          signals=PaperSignals(publication_types=["Randomized Controlled Trial"])),
]


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.data_dir = tmp_path / "cache", tmp_path / "out", tmp_path / "data"
    return s


@pytest.fixture()
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    write_papers(corpus() + EXTRA, d / "papers.jsonl")
    return d


def _entries(template):
    return [json.loads(x) for x in template.read_text().splitlines()][1:]


# --- the sheets ---------------------------------------------------------------------------------


def test_papers_are_picked_by_quota_from_what_they_say_they_are(corpus_dir, tmp_path, settings):
    template = write_sheets(corpus_dir, QUESTION, tmp_path / "gold", seed=0, settings=settings)
    kinds = {e["paper_id"]: e["kind"] for e in _entries(template)}
    assert kinds == {"P2": "human_clinical", "T2": "human_clinical", "A2": "animal", "V1": "in_vitro",
                     "P1": "negative_result", "P4": "review", "P6": "mentions_not_about"}
    assert set(kinds.values()) <= {k for k, _ in QUOTAS}
    index = (tmp_path / "gold" / "index.md").read_text()
    assert "human clinical trial (3 fewer than the quota)" in index, "a short category is said, not back-filled"


def test_proposed_kinds_must_be_confirmed_when_reading(corpus_dir, tmp_path, settings):
    template = write_sheets(corpus_dir, QUESTION, tmp_path / "gold", settings=settings)
    entries = {e["paper_id"]: e for e in _entries(template)}
    assert entries["P1"]["kind_confirmed"] is False
    assert entries["P1"]["kind_proposed_because"] == "its abstract says 'had no'"
    assert entries["P6"]["kind_confirmed"] is False
    assert "only in passing" in entries["P6"]["kind_proposed_because"], "the gate's own reason"
    assert entries["P2"]["kind_confirmed"] is True
    index = (tmp_path / "gold" / "index.md").read_text()
    assert index.count("**confirm when reading**") == 2


def test_the_selection_is_seeded(corpus_dir, tmp_path, settings):
    a = write_sheets(corpus_dir, QUESTION, tmp_path / "a", seed=3, settings=settings)
    b = write_sheets(corpus_dir, QUESTION, tmp_path / "b", seed=3, settings=settings)
    assert a.read_text().replace(str(tmp_path / "a"), "") == b.read_text().replace(str(tmp_path / "b"), "")
    assert (tmp_path / "a" / "index.md").read_text() == (tmp_path / "b" / "index.md").read_text()


def test_five_are_held_back_one_per_kind(corpus_dir, tmp_path, settings):
    entries = _entries(write_sheets(corpus_dir, QUESTION, tmp_path / "gold", settings=settings))
    held = [e for e in entries if e["holdout"]]
    assert len(held) == 5
    assert len({e["kind"] for e in held}) == 5, "the holdout is not five of one kind"
    sheet = (tmp_path / "gold" / held[0]["sheet"]).read_text()
    assert "Held back" in sheet


def test_a_sheet_shows_the_paper_and_nothing_the_pipeline_produced(corpus_dir, tmp_path, settings):
    entries = _entries(write_sheets(corpus_dir, QUESTION, tmp_path / "gold", settings=settings))
    forbidden = [*KIND_TITLES.values(), *KIND_TITLES, "confirm", "human_rct", "animal_interventional",
                 "narrative_review", "only in passing", "names psoriasis", "grade", "relevan"]
    papers = read_corpus(str(corpus_dir))
    for e in entries:
        sheet = (tmp_path / "gold" / e["sheet"]).read_text()
        assert e["title"] in sheet
        # What is left once the paper's own words are taken out is ours.
        paper = papers[e["paper_id"]]
        for text in (paper.title, *paper.sections.values(), QUESTION):
            sheet = sheet.replace(text, "")
        leaked = [w for w in forbidden if re.search(rf"\b{re.escape(w)}", sheet, re.I)]
        assert not leaked, f"{e['sheet']} shows {leaked}"


def test_a_sheet_shows_exactly_what_extraction_is_sent(corpus_dir, tmp_path, settings):
    """The sheet's chunks are the ones Stage 5 sends; checked against the real
    `extract_paper`, so the sheet cannot drift from the pipeline unnoticed."""
    entries = {e["paper_id"]: e for e in _entries(write_sheets(corpus_dir, QUESTION, tmp_path / "gold",
                                                               settings=settings))}
    paper = read_corpus(str(corpus_dir))["P1"]
    spec = read_question(QUESTION)
    prompts: list[str] = []

    def recorder(prompt, schema):
        prompts.append(prompt)
        return json.dumps({"findings": [], "is_review": False})

    from bedrock.litreview.facets import facets_for

    extract_paper(paper, extraction_job(paper, spec, settings), spec, facets_for(spec),
                  FakeLLMClient(responder=recorder), model="m")
    chunks, methods = read_by_pipeline(paper, spec, settings)
    sheet = (tmp_path / "gold" / entries["P1"]["sheet"]).read_text()
    assert len(prompts) == len(chunks) == 2
    for (_, _, text), prompt in zip(chunks, prompts):
        assert text in prompt and text in sheet
    assert methods in prompts[0] and methods in sheet


# --- refusing an unfinished gold set -------------------------------------------------------------


def _finding(**kw):
    base = dict(target="IL-23", direction="block", disease="psoriasis", evidence_type="in_vivo", species="mouse",
                outcome="improved", n=None, effect_detail="",
                quote="Anti-IL-23 antibody given before imiquimod challenge reduced ear thickness by 60% in "
                      "C57BL/6 mice.")
    base.update(kw)
    return base


def _entry(corpus_dir, **kw):
    base = dict(paper_id="P1", corpus=str(corpus_dir), question=QUESTION, kind="animal", holdout=False,
                findings=[_finding()])
    base.update(kw)
    return base


def _write(tmp_path, *rows):
    path = tmp_path / "goldset.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _problems(path):
    with pytest.raises(GoldsetError) as exc:
        load_goldset(path)
    return exc.value.problems


def test_a_finished_entry_loads_and_the_comment_line_is_skipped(corpus_dir, tmp_path):
    path = _write(tmp_path, {"_comment": "notes"}, _entry(corpus_dir))
    (entry,) = load_goldset(path)
    assert entry.paper_id == "P1" and entry.findings[0].species == "mouse"


def test_a_placeholder_is_refused(corpus_dir, tmp_path):
    path = _write(tmp_path, _entry(corpus_dir), {"paper_id": "EXAMPLE-1", "kind": "review", "findings": []})
    assert _problems(path) == ["line 2, paper EXAMPLE-1: a placeholder from the template, not a labelled "
                               "paper; delete it"]


def test_the_checked_in_placeholder_gold_set_is_refused(capsys):
    assert run_extract.main([]) == 1
    err = capsys.readouterr().err
    assert "paper EXAMPLE-1: a placeholder" in err and "not ready to score" in err


def test_an_empty_entry_needs_none_expected(corpus_dir, tmp_path):
    problems = _problems(_write(tmp_path, _entry(corpus_dir, findings=[])))
    assert problems == ["line 1, paper P1, field findings: empty. If the paper really has nothing to extract, "
                        "say so with \"none_expected\": true; otherwise it is not labelled yet"]
    assert load_goldset(_write(tmp_path, _entry(corpus_dir, findings=[], none_expected=True)))


def test_a_review_must_expect_zero_findings(corpus_dir, tmp_path):
    problems = _problems(_write(tmp_path, _entry(corpus_dir, kind="review")))
    assert any("a review article must expect zero findings" in p for p in problems)


def test_a_bad_field_names_the_paper_and_the_field(corpus_dir, tmp_path):
    rows = [_entry(corpus_dir, findings=[_finding(direction="inhibit")]),
            _entry(corpus_dir, paper_id="P2", findingz=[])]
    problems = _problems(_write(tmp_path, *rows))
    assert any(p.startswith("line 1, paper P1, field findings[0].direction:") for p in problems)
    assert any(p.startswith("line 2, paper P2, field findingz:") for p in problems)


def test_a_quote_not_in_the_paper_is_refused(corpus_dir, tmp_path):
    problems = _problems(_write(tmp_path, _entry(corpus_dir, findings=[_finding(quote="IL-23 cured everyone.")])))
    assert problems == ["line 1, paper P1, field findings[0].quote: not in the paper's text; copy the sentence "
                        "exactly"]


def test_an_unconfirmed_kind_is_refused(corpus_dir, tmp_path):
    problems = _problems(_write(tmp_path, _entry(corpus_dir, kind="negative_result", kind_confirmed=False,
                                                 kind_proposed_because="its abstract says 'did not'")))
    assert any("field kind_confirmed" in p and "its abstract says 'did not'" in p for p in problems)


# --- scoring end to end ---------------------------------------------------------------------------


def test_run_extract_end_to_end_with_the_fake_model(corpus_dir, tmp_path, settings):
    """Two papers: a mouse study with a positive and a null finding, and a
    review, which must produce nothing."""
    il17 = "Blocking IL-17A had no significant effect on epidermal thickness (p=0.41)."
    path = _write(
        tmp_path,
        _entry(corpus_dir, findings=[
            _finding(n=12),  # the fake model gives no n: one field wrong
            _finding(target="IL-17A", outcome="no_effect", quote=il17),
        ]),
        _entry(corpus_dir, paper_id="P4", kind="review", findings=[], none_expected=True),
    )
    results = run_extract.run(load_goldset(path), fake_client(), settings)
    lines: list[str] = []
    run_extract.report(results, out=lines.append)
    text = "\n".join(lines)

    assert results["recall"] == 1.0, results["missed"]
    assert results["quote_validity"] == 1.0
    assert results["fields"]["n"] == 0.5
    assert all(results["fields"][f] == 1.0 for f in run_extract.SCORED_FIELDS if f != "n")
    assert results["reviews_with_findings"] == []
    assert "Per-field accuracy (on findings paired by sentence)" in text
    assert "    n                0.50  <-- worst, fix this first" in text
    assert "  Reviews         0 of 1 produced findings" in text
    assert "  Recall          1.00   (2 of 2 expected were found)" in text


def test_findings_pair_by_sentence_so_a_wrong_field_shows(settings):
    """The old key paired only findings that agreed on five scored fields, so
    their accuracy was 1.00 by construction. Paired by sentence, a mouse study
    labelled as human shows up as an evidence_type error."""
    quote = "X reduced lesions in mice."
    expected = [{"target": "X", "direction": "block", "disease": "d", "evidence_type": "in_vivo",
                 "species": "mouse", "outcome": "improved", "n": None, "quote": quote}]
    predicted = [{**expected[0], "evidence_type": "human_clinical", "species": "human"}]
    result = run_extract.score(expected, predicted)
    assert result["matched"] == 1
    assert result["field_hits"]["evidence_type"] == [0] and result["field_hits"]["target"] == [1]
