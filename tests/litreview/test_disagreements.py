"""Code shows disagreements; whole quotes stay bound to their own papers."""

from __future__ import annotations

import pytest

from bedrock.litreview.facets import facets_for
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import EvidencePlan, ExtractionJob, Paper, PaperSignals, RankedPaper
from bedrock.litreview.report import synthesis
from bedrock.litreview.report.invariants import check_disagreements, check_report
from bedrock.litreview.report.layout import load_layout
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s6_score import summarize
from bedrock.litreview.stages.s7_write import WriteContext, write_report
from bedrock.litreview.stages.s8_render import to_html, to_markdown
from tests.litreview.fixtures import QUESTION
from tests.litreview.test_guard_cache_config import unit


def finding(pid, outcome="improved", *, species="human", target="IL23A", qualifier=None, **kw):
    name = {"IL23A": "IL-23", "IL17A": "IL-17A"}.get(target, target)
    who, n = ("patients", 40) if species == "human" else ("mice", 12)
    effect = {"improved": "reduced psoriasis scores by 30%", "no_effect": "had no significant effect on psoriasis scores",
              "worsened": "increased psoriasis scores by 20%", "association": "was associated with psoriasis scores"}[outcome]
    quote = f"Blocking {name} {effect} in {n} {who}."
    values = dict(id=f"u-{pid}", paper_id=pid, target=target, target_raw=name, outcome=outcome,
                  species=species, species_raw=who, evidence_type="human_clinical" if species == "human" else "in_vivo",
                  n=n, intervention=None, disease_raw="", effect_detail="", quote=quote, quote_start=0,
                  quote_end=len(quote), provenance={"target": "quote", "species": "quote", "n": "quote"},
                  qualifier_quote=qualifier, qualifier_start=len(quote) + 1 if qualifier else None,
                  qualifier_end=len(quote) + 1 + len(qualifier) if qualifier else None)
    values.update(kw)
    return unit(**values)


def context(units, *, titles=None, question=QUESTION, full=()):
    spec = parse_question(question)
    facets = facets_for(spec)
    titles = titles or {}
    papers = {}
    for u in units:
        text = u.quote + (" " + u.qualifier_quote if u.qualifier_quote else "")
        p = papers.setdefault(u.paper_id, Paper(id=u.paper_id, title=titles.get(u.paper_id, f"IL-23 and psoriasis {u.paper_id}"),
                                                abstract="", sections={"abstract": ""}, text_source="abstract_only",
                                                signals=PaperSignals(publication_types=["Randomized Controlled Trial"]
                                                                    if u.species == "human" else [])))
        start = len(p.sections["abstract"])
        p.sections["abstract"] += text + " "
        u.quote_start, u.quote_end = start, start + len(u.quote)
        if u.qualifier_quote:
            u.qualifier_start, u.qualifier_end = u.quote_end + 1, u.quote_end + 1 + len(u.qualifier_quote)
        p.abstract = p.sections["abstract"]
        if u.paper_id in full:
            p.sections["results"] = p.abstract
            p.text_source = "europepmc_xml"
    plan = EvidencePlan(question_id=spec.id, facets=[], best_source_id=next(iter(papers), None),
                        jobs=[ExtractionJob(paper_id=p, sections=["abstract"], facets=[], abstract_only=p not in full)
                              for p in papers])
    ranked = [RankedPaper(paper_id=p, rank=i, gate="pass", grade=3, study_kind="primary_research")
              for i, p in enumerate(papers, 1)]
    summary = summarize(units, spec, facets, ranked, plan)
    return WriteContext(spec=spec, facets=facets, plan=plan, summary=summary, ranked=ranked, papers=papers, gates={})


def report_for(ctx):
    return write_report(ctx, model_versions={}, prompt_versions={})


def section(report, sid="disagreements"):
    return next(s for s in report.sections if s.id == sid)


def test_a_human_disagreement_names_both_report_lines_and_shows_the_studies():
    benefit, null = finding("benefit"), finding("null", "no_effect")
    ctx = context([benefit, null], titles={"benefit": "IL-23 in psoriasis: the BENEFIT trial"}, full={"benefit"})
    report = report_for(ctx)
    body = section(report).body_markdown
    assert "**Studies disagree** —" in section(report, "synthesis").body_markdown
    assert "Papers disagree on this target" in next(s for s in section(report, "targets").subsections
                                                   if s.id == "targets:IL23A").note
    assert "How the studies line up: **IL23A** (Studies disagree)" in body
    assert "Candidate targets: **IL23A** (Papers disagree on this target; block)" in body
    assert "randomised controlled trial (BENEFIT), patients (n=40), full text read" in body
    assert "patients (n=40), abstract only" in body
    assert body.index("direction: **improved**") < body.index("direction: **no significant effect**")
    assert body.count("Independent study") == 2  # the pair belongs to both rules, shown once
    assert f"> {benefit.quote}" in body and f"> {null.quote}" in body
    assert check_report(report, list(ctx.papers.values()), ctx.spec, load_knowledge()) == []
    # Quotes from the second paper are not the best source's words.
    assert report.best_source_id != null.paper_id


def test_a_target_conflict_across_systems_is_shown_without_changing_the_human_basis():
    ctx = context([finding("mouse", species="mouse"), finding("human", "no_effect")])
    report = report_for(ctx)
    body = section(report).body_markdown
    assert "**Randomised comparison: 1 trial** —" in section(report, "synthesis").body_markdown
    assert "Studies disagree" not in section(report, "synthesis").body_markdown.split("\n\n")[0]
    assert "Candidate targets: **IL23A**" in body and "How the studies line up:" not in body
    assert "animal study, mice (n=12)" in body and "randomised controlled trial, patients (n=40)" in body
    assert "which differences, if any, explain it" in body
    assert check_disagreements(report) == []


def test_a_disagreement_within_animal_studies_is_shown():
    ctx = context([finding("benefit", species="mouse"), finding("worse", "worsened", species="mouse")])
    report = report_for(ctx)
    assert "How the studies line up: **IL23A** (Studies disagree)" in section(report).body_markdown
    assert "direction: **worsened**" in section(report).body_markdown


def test_a_synthesis_disagreement_without_a_target_conflict_still_has_an_entry():
    # Stage 6 groups by target AND action; synthesis groups by measure.
    ctx = context([finding("benefit"), finding("null", "no_effect", direction="activate")])
    assert ctx.summary.conflicts == []
    body = section(report_for(ctx)).body_markdown
    assert "How the studies line up: **IL23A** (Studies disagree)" in body
    assert "Candidate targets:" not in body


def test_every_disagreeing_measure_is_shown_even_past_the_synthesis_display_limit():
    units = [finding(f"p{i}-{j}", target=f"Measure{i}") for i in range(11) for j in range(3)]
    units += [finding("last-benefit", target="ZZZ"), finding("last-null", "no_effect", target="ZZZ")]
    ctx = context(units)
    studies = synthesis.profiles(ctx)
    rows = synthesis.synthesis_rows(ctx, studies)
    refs = {p: i for i, p in enumerate(ctx.papers, 1)}
    body = synthesis.disagreements(ctx, refs, studies, rows)
    assert rows.index(next(row for row in rows if row.basis == "Studies disagree")) >= 10
    for row in rows:
        if row.basis == "Studies disagree":
            assert f"How the studies line up: **{row.label}**" in body
    assert len(rows) > 10


def test_publications_from_one_trial_stay_in_one_independent_study_block():
    ctx = context([finding("a"), finding("b", "no_effect")], titles={
        "a": "IL-23 in psoriasis: the SHARED trial", "b": "IL-23 in psoriasis: post hoc analysis of SHARED trial"})
    report = report_for(ctx)
    body = section(report).body_markdown
    assert "**Randomised comparison: 1 trial** —" in section(report, "synthesis").body_markdown
    # Stage 6's target flag must still be explained, without treating the
    # two papers as independent replications.
    assert body.count("Independent study") == 1
    assert "post hoc or secondary analysis" in body and body.count("(SHARED)") == 2
    assert check_disagreements(report) == []


def test_quotes_are_whole_and_the_qualifier_travels_with_its_finding():
    qualifier = "Although this result did not reach statistical significance (p = 0.09), further study is needed."
    long_quote = "Blocking IL-23 reduced psoriasis scores in 40 patients " + "at the prespecified visit " * 30 + "."
    a = finding("benefit", quote=long_quote, qualifier=qualifier)
    ctx = context([a, finding("null", "no_effect")])
    report = report_for(ctx)
    body = section(report).body_markdown
    assert f"> {long_quote}" in body and f"> {qualifier}" in body
    assert body.index(long_quote) < body.index(qualifier) < body.index(ctx.summary.units[-1].quote)
    assert check_disagreements(report) == []


@pytest.mark.parametrize("change", ["quote", "qualifier", "reference", "truncate", "no_reference"])
def test_the_invariant_catches_tampered_words_or_a_quote_bound_to_another_paper(change):
    qualifier = "Although this result did not reach statistical significance (p = 0.09), further study is needed."
    ctx = context([finding("benefit", qualifier=qualifier), finding("null", "no_effect")])
    report = report_for(ctx)
    body = section(report).body_markdown
    if change == "quote":
        body = body.replace("reduced psoriasis scores by 30%", "cured psoriasis")
    elif change == "qualifier":
        body = body.replace("did not reach statistical significance", "reached statistical significance")
    elif change == "reference":
        body = body.replace("\n[1] ", "\n[2] ", 1)
    elif change == "truncate":
        body = body.replace(f"> {ctx.summary.units[0].quote}", "> Blocking IL-23 reduced psoriasis scores.")
    else:
        body = body.replace("\n[1] ", "\n", 1)
    section(report).body_markdown = body
    assert check_disagreements(report)
    assert any("[disagreements]" in v for v in check_report(report, list(ctx.papers.values()), ctx.spec, load_knowledge()))


def test_a_field_with_no_provenance_stays_off_the_page():
    a = finding("benefit", species_raw="imaginary population", n=999, intervention="imaginary agent",
                provenance={"target": "quote", "species": "none", "n": "none", "intervention": "none"})
    ctx = context([a, finding("null", "no_effect")])
    body = section(report_for(ctx)).body_markdown
    assert "imaginary" not in body and "999" not in body


def test_no_disagreement_has_no_heading_or_toc_link_in_markdown_and_html():
    report = report_for(context([finding("a"), finding("b")]))
    assert section(report).body_markdown == ""
    assert "Where the studies disagree" not in to_markdown(report)
    assert 'id="s-disagreements"' not in to_html(report)
    assert 'href="#s-disagreements"' not in to_html(report)


def test_the_section_renders_in_markdown_and_html_and_expands_its_abbreviations():
    a = finding("benefit", quote="Blocking IL-23 reduced psoriasis scores in 40 patients (CI 10 to 40).")
    report = report_for(context([a, finding("null", "no_effect")]))
    md, html = to_markdown(report), to_html(report)
    assert md.index("## How the studies line up") < md.index("## Where the studies disagree") < md.index("## References")
    assert '<h2 id="s-disagreements">Where the studies disagree</h2>' in html
    assert f"<blockquote>{a.quote}</blockquote>" in html
    assert "**CI** — confidence interval" in section(report, "glossary").body_markdown


@pytest.mark.parametrize("kind", ["target_identification", "intervention_efficacy", "mechanism", "safety", "other"])
def test_every_layout_places_the_generated_section_immediately_after_synthesis(kind):
    layout = load_layout(kind)
    i = next(i for i, s in enumerate(layout) if s.id == "synthesis")
    assert layout[i + 1].id == "disagreements" and layout[i + 1].kind == "generated"


def test_gaps_name_animal_measures_and_human_associations_with_all_their_references():
    ctx = context([finding('animal', species='mouse', target='IL17A'),
                   finding('association', 'association'), finding('association2', 'association')])
    report = report_for(ctx)
    refs = {r.paper_id: r.number for r in report.references}
    body = section(report, 'gaps').body_markdown
    expected_refs = ''.join(f'[{n}]' for n in sorted([refs['association'], refs['association2']]))
    assert f'No tested effect in humans among the papers read for targets: IL23A {expected_refs}; ' \
           f'IL17A [{refs["animal"]}].' in body
    rows = synthesis.synthesis_rows(ctx, ctx.studies)
    assert {row.basis for row in rows} == {'Animal or cell studies only', 'Associations only'}


def test_gaps_word_the_disclosure_for_measures_on_other_question_types():
    ctx = context([finding('mouse', species='mouse')], question='Does guselkumab treat psoriasis?')
    report = report_for(ctx)
    assert ctx.spec.question_type == 'intervention_efficacy'
    assert 'No tested effect in humans among the papers read for: IL23A [1].' in section(report, 'gaps').body_markdown
    assert 'for targets:' not in section(report, 'gaps').body_markdown


def test_gaps_do_not_claim_missing_human_effects_when_human_studies_tested_them():
    ctx = context([finding('human'), finding('mouse', species='mouse'), finding('null', 'no_effect', target='IL17A')])
    report = report_for(ctx)
    assert 'No tested effect in humans' not in section(report, 'gaps').body_markdown


def test_gaps_include_every_measure_beyond_the_synthesis_display_limit():
    units = [finding(f'p{i}', species='mouse', target=f'Measure{i}') for i in range(15)]
    ctx = context(units)
    report = report_for(ctx)
    assert '5 further measures' in section(report, 'synthesis').body_markdown
    bullet = next(line for line in section(report, 'gaps').body_markdown.splitlines()
                  if line.startswith('- No tested effect in humans'))
    refs = {r.paper_id: r.number for r in report.references}
    for u in units:
        assert f'{u.target} [{refs[u.paper_id]}]' in bullet


def test_an_animal_only_disagreement_still_has_no_tested_human_effect():
    ctx = context([finding('benefit', species='mouse'), finding('null', 'no_effect', species='mouse')])
    report = report_for(ctx)
    assert '**Studies disagree** —' in section(report, 'synthesis').body_markdown
    assert 'No tested effect in humans among the papers read for targets: IL23A [1][2].' in section(report, 'gaps').body_markdown
