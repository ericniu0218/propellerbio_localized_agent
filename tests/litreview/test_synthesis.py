"""The code-written report sections (report/synthesis.py) and the reading layout.

A scientist should not have to open fifty references to learn how strong the
evidence is: which studies there are, how many are really independent, whether
they agree, what the effect sizes were and in whom, and what the letters mean.
Every one of those is computed from verified fields and the papers' own
sentences; none is written by a model.
"""

from __future__ import annotations

from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.models import Paper, PaperSignals
from bedrock.litreview.report import synthesis
from bedrock.litreview.runner import answer_question
from bedrock.litreview.stages.s8_render import to_html, to_markdown
from tests.litreview.fixtures import QUESTION, corpus, fake_client


def _paper(pid: str, title: str, abstract: str = "", types=()) -> Paper:
    return Paper(id=pid, title=title, abstract=abstract, signals=PaperSignals(publication_types=list(types)))


# --- which publications are one study ---------------------------------------------------------------


def test_a_trial_and_its_post_hoc_analyses_are_one_study():
    papers = {
        "a": _paper("a", "Effect of tirzepatide versus insulin degludec on liver fat content (SURPASS-3 MRI): a "
                         "substudy of the randomised, open-label, parallel-group, phase 3 SURPASS-3 trial"),
        "b": _paper("b", "Liver Fat Changes in Patients With Type 2 Diabetes: A Post Hoc Analysis of SURPASS-3 MRI."),
        "c": _paper("c", "Tirzepatide and MASLD in obesity: a real-world multicenter study."),
        "d": _paper("d", "An unrelated trial", abstract="Registered at ClinicalTrials.gov, NCT04166773."),
        "e": _paper("e", "Another report of it", abstract="Trial registration NCT04166773."),
    }
    groups = synthesis.group_studies(list(papers), papers)
    assert groups["a"][0] == groups["b"][0] and groups["a"][1] == "SURPASS-3"
    assert groups["c"][0] not in (groups["a"][0], groups["d"][0]) and groups["c"][1] is None
    assert groups["d"][0] == groups["e"][0]


def test_measure_names_and_class_names_are_not_trials():
    papers = {"a": _paper("a", "HOMA-IR and GLP-1 responses in COVID-19 (MRI-PDFF)"),
              "b": _paper("b", "HOMA-IR changes with GLP-1 therapy")}
    groups = synthesis.group_studies(list(papers), papers)
    assert groups["a"][0] != groups["b"][0]


# --- the evidence level is a written rule ---------------------------------------------------------------


def test_the_basis_states_what_kind_of_evidence_it_is():
    level = synthesis._level
    assert level({"improved": {"s1", "s2"}}, {"s1": "rct", "s2": "rct"}, False)[0] == "Randomised comparisons: 2 trials"
    assert level({"improved": {"s1", "s2"}}, {"s1": "rct", "s2": "observational"}, False)[0] == \
        "Randomised comparison: 1 trial"
    assert level({"improved": {"s1"}}, {"s1": "observational"}, False)[0] == "Non-randomised human studies only"
    assert level({"improved": {"s1"}}, {"s1": "case"}, False)[0] == "Case reports only"
    assert level({"improved": {"s1"}}, {"s1": "animal"}, False)[0] == "Animal or cell studies only"
    assert level({"improved": {"s1"}, "no_effect": {"s2"}}, {"s1": "rct", "s2": "rct"}, False)[0] == "Studies disagree"
    assert level({"association": {"s1"}}, {"s1": "observational"}, False)[0] == "Associations only"
    assert "abstract alone" in level({"improved": {"s1"}}, {"s1": "rct"}, True)[1]


def test_no_level_claims_certainty_it_has_not_assessed():
    assert "not a formal certainty rating" in synthesis.LEVEL_RULE
    assert "Established" not in synthesis.LEVEL_RULE


def test_a_mouse_result_does_not_make_the_human_evidence_disagree():
    tiers = {"trial": "rct", "mice": "animal"}
    basis, why = synthesis._level({"improved": {"trial"}, "worsened": {"mice"}}, tiers, False)
    assert basis == "Randomised comparison: 1 trial" and "animal or cell" in why


def test_animal_only_basis_does_not_deny_human_associations():  # re-run, "Liver enzymes" line
    tiers = {"mice": "animal", "cohort": "observational", "trial": "rct"}
    basis, why = synthesis._level({"improved": {"mice"}, "association": {"cohort", "trial"}}, tiers, False)
    assert basis == "Animal or cell studies only"
    assert "no human evidence" not in why and "2 human studies report associations only" in why


# --- abbreviations: the paper's own definition first --------------------------------------------------------


class _Ctx:
    def __init__(self, papers):
        self.papers = papers


def test_the_glossary_prefers_the_papers_own_definition_and_expands_standard_ones():
    paper = _paper("p", "t", abstract="The fatty liver index (FLI) fell. Alanine aminotransferase rose.")
    text = "FLI decreased and ALT fell, and ΔHbA1c too; RSV is not used here."
    out = synthesis.glossary(_Ctx({"p": paper}), text, {"p": 1})
    assert "**FLI** — fatty liver index (as defined in [1])" in out
    assert "**ALT** — alanine aminotransferase" in out
    assert "**HbA1c** — glycated haemoglobin" in out
    assert "RSV" not in out  # neither defined by a cited paper nor a standard entry: not guessed


# --- the whole report, laid out for reading --------------------------------------------------------------------


def _report(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "c", tmp_path / "o", tmp_path / "db"
    return answer_question(QUESTION, settings=s, client=fake_client(), papers=corpus(),
                           db=Database(tmp_path / "db")).report


def test_the_reading_view_comes_first_and_the_audit_trail_in_an_appendix(tmp_path):
    md = to_markdown(_report(tmp_path))
    main, appendix = md.split("\n# Appendix\n", 1)
    for heading in ("## Answer", "## Evidence at a glance", "## How the studies line up",
                    "## Key results in context", "## Limitations, gaps and open questions", "## References"):
        assert heading in main
    for heading in ("## How the question was read", "## Papers excluded", "## How this answer was assembled",
                    "## Evidence table", "## Evidence quotes"):
        assert heading in appendix and heading not in main
    assert main.index("## Answer") < main.index("## Evidence at a glance") < main.index("## References")
    assert "No evidence for this section in the papers read." not in main


def test_written_sections_say_which_studies_they_rest_on(tmp_path):
    report = _report(tmp_path)
    kept = [s for s in report.sections if s.kind == "written" and any(x.kept for x in s.sentences)]
    assert kept and all("Evidence basis: [" in (s.note or "") for s in kept)
    assert "animal study, mice, full text" in to_markdown(report)


def test_key_results_quote_the_paper_and_name_the_study(tmp_path):
    md = to_markdown(_report(tmp_path))
    numbers = md.split("## Key results in context", 1)[1].split("\n## ", 1)[0]
    assert "“Anti-IL-23 antibody given before imiquimod challenge reduced ear thickness by 60% in C57BL/6 mice.”" \
        in numbers
    assert "animal study, mice, full text" in numbers


def test_the_html_has_the_same_structure(tmp_path):
    html = to_html(_report(tmp_path))
    assert html.index('id="s-glance"') < html.index('id="references"') < html.index('id="appendix"') \
        < html.index('id="s-excluded"')


def test_the_html_report_is_navigable_without_scripts(tmp_path):
    """The Research tab shows the report in a frame with scripts off: every path
    is a link or a <details>, which also work downloaded and offline."""
    report = _report(tmp_path)
    html = to_html(report)
    assert "<script" not in html
    toc = html.split('<nav class="toc"', 1)[1].split("</nav>", 1)[0]
    for sec in report.sections:
        if sec.id in ("glance", "synthesis", "numbers", "gaps", "evidence_table", "excluded"):
            assert f'href="#s-{sec.id}"' in toc and f'id="s-{sec.id}"' in html
    # claim -> evidence -> source, and source -> claims
    assert "Evidence behind this section" in html and 'href="#E1"' in html and 'id="E1"' in html
    assert "cited in <a href=\"#s-" in html
    # every anchor the page links to exists on the page
    import re
    targets = set(re.findall(r'href="#([^"]+)"', html))
    ids = set(re.findall(r'id="([^"]+)"', html))
    assert targets <= ids, targets - ids


def test_the_report_passes_its_integrity_check_and_is_byte_identical_twice(tmp_path):
    a = _report(tmp_path / "one")
    b = _report(tmp_path / "two")
    assert a.integrity == [] and a.model_dump_json() == b.model_dump_json()


def test_a_section_leads_with_the_papers_finding_about_the_subject():
    """Before, a paper's representative finding was whichever quote sorted
    first: body weight ("After three months...") ahead of its liver-fat result."""
    from bedrock.litreview.stages.s6_score import best_units
    from tests.litreview.test_guard_cache_config import unit

    weight = unit(id="u1", paper_id="p1", quote="After three months, body weight fell by 12%.", target="body weight")
    liver = unit(id="u2", paper_id="p1", quote="Liver fat content fell by 8.1% with tirzepatide.", target="liver fat")
    assert best_units([weight, liver], {"p1": 1}, 1) == [weight]
    assert best_units([weight, liver], {"p1": 1}, 1, prefer=lambda u: u.target != "liver fat") == [liver]


def test_non_results_are_kept_out_of_the_synthesis_and_the_front_of_sections():
    """The SURPASS-3 post hoc paper's E35 ("A total of 276 participants were
    included") was offered to the Answer ahead of its E38, the actual result,
    because "A total..." sorts first; the guard then rightly refused the
    sentence, and the Answer was written from two mouse studies."""
    from tests.litreview.test_guard_cache_config import unit

    for quote in ("A total of 276 participants were included in this analysis.",
                  "Findings were consistent across genotype subgroups.",
                  "This study investigated changes in APRI and FIB-4 scores during one year of treatment.",
                  "Baseline amylase levels did not differ significantly among treatment groups (p > 0.05).",
                  "Treatment with weekly semaglutide or tirzepatide resulted in a significant reduction in BMI (p"):
        assert not synthesis.substantive(unit(quote=quote)), quote
    for quote in ("In the SURPASS-3 MRI substudy, tirzepatide significantly reduced liver fat content (LFC) versus "
                  "insulin degludec.",
                  "Body weight decreased from 97.7 ± 10.9 kg at baseline to 85.7 ± 9.5 kg after 3 months (p < 0.001).",
                  "Blocking IL-17A had no significant effect on epidermal thickness (p=0.41)."):
        assert synthesis.substantive(unit(quote=quote)), quote


def test_subgroups_of_one_trial_going_different_ways_are_not_a_conflict():
    level = synthesis._level
    assert level({"improved": {"s1"}, "no_effect": {"s1"}}, {"s1": "rct"}, False)[0] == "Randomised comparison: 1 trial"


def test_a_mouse_study_is_never_called_a_human_study():
    from tests.litreview.test_guard_cache_config import unit

    paper = _paper("m", "Tirzepatide ameliorates hepatic steatosis in a MASLD mouse model")
    units = [unit(paper_id="m", species="mouse", evidence_type="human_clinical"),
             unit(paper_id="m", species="mouse", evidence_type="in_vivo")]
    assert synthesis._tier(paper, None, units) == "animal"


def test_design_names_are_pluralised_properly():
    assert synthesis._plural("observational study", 3) == "observational studies"
    assert synthesis._plural("clinical trial (randomisation not stated)", 2) == "clinical trials (randomisation not stated)"
    assert synthesis._plural("case report", 1) == "case report"


def test_among_equally_relevant_papers_a_randomised_trial_is_read_before_a_mouse_study():
    from bedrock.litreview.study_design import design_rank, design_signals

    rct = _paper("r", "Effect of tirzepatide versus insulin degludec on liver fat content (SURPASS-3 MRI)",
                 abstract="Participants were randomly assigned to tirzepatide or insulin degludec.",
                 types=["Randomized Controlled Trial"])
    mouse = _paper("m", "Tirzepatide ameliorates hepatic steatosis in mice",
                   abstract="C57BL/6 mice were treated with tirzepatide for 8 weeks.")
    case = _paper("c", "A Rare Case of Tirzepatide-Induced Hepatotoxicity", types=["Case Reports"])
    ranks = {p.id: design_rank(design_signals(p), p) for p in (rct, mouse, case)}
    assert ranks["r"] < ranks["c"] < ranks["m"]


def test_a_writer_expansion_the_paper_contradicts_is_removed():
    """Real case: the paper compares tirzepatide with bariatric metabolic
    surgery (BMS); the writer's sentence said "best medical standard (BMS)"."""
    from bedrock.litreview.stages.s7_write import _invented_expansion
    from tests.litreview.test_guard_cache_config import unit

    paper = _paper("b", "Comparative effectiveness of tirzepatide versus bariatric metabolic surgery (BMS) in adults "
                        "with MASLD")
    ctx = _Ctx({"b": paper})
    cited = [unit(paper_id="b", quote="The tirzepatide group showed a lower risk than the BMS group (HR 0.47).")]
    why = _invented_expansion("Tirzepatide lowered risk compared to best medical standard (BMS).", cited, ctx)
    assert why and "bariatric metabolic surgery" in why
    assert _invented_expansion("Tirzepatide lowered risk compared to bariatric metabolic surgery (BMS).", cited, ctx) is None
    # A standard expansion is fine even when the paper does not define it.
    assert _invented_expansion("Glycated haemoglobin (HbA1c) fell.", [unit(paper_id="b", quote="HbA1c fell.")], ctx) is None


def test_a_substudy_is_not_called_post_hoc():
    rct = _paper("r", "Effect of tirzepatide versus insulin degludec on liver fat content (SURPASS-3 MRI): a substudy "
                      "of the randomised, open-label, parallel-group, phase 3 SURPASS-3 trial")
    assert synthesis._design_text("rct", rct) == "substudy of a randomised controlled trial"
    post = _paper("p", "Liver Fat Changes by Genotype: A Post Hoc Analysis of SURPASS-3 MRI")
    assert synthesis._design_text("rct", post).startswith("post hoc")


def test_one_trials_mixed_measures_do_not_make_the_evidence_conflicting():
    level = synthesis._level
    tiers = {"trial": "rct", "cohort": "observational"}
    assert level({"improved": {"trial", "cohort"}, "no_effect": {"trial"}}, tiers, False)[0] == \
        "Randomised comparison: 1 trial"
    assert level({"improved": {"trial"}, "no_effect": {"cohort"}}, tiers, False)[0] == "Studies disagree"


def test_statistical_abbreviations_keep_their_statistical_meaning():
    paper = _paper("p", "t", abstract="Patients with heart failure (HR) were excluded. Fatty liver index (FLI) fell.")
    out = synthesis.glossary(_Ctx({"p": paper}), "HR, 0.66; 95% CI; FLI", {"p": 1})
    assert "**HR** — hazard ratio" in out and "heart failure" not in out


def test_two_papers_that_define_the_letters_differently_are_both_shown():
    a = _paper("a", "t", abstract="Liver stiffness measurement (LSM) fell.")
    b = _paper("b", "t", abstract="Lean soft mass (LSM) rose.")
    out = synthesis.glossary(_Ctx({"a": a, "b": b}), "LSM changed", {"a": 1, "b": 2})
    assert "use it differently" in out and "[1]" in out and "[2]" in out


def test_demographics_are_not_results():
    from tests.litreview.test_guard_cache_config import unit

    for quote in ("The study population had a mean age of 42.5 ± 11.4 years, and 76.8% were women.",
                  "Lipid-lowering therapy was taken by 59% and 27% of participants with T2D and those without T2D.",
                  "The most common glucose-lowering agent was metformin, predominantly in participants with T2D (76%)."):
        assert not synthesis.substantive(unit(quote=quote)), quote
