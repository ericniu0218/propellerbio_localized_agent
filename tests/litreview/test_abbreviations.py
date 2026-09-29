"""Issue #5: abbreviations and outcome measures never become targets.

Two hand-audit cases, both from stored qwen3:8b reports:

- asthma S34 and S52: "AHR" in PMID36077141 is airway hyperresponsiveness -
  the paper says so in its own words, "Airway hyper-responsiveness (AHR) was
  detected using noninvasive whole-body plethysmography" - yet its findings
  were grouped with the aryl hydrocarbon receptor (AhR) of another paper and
  written up under "targets:AhR".
- migraine S9: "MMDs" (monthly migraine days, PMID40542538) got its own row
  of candidate targets.

The paper's own definitions are read first (`knowledge/abbreviations.py`);
a readout is kept as evidence but flagged `outcome_measure`, and Stage 6
gives it no row. Every sentence below is the paper's own text, verbatim from
the corpus, unless it says otherwise.
"""

import json
from collections import Counter

from bedrock.litreview.knowledge import Knowledge, KnownEntity, load_knowledge
from bedrock.litreview.knowledge.abbreviations import (
    best_long_form,
    defined_as,
    find_abbreviations,
    is_outcome_measure,
    load_measures,
    resolve_target,
)
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import EvidenceUnit, ExtractionJob, Facet, Paper
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import extract_paper
from bedrock.litreview.stages.s6_score import summarize

KB = load_knowledge()
ASTHMA = parse_question("If I want to prevent asthma exacerbations, what should I target?")
MIGRAINE = parse_question("If I want to prevent migraine attacks, what areas should I target?")
FACETS = [Facet(id="causal", title="Causal", need="what changing the target does")]

# PMID36077141 (asthma corpus), results: the definition, and the paragraph E59
# and E248 were quoted from.
AHR_DEFINITION = ("Airway hyper‐responsiveness (AHR) was detected using noninvasive whole-body "
                  "plethysmography (Model PLY-UNR-MS2; EMKA Technologies, Paris, France) 24 h after the last OVA "
                  "treatment.")
E59_QUOTE = ("The effects of GRI977143 and H2L5186303 on AHR were measured. Results showed that the Penh value of "
             "the OVA-challenged mice was higher than the control group.")
E248_QUOTE = ("A significantly reduced AHR was observed in OVA-challenged mice treated with GRI977143 before OVA "
              "challenge but not sensitization ().")
AHR_RESULTS = (E59_QUOTE + " " + E248_QUOTE + " Moreover, H2L5186303 treatment before OVA sensitization and "
               "challenge significantly reduced AHR in OVA-challenged mice ().")

# PMID41596081 (asthma corpus), abstract: the receptor, spelled AhR.
AHR_RECEPTOR_ABSTRACT = (
    "The aryl hydrocarbon receptor (AhR), a ligand-activated transcription factor responsive to environmental "
    "stimuli, may modulate this process. It was documented that the regulation of AhR on senescence was "
    "attenuated by VAF347, whereas AhR deficiency exacerbated ROS generation and inflammation in AhRΔScgb1a1 "
    "mice."
)
# PMID42376192 (obesity corpus), abstract: the receptor, spelled AHR.
AHR_RECEPTOR_CAPITALS = (
    "In this study, we aim to investigate the role of the ligand-activated transcription factor, aryl "
    "hydrocarbon receptor (AHR), in the development of obesity and metabolic syndrome."
)
# PMID42421242 (asthma corpus), abstract: the gene AHR, never spelled out.
AHR_GENE_UNDEFINED = (
    "This study provides genetic evidence supporting causal links between PPI-related genes (AHR, DDAH1, MAPT) "
    "and asthma."
)

# PMID40542538 (migraine corpus): the definition, and E78's quote.
MMD_DEFINITION = (
    "The hypothesis was that rimegepant 75 mg every other day (EOD) would be superior to placebo in reducing the "
    "number of monthly migraine days (MMDs) from the observation phase (OP) to the last 4 weeks of the DBT phase."
)
E78_QUOTE = (
    "The response rates for the first secondary efficacy endpoint (i.e., participants with ≥50% reduction from "
    "the OP in moderate to severe MMDs in the last 4 weeks of the DBT phase) numerically favored rimegepant "
    "versus placebo and were 41.7% (95% CI = 35.5%–47.9%) in the rimegepant group and 34.4% (95% CI = "
    "28.5%–40.4%) in the placebo group ( p = 0.099; Table )."
)

# A table that knows the receptor by all three spellings, as entities.yaml
# would if the receptor were added to it: the airway "AHR" must still not
# land on it.
KB_WITH_AHR = Knowledge([
    *KB.entities,
    KnownEntity(canonical="AHR", kind="target", terms=("AhR", "aryl hydrocarbon receptor")),
])


# --- the definition finder, on real sentences ---------------------------------------


def test_the_papers_own_definitions_are_found():
    assert find_abbreviations(AHR_DEFINITION)["AHR"] == "Airway hyper-responsiveness"
    assert find_abbreviations(MMD_DEFINITION)["MMDs"] == "monthly migraine days"
    assert find_abbreviations(MMD_DEFINITION)["EOD"] == "every other day"
    assert find_abbreviations(AHR_RECEPTOR_ABSTRACT)["AhR"] == "aryl hydrocarbon receptor"
    assert find_abbreviations(AHR_RECEPTOR_CAPITALS)["AHR"] == "aryl hydrocarbon receptor"


def test_a_long_form_written_as_one_word_is_found():
    """Schwartz-Hearst matches letters, not word initials: R and H both sit
    inside "hyperresponsiveness" (PMID39245681 writes it that way)."""
    assert best_long_form("AHR", "we measured airway hyperresponsiveness") == "airway hyperresponsiveness"
    assert find_abbreviations("As a result, airway hyperresponsiveness (AHR), elevated serum IgE were seen.") \
        == {"AHR": "airway hyperresponsiveness"}


def test_the_reverse_order_is_found():
    """'SHORT (long form)' counts only when the whole bracket spells it."""
    assert find_abbreviations("AHR (airway hyperresponsiveness) was measured.") == {"AHR": "airway hyperresponsiveness"}
    assert find_abbreviations("IL-17 (a cytokine made by Th17 cells) rose.") == {}


def test_brackets_that_define_nothing_are_ignored():
    # PMID42421242: a list of genes in a bracket is not a definition.
    assert "AHR" not in find_abbreviations(AHR_GENE_UNDEFINED)
    assert find_abbreviations("Eosinophils were counted in the BALF of treated mice (n = 6).") == {}
    # A genotype is not a definition of the gene (seen in PMID41495031).
    assert "Sirt6-/-" not in find_abbreviations("We used Sirt6 KO mice (Sirt6-/-) throughout.")
    assert find_abbreviations("Cells are shown in panel three (Fig. C).") == {}


def test_a_definition_answers_for_its_plural_and_its_other_case():
    abbreviations = {"MMDs": "monthly migraine days", "AhR": "aryl hydrocarbon receptor"}
    assert defined_as("MMD", abbreviations) == "monthly migraine days"
    assert defined_as("AHR", abbreviations) == "aryl hydrocarbon receptor", "only the case differs"
    assert defined_as("Sirt6", {"Sirt6-/-": "Sirt6 KO mice"}) is None, "a character differs"


# --- what a target names --------------------------------------------------------------


def test_airway_ahr_is_a_measure_and_never_the_receptor():
    abbreviations = find_abbreviations(AHR_DEFINITION)
    for kb in (KB, KB_WITH_AHR):
        resolved = resolve_target("AHR", kb, abbreviations)
        assert resolved.outcome_measure
        assert resolved.name == "AHR", "the paper's own name, not the table's receptor"
        assert resolved.long_form == "Airway hyper-responsiveness"


def test_the_receptor_is_still_the_receptor():
    for text, raw in ((AHR_RECEPTOR_ABSTRACT, "AhR"), (AHR_RECEPTOR_CAPITALS, "AHR"), (AHR_RECEPTOR_ABSTRACT, "AHR")):
        abbreviations = find_abbreviations(text)
        assert not resolve_target(raw, KB, abbreviations).outcome_measure, (raw, text[:40])
        assert resolve_target(raw, KB_WITH_AHR, abbreviations).name == "AHR"
    assert resolve_target("AhR", KB, find_abbreviations(AHR_RECEPTOR_ABSTRACT)).name == "AhR"


def test_undefined_ahr_is_not_a_measure_because_it_is_also_the_genes_symbol():
    """PMID42421242 lists 'PPI-related genes (AHR, DDAH1, MAPT)' and never
    spells AHR out: that AHR is the receptor's gene, a real target."""
    assert "AHR" in load_measures().ambiguous
    resolved = resolve_target("AHR", KB, find_abbreviations(AHR_GENE_UNDEFINED))
    assert not resolved.outcome_measure and resolved.name == "AHR"


def test_mmds_is_a_measure_defined_or_not():
    assert resolve_target("MMDs", KB, find_abbreviations(MMD_DEFINITION)).outcome_measure
    assert resolve_target("MMDs", KB, {}).outcome_measure
    assert resolve_target("MMD", KB, {}).outcome_measure


def test_measures_from_the_stored_reports():
    """Every unit the measurement flagged, across the three stored reports."""
    for name in ("BMI", "FEV1", "MMDs", "AEs", "SAEs", "serious adverse events", "moderate to severe MMDs",
                 "PASI", "HbA1c", "Penh value"):
        assert is_outcome_measure(name, KB), name


def test_a_target_beside_a_measure_is_still_a_target():
    for name in ("AhR", "IgA", "AHR signaling", "leptin-induced weight loss", "IL-33-driven exacerbation",
                 "airway hyperresponsiveness (AHR), elevated serum IgE", "MC4R", "OX40L", "CGRP"):
        assert not is_outcome_measure(name, KB), name


def test_a_papers_own_definition_beats_the_list():
    """PMID38544478 (psoriasis corpus) uses VAS for its Vaseline control."""
    assert not resolve_target("VAS", KB, {"VAS": "Vaseline"}).outcome_measure
    assert resolve_target("VAS", KB, {}).outcome_measure


def test_a_long_form_is_looked_up_whole_never_by_a_name_inside_it():
    """PMID42409961 defines 'OX40 ligand (OX40L)'. 'OX40' inside it is the
    receptor TNFRSF4; the ligand must not be filed under it."""
    abbreviations = {"OX40L": "OX40 ligand"}
    assert resolve_target("OX40L", KB, abbreviations).name == "OX40L"
    # And a long form the table knows still normalises (PMID42264575).
    assert resolve_target("CGRP", KB, {"CGRP": "Calcitonin gene-related peptide"}).name == \
        KB.normalize("CGRP", "target")


# --- through Stage 5 and Stage 6 ------------------------------------------------------------


def _finding(quote: str, target: str, intervention: str | None, disease: str, species: str,
             evidence_type: str = "in_vivo") -> dict:
    return {
        "quote": quote, "target": target, "change": "", "direction": "block", "modality": "small_molecule",
        "intervention": intervention, "disease": disease, "evidence_type": evidence_type, "species": species,
        "effect": "reduced", "outcome": "improved", "setting": "prevention", "effect_detail": "", "n": None,
        "combo_with": None, "target_is_primary": True, "facet": "causal",
    }


def _extract(paper: Paper, section: str, spec, finding: dict, stats: Counter) -> list[EvidenceUnit]:
    reply = {"is_review": False, "findings": [finding]}
    client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
    job = ExtractionJob(paper_id=paper.id, sections=[section], facets=["causal"])
    return extract_paper(paper, job, spec, FACETS, client, model="m", knowledge=KB, stats=stats)


def test_stage5_flags_airway_ahr_and_keeps_the_finding():
    paper = Paper(id="PMID36077141", title="Efficacy Comparison of LPA2 Antagonist H2L5186303 and Agonist "
                  "GRI977143 on Ovalbumin-Induced Allergic Asthma in BALB/c Mice.",
                  abstract="", sections={"results": AHR_DEFINITION + "\n\n" + AHR_RESULTS})
    stats: Counter = Counter()
    units = _extract(paper, "results", ASTHMA, _finding(E248_QUOTE, "AHR", "GRI977143", "asthma", "mice"), stats)
    assert len(units) == 1, "the result is kept"
    assert units[0].outcome_measure and units[0].target == "AHR" and units[0].target_raw == "AHR"
    assert stats["target_outcome_measure"] == 1


def test_stage5_flags_mmds():
    paper = Paper(id="PMID40542538", title="Efficacy and safety of rimegepant for the preventive treatment of "
                  "migraine in Japan: A double-blind, randomized controlled trial.", abstract="",
                  sections={"other": MMD_DEFINITION, "results": E78_QUOTE})
    stats: Counter = Counter()
    units = _extract(paper, "results", MIGRAINE,
                     _finding(E78_QUOTE, "MMDs", "rimegepant", "migraine", "participants", "human_clinical"), stats)
    assert len(units) == 1 and units[0].outcome_measure and units[0].target == "MMDs"


def test_stage5_leaves_the_receptor_alone():
    paper = Paper(id="PMID41596081", title="Epithelial AhR Suppresses Allergen-Induced Oxidative Stress and "
                  "Senescence via c-Myc Regulation.", abstract=AHR_RECEPTOR_ABSTRACT,
                  sections={"abstract": AHR_RECEPTOR_ABSTRACT})
    quote = ("It was documented that the regulation of AhR on senescence was attenuated by VAF347, whereas AhR "
             "deficiency exacerbated ROS generation and inflammation in AhRΔScgb1a1 mice.")
    units = _extract(paper, "abstract", ASTHMA, _finding(quote, "AhR", None, "asthma", "mice"), Counter())
    assert len(units) == 1 and not units[0].outcome_measure and units[0].target == "AhR"


def _unit(uid: str, target: str, paper: str, *, measure: bool = False) -> EvidenceUnit:
    return EvidenceUnit(
        id=uid, paper_id=paper, question_id=ASTHMA.id, target=target, target_raw=target, direction="block",
        modality="small_molecule", intervention=None, disease="asthma", disease_raw="asthma",
        evidence_type="in_vivo", species="mouse", species_raw="mice", outcome="improved", effect_detail="",
        n=None, combo_with=None, target_is_primary=True, quote="q", quote_status="exact", section="results",
        extracted_by_model="fake", prompt_version="1", outcome_measure=measure,
    )


def test_stage6_gives_a_measure_no_row_and_never_merges_it_with_the_receptor():
    """The stored asthma report grouped 'AHR' with 'AhR' by folded spelling:
    the targets:AhR row held 15 units, 2 of them airway hyperresponsiveness
    (E59, E248). Flagged, they stay in the evidence and leave the row."""
    units = [
        _unit("u1", "AhR", "PMID41596081"), _unit("u2", "AhR", "PMID41596081"),
        _unit("u3", "AHR", "PMID36077141", measure=True), _unit("u4", "AHR", "PMID36077141", measure=True),
        _unit("u5", "MMDs", "PMID40542538", measure=True),
    ]
    summary = summarize(units, ASTHMA, [], [])
    assert [t.target for t in summary.targets] == ["AhR"]
    assert summary.targets[0].unit_ids == ["u1", "u2"]
    assert summary.targets[0].names_seen == ["AhR"]
    assert len(summary.units) == 5, "every finding is still evidence"


# --- stored reports keep their bytes -------------------------------------------------------


def test_a_unit_without_the_flag_serialises_without_the_key():
    """A stored run is verified by re-serialising its report and comparing
    bytes (`check_pinned`), so the new field must not appear on old units."""
    plain = _unit("u1", "AhR", "PMID41596081")
    assert "outcome_measure" not in plain.model_dump_json()
    flagged = _unit("u2", "AHR", "PMID36077141", measure=True)
    assert '"outcome_measure":true' in flagged.model_dump_json()
    assert EvidenceUnit.model_validate_json(flagged.model_dump_json()) == flagged


def test_a_stored_report_round_trips_byte_for_byte():
    """What `check_pinned` does: re-serialise with `to_json`, compare bytes.
    The stored report here is written as an old run wrote it - its unit has
    no `outcome_measure` key."""
    from bedrock.litreview.models import Report
    from bedrock.litreview.stages.s8_render import to_json

    old_unit = json.loads(_unit("u1", "AHR", "PMID36077141").model_dump_json())
    assert "outcome_measure" not in old_unit
    stored = Report(question_id="q", question="?", sections=[]).model_dump(mode="json")
    stored["evidence"] = [old_unit]
    original = to_json(Report.model_validate(stored))
    assert json.loads(original)["evidence"][0] == old_unit
    assert to_json(Report.model_validate_json(original)) == original
