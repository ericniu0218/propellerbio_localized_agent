"""Tests for the code-side normalisation.

These exist because normalisation is what makes the pipeline deterministic.
A lookup table gives the same answer every time; a model does not, and the
client's loudest complaint is inconsistency.
"""

from bedrock.litreview.extract.normalize import (
    extract_n,
    infer_evidence_type,
    normalize_species,
    normalize_target,
)

ALIASES = {"IL23A": ["IL-23", "interleukin-23", "p19", "IL-23p19"], "IL17A": ["IL-17", "IL-17A"]}


def test_species_variants_collapse():
    assert normalize_species("C57BL/6 mice") == "mouse"
    assert normalize_species("male Sprague-Dawley rats") == "rat"
    assert normalize_species("healthy human volunteers") == "human"
    assert normalize_species("cynomolgus monkeys") == "cyno"


def test_a_human_derived_cell_line_is_not_a_human_subject():
    """The ordering bug this guards against is subtle and expensive: a human
    cell line read as a human subject turns in-vitro evidence into clinical
    evidence, which is the single most misleading error this system can make."""
    assert normalize_species("HEK293 cells derived from human tissue") == "cell_line"
    assert normalize_species("primary human keratinocytes in culture") == "cell_line"


def test_unknown_species_falls_back_rather_than_crashing():
    assert normalize_species("axolotl") == "other"
    assert normalize_species("") == "other"


def test_target_aliases_resolve_to_one_symbol():
    for written in ("IL-23", "il 23", "interleukin-23", "p19", "IL_23"):
        assert normalize_target(written, ALIASES) == "IL23A"


def test_an_unknown_target_is_kept_verbatim():
    """Papers mention molecules outside the project all the time. Dropping the
    value would lose information; Stage 6 filters on names it knows."""
    assert normalize_target("TNF-alpha", ALIASES) == "TNF-alpha"


def test_sample_size_is_pulled_out_of_a_sentence():
    assert extract_n("...versus 8% receiving placebo (n=507).") == 507
    assert extract_n("A total of n = 1,204 patients enrolled.") == 1204
    assert extract_n("No sample size was stated.") is None


def test_a_mouse_study_labelled_clinical_is_corrected():
    """The most common extraction error there is, and the reason the gold set
    deliberately over-samples animal studies."""
    assert infer_evidence_type("mouse", "human_clinical") == "in_vivo"


def test_a_cell_line_is_always_in_vitro():
    assert infer_evidence_type("cell_line", "in_vivo") == "in_vitro"


def test_a_correct_label_is_left_alone():
    assert infer_evidence_type("human", "human_clinical") == "human_clinical"
    assert infer_evidence_type("mouse", "in_vivo") == "in_vivo"
