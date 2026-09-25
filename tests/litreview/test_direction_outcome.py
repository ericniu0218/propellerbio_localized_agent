"""Direction and outcome: the model labels them, and code checks the labels
against the words the model copied from its own quote.

The failure these pin down, from the MedGemma 4B benchmark run: "PGGT1B
deficiency aggravated the psoriasis-like lesions" labelled direction
`unclear`, outcome `improved`. A report built on that says knocking out
PGGT1B helps, which is the opposite of the paper.
"""

from bedrock.litreview.extract.normalize import classify_direction, classify_outcome
from bedrock.litreview.models import EvidenceUnit, ExtractedFinding
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s6_score import summarize
from bedrock.litreview.stages.s5_extract import reconcile


def finding(**kw) -> ExtractedFinding:
    base = dict(
        quote="Bone marrow PGGT1B deficiency aggravated the psoriasis-like lesions induced by imiquimod in mice.",
        target="PGGT1B", change="deficiency", direction="unclear", modality="genetic", intervention=None,
        disease="psoriasis-like lesions", evidence_type="in_vivo", species="mice", effect="aggravated",
        outcome="improved", setting="unclear", effect_detail="aggravated lesions", n=None, combo_with=None,
        target_is_primary=True, facet="causal",
    )
    base.update(kw)
    return ExtractedFinding(**base)


def test_an_aggravating_knockout_is_block_and_worsened():
    direction, outcome, _, notes = reconcile(finding())
    assert (direction, outcome) == ("block", "worsened")
    assert len(notes) == 2


def test_negation_is_read_before_the_verb_it_negates():
    assert classify_outcome("did not significantly reduce") == "no_effect"
    assert classify_outcome("failed to prevent") == "no_effect"
    assert classify_outcome("did not develop lesions") is None, "not developing disease is not a null result"


def test_words_that_flip_polarity_leave_the_model_in_charge():
    assert classify_outcome("increased survival") is None
    assert classify_outcome("lower survival") is None
    assert classify_outcome("increased neutralizing antibody titers") is None


def test_words_not_in_the_quote_are_ignored():
    direction, outcome, _, notes = reconcile(finding(change="knockout", effect="exacerbated"))
    assert (direction, outcome, notes) == ("unclear", "improved", [])


def test_an_observation_stays_an_association():
    """'IL-17A was increased in lesions' changed nothing, so it worsened nothing."""
    f = finding(quote="IL-17A expression was increased in psoriatic lesions.", target="IL-17A", change="",
                direction="unclear", effect="increased", outcome="association")
    assert reconcile(f)[:2] == ("unclear", "association")


def test_directions_from_copied_words():
    assert classify_direction("knockout") == "block"
    assert classify_direction("anti-IL-17A antibody") == "block"
    assert classify_direction("overexpression") == "activate"
    assert classify_direction("treatment with") is None


def test_a_cell_line_is_not_a_target():
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    unit = EvidenceUnit(
        id="u1", paper_id="P1", question_id=spec.id, target="HaCaT cells", target_raw="HaCaT cells",
        direction="block", modality="genetic", disease="psoriasis", disease_raw="psoriasis",
        evidence_type="in_vitro", species="human", species_raw="HaCaT cells", outcome="improved",
        effect_detail="x", n=None, combo_with=None, target_is_primary=True, quote="q", quote_status="exact",
        section="results",
        extracted_by_model="fake", prompt_version="3",
    )
    summary = summarize([unit], spec, [], [])
    assert summary.targets == [] and len(summary.units) == 1


def test_a_mouse_result_is_not_filed_as_clinical_evidence():
    from bedrock.litreview.stages.s5_extract import settle_facet

    ids = ["mechanism", "causal", "clinical", "prevention", "negative", "safety"]
    common = dict(direction="block", outcome="improved", setting="unclear", facet_ids=ids)
    assert settle_facet("clinical", evidence_type="in_vivo", **common) == "causal"
    assert settle_facet("clinical", evidence_type="human_clinical", **common) == "clinical"
    assert settle_facet("prevention", evidence_type="in_vivo", **common) == "causal", "not given before onset"
    assert settle_facet("negative", evidence_type="in_vivo", **common) == "causal", "it improved"
    assert settle_facet("causal", evidence_type="in_vivo", **{**common, "direction": "unclear"}) == "mechanism"


def test_words_that_name_nothing_to_act_on_are_not_targets():
    """From the MedGemma run: a statistics paper gave 'dropout',
    'discontinuation' and 'participants' target rows."""
    from bedrock.litreview.knowledge import looks_like_target

    for name in ("dropout", "discontinuation", "participants", "inflammation"):
        assert not looks_like_target(name), name
    # Measurements read as targets unless code says otherwise, and a name
    # this long is a clause copied out of a results sentence.
    for name in ("RSVPreF3-specific CD4+ T-cell frequencies", "RSV-A/RSV-B geometric mean titer ratios",
                 "markers of keratinocyte activation, epidermal thickness, KRT16 and Ki-67 expression"):
        assert not looks_like_target(name), name
    assert looks_like_target("IL-17A, IL-17F, and IL-12B mRNA"), "a list of real targets is still a target"
    for name in ("IL-17A", "NF-κB c-Rel", "UBE2L3", "HuR", "mast cells", "nirsevimab", "prefusion F protein"):
        assert looks_like_target(name), name


def test_one_target_spelled_two_ways_is_one_row():
    """The RSV sample report listed 'nirsevimab' and 'Nirsevimab' separately,
    which reads as two independent lines of evidence for one drug."""
    from bedrock.litreview.stages.s0_question import parse_question
    from bedrock.litreview.stages.s6_score import summarize

    spec = parse_question("What should I target to prevent RSV infection in infants?")

    def unit(uid, target, paper):
        return EvidenceUnit(
            id=uid, paper_id=paper, question_id=spec.id, target=target, target_raw=target, direction="block",
            modality="antibody", intervention="nirsevimab", disease="RSV infection", disease_raw="RSV",
            evidence_type="human_clinical", species="human", species_raw="infants", outcome="improved",
            effect_detail="efficacy", n=None, combo_with=None, target_is_primary=True,
            quote="Nirsevimab prevented RSV infection in infants.", quote_status="exact", section="abstract",
            extracted_by_model="fake", prompt_version="4",
        )

    summary = summarize([unit("u1", "Nirsevimab", "P1"), unit("u2", "nirsevimab", "P2")], spec, [], [])
    assert [t.target for t in summary.targets] == ["Nirsevimab"]
    assert summary.targets[0].paper_ids == ["P1", "P2"], "both papers count towards the one target"


def test_prophylaxis_is_prevention_whatever_the_model_labelled_it():
    """MedGemma labelled all 74 RSV findings 'unclear' or 'treatment',
    including nirsevimab prophylaxis, so a prevention question produced a
    report with an empty prevention section."""
    f = finding(
        quote="Nirsevimab, a long-acting MoAb, has shown high efficacy in RSV prophylaxis in newborns.",
        target="RSV F protein", change="", direction="block", effect="high efficacy", outcome="improved",
        setting="treatment", effect_detail="high efficacy in RSV prophylaxis",
    )
    direction, outcome, setting, notes = reconcile(f)
    assert setting == "prevention" and any("setting" in n for n in notes)
    treated = finding(quote="Guselkumab cleared established plaques in treated patients.", change="guselkumab",
                      effect="cleared", outcome="improved", setting="treatment", effect_detail="PASI-90")
    assert reconcile(treated)[2] == "treatment", "treating established disease is not prevention"
