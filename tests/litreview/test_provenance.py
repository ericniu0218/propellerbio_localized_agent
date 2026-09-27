"""Every field the model writes is found in the text it read, or blanked.

A verified quote proves the sentence is in the paper. These tests are for the
fields beside it — target, sample size, who was studied, the result summary —
which a model can fill from memory while copying a real sentence.
"""

import json

import pytest

from bedrock.litreview.extract.provenance import trace
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import ExtractionJob, Facet, Paper
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import extract_paper

KB = load_knowledge()

RESULTS = (
    "Mice lacking IL-23 were protected from imiquimod-induced skin inflammation.\n\n"
    "Epidermal thickness was reduced by 45% in treated animals compared with controls."
)
METHODS = "Female C57BL/6 mice (n = 12 per group) were treated daily with imiquimod cream."


def _trace(**kw):
    base = dict(
        quote="Epidermal thickness was reduced by 45% in treated animals compared with controls.",
        chunk=RESULTS, methods=METHODS, section="results", section_text=RESULTS,
        methods_section_text=METHODS, target="IL-23", intervention=None, combo_with=None,
        disease_raw="imiquimod-induced skin inflammation", species_raw="C57BL/6 mice", n=12,
        effect_detail="epidermal thickness reduced 45%", kb=KB,
    )
    base.update(kw)
    return trace(**base)


def test_subjects_and_sample_size_come_from_an_anchored_methods_sentence():
    p = _trace()
    # IL-23 is not in the quote: it is shown by the passage's sentence that
    # names it (`naming_quote`), and provenance says so.
    assert p.where["target"] == "naming" and p.where["naming"] == "chunk"
    assert p.naming is not None and RESULTS[p.naming.start:p.naming.end] == p.naming.text
    assert p.naming.text.startswith("Mice lacking IL-23")
    assert p.where["species"] == "methods" and p.where["n"] == "methods"
    assert p.context is not None and p.context.text == METHODS
    assert METHODS[p.context.start:p.context.end] == p.context.text
    assert p.values["species_raw"] == "C57BL/6 mice" and p.values["n"] == 12


def test_a_sample_size_the_text_never_states_is_blanked():
    p = _trace(n=40)
    assert p.where["n"] == "none" and p.values["n"] is None


def test_a_species_the_text_never_states_is_blanked():
    p = _trace(species_raw="psoriasis patients")
    assert p.where["species"] == "none" and p.values["species_raw"] == ""


def test_a_result_summary_with_a_number_not_in_the_text_is_blanked():
    p = _trace(effect_detail="epidermal thickness reduced 60%")
    assert p.where["effect_detail"] == "none" and p.values["effect_detail"] == ""


def test_a_target_the_text_never_names_drops_the_finding():
    assert _trace(target="TNF").drop == "target_unlocated"


def test_an_agent_the_text_never_names_is_blanked():
    p = _trace(intervention="guselkumab")
    assert p.where["intervention"] == "none" and p.values["intervention"] is None


def test_extraction_stores_the_papers_text_and_its_trace():
    """End to end through Stage 5 with a model that copies a real sentence
    from the middle, and invents a sample size and an agent beside it."""
    paper = Paper(id="P1", title="IL-23 in psoriasis", abstract="IL-23 drives psoriasis.",
                  sections={"abstract": "IL-23 drives psoriasis.", "results": RESULTS, "methods": METHODS})
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    facets = [Facet(id="causal", title="Causal", need="what changing the target does")]
    reply = {"is_review": False, "findings": [{
        "quote": "thickness was reduced by 45% in treated animals",
        "target": "IL-23", "change": "", "direction": "block", "modality": "antibody",
        "intervention": "risankizumab", "disease": "psoriasis", "evidence_type": "in_vivo",
        "species": "C57BL/6 mice", "effect": "reduced", "outcome": "improved", "setting": "treatment",
        "effect_detail": "thickness reduced 45% (n = 30)", "n": 30, "combo_with": None,
        "target_is_primary": True, "facet": "causal",
    }]}
    client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
    job = ExtractionJob(paper_id="P1", sections=["results"], facets=["causal"])
    units = extract_paper(paper, job, spec, facets, client, model="m", knowledge=KB)
    assert len(units) == 1
    u = units[0]
    assert u.quote == "Epidermal thickness was reduced by 45% in treated animals compared with controls."
    assert RESULTS[u.quote_start:u.quote_end] == u.quote
    assert u.quote_model == "thickness was reduced by 45% in treated animals"
    assert u.n is None and u.intervention is None and u.effect_detail == ""
    assert u.species_raw == "C57BL/6 mice" and u.context_quote == METHODS
    assert u.provenance["n"] == "none" and u.provenance["species"] == "methods"


# --- found in review ------------------------------------------------------------------


def test_a_human_species_is_not_located_in_a_sentence_about_pregnant_mice():
    p = _trace(species_raw="human", methods="Pregnant C57BL/6 mice (adult females) were treated with imiquimod.",
               methods_section_text="Pregnant C57BL/6 mice (adult females) were treated with imiquimod.")
    assert p.where["species"] == "none"


def test_a_cell_line_is_not_located_by_a_bare_cells():
    p = _trace(species_raw="HaCaT cells", quote="CD4+ T cells rose in patients after treatment with the drug.",
               chunk="CD4+ T cells rose in patients after treatment with the drug.", target="CD4",
               methods="", methods_section_text="")
    assert p.where["species"] == "none"


def test_a_percentage_is_not_a_sample_size():
    p = _trace(n=45)  # the quote says "reduced by 45%"
    assert p.where["n"] == "none" and p.values["n"] is None


def test_two_findings_from_one_sentence_keep_distinct_ids():
    sentence = "Anti-IL-23 reduced epidermal thickness by 45% and IL-17A mRNA by 60% in treated mice."
    paper = Paper(id="P2", title="t", abstract="a", sections={"abstract": "a", "results": sentence})
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    facets = [Facet(id="causal", title="Causal", need="what changing the target does")]

    def finding(quote, detail):
        return {"quote": quote, "target": "IL-23", "change": "", "direction": "block", "modality": "antibody",
                "intervention": None, "disease": "psoriasis", "evidence_type": "in_vivo", "species": "mice",
                "effect": "reduced", "outcome": "improved", "setting": "treatment", "effect_detail": detail,
                "n": None, "combo_with": None, "target_is_primary": True, "facet": "causal"}

    reply = {"is_review": False, "findings": [
        finding("Anti-IL-23 reduced epidermal thickness by 45%", "thickness 45%"),
        finding("and IL-17A mRNA by 60% in treated mice", "mRNA 60%"),
    ]}
    client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
    job = ExtractionJob(paper_id="P2", sections=["results"], facets=["causal"])
    units = extract_paper(paper, job, spec, facets, client, model="m", knowledge=KB)
    assert len(units) == 2 and len({u.id for u in units}) == 2


# --- found in the second review ---------------------------------------------------------


def test_a_name_inside_a_longer_name_is_not_found():
    """Exact means exact: IL-1 is not IL-17A, RSV is not bovine RSV."""
    from bedrock.litreview.extract.provenance import name_in

    assert not name_in("IL-1", "an anti-IL-17A antibody", KB, "target")
    assert not name_in("RSV", "calves infected with bRSV", KB)
    assert not name_in("TNF", "TNFR2 deficiency", KB, "target")
    assert name_in("IL-23", "interleukin-23 blockade", KB, "target")


def test_a_normalised_name_is_found_through_its_listed_names():
    """combo_with is stored canonical ("RSV F"); the paper says "fusion protein"."""
    from bedrock.litreview.extract.provenance import name_in

    assert name_in(KB.normalize("fusion protein", "target"), "the fusion protein binds nirsevimab", KB)


def test_a_model_label_is_weakened_by_its_own_species_but_never_strengthened():
    paper = Paper(id="P3", title="t", abstract="a", sections={
        "abstract": "a", "results": "IL-23 blockade reduced skin thickness in treated animals by day 7."})
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    facets = [Facet(id="causal", title="Causal", need="what changing the target does"),
              Facet(id="clinical", title="Clinical", need="human results")]

    def run(species, evidence_type, effect_detail=""):
        reply = {"is_review": False, "findings": [{
            "quote": "IL-23 blockade reduced skin thickness in treated animals by day 7.", "target": "IL-23",
            "change": "blockade", "direction": "block", "modality": "antibody", "intervention": None,
            "disease": "psoriasis", "evidence_type": evidence_type, "species": species, "effect": "reduced",
            "outcome": "improved", "setting": "treatment", "effect_detail": effect_detail, "n": None,
            "combo_with": None, "target_is_primary": True, "facet": "clinical"}]}
        client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
        job = ExtractionJob(paper_id="P3", sections=["results"], facets=["causal", "clinical"])
        return extract_paper(paper, job, spec, facets, client, model="m", knowledge=KB)[0]

    mouse = run("C57BL/6 mice", "human_clinical")
    assert mouse.species_raw == "" and mouse.evidence_type == "in_vivo" and mouse.facet != "clinical"
    assert run("patients", "in_vivo").evidence_type == "in_vivo", "an unstated human is no upgrade"
    unsupported = run("", "in_vivo", effect_detail="prophylactic dosing prevented disease onset")
    assert unsupported.effect_detail == "" and unsupported.setting != "prevention"


# --- found in the final review ---------------------------------------------------------


def test_a_negated_summary_is_not_a_restatement():
    from bedrock.litreview.extract.provenance import summary_supported

    assert not summary_supported("IL-17A blockade was not effective in mice", "IL-17A blockade was effective in mice")
    assert summary_supported("IL-17A blockade was effective in mice", "IL-17A blockade was effective in mice")


@pytest.mark.parametrize("n, text", [(12, "reduced by 12 in mice"), (40, "fell 40 percent in treated mice")])
def test_a_count_needs_the_thing_it_counts_beside_it(n, text):
    from bedrock.litreview.extract.provenance import n_in

    assert not n_in(n, text)


def test_a_quote_is_placed_in_the_passage_the_model_read():
    """The same words in two paragraphs: the stored sentence is the one from
    the chunk the model was given, qualifiers and all."""
    from bedrock.litreview.stages.s5_extract import _locate_in_chunk

    first = "IL-23 blockade reduced epidermal thickness in wild-type mice."
    second = "In Il17ra knockout mice, IL-23 blockade reduced epidermal thickness in wild-type mice as well."
    section = first + "\n\n" + second
    located = _locate_in_chunk("IL-23 blockade reduced epidermal thickness in wild-type mice", second, section)
    assert located.start >= len(first) and located.text == second
    assert section[located.start:located.end] == located.text


def test_a_model_species_rules_a_finding_out_of_a_species_question_but_never_in():
    paper = Paper(id="P4", title="t", abstract="a", sections={
        "abstract": "a", "results": "IL-23 blockade reduced skin thickness by day 7 in the treated group."})
    from bedrock.litreview.models import QuestionEntity

    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    spec = spec.model_copy(update={"entities": [*spec.entities, QuestionEntity(
        text="mice", kind="species", canonical="mouse", exact_terms=["mice"], required=False)]})
    facets = [Facet(id="causal", title="Causal", need="what changing the target does")]

    def run(species):
        reply = {"is_review": False, "findings": [{
            "quote": "IL-23 blockade reduced skin thickness by day 7 in the treated group.", "target": "IL-23",
            "change": "blockade", "direction": "block", "modality": "antibody", "intervention": None,
            "disease": "psoriasis", "evidence_type": "in_vivo", "species": species, "effect": "reduced",
            "outcome": "improved", "setting": "treatment", "effect_detail": "", "n": None,
            "combo_with": None, "target_is_primary": True, "facet": "causal"}]}
        client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
        job = ExtractionJob(paper_id="P4", sections=["results"], facets=["causal"])
        return extract_paper(paper, job, spec, facets, client, model="m", knowledge=KB)

    assert run("patients") == [], "the model said patients; this is a mouse question"
    kept = run("unclear")
    assert len(kept) == 1 and kept[0].species_raw == ""


# --- a summary that drops the paper's negation (review 2026-09-24, finding 2) ---------


NEGATED = "Treatment did not significantly reduce inflammation in mice."


@pytest.mark.parametrize("summary, evidence", [
    ("Treatment did significantly reduce inflammation in mice.", NEGATED),
    ("Anti-IL-23 reduced ear thickening.", "Anti-IL-23 did not reduce ear thickening compared with vehicle."),
    ("Treatment did reduce inflammation in mice.", "Treatment didn't reduce inflammation in mice."),
    ("There was a significant difference in AE types and frequencies between groups.",
     "There was no significant difference in AE types and frequencies between groups."),
    ("Anti-IL-23 did significantly reduce ear thickening.",
     "Anti-IL-23 did not affect body weight and did not significantly reduce ear thickening."),
    ("Treatment did not affect body weight and did significantly reduce inflammation in mice.",
     "Treatment did not affect body weight and did not significantly reduce inflammation in mice."),
    ("Treatment did not significantly increase IL-17 and did significantly reduce inflammation in mice.",
     "Treatment did not significantly increase IL-17 and did not significantly reduce inflammation in mice."),
], ids=["dropped-not", "dropped-not-new-verb-form", "dropped-contraction", "dropped-no-difference",
        "dropped-second-negation", "kept-one-negation-dropped-the-other", "kept-a-different-negation"])
def test_a_summary_that_drops_the_papers_negation_is_not_supported(summary, evidence):
    """A null result restated without its negation reads as the opposite
    finding, and every word it uses is still in the paper."""
    from bedrock.litreview.extract.provenance import text_supported

    assert not text_supported(summary, evidence, KB)


@pytest.mark.parametrize("summary, evidence", [
    (NEGATED, NEGATED),
    ("X reduced lesion area.", "X did not affect body weight, but reduced lesion area."),
    ("PASI-75 was achieved by 78% of patients.",
     "Patients did not receive corticosteroids. PASI-75 was achieved by 78% of patients."),
], ids=["faithful-negation", "negation-governs-another-clause", "negation-in-another-sentence"])
def test_a_negation_the_summary_keeps_or_is_not_about_does_not_block_it(summary, evidence):
    from bedrock.litreview.extract.provenance import text_supported

    assert text_supported(summary, evidence, KB)


def test_extraction_blanks_a_summary_that_dropped_the_negation_and_keeps_the_finding():
    quote = "IL-23 blockade did not significantly reduce inflammation in mice."
    text = quote + "\n\nEpidermal thickness was measured daily."
    kw = dict(quote=quote, chunk=text, section_text=text)
    flipped = _trace(effect_detail="IL-23 blockade did significantly reduce inflammation in mice.", **kw)
    assert flipped.drop is None, "the finding and its verified quote stay"
    assert flipped.where["effect_detail"] == "none" and flipped.values["effect_detail"] == ""
    faithful = _trace(effect_detail="IL-23 blockade did not significantly reduce inflammation in mice.", **kw)
    assert faithful.where["effect_detail"] != "none"


@pytest.mark.parametrize("summary, evidence", [
    ("Treatment reduced ear lesions in IL-23R-deficient mice.",
     "Treatment reduced ear lesions in wild-type mice but not in IL-23R-deficient mice after imiquimod."),
    ("Weight loss with the GLP-1 agonist occurred with changes in daily food intake.",
     "Weight loss with the GLP-1 agonist occurred without changes in daily food intake in obese mice."),
    ("IL-36 gamma was strongly downregulated in lesional skin.",
     "Expression of IL-36 gamma was strongly upregulated in lesional skin of patients with psoriasis."),
    ("IL-17 deficiency altered epidermal thickness.",
     "Neither IL-17 nor IL-22 deficiency altered epidermal thickness in imiquimod-treated mice."),
], ids=["dropped-but-not-in", "dropped-without", "flipped-up-down", "dropped-neither"])
def test_a_summary_that_drops_a_scope_or_flips_a_direction_is_not_supported(summary, evidence):
    """Found by an adversarial probe: each passed as a restatement before."""
    from bedrock.litreview.extract.provenance import text_supported

    assert not text_supported(summary, evidence, KB)


@pytest.mark.parametrize("summary, evidence", [
    ("Treatment reduced ear lesions in wild-type mice.",
     "Treatment reduced ear lesions in wild-type mice but not in IL-23R-deficient mice after imiquimod."),
    ("Weight loss occurred without changes in daily food intake.",
     "Weight loss with the GLP-1 agonist occurred without changes in daily food intake in obese mice."),
    ("Lesion scores improved in patients with psoriasis.",
     "Lesion scores improved in patients with or without psoriasis of the scalp."),
], ids=["kept-the-positive-half", "kept-the-without", "with-or-without-negates-nothing"])
def test_a_faithful_summary_beside_the_new_cues_is_still_supported(summary, evidence):
    from bedrock.litreview.extract.provenance import text_supported

    assert text_supported(summary, evidence, KB)
