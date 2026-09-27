"""Issue #9: the model's evidence-type and outcome labels were sometimes wrong,
and the report used them.

From the qwen3:8b hand audit (evals/litreview/hand_audit/):

- obesity S34, E29: "Western blotting revealed a significant decrease in
  Sirt6 protein levels in mast cells from individuals with obesity" was
  labelled human_genetic. It is human tissue; nothing about it is genetic.
- asthma S13, E220: "HBECs were exposed to CRE ... EN4 effectively suppressed
  CRE-induced c-Myc expression" was labelled in_vivo, species "mice" - the
  mice were in the context sentence. The quote is cells in a dish.
- obesity S3, E106: "similar increases in hypothalamic Sik3 mRNA levels in
  ... db/db mice" was labelled worsened. Nothing was done to SIK3; it is an
  association, and a worsened finding reads as a negative result.

Now the quote's own words about what was studied set the evidence type
(`extract/normalize.py` `system_in_quote`, `stages/s5_extract.py`
`settle_evidence_type`), and an improved or worsened finding with nothing
done to its target is an association (`reconcile`). Every correction is
written on the unit and counted.

Every quote and context sentence below is verbatim from the stored qwen3:8b
reports and the searcher corpora; the model's labels are the ones it gave.
"""

import json
from collections import Counter

from bedrock.litreview.extract.normalize import system_in_quote, target_changed_in
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import ExtractedFinding, ExtractionJob, Facet, Paper
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import extract_paper, reconcile, settle_evidence_type, settle_facet

KB = load_knowledge()
ASTHMA = parse_question("If I want to prevent asthma exacerbations, what should I target?")
OBESITY = parse_question("If I want to treat obesity, which targets should I pursue?")
FACETS = [
    Facet(id="causal", title="Causal", need="what changing the target does"),
    Facet(id="mechanism", title="Mechanism", need="how the target is involved"),
    Facet(id="negative", title="Negative", need="what did not work or made it worse"),
]

# --- the three audit cases ----------------------------------------------------------

# PMID41495031 (obesity corpus), the Results paragraph around E29, verbatim.
E29_QUOTE = ("Western blotting revealed a significant decrease in Sirt6 protein levels in mast cells from "
             "individuals with obesity (Fig. ).")
E29_PARAGRAPH = (
    "We and other researchers have reported that Sirt6 expression in adipocytes and macrophages is lower in "
    "humans and mice with obesity compared to lean controls – . Therefore, we investigated whether Sirt6 is also "
    "repressed in mast cells within obese adipose tissue. Immunostaining of human omental adipose tissue sections "
    "confirmed that individuals with obesity (BMI ≥ 30) had a higher number of total and degranulated mast cells "
    "compared to individuals with a BMI < 25 (Fig. and Supplementary Fig. ), consistent with the previous "
    "reports , . " + E29_QUOTE + " Moreover, Sirt6 protein levels in mast cells from the omental fat depot were "
    "negatively correlated with BMI and plasma leptin levels, while plasma adiponectin levels remained unchanged "
    "(Fig. )."
)

# PMID41596081 (asthma corpus): the Results paragraph holding E220's context
# sentence, then the paragraph holding its quote, both verbatim.
E220_CONTEXT = ("The VAF347-induced downregulation of c-Myc was confirmed in vivo: lung tissues from CRE-exposed "
                "mice treated with VAF347 showed decreased c-Myc expression compared with controls (C,D).")
E220_QUOTE = (
    "Given that c-Myc emerged as a downstream target of AhR, we hypothesized that it contributes to "
    "allergen-induced cellular senescence. HBECs were exposed to CRE (50 μg/mL, 24 h) with or without the c-Myc "
    "inhibitor EN4 (50 μM). Immunostaining revealed that EN4 effectively suppressed CRE-induced c-Myc expression "
    "(A,B)."
)
E220_RESULTS = (
    "To examine whether AhR regulates c-Myc expression, HBECs were exposed to CRE (50 μg/mL, 24 h) in the "
    "presence or absence of the AhR agonist VAF347 (20 μM) or antagonist CH223191 (10 μM). c-Myc expression was "
    "evaluated by immunostaining (A) and quantified (B). CRE exposure increased c-Myc expression, which was "
    "further elevated by CH223191 and inhibited by VAF347. " + E220_CONTEXT + "\n\n"
    + E220_QUOTE + " Notably, EN4 treatment reduced CRE-induced senescence, as evidenced by decreased SA-β-Gal "
    "activity and reduced expression of senescence markers Cdkn2a (p16), Cdkn1a (p21), and γ-H2AX."
)

# PMID42230648 (obesity corpus), E108 and E106, consecutive Results sentences.
E108_QUOTE = ("Hypothalamic Sik3 mRNA levels were significantly increased after 10 weeks of high-fat diet (HFD) "
              "feeding in mice compared to their lean chow-fed counterparts (Fig. ).")
E106_QUOTE = ("Consistently, we observed similar increases in hypothalamic Sik3 mRNA levels in 12-week-old "
              "genetically induced leptin receptor-deficient ( db/db ) mice when compared to their lean ( db/+ ) "
              "counterparts (Fig. ).")
SIK3_RESULTS = (
    "To establish the role of the SIK3 signaling pathway in obesity development, we first assessed whole "
    "hypothalamic Sik3 mRNA expression in both diet-induced and genetically induced obese mouse models. "
    + E108_QUOTE + " " + E106_QUOTE
)

# --- negatives: labels that were right and must stay ---------------------------------

# PMID37773161: a knockout that worsened disease, in mice.
E206_QUOTE = ("Assessment of glucose homeostasis in these animals revealed that the TMEM135 knockout results in "
              "glucose intolerance, hyperinsulinemia and insulin resistance (Fig. ).")
# PMID37773161: a mouse protein expressed in a human cell line.
E124_QUOTE = (
    "To assess the functional impact of this variant on mitochondrial fission, we ectopically expressed mouse WT "
    "or G369E mutant in TMEM135 knockout hBAT cells and determined the effect on mitochondrial morphology. "
    "Western blot analysis confirmed that the WT and G369E mutant TMEM135 were equally expressed in the knockout "
    "cells (Fig. )."
)
# PMID42103155 and PMID29031731: human genetic studies.
E154_QUOTE = ("Human genetic studies have identified GPR75 loss-of-function variants to be strongly protective "
              "against obesity, establishing GPR75 as a compelling therapeutic target.")
E5_QUOTE = ("We found that MC4R variant homozygotes were significantly more obese than heterozygotes (p < 0.001); "
            "there was no significant difference in BMI between those carrying heterozygous complete vs partial "
            "LOF variants (E).")
# PMID29031731: a clinical trial whose population is defined by a gene.
E214_QUOTE = "In a 28-day Phase 1b clinical trial, Setmelanotide led to weight loss in obese MC4R variant carriers."
# PMID41858165 and PMID35707726: people's cells in a dish, named for who gave them.
E229_QUOTE = (
    "We found that combined treatment with estrogen and GPER‐1 blockade (G36) reduced the number of "
    "IFN‐γ‐producing MAIT cells compared to estrogen treatment alone. This effect was only detected, however, in "
    "5‐OP‐RU ligand‐activated MAIT cells from healthy individuals (Figure D)."
)
E183_QUOTE = ("PBMCs from asthmatic children showed increased IL-3 secretion, which directly correlated with "
              "improved lung function.")
# PMID42327952: an animal model and cells, in one sentence.
E91_QUOTE = (
    "Experiments confirmed that in both the HDM asthma model and the in vitro cell models, lactate accumulation "
    "induced a significant increase in global protein lysine lactylation (pan-Kla), and this modification was "
    "predominantly enriched in the nucleus (Fig. N and O and Fig. E and F)."
)
# PMID36262478: a drug, whose direction on its target the model left unclear.
E54_QUOTE = ("Rimegepant, a second-generation gepant, has shown efficacy in several clinical trials in treating "
             "acute migraine.")


def _finding(quote: str, target: str, **kw) -> ExtractedFinding:
    base = dict(
        quote=quote, target=target, change="", direction="unclear", modality="other", intervention=None,
        disease="obesity", evidence_type="in_vivo", species="mice", effect="", outcome="association",
        setting="unclear", effect_detail="", n=None, combo_with=None, target_is_primary=True, facet="mechanism",
    )
    base.update(kw)
    return ExtractedFinding(**base)


def _settle(finding: ExtractedFinding, *, species: str, species_raw: str = "", context: str = "") -> tuple:
    return settle_evidence_type(
        finding.evidence_type, species=species, species_raw=species_raw, model_species=finding.species,
        quote=finding.quote, context=context,
    )


def _extract(paper: Paper, spec, finding: ExtractedFinding, stats: Counter):
    reply = {"is_review": False, "findings": [finding.model_dump()]}
    client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
    job = ExtractionJob(paper_id=paper.id, sections=["results"], facets=[f.id for f in FACETS])
    units = extract_paper(paper, job, spec, FACETS, client, model="m", knowledge=KB, stats=stats)
    assert len(units) == 1
    return units[0]


# --- E29: human tissue is not human genetic -----------------------------------------


def test_e29_human_tissue_is_human_clinical_not_human_genetic():
    f = _finding(E29_QUOTE, "Sirt6", evidence_type="human_genetic", species="humans")
    assert system_in_quote(E29_QUOTE, "human_genetic") == ("human_clinical", "individuals")
    evidence_type, note = _settle(f, species="human", species_raw="humans")
    assert evidence_type == "human_clinical"
    assert note == 'evidence_type: human_genetic -> human_clinical ("individuals")'


def test_e29_in_the_pipeline():
    paper = Paper(id="PMID41495031", title="Sirt6 in mast cells", abstract="",
                  sections={"results": E29_PARAGRAPH})
    stats = Counter()
    u = _extract(paper, OBESITY, _finding(E29_QUOTE, "Sirt6", change="significant decrease",
                                          evidence_type="human_genetic", species="humans",
                                          effect="significant decrease"), stats)
    assert (u.quote, u.evidence_type, u.outcome) == (E29_QUOTE, "human_clinical", "association")
    assert u.corrected == ['evidence_type: human_genetic -> human_clinical ("individuals")']
    assert stats["evidence_type_corrected"] == 1


# --- E220: cells in a dish, though the context sentence names mice -------------------


def test_e220_cells_in_a_dish_are_in_vitro_though_the_context_names_mice():
    paper = Paper(id="PMID41596081", title="Epithelial AhR and c-Myc", abstract="",
                  sections={"results": E220_RESULTS})
    stats = Counter()
    u = _extract(paper, ASTHMA, _finding(
        E220_QUOTE, "c-Myc", change="with or without the c-Myc inhibitor EN4", direction="block",
        intervention="EN4", disease="asthma", evidence_type="in_vivo", species="mice",
        effect="effectively suppressed", outcome="improved", facet="causal",
    ), stats)
    # The mice are shown, from the context sentence - but they are not what
    # the quote's experiment was done in.
    assert u.context_quote == E220_CONTEXT and u.species_raw == "mice"
    assert u.evidence_type == "in_vitro"
    assert u.corrected == ['evidence_type: in_vivo -> in_vitro ("HBECs")']
    assert (u.direction, u.outcome) == ("block", "improved")
    assert stats["evidence_type_corrected"] == 1


# --- E106 / E108: an observation is an association, not a worsened result -----------


def test_e106_an_observation_is_an_association_not_worsened():
    # The model's own labels: direction unclear, outcome improved, and a
    # "change" that is not in the quote (it has "( db/db ) mice").
    f = _finding(E106_QUOTE, "SIK3", modality="genetic", outcome="improved", setting="treatment", facet="causal",
                 change="increased in 12-week-old genetically induced leptin receptor-deficient mice",
                 effect="similar increases")
    direction, outcome, _, notes = reconcile(f)
    assert (direction, outcome) == ("unclear", "association")
    assert notes == ['outcome: improved -> worsened ("similar increases")',
                     "outcome: worsened -> association (nothing done to the target)"]
    # An association is never filed as a negative result.
    assert settle_facet("negative", evidence_type="in_vivo", direction=direction, outcome=outcome,
                        setting="treatment", facet_ids=["causal", "mechanism", "negative"]) == "mechanism"


def test_e106_and_e108_in_the_pipeline():
    paper = Paper(id="PMID42230648", title="Hypothalamic SIK3", abstract="", sections={"results": SIK3_RESULTS})
    for quote, change, effect in (
        (E106_QUOTE, "increased in 12-week-old genetically induced leptin receptor-deficient mice",
         "similar increases"),
        (E108_QUOTE, "increased after 10 weeks of high-fat diet feeding", "significantly increased"),
    ):
        stats = Counter()
        u = _extract(paper, OBESITY, _finding(quote, "SIK3", modality="genetic", outcome="improved",
                                              setting="treatment", facet="negative", change=change,
                                              effect=effect), stats)
        assert (u.direction, u.outcome, u.evidence_type) == ("unclear", "association", "in_vivo")
        assert u.facet != "negative"
        # Two notes on one field are one corrected outcome.
        assert stats["outcome_corrected"] == 1


# --- negatives -------------------------------------------------------------------------


def test_a_mouse_knockout_that_worsened_stays_worsened_and_in_vivo():
    f = _finding(E206_QUOTE, "TMEM135", change="knockout", direction="block", modality="genetic",
                 effect="results in glucose intolerance, hyperinsulinemia and insulin resistance",
                 outcome="worsened", facet="causal")
    assert reconcile(f)[:2] == ("block", "worsened")
    assert _settle(f, species="mouse") == ("in_vivo", None)
    # Even when the model leaves the direction unclear and copies words that
    # are not in the quote, the quote itself says "TMEM135 knockout".
    assert target_changed_in(E206_QUOTE, "TMEM135")
    muddled = f.model_copy(update={"direction": "unclear", "change": "deletion"})
    assert reconcile(muddled)[:2] == ("unclear", "worsened")


def test_a_human_genetic_study_stays_human_genetic():
    f = _finding(E154_QUOTE, "GPR75", evidence_type="human_genetic", species="human")
    assert _settle(f, species="human", species_raw="human") == ("human_genetic", None)
    # No people named in the quote: the species decides, and the genetic
    # words keep the label.
    f = _finding(E5_QUOTE, "MC4R", evidence_type="human_genetic", species="patients")
    assert system_in_quote(E5_QUOTE, "human_genetic") is None
    assert _settle(f, species="human") == ("human_genetic", None)


def test_a_trial_in_variant_carriers_stays_clinical():
    """Genetic words name the population, not the study: a Setmelanotide
    trial "in obese MC4R variant carriers" is a clinical trial, and stays in
    the clinical section."""
    f = _finding(E214_QUOTE, "MC4R", change="variant carriers", intervention="Setmelanotide",
                 evidence_type="human_clinical", species="patients", effect="led to weight loss",
                 outcome="improved", facet="clinical")
    evidence_type, note = _settle(f, species="human", species_raw="patients")
    assert (evidence_type, note) == ("human_clinical", None)
    direction, outcome, setting, _ = reconcile(f)
    assert outcome == "improved"
    assert settle_facet("clinical", evidence_type=evidence_type, direction=direction, outcome=outcome,
                        setting=setting, facet_ids=["clinical", "mechanism"]) == "clinical"


def test_an_in_vitro_label_on_peoples_cells_stands():
    for quote, species in ((E229_QUOTE, "humans"), (E183_QUOTE, "children")):
        f = _finding(quote, "x", evidence_type="in_vitro", species=species)
        assert system_in_quote(quote, "in_vitro") is None
        assert _settle(f, species="human", species_raw=species) == ("in_vitro", None)


def test_mouse_as_an_adjective_names_no_animal():
    f = _finding(E124_QUOTE, "TMEM135", evidence_type="in_vitro", species="hBAT cells")
    assert _settle(f, species="cell_line", species_raw="hBAT cells") == ("in_vitro", None)


def test_an_animal_model_and_cells_together_decide_nothing():
    assert system_in_quote(E91_QUOTE, "in_vivo") is None
    f = _finding(E91_QUOTE, "lactate", evidence_type="in_vivo", species="mice")
    assert _settle(f, species="mouse", species_raw="mice") == ("in_vivo", None)


def test_a_drug_whose_direction_is_unclear_keeps_its_outcome():
    f = _finding(E54_QUOTE, "Rimegepant", change="shown efficacy", intervention="Rimegepant",
                 evidence_type="human_clinical", species="patients", effect="shown efficacy", outcome="improved",
                 facet="clinical")
    assert reconcile(f)[:2] == ("unclear", "improved")
