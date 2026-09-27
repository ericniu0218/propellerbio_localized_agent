"""Issue #4: a conclusion sentence can overstate who was studied.

The asthma hand audit (S3, evals/litreview/hand_audit/qwen_asthma.md) graded
this report sentence misleading:

    "MUC1-CT reduces NLRP3 inflammasome-mediated pyroptosis by inhibiting the
     TLR4/MyD88/NF-κB pathway, thereby attenuating neutrophil airway
     inflammation in asthma patients."

It copies the abstract's conclusion (PMID37880668, E20), "... in patients
with asthma", when the mechanism was shown in BEAS-2B cells and mice. The
quote says "patients", so the who-was-studied check passed.

Now, when a quote is the authors' conclusion (`extract/verify.py`
`conclusion_reason`) and names people, Stage 5 reads the paper's Methods
(`extract/provenance.py` `methods_subjects`). Methods that name no human
subjects but do name animals or cells make the finding animal or cell
evidence, show the Methods sentence as its context, and mark the unit
`methods_subjects`; the citation guard then drops a sentence resting only on
such findings that says "patients" (check 4).

The real S3 paper also recruited patients (for induced sputum), so its own
Methods back the word and it is left alone - see the last test. The S3 case
below therefore gives it the Methods it would have had as a cells-and-mice
study: the paper's own cell and mouse paragraphs, verbatim, without the
patient paragraphs. The other papers are verbatim from the stored corpora.
"""

import json
from collections import Counter
from types import SimpleNamespace

from bedrock.litreview.extract.provenance import methods_subjects
from bedrock.litreview.extract.verify import abstract_methods, conclusion_reason, people_named
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import EvidenceUnit, ExtractionJob, Facet, Paper, Report
from bedrock.litreview.report.guard import CitationGuard
from bedrock.litreview.report.invariants import check_units
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import extract_paper
from bedrock.litreview.stages.s7_write import _evidence_lines
from bedrock.litreview.stages.s8_render import to_markdown

KB = load_knowledge()
ASTHMA = parse_question("If I want to prevent asthma exacerbations, what should I target?")
OBESITY = parse_question("If I want to treat obesity, which targets should I pursue?")
FACETS = [Facet(id="causal", title="Causal", need="what changing the target does")]

# --- the cases ------------------------------------------------------------------

# PMID37880668 (asthma corpus), the abstract's last two paragraphs, verbatim.
S3_RESULTS = (
    "The mRNA expression of MUC1 was downregulated in the induced sputum of patients with asthma and correlated "
    "with asthmatic neutrophilic airway inflammation. LPS activated the TLR4 pathway and NLRP3-mediated pyroptosis "
    "in BEAS-2B cells in vitro, which were significantly aggravated after MUC1-siRNA transfection."
)
S3_QUOTE = ("MUC1-CT reduces NLRP3 inflammasome-mediated pyroptosis by inhibiting the activation of the "
            "TLR4/MyD88/NF-κB pathway, thereby attenuating neutrophil airway inflammation in patients with asthma.")
S3_CONCLUSION = ("This study revealed that MUC1 downregulation plays an important role in asthmatic neutrophilic "
                 "airway inflammation. " + S3_QUOTE)
S3_ABSTRACT = S3_RESULTS + "\n\n" + S3_CONCLUSION
# The same paper's Methods paragraphs on its cells and its mice, verbatim.
S3_MICE = "Female C57BL/6J mice (6–8 weeks old) were obtained from Gempharmatech (Jiangsu, China)."
S3_CELLS_AND_MICE = (
    "Human bronchial epithelial cells (BEAS-2B, ATCC; Manassas, VA, USA) were cultured in Dulbecco’s Modified "
    "Eagle Medium (Gibco, USA) supplemented with 10% fetal bovine serum (Gibco, USA) in a 5% CO 2 atmosphere at "
    "37 °C.\n\n"
    "The transfection reagent lipofectamine 3000 (Invitrogen, Carlsbad, CA, USA) was used to transfect BEAS-2B "
    "cells with human MUC1-siRNAs or NC-siRNA (Guangzhou, China) for 24 h.\n\n"
    + S3_MICE + " All animals were maintained in a specific pathogen-free facility with free access to standard "
    "fodder and water and kept in a controlled environment (22 °C ± 2 °C, 55% ± 5% humidity) under a 12-h "
    "light/dark cycle."
)
# ... and the paragraphs it does have, before those: patients were recruited.
S3_PATIENTS = (
    "Adult patients with physician-diagnosed asthma were recruited if they met the following inclusion criteria: "
    "recurrent episodes of wheezing, dyspnea, cough, and sputum production.\n\n"
    "All participants provided informed consent. The Ethics Committee of Tongji Hospital, Tongji Medical College, "
    "Huazhong University of Science and Technology approved the study."
)
S3_SENTENCE = ("MUC1-CT reduces NLRP3 inflammasome-mediated pyroptosis by inhibiting the TLR4/MyD88/NF-κB pathway, "
               "thereby attenuating neutrophil airway inflammation in asthma patients.")

# PMID35707726 (asthma corpus): a conclusion about children, in a paper whose
# Methods collected blood from children. Abstract sentences and the first
# Methods sentence, verbatim.
IL3_QUOTE = "In conclusion, IL-3 downregulation was found associated with more severe allergic asthma in pre-school children."
IL3_ABSTRACT = (
    "PBMCs from asthmatic children showed increased IL-3 secretion, which directly correlated with improved lung "
    "function. IL-3 −/− asthmatic mice showed increased asthmatic traits. " + IL3_QUOTE
)
IL3_METHODS = (
    "Heparinized blood was collected from children with and without asthma at the Baseline Visit of the PreDicta "
    "study and subsequently PBMCs were isolated with Ficoll using density centrifugation.\n\n"
    "IL-3 −/− mice on a BALB/c background were bred in the animal facility."
)

# PMID28133727 (obesity corpus, abstract only): a structured abstract whose
# METHODS part studied rats, and whose conclusion speaks of obese individuals.
TRPV1_QUOTE = ("Changes in TRPV1-mediated vascular reactions and CGRP release are pathophysiological alterations "
               "that may be of relevance to the enhanced headache susceptibility of obese individuals.")
TRPV1_RATS = "Male Sprague-Dawley rats were fed a regular (n = 25) or HFHS diet (n = 26) for 20 weeks."
TRPV1_ABSTRACT = (
    "BACKGROUND: Clinical and experimental observations suggest a link between obesity and migraine. METHODS: "
    + TRPV1_RATS + " Capsaicin-induced changes in meningeal blood flow and release of calcitonin gene-related "
    "peptide (CGRP) from dural trigeminal afferents were measured in control and obese rats. RESULTS: In obese "
    "animals, dural application of the archetypal TRPV1 agonist capsaicin resulted in significantly augmented "
    "vasodilatory and vasoconstrictor responses as compared to controls. CONCLUSIONS: Our results suggest that "
    "obesity induced by long-term HFHS diet results in sensitization of the trigeminovascular system. "
    + TRPV1_QUOTE
)


def _finding(quote: str, target: str, disease: str, species: str, evidence_type: str = "human_clinical") -> dict:
    return {
        "quote": quote, "target": target, "change": "", "direction": "activate", "modality": "other",
        "intervention": None, "disease": disease, "evidence_type": evidence_type, "species": species,
        "effect": "", "outcome": "improved", "setting": "unclear", "effect_detail": "", "n": None,
        "combo_with": None, "target_is_primary": True, "facet": "causal",
    }


def _extract(paper: Paper, section: str, spec, finding: dict, stats: Counter | None = None) -> list[EvidenceUnit]:
    reply = {"is_review": False, "findings": [finding]}
    client = FakeLLMClient(responder=lambda prompt, schema: json.dumps(reply), version="fake-1")
    job = ExtractionJob(paper_id=paper.id, sections=[section], facets=["causal"])
    return extract_paper(paper, job, spec, FACETS, client, model="m", knowledge=KB,
                         stats=stats if stats is not None else Counter())


def _guard(spec, *units: EvidenceUnit) -> CitationGuard:
    table = {f"E{i}": u for i, u in enumerate(units, 1)}
    return CitationGuard(spec, KB, table, paper_labels={u.paper_id: "Author 2023 Journal" for u in units})


def _s3_paper(methods: str | None = S3_CELLS_AND_MICE) -> Paper:
    sections = {"abstract": S3_ABSTRACT}
    if methods is not None:
        sections["methods"] = methods
    return Paper(id="PMID37880668", title="MUC1 in neutrophilic asthma", abstract=S3_ABSTRACT, sections=sections)


def _s3_unit(methods: str | None = S3_CELLS_AND_MICE, stats: Counter | None = None) -> EvidenceUnit:
    units = _extract(_s3_paper(methods), "abstract", ASTHMA, _finding(S3_QUOTE, "MUC1-CT", "asthma", "patients"),
                     stats)
    assert len(units) == 1
    return units[0]


# --- S3: a cells-and-mice paper's conclusion about patients ----------------------


def test_s3_is_not_human_evidence_when_the_methods_studied_cells_and_mice():
    stats = Counter()
    u = _s3_unit(stats=stats)
    assert u.quote == S3_QUOTE
    assert u.methods_subjects == "mice and cells"
    assert (u.species, u.species_raw, u.evidence_type) == ("mouse", "mice", "in_vivo")
    # The Methods sentence that says so is shown beside the quote, anchored.
    assert u.context_quote == S3_MICE and u.context_section == "methods"
    assert S3_CELLS_AND_MICE[u.context_start:u.context_end] == S3_MICE
    assert u.provenance["species"] == "methods"
    assert stats["people_not_in_methods"] == 1
    assert check_units([u], {"PMID37880668": _s3_paper()}, KB) == []


def test_s3_sentence_is_dropped():
    verdict = _guard(ASTHMA, _s3_unit()).check(S3_SENTENCE, ["E1"])
    assert not verdict.kept
    assert verdict.reason == "says 'patients', but the paper's methods studied only mice and cells"


def test_s3_without_patients_is_kept():
    guard = _guard(ASTHMA, _s3_unit())
    sentence = S3_SENTENCE.replace("in asthma patients", "in asthma")
    assert guard.check(sentence, ["E1"]).kept, guard.check(sentence, ["E1"]).reason
    in_mice = S3_SENTENCE.replace("in asthma patients", "in mice")
    assert guard.check(in_mice, ["E1"]).kept, guard.check(in_mice, ["E1"]).reason


def test_human_cells_are_not_people():
    """"human bronchial epithelial cells" is a claim about cells."""
    sentence = ("MUC1-CT reduces NLRP3 inflammasome-mediated pyroptosis in human bronchial epithelial cells, "
                "thereby attenuating neutrophil airway inflammation in asthma.")
    verdict = _guard(ASTHMA, _s3_unit()).check(sentence, ["E1"])
    assert verdict.reason is None or "methods studied only" not in verdict.reason


def test_another_cited_finding_about_patients_keeps_the_sentence():
    """The sentence may rest on a finding whose own text names patients."""
    other = _s3_unit().model_copy(update={
        "id": "u2", "quote": ("The mRNA expression of MUC1 was downregulated in the induced sputum of patients with "
                              "asthma and correlated with asthmatic neutrophilic airway inflammation."),
        "methods_subjects": None, "context_quote": None, "species": "human", "species_raw": "patients",
        "evidence_type": "human_clinical"})
    assert _guard(ASTHMA, _s3_unit(), other).check(S3_SENTENCE, ["E1", "E2"]).reason != (
        "says 'patients', but the paper's methods studied only mice and cells")


def test_the_mark_is_rendered_and_given_to_the_writer():
    u = _s3_unit()
    report = Report(question_id="q", question="q", sections=[], evidence=[u], short_ids={u.id: "E1"})
    assert "the paper's methods studied only mice and cells" in to_markdown(report)
    ctx = SimpleNamespace(summary=SimpleNamespace(short_ids={u.id: "E1"}), labels={u.paper_id: "Author 2023"})
    lines = _evidence_lines([u], ctx)
    assert "Subjects: the quote is the authors' conclusion" in lines and "do not say patients" in lines
    assert "studied in animals (mice)" in lines


# --- left alone -------------------------------------------------------------------


def test_the_real_s3_paper_recruited_patients_so_it_is_left_alone():
    """PMID37880668's Methods open with recruited patients and informed
    consent: the conclusion's word is backed, whatever else it overstates."""
    u = _s3_unit(methods=S3_PATIENTS + "\n\n" + S3_CELLS_AND_MICE)
    assert u.methods_subjects is None
    assert (u.species, u.species_raw, u.evidence_type) == ("human", "patients", "human_clinical")
    assert _guard(ASTHMA, u).check(S3_SENTENCE, ["E1"]).kept


def test_no_methods_text_is_left_alone():
    """Abstract-only papers are common; nothing is guessed from nothing."""
    u = _s3_unit(methods=None)
    assert u.methods_subjects is None and u.species == "human"
    assert _guard(ASTHMA, u).check(S3_SENTENCE, ["E1"]).kept


def test_a_conclusion_about_children_whose_methods_studied_children_is_unchanged():
    paper = Paper(id="PMID35707726", title="IL-3 in asthma", abstract=IL3_ABSTRACT,
                  sections={"abstract": IL3_ABSTRACT, "methods": IL3_METHODS})
    units = _extract(paper, "abstract", ASTHMA, _finding(IL3_QUOTE, "IL-3", "allergic asthma", "children"))
    assert len(units) == 1
    u = units[0]
    assert u.methods_subjects is None and (u.species_raw, u.evidence_type) == ("children", "human_clinical")
    assert "methods_subjects" not in u.model_dump_json()


def test_a_result_that_is_not_a_conclusion_is_unchanged():
    """Only a conclusion is read against the Methods: a results sentence that
    names patients is the paper reporting what it measured."""
    quote = ("The mRNA expression of MUC1 was downregulated in the induced sputum of patients with asthma and "
             "correlated with asthmatic neutrophilic airway inflammation.")
    units = _extract(_s3_paper(), "abstract", ASTHMA, _finding(quote, "MUC1", "asthma", "patients"))
    assert len(units) == 1 and units[0].methods_subjects is None


# --- a structured abstract's METHODS part -------------------------------------------


def test_a_structured_abstracts_methods_part_is_read():
    """PMID28133727 has no Methods section; its abstract's METHODS part
    studied rats, and its conclusion speaks of obese individuals."""
    paper = Paper(id="PMID28133727", title="TRPV1 in obesity", abstract=TRPV1_ABSTRACT,
                  sections={"abstract": TRPV1_ABSTRACT})
    units = _extract(paper, "abstract", OBESITY, _finding(TRPV1_QUOTE, "TRPV1", "obesity", "obese individuals"))
    assert len(units) == 1
    u = units[0]
    assert u.methods_subjects == "rats"
    assert (u.species, u.species_raw, u.evidence_type) == ("rat", "rats", "in_vivo")
    # Widened to the paper's whole sentence, which the label runs into.
    assert u.context_quote == "METHODS: " + TRPV1_RATS and u.context_section == "abstract"
    assert check_units([u], {"PMID28133727": paper}, KB) == []
    sentence = "TRPV1-mediated vascular changes may be relevant to headache susceptibility in obese individuals."
    assert _guard(OBESITY, u).check(sentence, ["E1"]).reason == (
        "says 'individuals', but the paper's methods studied only rats")


# --- which sentences are conclusions ----------------------------------------------


def _reason(text: str, quote: str, section: str = "abstract") -> str | None:
    start = text.index(quote)
    return conclusion_reason(text, start, start + len(quote), section)


def test_conclusion_cues():
    assert _reason(IL3_ABSTRACT, IL3_QUOTE) == "conclusion:cue"
    text = "Overall, our findings here demonstrate that adult-onset Gpr75 deletion protects from obesity."
    assert _reason(text, text, "results") == "conclusion:cue"
    # Not conclusions: "Overall survival", "The results showed" opening a result.
    for text in ("Overall survival was longer in patients given the antibody.",
                 "The results showed that IL-6 was elevated in patients with asthma."):
        assert _reason(text, text, "results") is None


def test_a_conclusion_paragraph_in_an_unlabelled_abstract():
    assert _reason(S3_ABSTRACT, S3_QUOTE) == "conclusion:paragraph"
    # The paragraph before it is not the conclusion; a cue in the Results
    # section's paragraph does not reach past the abstract.
    first = S3_RESULTS.split(". ")[0] + "."
    assert _reason(S3_ABSTRACT, first) is None
    assert _reason(S3_CONCLUSION, S3_QUOTE, "results") is None


def test_structured_abstract_labels_decide():
    assert _reason(TRPV1_ABSTRACT, TRPV1_QUOTE) == "conclusion:label"
    results = "In obese animals, dural application of the archetypal TRPV1 agonist capsaicin resulted"
    assert _reason(TRPV1_ABSTRACT, results) is None
    # Glued on, as Europe PMC leaves it (PMID41546592, verbatim).
    glued = ("Primary outcomes included clinical efficacy and inflammatory markers.ResultsThe combination therapy "
             "showed higher total efficacy (95.0% vs. 85.0%, p=0.136).ConclusionDPL combined with BUD can quickly "
             "relieve the clinical symptoms and reduce the level of inflammation in children with ABP, with good "
             "safety.")
    assert _reason(glued, "DPL combined with BUD can quickly relieve") == "conclusion:label"
    assert _reason(glued, "The combination therapy showed") is None
    # Under a RESULTS label, "Our results show" is a result, and so is the next sentence.
    labelled = "RESULTS: Our results show X was reduced in patients. Y was also reduced in patients. CONCLUSIONS: Z."
    assert _reason(labelled, "Y was also reduced in patients.") is None


def test_abstract_methods_part():
    [(lo, hi)] = abstract_methods(TRPV1_ABSTRACT)
    assert TRPV1_ABSTRACT[lo:hi].strip().startswith(TRPV1_RATS)
    assert "RESULTS" not in TRPV1_ABSTRACT[lo:hi]
    assert abstract_methods(S3_ABSTRACT) == []


# --- who the words name -----------------------------------------------------------


def test_people_named():
    assert people_named("in patients with asthma") == "patients"
    assert people_named("in pre-school children") == "children"
    assert people_named("Human bronchial epithelial cells were cultured") is None
    assert people_named("adult-onset Gpr75 deletion protects from diet-induced obesity") is None
    assert people_named("patient-derived organoids") is None
    assert people_named("in humans") == "humans"


def test_methods_that_describe_people_back_the_conclusion():
    for words in ("All participants provided informed consent.", "Biopsies from 12 donors were cultured.",
                  "The study was approved by the institutional review board."):
        assert methods_subjects(S3_QUOTE, methods_section_text=S3_CELLS_AND_MICE + "\n\n" + words,
                                abstract="") is None


def test_methods_naming_no_subjects_at_all_decide_nothing():
    """A Methods section about statistics alone says nothing about who was
    studied."""
    stats_only = "Data are presented as mean ± SD and were compared with one-way ANOVA."
    assert methods_subjects(S3_QUOTE, methods_section_text=stats_only, abstract="") is None


def test_a_quote_naming_no_people_is_not_read_against_the_methods():
    assert methods_subjects("MUC1-CT reduces pyroptosis in mice.", methods_section_text=S3_CELLS_AND_MICE,
                            abstract="") is None


# --- stored compatibly, and checked -------------------------------------------------


def test_a_marked_unit_round_trips_and_an_unmarked_one_omits_the_field():
    u = _s3_unit()
    assert EvidenceUnit.model_validate_json(u.model_dump_json()) == u
    assert '"methods_subjects":"mice and cells"' in u.model_dump_json()
    plain = u.model_copy(update={"methods_subjects": None})
    assert "methods_subjects" not in plain.model_dump_json()


def test_a_marked_unit_labelled_human_is_caught():
    u = _s3_unit().model_copy(update={"evidence_type": "human_clinical"})
    assert any("is labelled human, but its paper's methods studied only mice and cells" in v
               for v in check_units([u], {"PMID37880668": _s3_paper()}, KB))
