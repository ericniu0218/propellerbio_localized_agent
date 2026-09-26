"""Issue #6: a target named only elsewhere in the passage cannot take the
quote's result.

Provenance used to accept a target found anywhere in the chunk the model read,
and the citation guard then let a sentence name it - though the reader saw
only the quote. The adversarial probe: a quote about anti-IL-23, a passage
that also names IL-17A, a finding labelled IL-17A, and "Blocking IL-17A
reduced ear thickness by 60%" passed every check. In the Claude-as-model
obesity report the writer named the finding's target beside the agent
("tubastatin A, acting on HDAC6", "REGN4461, which acts on LEPR"): true of the
passage, not stated in the quote the reader sees.

Now a name the quote does not state is shown: in the context sentence, the
qualifier, or the passage's sentence that names it (`naming_quote`,
`extract/provenance.py`). And the guard's check 9 drops a sentence that gives
the quote's result to that name while naming nothing the quote names. The
obesity abstracts below are the papers' own text, verbatim from the corpus.
"""

import json
from collections import Counter
from types import SimpleNamespace

from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import EvidenceUnit, ExtractionJob, Facet, Paper, Report
from bedrock.litreview.report.guard import CitationGuard, misattributes, source_text
from bedrock.litreview.report.invariants import check_units
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import extract_paper
from bedrock.litreview.stages.s7_write import _evidence_lines
from bedrock.litreview.stages.s8_render import to_markdown

KB = load_knowledge()
PSORIASIS = parse_question("If I want to prevent psoriasis, what should I target?")
OBESITY = parse_question("If I want to treat obesity, which targets should I pursue?")
FACETS = [Facet(id="causal", title="Causal", need="what changing the target does")]

# --- the cases ------------------------------------------------------------------

# The adversarial probe: the passage names IL-17A, the quote is about IL-23.
PROBE_QUOTE = "Anti-IL-23 treatment reduced ear thickness by 60% in imiquimod-treated mice."
PROBE_NAMING = "Imiquimod application increased IL-17A expression in the skin of wild-type mice."
PROBE_RESULTS = PROBE_NAMING + " " + PROBE_QUOTE
PROBE_SENTENCE = "Blocking IL-17A reduced ear thickness by 60% in imiquimod-treated mice."

# PMID35039672 (obesity corpus), abstract, verbatim.
HDAC6_ABSTRACT = (
    "The adipose tissue-derived hormone leptin can drive decreases in food intake while increasing energy "
    "expenditure. In diet-induced obesity, circulating leptin levels rise proportionally to adiposity. Despite this "
    "hyperleptinemia, rodents and humans with obesity maintain increased adiposity and are resistant to leptin's "
    "actions. Here we show that inhibitors of the cytosolic enzyme histone deacetylase 6 (HDAC6) act as potent "
    "leptin sensitizers and anti-obesity agents in diet-induced obese mice. Specifically, HDAC6 inhibitors, such as "
    "tubastatin A, reduce food intake, fat mass, hepatic steatosis and improve systemic glucose homeostasis in an "
    "HDAC6-dependent manner. Mechanistically, peripheral, but not central, inhibition of HDAC6 confers central "
    "leptin sensitivity. Additionally, the anti-obesity effect of tubastatin A is attenuated in animals with a "
    "defective central leptin-melanocortin circuitry, including db/db and MC4R knockout mice. Our results suggest "
    "the existence of an HDAC6-regulated adipokine that serves as a leptin-sensitizing agent and reveals HDAC6 as a "
    "potential target for the treatment of obesity."
)
HDAC6_QUOTE = ("Additionally, the anti-obesity effect of tubastatin A is attenuated in animals with a defective "
               "central leptin-melanocortin circuitry, including db/db and MC4R knockout mice.")
# The nearest sentence naming HDAC6 is the one before the quote; the one kept
# is the one that also names tubastatin A, which says how the two are linked.
HDAC6_NAMING = ("Specifically, HDAC6 inhibitors, such as tubastatin A, reduce food intake, fat mass, hepatic "
                "steatosis and improve systemic glucose homeostasis in an HDAC6-dependent manner.")

# PMID37992152 (obesity corpus), abstract sentences, verbatim: E130 in the
# stored qwen3:8b report (target LEPR, found only in the chunk).
REGN_NAMING = ("We developed a fully human monoclonal antibody (mAb), REGN4461 (mibavademab), that activates the "
               "human LEPR in the absence or presence of leptin.")
REGN_QUOTE = ("In obese leptin knockout mice, REGN4461 normalized body weight, food intake, blood glucose, and "
              "insulin sensitivity.")
REGN_ABSTRACT = (
    "Therapies that restore leptin-LEPR signaling may resolve these metabolic sequelae. " + REGN_NAMING + " "
    + REGN_QUOTE + " In a mouse model of generalized lipodystrophy, REGN4461 alleviated hyperphagia, "
    "hyperglycemia, insulin resistance, dyslipidemia, and hepatic steatosis."
)


def _finding(quote: str, target: str, intervention: str | None, disease: str, species: str) -> dict:
    return {
        "quote": quote, "target": target, "change": "", "direction": "block", "modality": "small_molecule",
        "intervention": intervention, "disease": disease, "evidence_type": "in_vivo", "species": species,
        "effect": "reduced", "outcome": "improved", "setting": "treatment", "effect_detail": "", "n": None,
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
    return CitationGuard(spec, KB, table, paper_labels={u.paper_id: "Author 2022 Journal" for u in units})


def _probe_paper() -> Paper:
    return Paper(id="P1", title="IL-23 in skin inflammation", abstract="IL-23 drives skin inflammation.",
                 sections={"abstract": "IL-23 drives skin inflammation.", "results": PROBE_RESULTS})


def _probe_unit() -> EvidenceUnit:
    units = _extract(_probe_paper(), "results", PSORIASIS,
                     _finding(PROBE_QUOTE, "IL-17A", None, "psoriasis", "mice"))
    assert len(units) == 1
    return units[0]


# --- the probe, end to end --------------------------------------------------------


def test_the_probe_stores_the_sentence_that_names_il17a():
    u = _probe_unit()
    assert u.quote == PROBE_QUOTE and u.target_raw == "IL-17A"
    assert u.naming_quote == PROBE_NAMING and u.naming_section == "results"
    assert PROBE_RESULTS[u.naming_start:u.naming_end] == PROBE_NAMING
    assert u.provenance["target"] == "naming" and u.provenance["naming"] == "chunk"
    assert check_units([u], {"P1": _probe_paper()}, KB) == []


def test_the_probe_sentence_is_dropped():
    verdict = _guard(PSORIASIS, _probe_unit()).check(PROBE_SENTENCE, ["E1"])
    assert not verdict.kept
    assert verdict.reason == "attributes the quote's result to IL-17A, which the quote does not name"


def test_the_probe_is_dropped_without_a_number_too():
    verdict = _guard(PSORIASIS, _probe_unit()).check("Blocking IL-17A reduced ear thickness in mice.", ["E1"])
    assert verdict.reason == "attributes the quote's result to IL-17A, which the quote does not name"


def test_the_quotes_own_agent_keeps_the_sentence():
    guard = _guard(PSORIASIS, _probe_unit())
    assert guard.check("Anti-IL-23 treatment reduced ear thickness by 60% in imiquimod-treated mice.", ["E1"]).kept
    # Naming IL-17A is fine once the sentence gives the result to what the
    # quote names: the naming sentence makes the name citable.
    sentence = ("In mice whose skin expressed IL-17A after imiquimod, anti-IL-23 treatment reduced ear thickness "
                "by 60%.")
    assert guard.check(sentence, ["E1"]).kept


def test_a_stored_unit_without_a_naming_sentence_is_checked_the_same():
    """A run extracted before naming sentences existed says the target was
    found in the 'chunk'. `verify-run` re-runs the guard on it, and the
    probe sentence is dropped there too."""
    u = _probe_unit().model_copy(update={"naming_quote": None, "naming_section": None, "naming_start": None,
                                         "naming_end": None, "provenance": {"target": "chunk"}})
    verdict = _guard(PSORIASIS, u).check(PROBE_SENTENCE, ["E1"])
    assert verdict.reason == "attributes the quote's result to IL-17A, which the quote does not name"


def test_another_cited_quote_about_il17a_keeps_the_sentence():
    """The sentence may rest on a second quote that does name IL-17A and
    reports a result."""
    other = EvidenceUnit(**{**_probe_unit().model_dump(), "id": "u2",
                            "quote": "Blocking IL-17A reduced ear thickness by 60% in imiquimod-treated mice.",
                            "naming_quote": None, "provenance": {"target": "quote"}})
    assert misattributes(PROBE_SENTENCE, PROBE_SENTENCE, [_probe_unit(), other], KB) is None


def test_a_number_from_the_naming_sentence_is_that_sentences_result():
    """"IL-17A levels fell by 40%" rests on the shown naming sentence, not on
    the quote, so check 9 does not claim it took the quote's result."""
    naming = "IL-17A levels in the skin fell by 40% after anti-IL-23 treatment."
    u = _probe_unit().model_copy(update={"naming_quote": naming})
    assert misattributes("IL-17A levels in the skin fell by 40%.", "IL-17A levels in the skin fell by 40%.",
                         [u], KB) is None


def test_a_quote_that_names_no_other_agent_is_left_alone():
    """Real case (migraine E92): the quote names nothing at all, the agent
    comes from the passage, and a sentence naming it is not misattributing
    anything - there is nothing else the result could belong to."""
    quote = "No patient reported treatment-related adverse events during the entire observation period."
    u = _probe_unit().model_copy(update={"quote": quote, "target": "rimegepant", "target_raw": "rimegepant",
                                         "intervention": "rimegepant"})
    sentence = "No patient reported treatment-related adverse events during the observation period in a study of rimegepant."
    assert misattributes(sentence, sentence, [u], KB) is None


# --- the tubastatin A / HDAC6 shape (Claude-model obesity report) -------------------


def _hdac6_paper() -> Paper:
    return Paper(id="PMID35039672", title="Histone deacetylase 6 inhibition restores leptin sensitivity and reduces "
                 "obesity.", abstract=HDAC6_ABSTRACT, sections={"abstract": HDAC6_ABSTRACT})


def _hdac6_unit(stats: Counter | None = None) -> EvidenceUnit:
    units = _extract(_hdac6_paper(), "abstract", OBESITY,
                     _finding(HDAC6_QUOTE, "HDAC6", "tubastatin A", "obesity", "mice"), stats)
    assert len(units) == 1
    return units[0]


def test_hdac6_is_shown_by_the_sentence_that_links_it_to_tubastatin_a():
    stats = Counter()
    u = _hdac6_unit(stats)
    assert u.provenance["target"] == "naming" and u.provenance["intervention"] == "quote"
    assert u.naming_quote == HDAC6_NAMING and u.naming_section == "abstract"
    assert HDAC6_ABSTRACT[u.naming_start:u.naming_end] == HDAC6_NAMING
    assert stats["naming_sentence"] == 1
    assert check_units([u], {u.paper_id: _hdac6_paper()}, KB) == []


def test_tubastatin_a_acting_on_hdac6_is_kept():
    sentence = ("The anti-obesity effect of tubastatin A, acting on HDAC6, is attenuated in db/db and MC4R knockout "
                "mice.")
    verdict = _guard(OBESITY, _hdac6_unit()).check(sentence, ["E1"])
    assert verdict.kept, verdict.reason


def test_the_naming_sentence_is_rendered_and_given_to_the_writer():
    u = _hdac6_unit()
    report = Report(question_id="q", question="q", sections=[], evidence=[u], short_ids={u.id: "E1"})
    assert f'names HDAC6 (abstract): "{HDAC6_NAMING}"' in to_markdown(report)
    ctx = SimpleNamespace(summary=SimpleNamespace(short_ids={u.id: "E1"}), labels={u.paper_id: "Author 2022"})
    lines = _evidence_lines([u], ctx)
    assert "Names (the paper's sentence that names what the quote does not" in lines and HDAC6_NAMING in lines
    assert HDAC6_NAMING in source_text(u)


def test_regn4461_names_lepr_by_the_sentence_that_says_it_activates_it():
    """E130 in the stored qwen3:8b obesity report: LEPR is not in the quote.
    Of the sentences naming LEPR, the one kept also names REGN4461."""
    paper = Paper(id="PMID37992152", title="REGN4461", abstract=REGN_ABSTRACT, sections={"abstract": REGN_ABSTRACT})
    units = _extract(paper, "abstract", OBESITY, _finding(REGN_QUOTE, "LEPR", "REGN4461", "obesity", "mice"))
    assert len(units) == 1 and units[0].naming_quote == REGN_NAMING
    sentence = "In obese leptin knockout mice, REGN4461, which acts on LEPR, normalized body weight and food intake."
    assert _guard(OBESITY, units[0]).check(sentence, ["E1"]).kept


# --- a target in the quote needs nothing --------------------------------------------


def test_a_target_in_the_quote_stores_no_naming_sentence():
    units = _extract(_probe_paper(), "results", PSORIASIS, _finding(PROBE_QUOTE, "IL-23", None, "psoriasis", "mice"))
    assert len(units) == 1
    u = units[0]
    assert u.provenance["target"] == "quote" and u.naming_quote is None
    assert "naming" not in u.model_dump_json()


def test_a_target_found_in_the_context_sentence_is_shown_by_it():
    methods = "IL-17A reporter C57BL/6 mice (n = 12 per group) were treated daily with imiquimod cream."
    paper = Paper(id="P1", title="t", abstract="a",
                  sections={"abstract": "a", "results": PROBE_QUOTE, "methods": methods})
    finding = {**_finding(PROBE_QUOTE, "IL-17A", None, "psoriasis", "C57BL/6 mice"), "n": 12}
    u = _extract(paper, "results", PSORIASIS, finding)[0]
    assert u.context_quote == methods and u.provenance["target"] == "context" and u.naming_quote is None


# --- stored compatibly, and checked -------------------------------------------------


def test_a_naming_sentence_that_is_not_the_papers_text_is_caught():
    u = _probe_unit().model_copy(update={"naming_quote": PROBE_NAMING.replace("increased", "decreased")})
    assert any("naming sentence" in v for v in check_units([u], {"P1": _probe_paper()}, KB))


def test_a_naming_provenance_without_the_sentence_is_caught():
    u = _probe_unit().model_copy(update={"naming_quote": None, "naming_section": None, "naming_start": None,
                                         "naming_end": None})
    assert any("target 'IL-17A' is not in its naming" in v for v in check_units([u], {"P1": _probe_paper()}, KB))


def test_a_unit_with_a_naming_sentence_round_trips():
    u = _probe_unit()
    assert EvidenceUnit.model_validate_json(u.model_dump_json()) == u
