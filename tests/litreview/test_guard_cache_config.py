"""The citation guard, the call cache and the air-gap check."""

import pytest

from bedrock.litreview.config import ConfigError, assert_local_url, load_settings
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient, strict_schema
from bedrock.litreview.llm.cache import CachedClient, CallCache
from bedrock.litreview.models import DraftSection, EvidenceUnit
from bedrock.litreview.report.guard import CitationGuard, numbers_in
from bedrock.litreview.stages.s0_question import parse_question


def unit(**kw) -> EvidenceUnit:
    base = dict(
        id="u1", paper_id="P1", question_id="q", target="IL23A", target_raw="IL-23", direction="block",
        modality="antibody", intervention="guselkumab", disease="psoriasis", disease_raw="plaque psoriasis",
        evidence_type="human_clinical", species="human", species_raw="patients", outcome="improved",
        effect_detail="PASI-90 in 73% vs 3%", n=837, combo_with=None, target_is_primary=True,
        quote="At week 16, PASI-90 was achieved by 73% of patients receiving guselkumab versus 3% on placebo.",
        quote_status="exact", section="abstract", extracted_by_model="fake", prompt_version="2",
    )
    base.update(kw)
    return EvidenceUnit(**base)


TRACED = dict(
    context_quote="A total of 837 patients with moderate-to-severe plaque psoriasis were randomised.",
    context_section="methods", context_start=0, context_end=82,
    provenance={"target": "chunk", "intervention": "quote", "disease_raw": "methods", "context": "methods",
                "species": "quote", "n": "methods", "effect_detail": "quote"},
)


@pytest.fixture()
def guard():
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    return CitationGuard(spec, load_knowledge(), {"E1": unit(**TRACED)}, paper_labels={"P1": "Garcia 2017 Lancet"})


def test_a_grounded_sentence_passes(guard):
    """837 is not in the quote; it is in the Methods sentence provenance found
    and anchored, which a reader can see beside the quote."""
    verdict = guard.check("Guselkumab, which blocks IL-23, produced PASI-90 in 73% of 837 patients by week 16.", ["E1"])
    assert verdict.kept, verdict.reason


def test_a_sample_size_only_the_model_wrote_is_not_evidence():
    """The loophole: `n=837` sat on a unit whose quote and context say nothing
    of 837, and the guard used to accept it."""
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    g = CitationGuard(spec, load_knowledge(), {"E1": unit(provenance={"target": "chunk", "n": "none"})})
    verdict = g.check("Guselkumab produced PASI-90 in 73% of 837 patients by week 16.", ["E1"])
    assert not verdict.kept and "837" in verdict.reason


def test_a_result_only_the_model_wrote_is_not_evidence():
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    g = CitationGuard(spec, load_knowledge(), {"E1": unit(effect_detail="PASI-90 in 73% vs 3%; PASI-75 in 91%")})
    assert not g.check("PASI-75 was reached by 91% of patients on guselkumab.", ["E1"]).kept


ALOX5 = dict(
    target="ALOX5", target_raw="ALOX5", intervention=None, species="other", species_raw="", n=None,
    evidence_type="in_vitro", effect_detail="",
    quote="ALOX5 and IL17RA expression correlated positively (r = 0.89) across skin and spleen tissues.",
    provenance={"target": "quote"},
)


@pytest.mark.parametrize("sentence", [
    "ALOX5 and IL17RA expression correlated positively (r=0.89) in human tissues.",
    "ALOX5 and IL17RA expression correlated positively (r=0.89) in mice.",
    "ALOX5 and IL17RA expression correlated positively (r=0.89) in patients.",
])
def test_who_was_studied_must_be_said_by_the_cited_text(sentence):
    """The audit's cases: the quote names no species, so neither may the report."""
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    g = CitationGuard(spec, load_knowledge(), {"E1": unit(**ALOX5)})
    verdict = g.check(sentence, ["E1"])
    assert not verdict.kept and "who was studied" in verdict.reason


def test_who_was_studied_passes_when_the_paper_says_it():
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    g = CitationGuard(spec, load_knowledge(), {"E1": unit(**{**ALOX5, "context_quote": "Tissues were taken from C57BL/6 mice.", "provenance": {"target": "quote", "species": "methods", "context": "methods"}})})
    assert g.check("ALOX5 and IL17RA expression correlated positively (r=0.89) in mice.", ["E1"]).kept


def test_an_invented_number_is_caught(guard):
    verdict = guard.check("Guselkumab produced PASI-90 in 91% of patients.", ["E1"])
    assert not verdict.kept and "91" in verdict.reason


def test_an_outside_drug_is_caught(guard):
    verdict = guard.check("Risankizumab showed the same effect.", ["E1"])
    assert not verdict.kept and "Risankizumab" in verdict.reason


def test_the_subject_and_the_papers_year_are_always_citable(guard):
    assert guard.check("In psoriasis, a 2017 trial of guselkumab reported benefit.", ["E1"]).kept


def test_numbers_inside_names_are_not_statistics(guard):
    # Raw extraction sees the "2" in "type 2" (the "23" and "90" are glued to
    # names); the guard masks known names before it looks for numbers.
    assert numbers_in("IL-23 and type 2 diabetes, PASI-90 in 73%") == {"2", "73"}
    assert guard.check("Blocking IL-23 with guselkumab helped patients.", ["E1"]).kept


def test_uncited_and_unknown_citations_are_caught(guard):
    assert guard.check("Something true.", []).reason == "no citation"
    assert "unknown" in guard.check("Something.", ["E9"]).reason


def test_citation_ids_are_constrained_by_the_schema():
    schema = strict_schema(DraftSection.model_json_schema(), enums={"citations": ["E1", "E2"]})
    items = schema["properties"]["sentences"]["items"]
    assert items["properties"]["citations"]["items"]["enum"] == ["E1", "E2"]
    assert items["required"] == ["citations", "text"], "citations are generated before the sentence"
    assert "$defs" not in schema


def test_the_cache_replays_and_keys_on_the_model_called(tmp_path):
    fake = FakeLLMClient(replies=["one", "two", "three"])
    client = CachedClient(fake, CallCache(tmp_path))
    assert client.complete("p", model="a") == "one"
    assert client.complete("p", model="a") == "one", "a repeat is served from the cache"
    assert client.complete("p", model="b") == "two", "a different model is a different call"
    assert len(fake.prompts) == 2


def test_a_cache_that_does_not_read_re_asks(tmp_path):
    fake = FakeLLMClient(replies=["one", "two"])
    CachedClient(fake, CallCache(tmp_path)).complete("p", model="a")
    assert CachedClient(fake, CallCache(tmp_path), read=False).complete("p", model="a") == "two"


def test_a_truncated_cache_entry_is_a_miss_not_a_crash(tmp_path):
    cache = CallCache(tmp_path)
    key = CallCache.key("p", "m", {})
    cache.put(key, "p", "r", "m")
    cache._path(key).write_text("{not json")
    assert cache.get(key) is None


def test_public_model_endpoints_are_refused():
    assert_local_url("http://localhost:8000/v1")
    assert_local_url("http://10.0.4.12:8000/v1")
    assert_local_url("http://gpu-box.local:11434")
    with pytest.raises(ConfigError):
        assert_local_url("https://api.openai.com/v1")
    with pytest.raises(ConfigError):
        assert_local_url("http://8.8.8.8:8000")


def test_a_non_zero_temperature_is_refused(tmp_path):
    """A whole-file models.yaml override cannot buy its way out of determinism."""
    (tmp_path / "models.yaml").write_text(
        "min_context_window: 12288\nmodel_sets:\n  x:\n    medium: {name: m}\n"
        "jobs:\n  extract: medium\ndefaults:\n  temperature: 0.7\n"
    )
    with pytest.raises(ConfigError):
        load_settings(model_set="x", env={"BEDROCK_LITREVIEW_CONFIG_DIR": str(tmp_path)})


def test_ollama_cloud_models_are_refused_even_behind_localhost():
    from bedrock.litreview.config import assert_local_model

    assert_local_model("medgemma1.5:4b", {})
    with pytest.raises(ConfigError):
        assert_local_model("gpt-oss:120b-cloud", {})
    assert_local_model("gpt-oss:120b-cloud", {"LLM_ALLOW_CLOUD_MODELS": "1"})


def test_a_null_claim_needs_a_null_result(guard):
    """The MedGemma run wrote 'did not improve' citing only findings where it did."""
    verdict = guard.check("Guselkumab did not improve PASI-90 in patients.", ["E1"])
    assert not verdict.kept and "no effect" in verdict.reason


def test_label_talk_is_not_evidence(guard):
    verdict = guard.check("Blocking IL-23 with guselkumab improved patient outcomes.", ["E1"])
    assert not verdict.kept and "label" in verdict.reason


def test_a_repeated_sentence_is_recognised():
    from bedrock.litreview.stages.s7_write import REPEAT_OVERLAP, _content_words, _overlap

    a = _content_words("The findings underscore the critical role of m6A RNA modification in psoriasis pathogenesis.")
    b = _content_words("The findings underscore the critical role of m6A RNA modification in psoriasis pathogenesis "
                       "and unveil a novel regulatory mechanism of TASL-IRF5 signaling.")
    c = _content_words("c-Rel deficiency protected mice from the disease.")
    assert _overlap(a, b) >= REPEAT_OVERLAP
    assert _overlap(a, c) < REPEAT_OVERLAP


def test_thinking_traces_are_not_the_answer():
    from bedrock.litreview.llm.client import parse_json_reply, strip_thinking

    assert strip_thinking("<unused94>thought\nplan...<unused95>Target IL-23.") == "Target IL-23."
    assert strip_thinking("<think>hmm</think>\nAnswer.") == "Answer."
    assert strip_thinking("<unused94>thought\ncut off mid-thought") == "", "no answer began"
    assert parse_json_reply('<unused94>x<unused95>{"a": 1}') == {"a": 1}


def test_a_bound_presented_at_the_wrong_end_is_caught():
    """From the RSV report: "280.2 (95% UR: 268.8-290.6)" makes 268.8 the low
    end, and the report called it the upper range. Every number was in the
    evidence, so only the direction gives it away."""
    from bedrock.litreview.report.guard import bound_misuse

    ev = ("Across the countries studied, estimated YLL exceeded 280.2 (95% UR: 268.8-290.6) per 100,000 "
          "older adults and 311.5 (95% UR: 271.8-377.9) per 100,000 infants.")
    assert bound_misuse("YLL exceeded 268.8 per 100,000 older adults in the upper range of 95% uncertainty.", ev) \
        == "268.8"
    assert bound_misuse("YLL exceeded 280.2 per 100,000 older adults and 311.5 per 100,000 infants.", ev) is None
    assert bound_misuse("The lower bound of the range was 268.8.", ev) is None, "the right end is fine"
    assert bound_misuse("The upper bound was 290.6.", ev) is None


# --- found in review ------------------------------------------------------------------


def test_cell_types_and_life_stages_are_not_claims_about_who_was_studied():
    """"Th17 cells" is biology, not work in a dish; "adult mice" are mice."""
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    quote = "IL-23 blockade reduced IL-17-producing lymphocytes in the skin of 8-week-old C57BL/6 mice."
    g = CitationGuard(spec, load_knowledge(), {"E1": unit(**{**ALOX5, "target": "IL23A", "target_raw": "IL-23",
                                                             "quote": quote})})
    assert g.check("IL-23 blockade reduced Th17 cells in psoriatic skin.", ["E1"]).kept
    assert g.check("IL-23 blockade reduced IL-17-producing lymphocytes in adult mice.", ["E1"]).kept


# --- found by the benchmark ------------------------------------------------------------


def test_in_animals_needs_cited_text_about_animals():
    """RSV-A printed "studied ... in animals" citing a human modelling study:
    "animals" was in no class, so the check never fired."""
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    human = unit(**{**ALOX5, "quote": "ALOX5 expression rose in children, per the transmission model of the 2019 season."})
    mouse = unit(**{**ALOX5, "quote": "ALOX5 expression rose in the skin of C57BL/6 mice after imiquimod."})
    g = CitationGuard(spec, load_knowledge(), {"E1": human, "E2": mouse})
    assert not g.check("ALOX5 expression rose in animals.", ["E1"]).kept
    assert g.check("ALOX5 expression rose in animals.", ["E2"]).kept


def _mouse_guard(**quotes: str) -> CitationGuard:
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    units = {cid: unit(id=f"u{cid}", quote=q, target="IL17A", target_raw="IL-17A", intervention=None,
                       evidence_type="in_vivo", species="mouse", species_raw="mice", effect_detail="", n=None,
                       outcome="improved", provenance={"target": "quote", "species": "quote"})
             for cid, q in quotes.items()}
    return CitationGuard(spec, load_knowledge(), units, paper_labels={"P1": "Smith 2020 J Invest Dermatol"})


NULL_QUOTE = "Blocking IL-17A had no significant effect on epidermal thickness in imiquimod-treated mice."


@pytest.mark.parametrize("sentence, quote", [
    ("Blocking IL-17A reduced epidermal thickness in imiquimod-treated mice.", NULL_QUOTE),
    ("Blocking IL-17A significantly reduced epidermal thickness in mice.",
     "Blocking IL-17A did not significantly reduce epidermal thickness in mice."),
    ("IL-36 gamma expression was decreased in lesional skin of imiquimod-treated mice.",
     "IL-36 gamma expression was increased in lesional skin of imiquimod-treated mice."),
], ids=["effect-claimed-on-null", "negation-dropped", "direction-flipped"])
def test_a_sentence_that_reverses_its_cited_result_is_dropped(sentence, quote):
    """Found by an adversarial probe: every name and number in these is in
    the quote, and each says the opposite of it. The null result was kept even
    when the finding was labelled no_effect."""
    verdict = _mouse_guard(E1=quote).check(sentence, ["E1"])
    assert not verdict.kept


@pytest.mark.parametrize("sentence, quotes", [
    ("Blocking IL-17A had no significant effect on epidermal thickness in mice.", {"E1": NULL_QUOTE}),
    ("Blocking IL-17A prevented the increase in epidermal thickness in mice.",
     {"E1": "Blocking IL-17A reduced epidermal thickness in imiquimod-treated mice."}),
    ("Blocking IL-17A lowered epidermal thickness in mice.",
     {"E1": "Blocking IL-17A reduced epidermal thickness in imiquimod-treated mice."}),
    ("Blocking IL-17A reduced ear thickness in mice.",
     {"E1": "Blocking IL-17A reduced ear thickness but had no effect on scaling in imiquimod-treated mice."}),
    ("Blocking IL-17A reduced ear thickness in mice.",
     {"E1": NULL_QUOTE, "E2": "Blocking IL-17A reduced ear thickness in imiquimod-treated mice."}),
    ("Epidermal thickness was higher in untreated mice than in mice given anti-IL-17A.",
     {"E1": "Blocking IL-17A reduced epidermal thickness in imiquimod-treated mice."}),
], ids=["faithful-null", "prevented-the-increase", "synonym", "mixed-quote", "one-positive-citation",
        "comparison-rephrased"])
def test_a_faithful_sentence_beside_the_direction_checks_is_kept(sentence, quotes):
    verdict = _mouse_guard(**quotes).check(sentence, list(quotes))
    assert verdict.kept, verdict.reason


def test_a_result_cited_to_a_sentence_that_reports_none_is_dropped():
    """Found in a hand audit of an asthma report (qwen3:8b): the quote is a
    dose-finding trial's enrolment sentence, and the sentence invents an
    efficacy result while reusing its names and numbers."""
    quote = ("Key Results: Among 45 participants randomized to montelukast and 44 to placebo, median [IQR] ages "
             "were 6.6 [5.4, 10] and 6.7 [5.3, 9.9] years, with high-moderate pre-treatment exacerbation severity.")
    # The question here is psoriasis, so the disease is left out of the sentence.
    sentence = ("Montelukast reduced exacerbations in children with high-moderate pre-treatment severity, "
                "as shown in a study of 45 participants randomized to montelukast and 44 to placebo.")
    verdict = _mouse_guard(E1=quote).check(sentence, ["E1"])
    # In this default findings section, issue #8's broader check (no cited
    # quote reports any result at all) now catches this before the narrower
    # "claims a result its evidence does not report" check would.
    assert not verdict.kept and "reports a result" in verdict.reason


@pytest.mark.parametrize("sentence, quote", [
    ("Adverse events were infrequent and mild in mice treated with anti-IL-17A.",
     "Adverse events were infrequent and mild in mice given anti-IL-17A."),
    ("Blocking IL-17A caused no increase in heart rate in mice.",
     "There was no increase in heart rate at any dose of anti-IL-17A in mice."),
    ("Anti-IL-17A reduced inflammation in mice.",
     "Anti-IL-17A can quickly relieve symptoms and reduce the level of inflammation in mice."),
], ids=["safety-wording", "no-increase", "base-verb-forms"])
def test_a_result_in_plain_wording_still_counts_as_a_result(sentence, quote):
    verdict = _mouse_guard(E1=quote).check(sentence, ["E1"])
    assert verdict.kept, verdict.reason
