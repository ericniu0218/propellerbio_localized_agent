"""Tests for the fabrication guard.

These matter more than any other tests in the repo. If quote verification
breaks, invented findings reach a scientist's report carrying a citation that
looks checked. Treat a failure here as a stop-work.

Runs with no GPU, no network, no model.
"""

from bedrock.litreview.extract.verify import normalize, verify_quote

SOURCE = (
    "Patients were randomised 2:1. At week 12, PASI-75 was achieved by 78% of "
    "patients receiving guselkumab versus 8% receiving placebo (n=507). "
    "Adverse events were balanced across arms."
)


def test_an_exact_sentence_verifies():
    quote = "At week 12, PASI-75 was achieved by 78% of patients receiving guselkumab versus 8% receiving placebo (n=507)."
    assert verify_quote(quote, SOURCE) == "exact"


def test_an_invented_sentence_is_rejected():
    """The whole point. A fluent, plausible sentence that nobody wrote."""
    quote = "At week 24, PASI-90 was achieved by 91% of patients receiving guselkumab."
    assert verify_quote(quote, SOURCE) == "failed"


def test_a_paraphrase_is_rejected():
    """Paraphrase is not citation. If the model reworded it, we cannot point a
    reviewer at a sentence, so the finding does not survive."""
    quote = "Guselkumab produced a much better skin response than placebo did at the twelve week mark."
    assert verify_quote(quote, SOURCE) == "failed"


def test_curly_quotes_and_ligatures_still_verify():
    """PDF typography must not destroy a real quote."""
    source = "The eﬀect was signiﬁcant in the treated group, and the authors’ conclusion followed."
    quote = "The effect was significant in the treated group, and the authors' conclusion followed."
    assert verify_quote(quote, source) in {"exact", "fuzzy"}


def test_a_word_hyphenated_across_a_line_break_still_verifies():
    """GROBID leaves these in constantly; a strict match would bin good findings."""
    source = "Treatment reduced joint inflamma-\ntion in the treated animals over fourteen days."
    quote = "Treatment reduced joint inflammation in the treated animals over fourteen days."
    assert verify_quote(quote, source) in {"exact", "fuzzy"}


def test_collapsed_whitespace_still_verifies():
    source = "Serum   levels\n\n  fell by   forty percent after the second dose was given."
    quote = "Serum levels fell by forty percent after the second dose was given."
    assert verify_quote(quote, source) in {"exact", "fuzzy"}


def test_a_very_short_quote_is_rejected():
    """A short string appears in almost any paper by chance, so matching one
    proves nothing. Rejecting it is the conservative call."""
    assert verify_quote("PASI-75", SOURCE) == "failed"


def test_empty_input_is_rejected():
    assert verify_quote("", SOURCE) == "failed"
    assert verify_quote("a real looking sentence of sufficient length", "") == "failed"


def test_normalize_is_idempotent():
    once = normalize(SOURCE)
    assert normalize(once) == once


# --- a fuzzy match is typography, never meaning ----------------------------------

import pytest  # noqa: E402

from bedrock.litreview.extract.verify import locate_quote, normalize_with_map  # noqa: E402

NULL_RESULT = (
    "Background sentence. Anti-IL-23 treatment did not significantly reduce joint swelling "
    "compared with vehicle control in 24 mice (p=0.41). The next sentence follows."
)


@pytest.mark.parametrize("changed", [
    "Anti-IL-23 treatment did significantly reduce joint swelling compared with vehicle control in 24 mice (p=0.41).",
    "Anti-IL-23 treatment did not significantly reduce joint swelling compared with vehicle control in 21 mice (p=0.41).",
    "Anti-IL-23 treatment did not significantly reduce joint swelling compared with vehicle control in 24 mice (p=0.01).",
    "Anti-IL-23 treatment did not significantly induce joint swelling compared with vehicle control in 24 mice (p=0.41).",
    "Anti-IL-23 treatment did not significantly reduce joint swelling compared with vehicle control in 24 rats (p=0.41).",
])
def test_a_quote_that_changes_the_meaning_fails(changed):
    """Each of these scored as a 'fuzzy' match before: one word or digit off
    a long sentence is well above the similarity threshold. Every one of them
    says something the paper did not."""
    assert verify_quote(changed, NULL_RESULT) == "failed"
    assert locate_quote(changed, NULL_RESULT).reason.startswith("meaning changed")


def test_a_harmless_word_swap_is_still_fuzzy_and_the_paper_text_is_kept():
    source = ("Moreover, the frequency of IL-17A producing gamma delta T cells in the draining lymph nodes of the "
              "treated animals was comparable between the two independent cohorts examined in this study.")
    quote = ("Furthermore, the frequency of IL-17A producing gamma delta T cells in the draining lymph nodes of the "
             "treated animals was comparable between the two independent cohorts examined in this study.")
    located = locate_quote(quote, source)
    assert located.status == "fuzzy"
    assert located.text == source  # the paper's words, not the model's


def test_the_located_text_is_the_source_at_its_offsets():
    located = locate_quote("did not significantly reduce joint swelling compared with vehicle control", NULL_RESULT)
    assert located.status == "exact"
    assert NULL_RESULT[located.start:located.end] == located.text


def test_a_quote_is_widened_to_its_whole_sentence():
    """The model quoted from the middle of the sentence and lost what was done
    to the target: the audit's FABP5 case. The whole sentence is shown."""
    source = "Moreover, specific deletion of Fabp5 in keratinocytes reduced the infiltration of neutrophils. Other."
    located = locate_quote("Fabp5 in keratinocytes reduced the infiltration of neutrophils", source)
    assert located.text == "Moreover, specific deletion of Fabp5 in keratinocytes reduced the infiltration of neutrophils."


def test_widening_stops_at_sentence_ends():
    located = locate_quote("did not significantly reduce joint swelling", NULL_RESULT)
    assert located.text.startswith("Anti-IL-23") and located.text.endswith("(p=0.41).")


def test_ligature_offsets_map_back_to_the_source():
    source = "The eﬀect was signiﬁcant in every treated group of animals studied."
    located = locate_quote("The effect was significant in every treated group of animals studied.", source)
    assert located.status == "exact" and located.text == source


@pytest.mark.parametrize("text", [
    NULL_RESULT, "Treatment reduced joint inflamma-\ntion in  the treated\n\nanimals.",
    "The eﬀect — “signiﬁcant” — was seen.", "  leading and trailing  ", "ΚΑΛΗΜΕΡΑ ΣΑΣ İstanbul", "été",
])
def test_the_mapped_normalisation_matches_the_fast_one(text):
    assert normalize_with_map(text)[0] == normalize(text)


# --- found in review ------------------------------------------------------------------


def test_a_quote_that_begins_its_sentence_is_not_glued_to_the_one_before():
    source = "Mice were dosed with 5 mg/kg. IL-17 blockade reduced skin thickness by 40% in treated animals. Next."
    located = locate_quote("IL-17 blockade reduced skin thickness by 40% in treated animals", source)
    assert located.text == "IL-17 blockade reduced skin thickness by 40% in treated animals."


def test_a_greek_letter_swap_changes_the_molecule():
    source = "IFN-γ production was strongly reduced in lesional skin of treated animals over two weeks."
    quote = "IFN-α production was strongly reduced in lesional skin of treated animals over two weeks."
    assert verify_quote(quote, source) == "failed"


def test_a_preposition_is_not_a_meaning_word():
    """Splitting "in vitro" once made "in" a meaning word, failing good quotes."""
    source = ("Expression of IL-17A was markedly elevated in the lesions of every patient examined in the "
              "discovery cohort and in the independent validation cohort as well.")
    quote = ("Expression of IL-17A was markedly elevated of the lesions of every patient examined in the "
             "discovery cohort and in the independent validation cohort as well.")
    assert locate_quote(quote, source).status == "fuzzy"


@pytest.mark.parametrize("quote", [
    "Anti-IL-23 treatment did not significantly reduce joint swelling compared with vehicle control in 24 mice (p=0.41).",
    "Anti-IL-23 treatment did significantly reduce joint swelling compared with vehicle control in 24 mice (p=0.41).",
    "treatment did not significantly reduce joint swelling",
])
def test_verify_quote_and_locate_quote_agree(quote):
    assert verify_quote(quote, NULL_RESULT) == locate_quote(quote, NULL_RESULT).status


def test_a_percent_sign_beside_an_unchanged_number_is_typography():
    source = ("The nirsevimab acceptance rate among the parents of all eligible infants in the region "
              "was 91.6% [89.1%-94.2%] during the first season of the programme.")
    quote = ("The nirsevimab acceptance rate among the parents of all eligible infants in the region "
             "was 91.6% [89.1-94.2%] during the first season of the programme.")
    assert locate_quote(quote, source).status == "fuzzy"


def test_a_wrong_number_beside_the_right_words_still_fails():
    """Found in the stored RSV report: the model attached 71.8% to a reason the
    paper gives 55.0% for."""
    source = ("The most important reason for 55.0% of respondents was that the immunization was too new, "
              "and 71.8% said they wanted more information first.")
    quote = "The most important reason for 71.8% of respondents was that the immunization was too new."
    assert verify_quote(quote, source) == "failed"


@pytest.mark.parametrize("source, quote, expected", [
    ("We saw it. Anti-IL-17A treatment reduced PASI scores compared with vehicle (Fig. 2A) but not in "
     "Il17ra knockout mice. Then more.",
     "Anti-IL-17A treatment reduced PASI scores compared with vehicle",
     "Anti-IL-17A treatment reduced PASI scores compared with vehicle (Fig. 2A) but not in Il17ra knockout mice."),
    ("Intro here. As shown by Smith et al. (2019) IL-23 blockade did not reduce the lesions in any group. End.",
     "IL-23 blockade did not reduce the lesions in any group",
     "As shown by Smith et al. (2019) IL-23 blockade did not reduce the lesions in any group."),
])
def test_abbreviations_are_not_sentence_ends(source, quote, expected):
    assert locate_quote(quote, source).text == expected


# --- found in the final review ---------------------------------------------------------


def test_a_contracted_negation_that_is_dropped_fails():
    source = ("Background. Treatment with the anti-IL-17A antibody didn't reduce the PASI score in adult "
              "patients over the twelve weeks.")
    quote = ("Treatment with the anti-IL-17A antibody did reduce the PASI score in adult patients over the "
             "twelve weeks.")
    assert verify_quote(quote, source) == "failed"


def test_a_sentence_ending_in_a_single_letter_is_still_an_end():
    source = "Cells were incubated at 37 °C. IL-17A reduced keratinocyte proliferation by 30% in culture. Next."
    located = locate_quote("IL-17A reduced keratinocyte proliferation by 30% in culture", source)
    assert located.text == "IL-17A reduced keratinocyte proliferation by 30% in culture."


PROBE_PAD = ("Psoriasis is a chronic inflammatory skin disease affecting millions of people worldwide. "
             "We therefore designed a series of experiments to examine the pathway. ")


@pytest.mark.parametrize("sentence, before, after", [
    ("Neutralisation of IL-17A reduced neutrophil infiltration into the epidermis of treated animals in this model.",
     "IL-17A", "IL-17F"),
    ("Expression of IL-36 gamma was strongly upregulated in lesional skin compared with non-lesional skin from the same patients.",
     "upregulated", "downregulated"),
    ("Serum IL-17A concentration was positively correlated with PASI score across the whole cohort of enrolled patients.",
     "positively", "negatively"),
    ("Mice that received two doses of the anti-IL-23 antibody before challenge developed markedly thinner ear lesions.",
     "two doses", "three doses"),
    ("Imiquimod application produced a twofold rise in epidermal thickness relative to vaseline-treated control skin.",
     "twofold", "threefold"),
    ("Weekly semaglutide produced sustained body weight reduction over the full treatment period in adults with obesity.",
     "semaglutide", "liraglutide"),
    ("Genetic ablation of TNF in keratinocytes protected the animals from imiquimod-induced psoriasiform dermatitis.",
     "TNF in", "TNFR in"),
    ("Anti-IL-23 antibody given before imiquimod challenge reduced ear thickness in the treated group of animals.",
     "before", "after"),
    ("The compound lowered fasting glucose in obese mice fed a high-fat diet for twelve consecutive weeks.",
     "obese", "lean"),
    ("The effect on body weight was observed only in male mice receiving the highest dose of the compound.",
     "male", "female"),
    ("Body weight was reduced by 15% after 12 weeks of daily treatment with the dual agonist in the trial.",
     "weeks", "days"),
    ("Knockout mice gained less body weight than littermates when fed a high-fat diet for sixteen consecutive weeks.",
     "high-fat", "low-fat"),
    ("Mice were dosed at 10 mg/kg intraperitoneally twice weekly for the whole duration of the imiquimod challenge.",
     "10 mg/kg", "10 g/kg"),
], ids=["numbered-name-letter", "up-down", "positive-negative", "number-word", "fold-word", "drug-name",
        "target-name", "before-after", "obese-lean", "male-female", "time-unit", "high-low", "dose-unit"])
def test_a_one_word_change_of_meaning_fails_the_fuzzy_match(sentence, before, after):
    """Each of these matched as 'fuzzy' in an adversarial probe (September
    2026): one word off a long sentence stays above the threshold, and none
    of these words was a number, a listed meaning word or a Greek letter."""
    source = PROBE_PAD + sentence + " Further work is needed to confirm these findings in other models."
    located = locate_quote(sentence.replace(before, after, 1), source)
    assert located.status == "failed"
    assert located.reason.startswith("meaning changed")
    assert locate_quote(sentence, source).status == "exact"


# --- results reported as the paper's own, when they are someone else's --------

from bedrock.litreview.extract.verify import other_work_reason  # noqa: E402

# Verbatim, from evals/litreview/hand_audit/qwen_asthma.md "## S9": a
# birth-cohort finding the paper quotes from its own Introduction, its
# reference number stripped by Europe PMC down to the bracket that held it.
S9_QUOTE = (
    "In a birth cohort study, MAIT cell frequency at 1 year of age was associated with a decreased risk of "
    "asthma by age 7, suggesting that increased MAIT cells early in life could be protective, potentially "
    "reflecting immune programing to T helper 1 (Th1) responses []."
)


def test_the_asthma_s9_quote_is_dropped():
    assert other_work_reason(S9_QUOTE, "results") is not None


@pytest.mark.parametrize("quote", [
    "In a birth cohort study, MAIT cell frequency at 1 year of age was associated with a decreased risk of "
    "asthma by age 7 [].",
    "MAIT cells may modulate asthmatic responses [12].",
    "MAIT cells may modulate asthmatic responses [3, 4].",
    "MAIT cells may modulate asthmatic responses [5-7].",
    "MAIT cells may modulate asthmatic responses [, ].",
    "MAIT cells may modulate asthmatic responses [–].",  # en dash: "[–]"
])
def test_stripped_reference_brackets_are_dropped(quote):
    assert other_work_reason(quote, "results") is not None


@pytest.mark.parametrize("quote", [
    "IL-23 blockade reduced ear thickness in mice, consistent with earlier work (Smith et al., 2019).",
    "IL-23 blockade reduced ear thickness in mice, consistent with earlier work (Smith and Jones 2019).",
    "IL-23 blockade reduced ear thickness in mice, consistent with earlier work (Smith, 2019a).",
])
def test_author_year_citations_are_dropped(quote):
    assert other_work_reason(quote, "results") is not None


@pytest.mark.parametrize("quote", [
    "Previous studies have shown that MAIT cells may modulate asthmatic responses.",
    "A recent study found that cold exposure increases brown fat activity in mice.",
    "Another study showed that IL-23 blockade reduces psoriasiform inflammation.",
    "It has been reported that IL-17A drives keratinocyte proliferation in psoriasis.",
    "Prior work has linked MAIT cell frequency to a decreased risk of asthma.",
    "Earlier studies have linked MAIT cell frequency to a decreased risk of asthma.",
])
def test_a_reported_other_work_lead_in_is_dropped(quote):
    assert other_work_reason(quote, "results") is not None


def test_the_lead_in_only_matters_at_the_start_of_the_sentence():
    """"has been shown to" is on the lead-in list, but a real result routinely
    contains the words mid-sentence ("X has been shown to...") - only a quote
    that opens with it is someone else's work."""
    assert other_work_reason("IL-23 has been shown to drive psoriasis in treated mice.", "results") is None


@pytest.mark.parametrize("quote", [
    # A confidence interval: letters and a percent sign inside the brackets.
    "The nirsevimab acceptance rate was 91.6% [95% CI 89.1-94.2%] in the treated cohort.",
    # A figure or table reference: parentheses, not a stripped citation.
    "Anti-IL-17A treatment reduced PASI scores compared with vehicle (Fig. 2A) in treated animals.",
    "The full dose-response data are given in (Table 1) for the treated cohort.",
    # A concentration, not a citation bracket: letters and a plus sign inside.
    "Intracellular [Ca2+]i rose sharply within minutes of ionomycin treatment in the stimulated cells.",
    # A sample size or a p-value: parentheses, and no capitalised name at all.
    "Anti-IL-23 treatment did not significantly reduce ear thickening compared with vehicle control (n = 12).",
    "Anti-IL-23 treatment did not significantly reduce ear thickening compared with vehicle control (p < 0.05).",
    # An abbreviation in parentheses, not an author and year.
    "Levels of the cytokine (IL-17A) were markedly elevated in lesional skin of the treated animals.",
    # An ordinary result with no bracket, citation or lead-in at all.
    "Anti-IL-23 antibody given before imiquimod challenge reduced ear thickness by 60% in treated mice.",
])
def test_findings_with_no_citation_marker_are_kept(quote):
    assert other_work_reason(quote, "results") is None


def test_a_section_the_parser_could_only_call_other_is_dropped():
    """parsing/sections.py folds an Introduction or Background heading into
    the canonical section "other" - there is no canonical "introduction" to
    check for, so "other" is the real name of the rule the audit asked for."""
    assert other_work_reason("A perfectly ordinary sentence with no bracket or lead-in at all here.", "other") \
        == "other_work:section"


def test_a_results_section_with_no_marker_is_not_dropped_by_section_alone():
    assert other_work_reason("A perfectly ordinary sentence with no bracket or lead-in at all here.", "results") \
        is None


# --- a comparison to earlier work is not the same as reporting it -------------
#
# Found by running other_work_reason over 593 real evidence units from stored
# qwen3:8b reports: two units (E257, E270) are the paper's own finding, with a
# citation marker sitting inside a clause that only compares the result with
# earlier work, not the finding itself. Both quotes below are verbatim from
# those units. The four "must stay dropped" quotes are the audit's own
# counter-examples: each opens as if it were the same shape, but is actually
# summarising someone else's result, not fencing off a comparison next to the
# paper's own.

E257_QUOTE = (
    "We also observed that the LPA 2 agonist GRI977143 was effective in suppressing allergic responses only "
    "in treatment before antigen challenge, unlike the previously known negative regulation of dendritic "
    "cell activation by LPA 2 [,"
)

E270_QUOTE = (
    "Our findings of exacerbated allergic inflammation upon transfer of G36‐treated MAIT cells are "
    "consistent with research demonstrating the role of GPER‐1 in regulating immune responses []."
)


@pytest.mark.parametrize("quote", [E257_QUOTE, E270_QUOTE])
def test_a_comparison_clause_beside_the_papers_own_result_is_kept(quote):
    assert other_work_reason(quote, "results") is None


@pytest.mark.parametrize("quote", [
    # "we demonstrated" is past-tense prior-work framing, not this paper
    # reporting its own result now - deliberately not on the opener list.
    "Furthermore, we demonstrated that AhR in type II alveolar epithelial (AT2) cells preserves autophagic "
    "flux, preventing epithelial damage in the model [].",
    # "In our studies we have found" reads like the opener list but is not on
    # it: "we have found" (not "we found"), "our studies" (not "this study").
    "In our studies we have found that steady levels of estrogen decreased type 2 inflammation in the "
    "airway [].",
    # A citation to someone else's study by name, with no own-work opener at all.
    "For instance, a study by Hinks et al. [] observed increased AhR expression in bronchial epithelium.",
    # "These findings" is not "Our findings" - a real difference: the sentence
    # is summarising someone else's emerging data, not this paper's result.
    "These findings align with emerging data demonstrating that environmental pollutants such as diesel "
    "exhaust exacerbate airway hyperresponsiveness in asthma via AhR-dependent pathways [].",
])
def test_prior_work_framing_that_only_looks_like_the_papers_own_stays_dropped(quote):
    assert other_work_reason(quote, "results") is not None
