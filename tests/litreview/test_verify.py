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
