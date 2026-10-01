"""`bedrock.agentpack.citation_marker`: is a finished answer presenting
itself as literature-cited?

Unit tests only, the same split `tests/test_recall.py` uses for
`wants_recall`: the pure trigger is tested directly here, and where it gets
applied (`AgentPackApp._finish`) is exercised by the app-level tests.
"""

from __future__ import annotations

from bedrock.agentpack.citation_marker import CITATION_MARKER, looks_cited, with_citation_notice

# ---------------------------------------------------------------------------
# Answers that should be marked
# ---------------------------------------------------------------------------


def test_a_pmid_is_cited():
    assert looks_cited("This is supported by the literature (PMID: 34521234).")


def test_a_doi_is_cited():
    assert looks_cited("Full text: https://doi.org/10.1038/s41586-020-1234-5")


def test_et_al_is_cited():
    assert looks_cited("Smith et al. showed that the treatment reduced inflammation.")


def test_et_al_without_a_trailing_period_is_still_cited():
    assert looks_cited("Smith et al (2020) showed a reduction in symptoms.")


def test_a_bracketed_reference_number_is_cited():
    assert looks_cited("The treatment reduced inflammation significantly [3].")


def test_several_bracketed_reference_numbers_are_cited():
    assert looks_cited("Two independent trials confirmed this [1, 2].")


def test_according_to_with_a_year_is_cited():
    assert looks_cited("According to a 2019 study, the treatment reduced inflammation.")


def test_according_to_with_a_year_further_away_is_still_cited():
    assert looks_cited(
        "According to the most comprehensive meta-analysis on the subject, published in 2021, "
        "the effect was modest."
    )


# ---------------------------------------------------------------------------
# Answers that should NOT be marked
# ---------------------------------------------------------------------------


def test_an_ordinary_answer_with_no_citation_shape_is_not_cited():
    assert not looks_cited(
        "Based on general immunology, inhibiting this pathway could reduce inflammation. "
        "You would want to confirm this experimentally before moving forward."
    )


def test_mentioning_a_year_alone_is_not_cited():
    """Half of biology has a year in it - a bare year must not fire this."""
    assert not looks_cited("In 2020, the FDA approved several new therapies for psoriasis.")


def test_according_to_without_a_year_nearby_is_not_cited():
    assert not looks_cited("According to the lab's own protocol, samples are stored at -80C.")


def test_a_code_style_bracket_is_not_a_citation():
    """`arr[0]` has no space (or start-of-text) before the bracket - unlike a
    citation, which is what the lookbehind in `_BRACKET_CITATION` rules out."""
    assert not looks_cited("Check arr[0] for the first sample, and arr[12] for the last.")


def test_empty_text_is_not_cited():
    assert not looks_cited("")


# ---------------------------------------------------------------------------
# with_citation_notice
# ---------------------------------------------------------------------------


def test_the_marker_is_appended_when_cited():
    text = "Smith et al. found a significant reduction in symptoms."
    marked = with_citation_notice(text)
    assert marked.startswith(text)
    assert CITATION_MARKER in marked
    assert "/review" in marked


def test_the_marker_is_not_appended_when_not_cited():
    text = "This is a general answer with no citations at all."
    assert with_citation_notice(text) == text
