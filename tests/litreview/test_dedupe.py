"""Tests for finding deduplication.

The failure this guards against: a report saying "3 studies support this" when
it means "1 study, counted 3 times". That is not cosmetic — it is the report
being wrong about the weight of evidence, which is the one thing the client is
actually buying.
"""

from dataclasses import dataclass

from bedrock.litreview.extract.dedupe import dedupe_run, dedupe_within_paper


@dataclass
class FakeUnit:
    """Stands in for EvidenceUnit so these tests need no validation setup."""

    paper_id: str = "p1"
    target: str = "IL23A"
    direction: str = "block"
    disease: str = "psoriasis"
    evidence_type: str = "human_clinical"
    outcome: str = "improved"
    combo_with: str | None = None
    section: str = "results"
    quote_status: str = "exact"
    n: int | None = None
    effect_detail: str = ""
    quote: str = ""


def test_the_same_finding_in_abstract_and_results_collapses():
    units = [
        FakeUnit(section="abstract", effect_detail="improved skin scores"),
        FakeUnit(section="results", effect_detail="PASI-75 in 78% vs 8% placebo", n=507),
    ]
    kept = dedupe_within_paper(units)
    assert len(kept) == 1
    assert kept[0].section == "results", "the results version carries the real numbers"


def test_the_discussion_restatement_loses():
    """Authors overstate in the discussion, so it ranks last."""
    units = [FakeUnit(section="discussion"), FakeUnit(section="abstract")]
    assert dedupe_within_paper(units)[0].section == "abstract"


def test_richer_wins_within_one_section():
    units = [
        FakeUnit(section="results", effect_detail="improved", n=None, quote_status="fuzzy"),
        FakeUnit(section="results", effect_detail="PASI-75 in 78% vs 8%", n=507, quote_status="exact"),
    ]
    assert dedupe_within_paper(units)[0].n == 507


def test_different_outcomes_are_different_findings():
    """A drug that helped skin but not joints is two findings, not one."""
    units = [
        FakeUnit(disease="psoriasis", outcome="improved"),
        FakeUnit(disease="arthritis", outcome="no_effect"),
    ]
    assert len(dedupe_within_paper(units)) == 2


def test_two_papers_reporting_the_same_result_are_both_kept():
    """Independent replication is the strongest thing a literature review can
    show. Collapsing it across papers would hide exactly that."""
    units = [FakeUnit(paper_id="p1"), FakeUnit(paper_id="p2")]
    assert len(dedupe_run(units)) == 2


def test_a_combination_finding_is_distinct_from_a_single_target_one():
    """Pairwise work is the client's whole thesis; these must never merge."""
    units = [FakeUnit(combo_with=None), FakeUnit(combo_with="IL17A")]
    assert len(dedupe_within_paper(units)) == 2


def test_different_measurements_of_one_finding_are_both_kept():
    """Two readouts of one experiment carry two numbers the report needs.
    Weight of evidence is counted by papers, so keeping both inflates nothing."""
    units = [
        FakeUnit(effect_detail="epidermal thickness reduced by 45%"),
        FakeUnit(effect_detail="IL-17A mRNA reduced by 60%"),
        FakeUnit(section="abstract", effect_detail="reduced inflammation"),
        FakeUnit(section="abstract", effect_detail="thickness reduced by 45%"),
    ]
    kept = dedupe_within_paper(units)
    assert sorted(u.effect_detail for u in kept) == [
        "IL-17A mRNA reduced by 60%", "epidermal thickness reduced by 45%",
    ], "the restatements, with no new number, collapse"


def test_measurements_are_told_apart_by_the_quote_when_the_summary_was_blanked():
    """Provenance blanks an effect_detail the paper does not support; the two
    readouts must still survive, told apart by the paper's own sentences."""
    units = [
        FakeUnit(quote="Epidermal thickness was reduced by 45% in treated mice."),
        FakeUnit(quote="IL-17A mRNA was reduced by 60% in treated mice."),
    ]
    assert len(dedupe_within_paper(units)) == 2


def test_figure_numbers_and_p_values_are_not_new_measurements():
    units = [
        FakeUnit(section="abstract", quote="IL-17A deficiency reduced skin inflammation by 45% (P < 0.01)."),
        FakeUnit(quote="IL-17A deficiency reduced skin inflammation by 45% (Fig. 3B)."),
    ]
    assert len(dedupe_within_paper(units)) == 1
