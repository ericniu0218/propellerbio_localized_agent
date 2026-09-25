"""Exact means exact. The client asked for answers about the exact drug,
species or virus in the question, never about something close enough.

These tests pin down the matching rules that make that true — each one is a
near-miss that a naive substring match would let through.
"""

from bedrock.litreview.knowledge import TermMatcher, load_knowledge
from bedrock.litreview.models import ExtractedFinding
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s5_extract import entity_check

KB = load_knowledge()


def summary(canonical: str, text: str):
    return KB.get(canonical).matcher().summary(text)


def test_a_sibling_virus_is_not_the_virus():
    assert summary("respiratory syncytial virus", "human metapneumovirus (hMPV) infection") == (0, ["hMPV", "human metapneumovirus"])
    assert summary("respiratory syncytial virus", "BRSV in calves")[0] == 0
    assert summary("respiratory syncytial virus", "bovine respiratory syncytial virus")[0] == 0
    assert summary("respiratory syncytial virus", "anti-RSV antibodies and RSV-A strains")[0] == 2


def test_sars_cov_is_not_sars_cov_2_and_vice_versa():
    assert summary("sars-cov-2", "SARS-CoV infection in 2003")[0] == 0
    assert summary("sars-cov-2", "SARS-CoV-2 spike")[0] == 1
    assert summary("sars-cov-2", "severe acute respiratory syndrome coronavirus 2")[0] == 1


def test_a_longer_near_miss_swallows_the_shorter_exact_term():
    """'psoriatic' is psoriasis, 'psoriatic arthritis' is a different disease."""
    assert summary("psoriasis", "psoriatic arthritis patients") == (0, ["psoriatic arthritis"])
    assert summary("psoriasis", "psoriatic lesional skin")[0] == 1
    assert summary("psoriasis", "imiquimod-induced psoriasis-like inflammation")[0] == 1


def test_numbered_names_do_not_match_their_neighbours():
    il23 = TermMatcher(["IL-23"], near_miss_patterns=[r"IL-?\d+[A-Z]?"])
    assert il23.summary("IL-2 levels rose") == (0, ["IL-2"])
    assert il23.summary("IL-23R variants") == (0, ["IL-23R"])
    assert il23.exact_count("anti-IL-23 therapy and IL23 blockade") == 2


def test_influenza_subtypes_are_distinct():
    spec = parse_question("Which pathways should we block to prevent H5N1 infection?")
    entity = spec.required_entities[0]
    matcher = TermMatcher(entity.exact_terms, entity.near_miss_terms, entity.near_miss_patterns)
    assert matcher.summary("H1N1 pandemic strains") == (0, ["H1N1"])
    assert matcher.exact_count("highly pathogenic H5N1") == 1


def test_short_acronyms_are_case_sensitive():
    """'als' in running text is not ALS."""
    assert summary("amyotrophic lateral sclerosis", "we also measured totals")[0] == 0
    assert summary("amyotrophic lateral sclerosis", "patients with ALS")[0] == 1


def test_brand_and_code_names_are_the_same_drug_but_class_members_are_not():
    assert summary("semaglutide", "Wegovy 2.4 mg weekly")[0] == 1
    assert summary("semaglutide", "liraglutide 3.0 mg") == (0, ["liraglutide"])


def test_extracted_names_merge_onto_one_target():
    assert KB.normalize("anti-IL-23p19 antibody", "target") == "IL23A"
    assert KB.normalize("interleukin-23", "target") == "IL23A"
    assert KB.normalize("prefusion F", "target") == "RSV F"
    # Two names inside one string is ambiguous: keep it verbatim, never guess.
    assert KB.normalize("IL-23/IL-17 axis", "target") == "IL-23/IL-17 axis"


def _finding(**kw) -> ExtractedFinding:
    base = dict(quote="x" * 30, target="IL-4R", direction="block", modality="antibody", intervention=None,
                disease="skin inflammation", evidence_type="human_clinical", species="patients",
                outcome="improved", setting="treatment", effect_detail="", n=None, combo_with=None,
                target_is_primary=True, facet="clinical")
    base.update(kw)
    return ExtractedFinding(**base)


def test_a_finding_about_a_near_miss_disease_is_dropped_even_if_its_quote_mentions_the_subject():
    """Regression: the quote named psoriasis in passing, the finding was about atopic dermatitis."""
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    drop, _ = entity_check(
        _finding(disease="atopic dermatitis", quote="Unlike psoriasis, atopic dermatitis responded to IL-4R blockade."),
        spec, KB,
    )
    assert drop == "near_miss:atopic dermatitis"


def test_a_finding_about_another_known_disease_is_dropped():
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    drop, _ = entity_check(_finding(disease="rheumatoid arthritis"), spec, KB)
    assert drop == "other_disease:rheumatoid arthritis"


def test_a_neutral_disease_field_takes_its_context_from_the_paper():
    spec = parse_question("If I want to prevent psoriasis, what should I target?")
    assert entity_check(_finding(disease="imiquimod-induced skin inflammation"), spec, KB) == (None, "paper_context")
    assert entity_check(_finding(disease="plaque psoriasis"), spec, KB) == (None, "exact")


def test_a_paper_titled_after_a_sibling_is_excluded_even_if_it_mentions_the_subject():
    """Regression from the RSV sample corpus: a bovine RSV paper said 'RSV' once."""
    from bedrock.litreview.models import Paper
    from bedrock.litreview.stages.s2_gate import gate_papers

    spec = parse_question("What should I target to prevent RSV infection in infants?")
    paper = Paper(
        id="B1",
        title="Bovine respiratory syncytial virus utilizes the human IGF1R late in infection",
        abstract="BRSV is a major pathogen of calves. Like human RSV, BRSV fuses with host cells. "
                 "We show BRSV entry requires IGF1R.",
    )
    both = Paper(id="B2", title="RSV and bovine respiratory syncytial virus share entry factors",
                 abstract="BRSV and BRSV and BRSV.")
    gates = gate_papers([paper, both], spec)
    assert gates["B1"].status == "near_miss" and "mainly about" in gates["B1"].reason
    assert gates["B2"].status == "pass", "a title naming the subject itself is never overridden"


def test_a_report_sentence_about_a_near_miss_is_dropped():
    """From the RSV benchmark: swapping RSV for hMPV in a finished sentence
    got past the guard, because the table lists hMPV only as RSV's near-miss
    and `kb.scan` finds entities that are listed in their own right."""
    from bedrock.litreview.report.guard import CitationGuard
    from bedrock.litreview.models import EvidenceUnit

    spec = parse_question("What should I target to prevent RSV infection in infants?")
    unit = EvidenceUnit(
        id="u1", paper_id="P1", question_id=spec.id, target="RSV F protein", target_raw="RSV F",
        direction="block", modality="antibody", intervention="nirsevimab", disease="RSV infection",
        disease_raw="RSV", evidence_type="human_clinical", species="human", species_raw="infants",
        outcome="improved", effect_detail="high efficacy in RSV prophylaxis", n=None, combo_with=None,
        target_is_primary=True,
        quote="Nirsevimab, a long-acting MoAb, has shown high efficacy in RSV prophylaxis in newborns.",
        quote_status="exact", section="abstract", extracted_by_model="fake", prompt_version="4",
    )
    guard = CitationGuard(spec, KB, {"E1": unit}, paper_labels={"P1": "Diaz 2024 Lancet"})
    assert guard.check("Nirsevimab has shown high efficacy in RSV prophylaxis in newborns.", ["E1"]).kept
    swapped = guard.check("Nirsevimab has shown high efficacy in human metapneumovirus prophylaxis in newborns.", ["E1"])
    assert not swapped.kept and "metapneumovirus" in swapped.reason
