"""The target-proposal deck's content (report/slides.py), tested as data.

`outline()` builds the slides without python-pptx, so nothing here draws
anything (that is test_deck_render.py). What these check is what makes it safe
to hand a deck to someone who never opens the report: every statement is one
the report kept, with its citations; a field a paper does not support is never
shown; each kind of evidence lands on the slide the template gives it; a slot
with nothing in it says so; and the same report always gives the same slides.
"""

from __future__ import annotations


from bedrock.litreview.models import PaperSignals, Reference, Report, ReportSection, Sentence
from bedrock.litreview.report import slides
from bedrock.litreview.report.deck_model import FigureAsset
from tests.litreview.test_guard_cache_config import TRACED, unit

QUESTION = "Does blocking IL-23 clear plaque psoriasis in adults?"
SENTENCES = {  # text, section, cited short ids
    "answer": ("Guselkumab cleared skin in most adults with plaque psoriasis in a randomised trial.", ["E1"]),
    "human": ("In 837 adults, guselkumab produced PASI-90 in 73% of patients by week 16.", ["E1", "E2"]),
    "animal": ("Anti-IL-23 antibody reduced ear thickness by 60% in imiquimod-treated mice.", ["E3"]),
    "moa": ("IL-23 was raised in lesional skin and in cultured keratinocytes.", ["E4"]),
    "mixed": ("IL-23 blockade helped in mice and in adults.", ["E1", "E3"]),
    "negative": ("Briakinumab did not improve scores in adults compared with placebo.", ["E6"]),
    "prevention": ("Early IL-23 blockade delayed relapse in mice and in adults.", ["E1", "E3"]),
}
DROPPED = "Guselkumab cures psoriasis forever."


def _units(extra=()):
    traced = dict(TRACED["provenance"])
    base = dict(target="IL23A", **{k: v for k, v in TRACED.items() if k != "provenance"})
    rows = [
        unit(id="u1", paper_id="P1", provenance=traced, **base, modality="antibody"),
        unit(id="u2", paper_id="P2", target="IL17A", intervention="secukinumab", modality="antibody",
             provenance=dict(traced, target="quote"), disease="psoriasis"),
        unit(id="u3", paper_id="P3", evidence_type="in_vivo", species="mouse", species_raw="C57BL/6 mice", n=8,
             provenance=traced, outcome="improved", intervention="anti-IL-23 antibody"),
        unit(id="u4", paper_id="P4", evidence_type="in_vitro", species="cell_line", outcome="association",
             intervention=None, provenance=traced),
        # its agent was found nowhere in the paper: provenance "none", so it must never be shown
        unit(id="u5", paper_id="P5", intervention="ghostmab", provenance=dict(traced, intervention="none")),
        unit(id="u6", paper_id="P6", intervention=None, outcome="no_effect", provenance=traced),
        *extra,
    ]
    return rows


def _refs(n):
    return [Reference(number=i, paper_id=f"P{i}", citation=f"Author{i} AB, Lee K et al. A study of IL-23 {i}. *Journal* 20{10 + i}",
                      url=f"https://europepmc.org/article/MED/{i}", signals=PaperSignals(year=2010 + i),
                      evidence_ids=[f"E{i}"]) for i in range(1, n + 1)]


def _report(*, question_type="intervention_efficacy", entities=None, units=None, sections=None, n_refs=6, **kw):
    units = units if units is not None else _units()
    short = {u.id: f"E{i}" for i, u in enumerate(units, 1)}
    if entities is None:
        entities = [{"kind": "target", "canonical": "IL23A"}, {"kind": "disease", "canonical": "psoriasis"}]

    def w(key, section):
        text, cites = SENTENCES[key]
        return ReportSection(id=section, title=section.title(), kind="written", sentences=[Sentence(text=text, citations=cites)])

    secs = sections if sections is not None else [
        ReportSection(id="answer", title="Answer", kind="written", note="Evidence basis: [1] randomised trial.",
                      sentences=[Sentence(text=SENTENCES["answer"][0], citations=["E1"]),
                                 Sentence(text=DROPPED, citations=["E1"], kept=False, drop_reason="not in the evidence")]),
        ReportSection(id="targets", title="Candidate targets", kind="written", subsections=[
            ReportSection(id="targets:IL23A", title="IL23A", kind="written", note="2 papers.",
                          sentences=[Sentence(text=SENTENCES["mixed"][0], citations=["E1", "E3"])])]),
        w("moa", "mechanism"), w("human", "clinical"), w("animal", "preclinical"), w("negative", "negative"),
        w("prevention", "prevention"),
        ReportSection(id="safety", title="Safety", kind="written", sentences=[], note="No evidence for this section in the papers read."),
        ReportSection(id="gaps", title="Gaps", kind="generated", body_markdown="\n".join([
            "- 9 of 16 papers read were available as abstract only.",
            "- No tested effect in humans among the papers read for: IL17A [2]; Liver fat [3].",
            "- Figures and tables are not read."])),
        ReportSection(id="disagreements", title="Disagree", kind="generated", body_markdown="\n\n".join([
            "The study records below show design.",
            "How the studies line up: **IL23A** (Studies disagree).",
            "- **Independent study [1][6]**", "Garcia 2017; direction: **improved**.", "> quote"])),
        ReportSection(id="background", title="Background", kind="generated", body_markdown="\n\n".join([
            "Review articles found for this question, for context. **They are not evidence in this report**.",
            "**Wu 2020 Nat Rev** — *IL-23 in disease* · review · <https://example.org/wu>\n> IL-23 drives psoriasis.",
            "Also collected:\n- Roe 2019 — Other review <https://example.org/roe>"])),
    ]
    return Report(question_id="q", question=QUESTION, sections=secs, references=_refs(n_refs), evidence=units, short_ids=short,
                  interpretation={"question_type": question_type, "goal": "treat", "subject": "plaque psoriasis",
                                  "population": "adults", "entities": entities},
                  stats={"papers_collected": 40, "papers_read": 12, "papers_abstract_only": 3}, **kw)


def _by(deck, prefix):
    return next(s for s in deck if s.title.startswith(prefix))


def _text(slide):
    return [p.text for p in slide.paras]


# --- every statement is the report's ------------------------------------------------------------------------------


def test_sentence_slides_hold_only_kept_sentences_word_for_word_with_their_references():
    report = _report()
    refs, _ = slides._ref_map(report)
    kept = {}
    for section in report.sections:
        stack = [section]
        while stack:
            part = stack.pop()
            for s in part.sentences:
                marker = "".join(f"[{n}]" for n in slides._numbers(s, refs))
                kept[f"{s.text.rstrip()} {marker}".rstrip()] = s.kept
            stack += part.subsections
    deck = slides.outline(report)
    for prefix in ("Target biology", "Target POC (proof", "Target POC - animal"):
        for p in _by(deck, prefix).paras:
            assert kept.get(p.text) is True, p.text
    shown = [t for s in deck for t in _text(s)]
    assert not any(DROPPED in t for t in shown)
    # a subsection's sentence (per-target "Candidate targets") reaches the deck too
    assert any(SENTENCES["mixed"][0] in t for t in shown)


def test_the_deck_says_nothing_the_report_does_not():
    """Every paragraph anywhere is a kept sentence, a line the report's code wrote, or one of the deck's fixed
    gap and fill-in lines."""
    report = _report()
    refs, quotes = slides._ref_map(report)
    allowed = {"Key limitations", slides.NO_FINDING, slides.NO_COMBINATION, slides.NO_COMPETITOR,
               "Unmet need and opportunities: not in the papers read - fill in.",
               "Company, stage and unmet need: not in the papers read - fill in."}
    for section in report.sections:
        stack = [section]
        while stack:
            part = stack.pop()
            allowed |= {f"{s.text} " + "".join(f"[{n}]" for n in slides._numbers(s, refs)) for s in part.sentences if s.kept}
            stack += part.subsections
        allowed |= set(slides._bullets(section.body_markdown))
        if section.note:
            allowed.add(slides.plain(section.note))
    allowed |= {f"{c} [{refs[c]}] “{q}”" for c, q in quotes.items()}
    allowed |= {f"[{r.number}] {slides.plain(r.citation)}" for r in report.references}
    for slide in slides.outline(report):
        if slide.layout in ("title", "matrix") or slide.title.startswith(("Target SWOT", "Introduction")):
            continue
        for p in slide.paras:
            assert p.text in allowed, (slide.title, p.text)


def test_a_field_the_paper_does_not_support_is_never_shown():
    report = _report()
    deck = slides.outline(report)
    everything = "\n".join(t for s in deck for t in _text(s)) + "\n".join(
        c for s in deck for row in (s.table or []) for c in row)
    assert "ghostmab" not in everything  # provenance "none" on its agent
    assert "PASI-90 in 73% vs 3%" not in everything  # effect_detail is never shown
    # nor in a visual
    labels = " ".join(i.label + i.sublabel for s in deck if s.visual for i in s.visual.items)
    assert "ghostmab" not in labels


# --- the title and the slide list ---------------------------------------------------------------------------------


def test_titles_follow_the_question_a_single_target_a_bispecific_a_disease():
    single = slides.outline(_report(), author="Dr Lee", date="30 Sep 2026")
    assert single[0].layout == "title" and single[0].title == "IL23A"
    assert single[0].subtitle == f"Dr Lee | 30 Sep 2026\n{QUESTION}"
    assert _by(single, "Introduction").title == "Introduction of the target"
    assert not any(s.title.startswith("Biology rationale") for s in single)

    two = [{"kind": "target", "canonical": "IL23A"}, {"kind": "target", "canonical": "IL17A"}]
    bi = slides.outline(_report(entities=two))
    assert bi[0].title == "IL23A x IL17A"
    assert _by(bi, "Executive summary").title == "Executive summary for bispecific proposal"
    assert _by(bi, "Introduction").title == "Introduction of the targets"
    assert [s.title for s in bi].index("Biology rationale for bispecific antibody") == [s.title for s in bi].index(
        "Disease indications") - 1

    disease = slides.outline(_report(question_type="target_identification", entities=[
        {"kind": "disease", "canonical": "psoriasis"}]))
    assert disease[0].title == "Plaque psoriasis - candidate targets"  # first letter capitalised, nothing else changed
    assert _by(disease, "Candidate targets").visual.type == "stacked_list"
    assert disease[0].subtitle == QUESTION  # no author or date given
    asthma = _report(question_type="target_identification", entities=[])
    asthma.interpretation["subject"] = "asthma"
    assert slides.outline(asthma)[0].title == "Asthma - candidate targets"
    assert slides.outline(_report(question_type="intervention_efficacy", entities=[
        {"kind": "drug", "canonical": "tirzepatide"}]))[0].title == "tirzepatide"  # other casing is left alone


def test_the_slides_come_in_the_templates_order():
    titles = [s.title for s in slides.outline(_report())]
    assert titles[:3] == ["IL23A", "Executive summary", "Introduction of the target"]
    order = ["Target biology MOA (mechanism of action)", "Target POC (proof of concept) - human", "Target POC - animal",
             "Disease indications", "Competitive landscape", "Target SWOT analysis", "References",
             "Appendix: evidence quotes"]
    assert [t for t in titles if t in order] == order


def test_the_executive_summary_holds_the_answer_its_key_limitations_and_counts_made_by_code():
    report = _report()
    deck = slides.outline(report)
    summary = _by(deck, "Executive summary")
    text = _text(summary)
    assert text[0] == f"{SENTENCES['answer'][0]} [1]"
    assert "Key limitations" in text
    assert "No tested effect in humans among the papers read for: IL17A [2]; Liver fat [3]." in text
    stats = {i.label: i.value for i in summary.visual.items}
    assert stats == {"Papers read": 12, "Papers with human findings": 4, "Papers with animal findings": 1,
                     "Verified findings": 6}
    assert not any(DROPPED in t for t in text)


# --- routing --------------------------------------------------------------------------------------------------------


def test_a_sentence_goes_to_the_slide_that_lists_its_section_and_each_appears_once():
    deck = slides.outline(_report())
    on = lambda prefix: " ".join(_text(_by(deck, prefix)))  # noqa: E731
    human, animal, moa = on("Target POC (proof"), on("Target POC - animal"), on("Target biology")
    assert SENTENCES["human"][0] in human
    assert SENTENCES["animal"][0] in animal
    assert SENTENCES["moa"][0] in moa
    assert SENTENCES["prevention"][0] in on("Disease indications")  # human and animal work mixed: the section decides
    # "targets" is the introduction only in a target-identification report; here no slide lists it, so what the
    # sentence cites decides, and a mix goes to the default slide
    assert SENTENCES["mixed"][0] in moa
    ident = slides.outline(_report(question_type="target_identification"))
    assert _text(_by(ident, "Candidate target: IL23A")) == [f"{SENTENCES['mixed'][0]} [1][3]"]  # "targets:IL23A" is "targets"
    # the negative section's sentences are the SWOT's weaknesses and have no slide of their own
    assert not any(SENTENCES["negative"][0] in " ".join(_text(s)) for s in deck if s.layout != "matrix")
    sentence_slides = [s for s in deck if s.title.startswith(("Introduction", "Target biology", "Target POC", "Disease indications"))]
    every = [p.text for s in sentence_slides for p in s.paras if "[" in p.text]
    assert len(every) == len(set(every))
    for key, (text, _) in SENTENCES.items():
        if key not in ("answer", "negative"):
            assert sum(text in t for t in every) == 1, key


def test_a_section_the_deck_does_not_list_is_placed_by_what_its_sentences_cite():
    unlisted = ReportSection(id="evidence", title="Evidence", kind="written", sentences=[
        Sentence(text="Adults on guselkumab cleared skin.", citations=["E1"]),
        Sentence(text="Mice on the antibody had thinner ears.", citations=["E3"]),
        Sentence(text="Keratinocytes made more IL-23 in culture.", citations=["E4"]),
        Sentence(text="Both mice and adults improved.", citations=["E1", "E3"])])
    deck = slides.outline(_report(sections=[unlisted]))
    on = lambda prefix: _text(_by(deck, prefix))  # noqa: E731
    assert on("Target POC (proof") == ["Adults on guselkumab cleared skin. [1]"]
    assert on("Target POC - animal") == ["Mice on the antibody had thinner ears. [3]"]
    assert on("Target biology") == ["Keratinocytes made more IL-23 in culture. [4]", "Both mice and adults improved. [1][3]"]
    # whereas a listed section keeps its sentences on its own slide whatever they cite
    clinical = ReportSection(id="clinical", title="Clinical", kind="written", sentences=[
        Sentence(text="Mice and adults improved on guselkumab.", citations=["E1", "E3"]),
        Sentence(text="Mice alone improved.", citations=["E3"])])
    deck = slides.outline(_report(sections=[clinical]))
    assert _text(_by(deck, "Target POC (proof")) == ["Mice and adults improved on guselkumab. [1][3]", "Mice alone improved. [3]"]


def _clinical(n, cite=lambda i: "E1"):
    return ReportSection(id="clinical", title="Clinical", kind="written", sentences=[
        Sentence(text=f"Result number {i} in adults.", citations=[cite(i)]) for i in range(1, n + 1)])


def test_a_slide_with_a_chart_shows_four_sentences_and_the_rest_follow_the_swot():
    deck = slides.outline(_report(sections=[_clinical(9)]))
    human = _by(deck, "Target POC (proof")
    assert human.visual is not None  # the chart takes room, so four sentences, not six
    assert _text(human) == [f"Result number {i} in adults. [1]" for i in range(1, 5)]
    assert human.notes[-1].startswith("Sentence 4:")
    more = _by(deck, "Appendix: further findings")
    assert more.title == "Appendix: further findings - Target POC (proof of concept) - human"
    assert _text(more) == [f"Result number {i} in adults. [1]" for i in range(5, 10)]
    assert [n[:11] for n in more.notes] == [f"Sentence {i}:" for i in range(5, 10)]
    assert more.visual is None and more.kicker == human.kicker and [m.number for m in more.refs] == [1]
    titles = [s.title for s in deck]
    assert titles.index("Target SWOT analysis") < titles.index(more.title) < titles.index("References")
    # nothing is lost and nothing repeats
    both = _text(human) + _text(more)
    assert len(both) == 9 == len(set(both))
    # the quotes of what the extra slide cites are still in the appendix
    assert "E1 [1]" in " ".join(_text(_by(deck, "Appendix: evidence quotes")))
    # a report with four or fewer has no such slide
    assert not any(s.title.startswith("Appendix: further") for s in slides.outline(_report()))


def test_a_slide_with_a_figure_shows_six_and_the_figure_may_come_from_a_sentence_that_overflowed():
    # the figure belongs to sentence 8 (E2, paper P2), which is past the six shown
    report = _report(sections=[_clinical(9, cite=lambda i: "E2" if i == 8 else "E1")])
    deck = slides.outline(report, figures=[_fig("f1", ["E2"], paper="P2")])
    human = _by(deck, "Target POC (proof")
    assert human.visual is None and (human.figure.fig_id, human.figure_ref) == ("f1", 2)
    assert len(human.paras) == 6
    more = _by(deck, "Appendix: further findings")
    assert [p.text for p in more.paras][1] == "Result number 8 in adults. [2]" and len(more.paras) == 3


def test_a_bispecific_sends_combination_findings_to_the_rationale_slide():
    two = [{"kind": "target", "canonical": "IL23A"}, {"kind": "target", "canonical": "IL17A"}]
    combo = unit(id="u7", paper_id="P1", target="IL17A", combo_with="IL23A", provenance=dict(TRACED["provenance"], combo_with="quote"),
                 outcome="improved")
    report = _report(entities=two, units=_units([combo]))
    report.sections[3].sentences = [Sentence(text="Blocking IL-17A and IL-23 together cleared more skin than either alone.",
                                            citations=["E7"])]
    deck = slides.outline(report)
    rationale = _by(deck, "Biology rationale")
    assert _text(rationale) == ["Blocking IL-17A and IL-23 together cleared more skin than either alone. [1]"]
    counts = {i.label: i.value for i in rationale.visual.items}
    assert rationale.visual.type == "comparison"
    # improved findings: IL23A on its own in 3 papers (the trial, the mouse study, ghostmab's), IL17A on its own in 1
    assert counts == {"IL23A alone": 3, "IL17A alone": 1, "IL23A and IL17A together": 1}
    assert "together cleared" not in " ".join(_text(_by(deck, "Target POC (proof")))


def test_a_bispecific_with_no_combination_finding_says_so():
    two = [{"kind": "target", "canonical": "IL23A"}, {"kind": "target", "canonical": "IL17A"}]
    rationale = _by(slides.outline(_report(entities=two)), "Biology rationale")
    assert _text(rationale) == [slides.NO_COMBINATION] and rationale.visual is None


def _targets_section(*subs):
    """The report's `targets` section: no sentences of its own, one subsection per candidate target."""
    return ReportSection(id="targets", title="Candidate targets", kind="written", subsections=[
        ReportSection(id=f"targets:{title}", title=title, kind="written", note="Two papers.",
                      sentences=[Sentence(text=text, citations=cites) for text, cites in sentences])
        for title, sentences in subs])


_DISEASE = [{"kind": "disease", "canonical": "psoriasis", "exact_terms": ["psoriasis", "plaque psoriasis"]}]


def _ident(*subs, **kw):
    return _report(question_type="target_identification", entities=_DISEASE,
                   sections=[ReportSection(id="answer", title="Answer", kind="written", sentences=[]), _targets_section(*subs)],
                   **kw)


def test_candidate_targets_are_listed_in_the_reports_order_and_each_gets_a_slide():
    deck = slides.outline(_ident(
        ("SIK3", [("SIK3 loss helped adults.", ["E1"]), ("SIK3 knockout mice improved.", ["E3"])]),
        ("MC4R (Mc4r)", [("Mc4r agonism helped adults.", ["E2"])])))
    overview = _by(deck, "Candidate targets")
    assert overview.kicker == "From the report's candidate-target sections" and overview.paras == []  # no sentences of its own
    assert overview.visual.type == "stacked_list"
    assert [(i.label, i.sublabel) for i in overview.visual.items] == [
        ("SIK3", "2 findings from 2 papers: 1 human, 1 animal"), ("MC4R (Mc4r)", "1 finding from 1 paper: 1 human, 0 animal")]
    titles = [s.title for s in deck]
    assert titles[2:5] == ["Candidate targets", "Candidate target: SIK3", "Candidate target: MC4R (Mc4r)"]
    sik3 = _by(deck, "Candidate target: SIK3")
    assert _text(sik3) == ["SIK3 loss helped adults. [1]", "SIK3 knockout mice improved. [3]"]
    assert sik3.visual.type == "stats_callout"
    assert [(i.label, i.value) for i in sik3.visual.items] == [("Papers", 2), ("Human findings", 1), ("Animal findings", 1)]
    assert [m.number for m in sik3.refs] == [1, 3]
    # their sentences are on their own slides only, not scattered by what they cite
    elsewhere = " ".join(t for s in deck if not s.title.startswith("Candidate target") and s.layout != "matrix" for t in _text(s))
    assert not any(x in elsewhere for x in ("SIK3 loss helped", "SIK3 knockout", "Mc4r agonism"))


def test_a_candidate_slide_takes_the_figure_of_any_of_its_sentences_and_keeps_six_with_it():
    sentences = [(f"SIK3 result {i}.", ["E2" if i == 8 else "E1"]) for i in range(1, 9)]
    with_fig = slides.outline(_ident(("SIK3", sentences)), figures=[_fig("f1", ["E2"], paper="P2")])
    sik3 = _by(with_fig, "Candidate target: SIK3")
    assert sik3.visual is None and (sik3.figure.fig_id, sik3.figure_ref) == ("f1", 2) and len(sik3.paras) == 6
    more = _by(with_fig, "Appendix: further findings")
    assert more.title == "Appendix: further findings - Candidate target: SIK3" and len(more.paras) == 2
    without = _by(slides.outline(_ident(("SIK3", sentences))), "Candidate target: SIK3")
    assert without.visual.type == "stats_callout" and len(without.paras) == 4  # the stats take the room


def test_candidates_past_the_slide_limit_go_to_one_appendix_slide_and_the_list_counts_them():
    subs = [(f"Target{i}", [(f"Target{i} helped adults.", ["E1"])]) for i in range(1, 9)]
    deck = slides.outline(_ident(*subs))
    assert [s.title for s in deck if s.title.startswith("Candidate target:")] == [
        f"Candidate target: Target{i}" for i in range(1, 7)]
    overview = _by(deck, "Candidate targets")
    assert [i.label for i in overview.visual.items] == [f"Target{i}" for i in range(1, 7)]
    assert overview.paras == [] and overview.visual.caption.endswith("(+2 more in the report)")
    more = _by(deck, "Appendix: further candidate targets")
    assert [(p.text, p.bold) for p in more.paras] == [
        ("Target7", True), ("Target7 helped adults. [1]", False), ("Target8", True), ("Target8 helped adults. [1]", False)]
    titles = [s.title for s in deck]
    assert titles.index("Target SWOT analysis") < titles.index(more.title) < titles.index("References")


def test_without_subsections_the_target_list_counts_what_the_findings_name_minus_the_question_itself():
    units = _units([unit(id="u7", paper_id="P1", target="psoriasis", provenance=dict(TRACED["provenance"])),
                    unit(id="u8", paper_id="P2", target="Plaque Psoriasis", provenance=dict(TRACED["provenance"]))])
    report = _report(question_type="target_identification", entities=_DISEASE, units=units,
                     sections=[ReportSection(id="answer", title="Answer", kind="written", sentences=[])])
    items = _by(slides.outline(report), "Candidate targets").visual.items
    assert [i.label for i in items] == ["IL23A", "IL17A"]  # most papers first; "psoriasis" is the question, not a target
    assert items[0].sublabel == "5 papers: 3 human, 1 animal"


def test_with_no_animal_section_the_animal_slide_takes_live_animal_sentences_from_mechanism_and_prevention():
    secs = [ReportSection(id="mechanism", title="Mech", kind="written", sentences=[
                Sentence(text="IL-23 drove ear swelling in mice.", citations=["E3"]),
                Sentence(text="IL-23 is raised in keratinocytes.", citations=["E4"])]),
            ReportSection(id="prevention", title="Prev", kind="written", sentences=[
                Sentence(text="Early blockade protected mice.", citations=["E3"]),
                Sentence(text="Early blockade protected mice and adults.", citations=["E1", "E3"])])]
    ident = slides.outline(_report(question_type="target_identification", entities=_DISEASE, sections=secs))
    assert _text(_by(ident, "Target POC - animal")) == ["IL-23 drove ear swelling in mice. [3]", "Early blockade protected mice. [3]"]
    assert _text(_by(ident, "Target biology")) == ["IL-23 is raised in keratinocytes. [4]"]
    assert "Early blockade protected mice and adults. [1][3]" in _text(_by(ident, "Disease indications"))
    # a question type with a section for the animal slide keeps its mechanism sentences where they are
    other = slides.outline(_report(sections=secs))
    assert "IL-23 drove ear swelling in mice. [3]" in _text(_by(other, "Target biology"))


def test_the_introduction_of_a_non_target_question_quotes_review_articles_and_names_what_was_asked():
    intro = _by(slides.outline(_report()), "Introduction")
    assert intro.kicker == "Context from review articles - not primary evidence"
    review = next(p for p in intro.paras if p.text.startswith("Wu 2020"))
    assert review.link == "https://example.org/wu" and "https://" not in review.text
    assert "“IL-23 drives psoriasis.”" in _text(intro)
    assert intro.visual.type == "callout_cards"
    assert [(i.label, i.sublabel) for i in intro.visual.items] == [
        ("IL23A", "Target"), ("psoriasis", "Disease"), ("adults", "Population")]


def test_charts_count_verified_findings_by_code():
    deck = slides.outline(_report())
    moa, human, animal = (_by(deck, p).visual for p in ("Target biology", "Target POC (proof", "Target POC - animal"))
    assert moa.type == "doughnut_chart" and {i.label: i.value for i in moa.items} == {
        "Cell or in vitro": 1, "Animal (in vivo)": 1, "Human clinical": 4}
    assert human.type == "bar_chart" and [(i.sublabel, i.value, i.tone) for i in human.items] == [
        ("improved", 3, "positive"), ("no_effect", 1, "neutral")]
    assert animal.type == "doughnut_chart" and [(i.label, i.value) for i in animal.items] == [("Mouse", 1)]


# --- competitive landscape ---------------------------------------------------------------------------------------


def test_competitors_come_only_from_human_clinical_findings_whose_agent_the_paper_supports():
    deck = slides.outline(_report())
    slide = _by(deck, "Competitive landscape")
    assert slide.layout == "table" and slide.by_rule
    head, *rows = slide.table
    assert head == ["Drug/agent", "Target", "Modality", "Company", "Indication", "Stage/design", "Ref."]
    assert [r[0] for r in rows] == ["guselkumab", "secukinumab"]  # in reference order
    # not the mouse antibody (animal), nor ghostmab (agent not found in its paper), nor a finding with no agent
    assert all(r[3] == "- fill in" for r in rows)
    assert rows[0] == ["guselkumab", "IL23A", "antibody", "- fill in", "psoriasis", "human study", "[1]"]
    assert [m.number for m in slide.refs] == [1, 2]


def test_only_drug_type_agents_are_competitors_not_actions_hormones_or_the_target_itself():
    prov = dict(TRACED["provenance"])
    extra = [
        unit(id="u7", paper_id="P1", intervention="inhibition", target="c-Myc", modality="small_molecule", provenance=prov),
        unit(id="u8", paper_id="P2", intervention="Blockade", target="IL17A", modality="antibody", provenance=prov),
        unit(id="u9", paper_id="P3", intervention="estrogen", target="ERs", modality="other", provenance=prov),
        unit(id="u10", paper_id="P4", intervention="hormone replacement therapy", target="ERs", modality="other",
             provenance=prov),
        unit(id="u11", paper_id="P5", intervention="il23a", target="IL23A", modality="antibody", provenance=prov),
        unit(id="u12", paper_id="P6", intervention="risankizumab", target="IL23A", modality="small_molecule", provenance=prov),
        unit(id="u13", paper_id="P6", intervention="inhibitor", target="IL23A", modality="small_molecule", provenance=prov),
    ]
    slide = _by(slides.outline(_report(units=_units(extra))), "Competitive")
    rows = slide.table[1:]
    assert [r[0] for r in rows] == ["guselkumab", "secukinumab", "risankizumab"]
    assert rows[2][2] == "small molecule" and rows[0][2] == "antibody"  # the modality, readable, never "-"
    assert all(r[2] != "-" for r in rows)


def test_the_stage_comes_from_the_papers_own_publication_type():
    report = _report()
    report.references[0].signals.publication_types = ["Journal Article", "Randomized Controlled Trial"]
    report.references[1].signals.publication_types = ["Clinical Trial, Phase II"]
    rows = _by(slides.outline(report), "Competitive").table[1:]
    assert [r[5] for r in rows[:2]] == ["randomised controlled trial", "phase II trial"]


def test_duplicate_agent_target_disease_rows_are_one_row():
    again = unit(id="u8", paper_id="P2", intervention="Guselkumab", provenance=dict(TRACED["provenance"]), target="IL23A")
    rows = _by(slides.outline(_report(units=_units([again]))), "Competitive").table[1:]
    assert [r[0] for r in rows].count("guselkumab") == 1


def test_with_no_competitor_the_slide_says_so_and_asks_to_fill_in():
    only_animal = [unit(id="u1", paper_id="P1", evidence_type="in_vivo", species="mouse", provenance=dict(TRACED["provenance"]))]
    slide = _by(slides.outline(_report(units=only_animal)), "Competitive")
    assert slide.table is None and slide.by_rule
    assert _text(slide) == [slides.NO_COMPETITOR, "Company, stage and unmet need: not in the papers read - fill in."]


# --- SWOT ---------------------------------------------------------------------------------------------------------


def _quadrants(slide):
    out, label = {}, None
    for p in slide.paras:
        if p.level == 0:
            label = p.text
            out[label] = []
        else:
            out[label].append(p)
    return out


def test_the_swot_sorts_findings_into_four_quadrants_by_rule():
    deck = slides.outline(_report())
    swot = _by(deck, "Target SWOT analysis")
    assert swot.layout == "matrix" and swot.by_rule
    q = {k: [p.text for p in v] for k, v in _quadrants(swot).items()}
    assert list(q) == ["Strengths", "Weaknesses", "Opportunities", "Threats"]
    # strengths: sentences citing only human findings that improved
    assert q["Strengths"] == [f"{SENTENCES['human'][0]} [1][2]"]
    # weaknesses: the negative section's sentence
    assert q["Weaknesses"] == [f"{SENTENCES['negative'][0]} [6]"]
    # a safety section with nothing in it: absence is not safety, said in the notes
    assert any(n.startswith("No verified safety finding") and slides.ABSENCE in n for n in swot.notes)
    # opportunities: measures with no tested human effect; threats: agents in people, then disagreements
    assert q["Opportunities"] == ["No tested effect in humans among the papers read for: IL17A [2]; Liver fat [3]."]
    assert q["Threats"] == ["guselkumab (IL23A) tested in people [1]", "secukinumab (IL17A) tested in people [2]",
                            "How the studies line up: IL23A (Studies disagree). [1][6]"]
    assert [m.number for m in swot.refs] == sorted({n for ps in _quadrants(swot).values() for p in ps for n in p.refs})
    assert not any(s.title.endswith("(detail)") for s in deck)


def test_strengths_fall_back_to_non_human_findings_only_when_no_human_finding_improved():
    animal = ReportSection(id="preclinical", title="Pre", kind="written", sentences=[
        Sentence(text=SENTENCES["animal"][0], citations=["E3"])])
    q = _quadrants(_by(slides.outline(_report(sections=[animal])), "Target SWOT analysis"))
    assert [p.text for p in q["Strengths"]] == [f"{SENTENCES['animal'][0]} [3]"]


def test_a_sentence_citing_an_improved_finding_is_never_a_weakness():
    montelukast = unit(id="u7", paper_id="P1", intervention="montelukast", outcome="improved", provenance=dict(TRACED["provenance"]))
    secs = [ReportSection(id="negative", title="Neg", kind="written", sentences=[
                Sentence(text="Montelukast reduced exacerbations in adults.", citations=["E7"]),
                Sentence(text="Briakinumab did not help.", citations=["E6"])]),
            ReportSection(id="clinical", title="Clin", kind="written", sentences=[
                Sentence(text="Placebo-level scores persisted in adults.", citations=["E6"]),
                Sentence(text="Montelukast was no better than guselkumab.", citations=["E6", "E7"])])]
    q = _quadrants(_by(slides.outline(_report(units=_units([montelukast]), sections=secs)), "Target SWOT analysis"))
    weak = [p.text for p in q["Weaknesses"]]
    assert weak == ["Briakinumab did not help. [6]", "Placebo-level scores persisted in adults. [6]"]
    assert all("Montelukast" not in t for t in weak)


def test_a_quadrant_shows_three_items_and_counts_the_rest():
    many = [Sentence(text=f"Treatment {i} did not help adults with psoriasis.", citations=["E6"]) for i in range(5)]
    report = _report()
    report.sections[5].sentences = many
    deck = slides.outline(report)
    weak = _quadrants(_by(deck, "Target SWOT analysis"))["Weaknesses"]
    assert [p.text for p in weak] == [f"Treatment {i} did not help adults with psoriasis. [6]" for i in range(3)] + [
        "+2 more in the report"]
    assert weak[-1].muted
    assert not any(s.title.endswith("(detail)") for s in deck)


def test_empty_swot_quadrants_state_the_gap():
    quiet = _report(units=[unit(id="u1", paper_id="P1", provenance=dict(TRACED["provenance"]))], sections=[
        ReportSection(id="answer", title="Answer", kind="written", sentences=[])])
    q = _quadrants(_by(slides.outline(quiet), "Target SWOT analysis"))
    assert [p.text for p in q["Strengths"]] == [slides.NO_FINDING] and q["Strengths"][0].muted
    assert [p.text for p in q["Opportunities"]] == ["Not in the papers read - fill in."]


# --- the rest of the deck -------------------------------------------------------------------------------------------


def test_slides_assembled_by_rule_are_marked_and_the_others_are_not():
    flagged = {s.title for s in slides.outline(_report()) if s.by_rule}
    assert flagged == {"Disease indications", "Competitive landscape", "Target SWOT analysis"}


def test_disease_indications_count_diseases_and_leave_the_unmet_need_to_fill_in():
    slide = _by(slides.outline(_report()), "Disease indications")
    assert slide.visual.type == "stacked_list"
    assert [(i.label, i.sublabel) for i in slide.visual.items] == [("psoriasis", "6 papers: 4 human, 1 animal")]
    assert slide.paras[-1].text == "Unmet need and opportunities: not in the papers read - fill in."
    # a disease the paper does not support is not listed
    unsupported = [unit(id="u1", paper_id="P1", disease="eczema", provenance=dict(TRACED["provenance"], disease_raw="none"))]
    assert _by(slides.outline(_report(units=unsupported)), "Disease indications").visual is None


def test_references_link_to_records_and_the_appendix_quotes_what_the_deck_cites():
    report = _report()
    deck = slides.outline(report)
    refs = _by(deck, "References")
    assert refs.layout == "references" and refs.size == 10
    assert [(p.link, p.refs) for p in refs.paras] == [(r.url, [r.number]) for r in report.references]
    appendix = " ".join(_text(_by(deck, "Appendix")))
    quotes = {u.id: " ".join(u.quote.split()) for u in report.evidence}
    for short, uid in (("E1", "u1"), ("E3", "u3"), ("E6", "u6")):
        assert f"{short} [" in appendix and quotes[uid] in appendix
    # E5's agent was not found in its paper and nothing the deck says cites it
    assert "E5 [" not in appendix


def test_every_slide_lists_the_references_it_cites_by_a_short_name_from_the_paper_record():
    report = _report()
    deck = slides.outline(report)
    human = _by(deck, "Target POC (proof")
    assert [(m.number, m.short, m.url) for m in human.refs] == [
        (1, "Author1 2011", "https://europepmc.org/article/MED/1"), (2, "Author2 2012", "https://europepmc.org/article/MED/2")]
    for s in deck:
        if s.layout not in ("title", "references"):
            cited = {n for p in s.paras for n in p.refs} | {int(n) for r in (s.table or [])[1:] for c in r
                                                          for n in slides._NUM.findall(c)}
            assert {m.number for m in s.refs} >= cited, s.title


def test_short_names_are_surname_and_year_from_the_reference():
    def ref(citation, year=None):
        return Reference(number=3, paper_id="p", citation=citation, signals=PaperSignals(year=year))
    assert slides.short_name(ref("Smith JA, Lee K, Park S et al. A title. *J* 2020", 2020)) == "Smith 2020"
    assert slides.short_name(ref("Smith, John A title. *Lancet* 2019")) == "Smith 2019"
    assert slides.short_name(ref("van der Berg JA A lone author's paper. *Cell* 2018", 2018)) == "van der Berg 2018"
    assert slides.short_name(ref("Anonymous A title with no year.")) == "Anonymous"


def test_empty_slides_still_appear_and_state_the_gap():
    bare = _report(units=[], sections=[ReportSection(id="answer", title="Answer", kind="written", sentences=[])], n_refs=0)
    deck = slides.outline(bare)
    for prefix in ("Target biology", "Target POC (proof", "Target POC - animal"):
        assert _text(_by(deck, prefix)) == [slides.NO_FINDING], prefix
    assert _text(_by(deck, "Executive summary")) == [slides.NO_FINDING]
    assert _text(_by(deck, "Disease indications"))[0] == slides.NO_FINDING
    assert not any(s.title in ("References", "Appendix: evidence quotes") for s in deck)  # nothing to list


def test_the_integrity_warning_is_on_the_title_and_the_executive_summary_only():
    deck = slides.outline(_report(integrity=["E3: quote not found in paper P3"]))
    warned = [s.title for s in deck if s.warning]
    assert warned == ["IL23A", "Executive summary"]
    assert not any(s.warning for s in slides.outline(_report()))


# --- figures --------------------------------------------------------------------------------------------------------


def _fig(fig_id, evidence, path="/tmp/f.jpg", paper="P3"):
    return FigureAsset(paper_id=paper, pmcid="PMC1", fig_id=fig_id, label=f"Figure {fig_id}", caption="A caption.",
                       href=fig_id, licence="CC BY 4.0", evidence_ids=evidence, path=path)


def test_only_downloaded_figures_are_placed_each_once_and_they_replace_the_chart():
    figs = [_fig("f1", ["E3"], path=None),               # chosen but never downloaded: not shown
            _fig("f2", ["E3"]), _fig("f3", ["E3"]),     # two for one finding, so two slides citing it get one each
            _fig("f4", ["E4"], paper="P4")]              # a slide already holding a figure takes no second
    deck = slides.outline(_report(), figures=figs)
    placed = {s.title: s.figure.fig_id for s in deck if s.figure}
    assert placed == {"Target biology MOA (mechanism of action)": "f2", "Target POC - animal": "f3"}
    animal = _by(deck, "Target POC - animal")
    assert (animal.figure_ref, animal.visual) == (3, None) and 3 in [m.number for m in animal.refs]
    assert _by(deck, "Target POC (proof").visual is not None  # nothing to place there: the chart stays
    # no figure is placed for a finding no sentence on a figure slide cites, nor from a paper with no reference
    assert not any(s.figure for s in slides.outline(_report(), figures=[_fig("f9", ["E5"], paper="P5"),
                                                                       _fig("f8", ["E3"], paper="P99")]))
    assert _by(slides.outline(_report()), "Target POC - animal").visual is not None


# --- the same report, the same deck ---------------------------------------------------------------------------------


def test_the_outline_is_deterministic_and_leaves_the_report_unchanged():
    report = _report()
    before = report.model_dump_json()
    figs = [_fig("f2", ["E3"])]
    a = slides.outline(report, "review-x", figures=figs, author="A", date="D")
    b = slides.outline(report, "review-x", figures=figs, author="A", date="D")
    assert a == b and report.model_dump_json() == before
    assert any("review-x" in n for n in a[0].notes)
    assert not any("/Users/" in str(s) for s in a)


def test_to_pptx_draws_the_outline_with_the_renderer(monkeypatch):
    from bedrock.litreview.report import deck_render

    seen = {}
    monkeypatch.setattr(deck_render, "render", lambda deck: seen.setdefault("deck", deck) and b"pptx")
    report = _report()
    assert slides.to_pptx(report, "review-x", author="A", date="D") == b"pptx"
    assert seen["deck"] == slides.outline(report, "review-x", author="A", date="D")


# --- the pieces that moved over from the old condensed deck ---------------------------------------------------------


def test_key_limitations_lead_with_what_changes_how_the_answer_is_read():
    gaps = ["5 paper(s) were read for negative and conflicting results, but no verified finding was extracted.",
            "9 of 16 papers read were available as abstract only.",
            "Figures and tables are not read.",
            "E38 [3]: interval printed as 0.58 to 0.42: the lower bound is above the upper.",
            "The question is about adults with obesity. 7 of the 13 independent studies name obesity."]
    assert slides.key_limitations(gaps) == [gaps[4], gaps[3], gaps[1]]
    bullet = "No tested effect in humans among the papers read for: Liver fat [1]; Insulin resistance [2]."
    assert slides.key_limitations([bullet]) == [bullet]


def test_the_layout_file_can_be_replaced_whole(tmp_path):
    import pytest
    bad = tmp_path / "deck.yaml"
    bad.write_text("version: deck.v0\nslides: {}\n")
    with pytest.raises(ValueError, match="version"):
        slides.load_deck_layout(bad)
    bad.write_text("version: deck.v1\nslides: {moa: {title: x}}\n")
    with pytest.raises(ValueError, match="no entry for slide"):
        slides.load_deck_layout(bad)
