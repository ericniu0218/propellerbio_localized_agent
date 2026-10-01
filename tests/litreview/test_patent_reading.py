"""Reading and summarising patent filings: what is copied, what code decides, and
what the report may say about it.

A patent reports no results; it says what its filer claims or describes. So every
test here is about one of two things: a model's copy being checked against the
filing's own text (the quote, the target, the words of "covers"), or a sentence
being checked against the filings it cites (a number, a name, a company, and
whether it says whose claim this is).
"""

from __future__ import annotations

import json
import re

import pytest

from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import PatentRecord
from bedrock.litreview.report.invariants import check_patents, verify_run
from bedrock.litreview.runner import answer_question, load_result
from bedrock.litreview.stages.s6c_patent_extract import (
    assignee_key,
    assignee_name,
    claim_kind,
    found_in,
    is_english,
    load_reading,
    modality_supported,
)
from tests.litreview.fixtures import QUESTION, corpus, responder
from tests.litreview.test_patent_stage import Office

ABSTRACT_A = ("The invention relates to a bispecific antibody that binds the p19 subunit of human IL-23. "
              "The antibody is used to treat plaque psoriasis in a patient.")
CLAIM_A = "A method of treating plaque psoriasis in a subject, comprising administering an antibody that binds IL-23 p19."
ABSTRACT_MENTION = ("A kit for measuring cytokines in serum, wherein the cytokines are selected from IL-6, IL-17 and IL-23, "
                    "for use in the diagnosis of inflammatory conditions of the skin.")


def filings():
    return [
        PatentRecord(id="WO1A1", family_id="f1", title="Anti-IL-23 bispecific antibody and its use",
                     abstract=ABSTRACT_A, assignee="JANSSEN PHARMACEUTICA NV [BE]", kind_code="A1",
                     priority_date="2021-03-01", claim_1=CLAIM_A, matched_targets=["IL23A"]),
        PatentRecord(id="EP2B1", family_id="f2", title="Antibody for psoriasis",
                     abstract="An IL-23 antibody for use in psoriasis. The antibody is a human monoclonal antibody.",
                     assignee="Amgen Inc.", kind_code="B1", priority_date="2019-06-01", matched_targets=["IL23A"]),
        PatentRecord(id="US3A1", family_id="f3", title="Cytokine measurement kit",
                     abstract=ABSTRACT_MENTION, assignee="Acme Diagnostics LLC", kind_code="A1",
                     priority_date="2022-01-01", matched_targets=["IL23A"]),
        PatentRecord(id="CN4A", family_id="f4", title="IL-23p19抗体及其应用",
                     abstract="本发明涉及一种特异性结合IL-23p19的单克隆抗体及其在银屑病治疗中的应用。",
                     assignee="NANJING CO LTD", kind_code="A", priority_date="2023-01-01", matched_targets=["IL23A"]),
    ]


class Script:
    """A scripted model for the two patent prompts; everything else is the shared fake."""

    def __init__(self, *, bad_quote=False, bad_target=False, claim_wrong_modality=False, sentences=None):
        self.bad_quote, self.bad_target, self.claim_wrong_modality = bad_quote, bad_target, claim_wrong_modality
        self.sentences = sentences
        self.write_prompts: list[str] = []

    def __call__(self, prompt, schema):
        title = (schema or {}).get("title")
        if title == "PatentExtraction":
            head = prompt.split("Title:", 1)[1]
            if "Anti-IL-23 bispecific" in head:
                quote, src = ABSTRACT_A.split(". ")[0] + ".", "abstract"
                if self.bad_quote:
                    quote = "The antibody cured every patient within one week of dosing."
                return json.dumps({"findings": [{
                    "quote": quote, "source": src, "target": "IL-17A" if self.bad_target else "IL-23",
                    "modality": "small_molecule" if self.claim_wrong_modality else "antibody",
                    "covers": "a bispecific antibody that binds the p19 subunit of human IL-23",
                    "indication": "plaque psoriasis"}]})
            if "Antibody for psoriasis" in head:
                return json.dumps({"findings": [{
                    "quote": "An IL-23 antibody for use in psoriasis.", "source": "abstract", "target": "IL-23",
                    "modality": "antibody", "covers": "an IL-23 antibody with a made-up mechanism of action",
                    "indication": "psoriasis"}]})
            if "Cytokine measurement kit" in head:
                return json.dumps({"findings": [{
                    "quote": ABSTRACT_MENTION.split(",")[0] + ",", "source": "abstract", "target": "IL-23",
                    "modality": "other", "covers": "a kit for measuring cytokines", "indication": None}]})
            return json.dumps({"findings": []})
        if title == "DraftSection" and "summarising the patent filings" in prompt:
            self.write_prompts.append(prompt)
            # Filings are numbered by filer, so P1 is Amgen's and P2 Janssen's: the
            # script says which *filing* a sentence cites and finds its number here.
            number = {m[2]: m[1] for m in re.finditer(r"\[(P\d+)\] (\S+) ", prompt)}
            chosen = self.sentences or [
                ("Janssen's application WO1A1 claims a method of treating plaque psoriasis with an antibody "
                 "that binds IL-23 p19.", ["WO1A1"]),
                ("Amgen's granted patent EP2B1 is directed to an IL-23 antibody for use in psoriasis.", ["EP2B1"]),
            ]
            return json.dumps({"sentences": [{"citations": [number.get(c, c) for c in cites], "text": t}
                                             for t, cites in chosen]})
        return responder(prompt, schema)


def client(script: Script):
    return FakeLLMClient(responder=script, version="fake-1")


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "cache", tmp_path / "out", tmp_path / "db.sqlite3"
    return s


def run(settings, tmp_path, script=None, patents=None, **kwargs):
    script = script or Script()
    return answer_question(QUESTION, settings=settings, client=client(script), papers=corpus(),
                           db=Database(tmp_path / "db.sqlite3"), patents=patents if patents is not None else filings(),
                           **kwargs), script


# --- the small pure functions --------------------------------------------------------


def test_claim_kind_is_read_off_claim_1s_own_first_words():
    assert claim_kind("A method of treating plaque psoriasis in a subject") == "method_of_treatment"
    assert claim_kind("A method for preventing relapse of colitis") == "method_of_treatment"
    assert claim_kind("A process for producing an antibody") == "method"
    assert claim_kind("Use of an antibody that binds IL-23 for treating psoriasis") == "use"
    assert claim_kind("Vaccine for use in the prevention of an IL-23 related disease") == "use"
    assert claim_kind("An isolated antibody that binds human IL-23") == "composition"
    assert claim_kind(None) == "not_stated", "no claim 1 is not guessed from the abstract"


def test_a_filer_is_one_company_however_the_register_spells_it():
    names = ["JANSSEN PHARMACEUTICA NV [BE]", "Janssen Pharmaceutica N.V.", "janssen pharmaceutica nv"]
    assert len({assignee_key(n) for n in names}) == 1
    assert assignee_key("Amgen Inc.") != assignee_key("Amgen Research (Munich) GmbH"), "different companies stay apart"
    assert assignee_name("JANSSEN PHARMACEUTICA NV [BE]") == "Janssen Pharmaceutica Nv"
    assert assignee_name(None) == "Unknown / unassigned"


def test_english_is_told_from_chinese_and_french_and_short_junk():
    f = filings()
    assert is_english(f[0]) and is_english(f[1]) and is_english(f[2])
    assert not is_english(f[3])
    assert not is_english(PatentRecord(id="x", title="MÉTHODES ET COMPOSITIONS POUR TRAITER LE PSORIASIS",
                                       abstract="La présente invention concerne des méthodes pour traiter le psoriasis."))
    assert not is_english(PatentRecord(id="y", title="Short", abstract="Tiny."))


def test_covers_and_indication_must_be_in_the_text_or_they_are_blanked():
    text = "A bispecific antibody that binds the p19 subunit of human IL-23 for plaque psoriasis"
    assert found_in("a bispecific antibody that binds the p19 subunit", text)
    assert not found_in("an antibody with a made-up mechanism of action", text)
    assert not found_in("", text) and not found_in(None, text)


def test_a_modality_the_text_never_mentions_is_not_believed():
    assert modality_supported("antibody", "A human monoclonal antibody")
    assert not modality_supported("antibody", "A compound of formula (I) or a salt thereof")
    assert modality_supported("other", "anything")


def test_a_wrong_modality_is_replaced_only_when_the_text_points_to_exactly_one_kind():
    from bedrock.litreview.stages.s6c_patent_extract import modality_from_text

    assert modality_from_text("A human monoclonal antibody against IL-23") == "antibody"
    assert modality_from_text("A cyclic peptide inhibitor of IL-23") == "other", "peptide and inhibitor: two cues, no guess"
    assert modality_from_text("A kit for measuring serum") == "other"


def test_a_filing_is_about_a_target_when_it_names_it_up_front_and_not_in_a_list():
    from bedrock.litreview.stages.s6c_patent_extract import names_target_as_subject

    kb = load_knowledge()
    matcher = kb.get("IL23A").matcher()
    f = filings()
    assert names_target_as_subject(matcher, "IL23A", f[0], kb), "named in the title"
    assert names_target_as_subject(matcher, "IL23A", f[1], kb), "first sentence of the abstract"
    assert not names_target_as_subject(matcher, "IL23A", f[2], kb), "one of IL-6, IL-17 and IL-23"
    bispecific = PatentRecord(id="b", title="Bispecific antibody binding IL-23 and IL-17A", abstract="x")
    assert names_target_as_subject(matcher, "IL23A", bispecific, kb), "two targets is a bispecific, not a list"


# --- the stage, end to end ------------------------------------------------------------


def test_what_verifies_is_kept_in_the_filings_own_words_and_a_mention_is_not_a_subject(settings, tmp_path):
    result, _ = run(settings, tmp_path)
    reading = load_reading(Database(tmp_path / "db.sqlite3"), result.run.id)
    by_patent = {u.patent_id: u for u in reading.units}

    janssen = by_patent["WO1A1"]
    assert janssen.subject == "subject" and janssen.target == "IL23A" and janssen.claim_kind == "method_of_treatment"
    assert janssen.basis == "claim_1_and_abstract"
    assert janssen.quote == ABSTRACT_A.split(". ")[0] + ".", "the filing's words, not the model's copy"
    assert filings()[0].abstract[janssen.quote_start:janssen.quote_end] == janssen.quote
    assert janssen.covers and janssen.indication == "plaque psoriasis"

    assert by_patent["US3A1"].subject == "mention", "named in the abstract among others, not in title or claim 1"
    assert "US3A1" not in {u.patent_id for u in reading.subjects()}
    assert "CN4A" in reading.skipped and "not English" in reading.skipped["CN4A"]
    assert reading.stats["mention"] == 1 and reading.stats["subject"] == 2


def test_a_fabricated_quote_destroys_the_finding(settings, tmp_path):
    result, _ = run(settings, tmp_path, Script(bad_quote=True))
    reading = load_reading(Database(tmp_path / "db.sqlite3"), result.run.id)
    assert "WO1A1" not in {u.patent_id for u in reading.units}
    assert reading.stats["quote_failed"] >= 1


def test_a_target_the_filing_never_names_destroys_the_finding(settings, tmp_path):
    result, _ = run(settings, tmp_path, Script(bad_target=True))
    reading = load_reading(Database(tmp_path / "db.sqlite3"), result.run.id)
    assert "WO1A1" not in {u.patent_id for u in reading.units}
    assert reading.stats["target_not_in_text"] >= 1


def test_made_up_words_and_an_unsupported_modality_are_corrected_in_code(settings, tmp_path):
    result, _ = run(settings, tmp_path, Script(claim_wrong_modality=True))
    reading = load_reading(Database(tmp_path / "db.sqlite3"), result.run.id)
    amgen = next(u for u in reading.units if u.patent_id == "EP2B1")
    assert amgen.covers is None and any("covers: blanked" in c for c in amgen.corrected)
    assert amgen.claim_kind == "not_stated" and amgen.basis == "abstract_only"
    janssen = next(u for u in reading.units if u.patent_id == "WO1A1")
    assert janssen.modality == "antibody", "the model said small_molecule; the text only ever says antibody"
    assert any("modality: small_molecule -> antibody" in c for c in janssen.corrected)


def test_a_scanned_spelling_of_the_searched_target_is_not_a_target_of_its_own(settings, tmp_path):
    """Seen live: a filing's own text says "IL-23pl9" (a scan of "IL-23p19"), the
    knowledge table has no such name, and the landscape grew a fourth target."""
    scanned = PatentRecord(
        id="WO7A1", family_id="f7", title="Method of treating genital psoriasis", assignee="Sun Pharma",
        abstract="A method of treating genital psoriasis comprising administering an anti-IL-23pl9 antibody to a patient "
                 "in need thereof, wherein the antibody is a monoclonal antibody.",
        kind_code="A1", priority_date="2024-09-20", matched_targets=["IL23A"])

    class Reads(Script):
        def __call__(self, prompt, schema):
            if (schema or {}).get("title") == "PatentExtraction" and "genital psoriasis" in prompt:
                return json.dumps({"findings": [{
                    "quote": "A method of treating genital psoriasis comprising administering an anti-IL-23pl9 antibody to a patient in need thereof",
                    "source": "abstract", "target": "IL-23pl9", "modality": "antibody",
                    "covers": "a method of treating genital psoriasis", "indication": "genital psoriasis"}]})
            return super().__call__(prompt, schema)

    result, _ = run(settings, tmp_path, Reads(), patents=[scanned])
    unit = load_reading(Database(tmp_path / "db.sqlite3"), result.run.id).units[0]
    assert unit.target == "IL23A" and unit.target_raw == "IL-23pl9"
    assert any("the only target this filing was searched for" in c for c in unit.corrected)


def test_a_name_the_table_does_not_know_is_kept_when_the_filing_was_searched_for_several_targets(settings, tmp_path):
    record = filings()[0].model_copy(update={"matched_targets": ["IL23A", "IL17A"]})

    class Reads(Script):
        def __call__(self, prompt, schema):
            if (schema or {}).get("title") == "PatentExtraction" and "Anti-IL-23 bispecific" in prompt:
                return json.dumps({"findings": [{
                    "quote": ABSTRACT_A.split(". ")[0] + ".", "source": "abstract", "target": "p19 subunit",
                    "modality": "antibody", "covers": "a bispecific antibody", "indication": None}]})
            return super().__call__(prompt, schema)

    result, _ = run(settings, tmp_path, Reads(), patents=[record])
    units = load_reading(Database(tmp_path / "db.sqlite3"), result.run.id).units
    assert all(u.target == "p19 subunit" for u in units), "with two candidates there is nothing to choose between"


def test_filings_are_numbered_for_citation_subjects_only_in_a_fixed_order(settings, tmp_path):
    result, _ = run(settings, tmp_path)
    reading = load_reading(Database(tmp_path / "db.sqlite3"), result.run.id)
    numbered = {reading.short_ids[u.id]: u.patent_id for u in reading.subjects()}
    assert set(numbered) == {"P1", "P2"} and set(numbered.values()) == {"WO1A1", "EP2B1"}
    assert all(u.id not in reading.short_ids for u in reading.units if u.subject == "mention")


# --- the section ------------------------------------------------------------------------


def competitors(result) -> str:
    return next(s.body_markdown for s in result.report.sections if s.id == "competitors")


def test_the_section_says_what_the_filings_claim_with_their_own_words_beneath(settings, tmp_path):
    result, _ = run(settings, tmp_path)
    body = competitors(result)
    assert "claims a method of treating plaque psoriasis" in body
    assert "IL23A" in body and "not evidence that it works, and not a legal view" in body
    assert "[P1] WO1A1" in body or "[P1] EP2B1" in body
    assert any(line.startswith("> ") for line in body.splitlines())
    assert "1 filing(s) only mention a target among others and are not summarised." in body
    assert "1 not read (not English)." in body
    assert result.report.integrity == [], result.report.integrity


def test_the_filer_table_merges_spellings_of_one_company(settings, tmp_path):
    f = filings()
    f.append(PatentRecord(id="EP9A1", family_id="f9", title="Another IL-23 antibody",
                          abstract=ABSTRACT_A, assignee="Janssen Pharmaceutica N.V.", kind_code="A1",
                          priority_date="2022-02-02", matched_targets=["IL23A"]))
    result, _ = run(settings, tmp_path, patents=f)
    rows = [l for l in competitors(result).splitlines() if l.startswith("| Janssen")]
    assert len(rows) == 1 and "| 2 |" in rows[0], rows


def test_a_run_without_a_model_reads_no_filing_and_says_nothing_new(settings, tmp_path):
    result = answer_question(QUESTION, settings=settings, use_model=False, papers=corpus(),
                             db=Database(tmp_path / "db.sqlite3"), patents=filings())
    assert load_reading(Database(tmp_path / "db.sqlite3"), result.run.id) is None
    assert "not evidence that it works" not in competitors(result)


def test_patents_are_never_counted_as_evidence(settings, tmp_path):
    with_patents, _ = run(settings, tmp_path)
    plain = answer_question(QUESTION, settings=settings, client=client(Script()), papers=corpus(),
                            db=Database(tmp_path / "db-plain.sqlite3"))
    assert with_patents.summary == plain.summary
    assert with_patents.report.evidence == plain.report.evidence
    assert with_patents.report.references == plain.report.references
    assert with_patents.report.stats == plain.report.stats


# --- the guard ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentence, why", [
    ("Janssen's application WO1A1 claims a method that gives a 60% response rate in patients.", "number 60"),
    ("Pfizer's application WO1A1 claims a method of treating plaque psoriasis with an antibody.", "Pfizer"),
    ("Janssen's application WO1A1 is an effective treatment for plaque psoriasis using an antibody.", "outcome as fact"),
    ("A method of treating plaque psoriasis with an antibody that binds IL-23 p19 is described by the antibody.", None),
    ("An antibody binds IL-23 p19 and treats psoriasis in patients across many different trials.", "whose claim"),
    ("Janssen's WO1A1 claims a method that also targets IL-36R in patients with plaque psoriasis.", "IL36R"),
    ("Amgen's granted patent WO1A1 claims a method of treating plaque psoriasis with an antibody.", "granted"),
])
def test_the_patent_guard_drops_what_a_filing_cannot_support(settings, tmp_path, sentence, why):
    script = Script(sentences=[(sentence, ["WO1A1"])])
    result, _ = run(settings, tmp_path, script, patents=[filings()[0], filings()[1]])
    kept = [l for l in competitors(result).splitlines() if l.startswith("- ")]
    if why is None:
        return
    assert not any(sentence in l for l in kept), f"kept: {kept}"
    assert "No sentence could be written and verified" in competitors(result)
    assert result.report.integrity == []


def test_a_sentence_that_cites_nothing_or_an_unknown_filing_is_dropped(settings, tmp_path):
    script = Script(sentences=[("Janssen's application WO1A1 claims a method of treating plaque psoriasis.", []),
                               ("Janssen's application WO1A1 claims a method that also names something else.", ["P9"]),
                               ("Amgen's patent EP2B1 is directed to an antibody for use in psoriasis.", ["EP2B1"])])
    result, _ = run(settings, tmp_path, script, patents=[filings()[0], filings()[1]])
    body = competitors(result)
    assert "WO1A1 claims a method of treating plaque psoriasis." not in body
    assert "directed to an antibody for use in psoriasis" in body


# --- the invariant and reopening --------------------------------------------------------


def test_every_quote_in_the_section_is_re_proved_against_the_stored_filing(settings, tmp_path):
    result, _ = run(settings, tmp_path)
    assert check_patents(result.report, filings()) == []

    section = next(s for s in result.report.sections if s.id == "competitors")
    forged = result.report.model_copy(update={"sections": [
        s.model_copy(update={"body_markdown": (s.body_markdown or "") + "\n[P1] WO1A1 · x\n> The antibody cured everyone."})
        if s is section else s for s in result.report.sections]})
    assert any("not in the cited filing" in v for v in check_patents(forged, filings()))

    gone = check_patents(result.report, [f for f in filings() if f.id != "WO1A1"])
    assert any("not among the run's saved patents" in v for v in gone)
    assert check_patents(result.report, None) == [], "a caller with no filings checks nothing"


def test_a_stored_run_with_patents_verifies_clean_and_reopens_with_them(settings, tmp_path):
    result, _ = run(settings, tmp_path)
    db = Database(tmp_path / "db.sqlite3")
    assert [v for v in verify_run(db, result.run.id) if "patent" in v.lower() or "competitors" in v] == []
    assert load_result(db, result.run.id).patents == filings()


# --- found by running it end to end against real EPO filings and a real model ---------


def test_a_filing_the_model_describes_twice_is_one_statement_with_one_citation_number(settings, tmp_path):
    """The first real run listed [P5] three times: the model returned a finding per
    abstract sentence for one filing, each with its own quote and the same number."""
    record = filings()[0]

    class Twice(Script):
        def __call__(self, prompt, schema):
            if (schema or {}).get("title") == "PatentExtraction" and "Anti-IL-23 bispecific" in prompt:
                abstract = ABSTRACT_A.split(". ")
                one = lambda quote, source: {"quote": quote, "source": source, "target": "IL-23", "modality": "antibody",
                                            "covers": "a bispecific antibody", "indication": "plaque psoriasis"}
                return json.dumps({"findings": [one(abstract[0] + ".", "abstract"), one(CLAIM_A, "claim_1"),
                                                one(abstract[1], "abstract")]})
            return super().__call__(prompt, schema)

    result, _ = run(settings, tmp_path, Twice(), patents=[record])
    units = load_reading(Database(tmp_path / "db.sqlite3"), result.run.id).units
    assert len(units) == 1 and units[0].quote_source == "claim_1", "one per filing and target, the claim preferred"
    body = competitors(result)
    assert body.count("[P1] WO1A1") == 1
    assert result.report.integrity == []


def test_a_chinese_claim_under_an_english_abstract_is_not_read_or_quoted(settings, tmp_path):
    """Real WO filings (Innovent's, among them) have an English abstract and a
    Chinese claim 1. The first real run quoted the Chinese claim in an English report."""
    record = filings()[0].model_copy(update={"claim_1": "一种治疗受试者的斑块状银屑病的方法，包括向所述受试者施用有效量的抗IL23p19抗体"})
    prompts: list[str] = []

    class Spy(Script):
        def __call__(self, prompt, schema):
            if (schema or {}).get("title") == "PatentExtraction":
                prompts.append(prompt)
            return super().__call__(prompt, schema)

    result, _ = run(settings, tmp_path, Spy(), patents=[record])
    unit = load_reading(Database(tmp_path / "db.sqlite3"), result.run.id).units[0]
    assert unit.basis == "abstract_only" and unit.claim_kind == "not_stated"
    assert not any("银屑病" in p for p in prompts), "the model is not shown a claim it cannot be checked on"
    assert "银屑病" not in competitors(result)
    assert result.report.integrity == []


def test_every_filing_a_sentence_cites_is_listed_beneath_and_a_sentence_that_would_not_fit_is_left_out(settings, tmp_path):
    """The first real run failed its own integrity check: sentences cited P12-P16
    and the listing stopped at eight. The check is the guarantee, so the listing
    follows the citations and not the other way round."""
    from bedrock.litreview.stages.s7b_patent_write import PER_TARGET_CITED

    many = [PatentRecord(id=f"WO{i}A1", family_id=f"f{i}", title=f"Anti-IL-23 antibody number {i}",
                         abstract=ABSTRACT_A, assignee="Janssen Biotech Inc", kind_code="A1",
                         priority_date=f"20{10 + i}-01-01", matched_targets=["IL23A"]) for i in range(12)]
    first, second = [r.id for r in many[:6]], [r.id for r in many[6:]]
    script = Script(sentences=[
        ("Janssen's applications describe an antibody that binds the p19 subunit of human IL-23.", first),
        ("Janssen's later applications also describe an antibody for plaque psoriasis in a patient.", second)])

    class Reads(Script):
        def __call__(self, prompt, schema):
            if (schema or {}).get("title") == "PatentExtraction":
                return json.dumps({"findings": [{
                    "quote": ABSTRACT_A.split(". ")[0] + ".", "source": "abstract", "target": "IL-23",
                    "modality": "antibody", "covers": "a bispecific antibody", "indication": "plaque psoriasis"}]})
            return script(prompt, schema)

    result, _ = run(settings, tmp_path, Reads(), patents=many)
    body = competitors(result)
    listed = [l for l in body.splitlines() if l.startswith("[P")]
    assert len(listed) == 6 <= PER_TARGET_CITED, "the six cited by the sentence kept; the second would have made 12"
    assert "later applications also describe" not in body
    assert result.report.integrity == [], result.report.integrity


def test_when_there_are_more_filings_than_the_limit_the_likely_subjects_are_read_first(settings, tmp_path):
    from bedrock.litreview.stages.s6c_patent_extract import extract_patents

    f = filings()
    reading = extract_patents(
        [f[2], f[1], f[0]], db=Database(tmp_path / "db.sqlite3"), run_id="r", spec=_spec(), kb=load_knowledge(),
        client=client(Script()), model="m", target_order=["IL23A"], limit=2, say=lambda m: None)
    assert set(reading.skipped) == {"US3A1"}, "the kit that only lists IL-23 is the one left unread"
    assert "over the limit of 2" in reading.skipped["US3A1"]


def _spec():
    from bedrock.litreview.stages.s0_question import parse_question

    return parse_question(QUESTION, None, model=None, knowledge=load_knowledge())


def test_patent_sentences_dropped_are_counted_apart_from_the_papers(settings, tmp_path):
    result, _ = run(settings, tmp_path, Script(sentences=[
        ("Pfizer's application WO1A1 claims a method of treating plaque psoriasis with an antibody.", ["WO1A1"])]),
        patents=[filings()[0]])
    plain = answer_question(QUESTION, settings=settings, client=client(Script()), papers=corpus(),
                            db=Database(tmp_path / "db-plain.sqlite3"))
    assert result.report.stats == plain.report.stats, "a dropped patent sentence is not a dropped paper sentence"
