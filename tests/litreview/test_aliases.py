"""Alias import (`knowledge/aliases.py`, `bedrock-litreview aliases`):
names from files a person downloaded, proposed, and written into the table
only when a person approves them.

What this proves, from small HGNC and MeSH files written here:

- proposals hold only a gene's own HGNC names and a MeSH heading's preferred
  concept; MeSH's narrower concepts are never read
- code refuses what it can prove wrong, with the reason: a name another
  entity holds, a near-miss, a name HGNC gives another gene, a short name,
  an inverted MeSH form
- nothing is approved by default, and `approve` with nothing marked leaves
  the table byte for byte
- `approve` edits the table as text, so its comments survive, adds exactly
  the marked names, re-checks refusals against the table as it is then, and
  can add a new target
- the knowledge loader and the search signature see the approved names
"""

from __future__ import annotations

import pytest
import yaml

from bedrock.litreview import cli
from bedrock.litreview.knowledge import Knowledge
from bedrock.litreview.knowledge import aliases as al
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s1_collect import terms_signature

TABLE = """\
# A small table, with the comments a real one has.
diseases:
  obesity:
    terms: [obesity, obese]
    near_miss: [overweight]
  asthma:
    terms: [asthma, asthmatic]

# Targets are mostly answers.
targets:
  IL23A:
    terms: [IL-23, IL23A, interleukin-23,
            IL-23p19]
    near_miss: [IL-12]
  LEP:
    terms: [leptin, LEP]   # the hormone
"""

HGNC = "\t".join(("hgnc_id", "symbol", "name", "status", "alias_symbol", "alias_name", "prev_symbol", "prev_name")) + "\n"
HGNC += "\t".join(("HGNC:15488", "IL23A", "interleukin 23 subunit alpha", "Approved", '"SGRF|P19|IL-23"',
                   '"IL-23p19"', "", "")) + "\n"
HGNC += "\t".join(("HGNC:6553", "LEP", "leptin", "Approved", '"OB|OBS"', "", "", "")) + "\n"
HGNC += "\t".join(("HGNC:1787", "CDKN2D", "cyclin dependent kinase inhibitor 2D", "Approved", "P19", "", "", "")) + "\n"
HGNC += "\t".join(("HGNC:26167", "TMEM135", "transmembrane protein 135", "Approved", "PMP52", "", "", "")) + "\n"
HGNC += "\t".join(("HGNC:1", "OLD1", "withdrawn gene", "Entry Withdrawn", "", "", "", "")) + "\n"

MESH = """<?xml version="1.0"?>
<DescriptorRecordSet>
 <DescriptorRecord>
  <DescriptorUI>D009765</DescriptorUI>
  <DescriptorName><String>Obesity</String></DescriptorName>
  <ConceptList>
   <Concept PreferredConceptYN="Y"><TermList>
     <Term><String>Obesity</String></Term>
   </TermList></Concept>
   <Concept PreferredConceptYN="N"><TermList>
     <Term><String>Obesity, Abdominal</String></Term>
     <Term><String>Visceral obesity</String></Term>
   </TermList></Concept>
  </ConceptList>
 </DescriptorRecord>
 <DescriptorRecord>
  <DescriptorUI>D001249</DescriptorUI>
  <DescriptorName><String>Asthma</String></DescriptorName>
  <ConceptList>
   <Concept PreferredConceptYN="Y"><TermList>
     <Term><String>Asthma</String></Term>
     <Term><String>Asthmas</String></Term>
     <Term><String>Bronchial Asthma</String></Term>
     <Term><String>Asthma, Bronchial</String></Term>
   </TermList></Concept>
   <Concept PreferredConceptYN="N"><TermList>
     <Term><String>Exercise-Induced Asthma</String></Term>
   </TermList></Concept>
  </ConceptList>
 </DescriptorRecord>
</DescriptorRecordSet>
"""


@pytest.fixture()
def files(tmp_path):
    table, hgnc, mesh = tmp_path / "entities.yaml", tmp_path / "hgnc.txt", tmp_path / "desc.xml"
    table.write_text(TABLE, encoding="utf-8")
    hgnc.write_text(HGNC, encoding="utf-8")
    mesh.write_text(MESH, encoding="utf-8")
    return table, hgnc, mesh


def by_entity(proposals):
    return {p.entity: p for p in proposals}


def names(p):
    return [n["name"] for n in p.names]


def refused(p):
    return {n["name"]: n["reason"] for n in p.refused}


# --- reading ---------------------------------------------------------------------


def test_hgnc_is_read_with_every_name_column_and_withdrawn_entries_skipped(files):
    genes = {g.symbol: g for g in al.read_hgnc(files[1])}
    assert "OLD1" not in genes
    assert genes["IL23A"].names[:4] == [("IL23A", "symbol"), ("SGRF", "alias_symbol"), ("P19", "alias_symbol"),
                                        ("IL-23", "alias_symbol")]


def test_mesh_reads_the_preferred_concept_only(files):
    obesity, asthma = al.read_mesh(files[2])
    assert obesity.terms == ["Obesity"]
    assert "Exercise-Induced Asthma" not in asthma.terms and "Bronchial Asthma" in asthma.terms


def test_a_file_that_is_not_what_it_was_given_as_is_refused(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("not\ta\theader\n", encoding="utf-8")
    with pytest.raises(al.AliasError, match="HGNC"):
        al.read_hgnc(bad)
    with pytest.raises(al.AliasError, match="MeSH"):
        al.read_mesh(bad)


# --- proposing -------------------------------------------------------------------


def test_target_names_are_proposed_and_what_code_can_prove_wrong_is_refused(files):
    kb = Knowledge.load(files[0])
    got = by_entity(al.propose_targets(kb, al.read_hgnc(files[1])))
    il23 = got["IL23A"]
    # Already in the table: not proposed at all.
    assert "IL-23" not in names(il23) and "IL-23" not in refused(il23)
    assert names(il23) == ["SGRF", "interleukin 23 subunit alpha"]
    assert all(n["approve"] is False for n in il23.names)
    assert "HGNC also gives this name to CDKN2D" in refused(il23)["P19"]
    lep = got["LEP"]
    assert "too short" in refused(lep)["OB"]
    assert names(lep) == ["OBS"] and "4 characters or fewer" in lep.names[0]["note"]
    # No symbol asked for, so no new entry.
    assert "TMEM135" not in got


def test_a_new_target_is_proposed_only_when_asked_for(files):
    kb = Knowledge.load(files[0])
    got = by_entity(al.propose_targets(kb, al.read_hgnc(files[1]), ["TMEM135", "IL23A"]))
    assert got["TMEM135"].new_entity is True
    assert names(got["TMEM135"]) == ["TMEM135", "PMP52", "transmembrane protein 135"]
    assert got["IL23A"].new_entity is False
    with pytest.raises(al.AliasError, match="not an approved symbol"):
        al.propose_targets(kb, al.read_hgnc(files[1]), ["NOPE1"])


def test_disease_names_come_from_one_heading_and_inverted_forms_are_refused(files):
    kb = Knowledge.load(files[0])
    got = by_entity(al.propose_diseases(kb, al.read_mesh(files[2])))
    assert "obesity" not in got, "the preferred concept adds nothing the table lacks"
    asthma = got["asthma"]
    assert names(asthma) == ["Asthmas", "Bronchial Asthma"]
    assert refused(asthma)["Asthma, Bronchial"].startswith("an inverted index form")


def test_a_name_the_table_gives_another_entity_or_a_near_miss_is_refused(files):
    kb = Knowledge.load(files[0])
    owners = al._owners(kb)
    assert al.refusal("leptin", "IL23A", owners) == "the table already gives this name to LEP"
    assert al.refusal("IL-12", "IL23A", owners) == "a near-miss of IL23A in the table"
    assert al.refusal("overweight", "obesity", owners) == "a near-miss of obesity in the table"
    assert al.refusal("IL23-A", "IL23A", owners) == "already"


# --- approving -------------------------------------------------------------------


def proposals_file(tmp_path, files, mark=(), symbols=()):
    kb = Knowledge.load(files[0])
    proposals = al.propose_targets(kb, al.read_hgnc(files[1]), list(symbols))
    proposals += al.propose_diseases(kb, al.read_mesh(files[2]))
    out = tmp_path / "proposals.yaml"
    al.write_proposals(proposals, [{"kind": "hgnc", "file": "hgnc.txt", "sha256": al.sha256(files[1])}], out)
    body = yaml.safe_load(out.read_text(encoding="utf-8"))
    for p in body["proposals"]:
        for n in p["names"]:
            if (p["entity"], n["name"]) in mark:
                n["approve"] = True
    out.write_text(al.HEADER + yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def test_nothing_marked_leaves_the_table_byte_for_byte(files, tmp_path):
    before = files[0].read_bytes()
    assert al.approve(proposals_file(tmp_path, files), files[0]) == []
    assert files[0].read_bytes() == before


def test_approve_adds_exactly_the_marked_names_and_keeps_the_comments(files, tmp_path):
    marked = {("IL23A", "SGRF"), ("asthma", "Bronchial Asthma"), ("LEP", "OBS")}
    changes = al.approve(proposals_file(tmp_path, files, marked), files[0])
    assert changes == ["IL23A: added SGRF", "LEP: added OBS", "asthma: added Bronchial Asthma"]
    text = files[0].read_text(encoding="utf-8")
    assert "# A small table, with the comments a real one has." in text and "# the hormone" in text
    assert "            IL-23p19, SGRF]" in text
    kb = Knowledge.load(files[0])
    assert kb.get("IL23A").terms[-1] == "SGRF"
    assert kb.lookup("Bronchial Asthma", ("disease",)).canonical == "asthma"
    assert kb.get("LEP").terms == ("leptin", "LEP", "OBS")
    # Unmarked names stay out.
    assert kb.lookup("interleukin 23 subunit alpha") is None


def test_approve_adds_a_new_target_at_the_end_of_its_section(files, tmp_path):
    marked = {("TMEM135", "TMEM135"), ("TMEM135", "PMP52")}
    al.approve(proposals_file(tmp_path, files, marked, symbols=["TMEM135"]), files[0])
    text = files[0].read_text(encoding="utf-8")
    assert text.rstrip().endswith("  TMEM135:\n    terms: [TMEM135, PMP52]")
    assert Knowledge.load(files[0]).get("TMEM135").kind == "target"


def test_a_new_disease_entity_goes_before_the_next_sections_comment(files, tmp_path):
    lines = files[0].read_text(encoding="utf-8").split("\n")
    end, start, _ = al._entity_block(lines, "diseases", "nothing")
    assert start == -1 and lines[end - 1].startswith("    terms: [asthma")


def test_approve_rechecks_against_the_table_as_it_is_now(files, tmp_path):
    proposals = proposals_file(tmp_path, files, {("IL23A", "SGRF")})
    # Someone gave SGRF to another entity after the file was proposed.
    files[0].write_text(TABLE + "  OTHER:\n    terms: [SGRF]\n", encoding="utf-8")
    before = files[0].read_bytes()
    with pytest.raises(al.AliasError, match="already gives this name to OTHER"):
        al.approve(proposals, files[0])
    assert files[0].read_bytes() == before
    # A name hand-edited into the file is checked the same way.
    files[0].write_text(TABLE, encoding="utf-8")
    body = yaml.safe_load(proposals.read_text(encoding="utf-8"))
    body["proposals"][0]["names"] = []
    body["proposals"][0]["names"].append({"name": "IL-12", "approve": True})
    proposals.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(al.AliasError, match="near-miss"):
        al.approve(proposals, files[0])


def test_a_name_yaml_would_misread_is_quoted():
    assert al._yaml_item("IL-23") == "IL-23"
    assert al._yaml_item("yes") == '"yes"'
    assert al._yaml_item("a: b") == '"a: b"'
    assert al._yaml_item("Crohn's disease") == "Crohn's disease"


def test_approved_names_widen_the_search_signature(files, tmp_path):
    question = "If I want to treat asthma, what should I target?"
    before = terms_signature(parse_question(question, None, model=None, knowledge=Knowledge.load(files[0])))
    al.approve(proposals_file(tmp_path, files, {("asthma", "Bronchial Asthma")}), files[0])
    after = terms_signature(parse_question(question, None, model=None, knowledge=Knowledge.load(files[0])))
    assert before != after and "bronchial asthma" in after


# --- the command -----------------------------------------------------------------


def test_the_command_proposes_then_approves(files, tmp_path, capsys):
    table, hgnc, mesh = files
    out = tmp_path / "p.yaml"
    assert cli.main(["aliases", "propose", "--hgnc", str(hgnc), "--mesh", str(mesh), "--out", str(out)]) == 0
    assert "Nothing is used until a person marks names" in capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    assert text.startswith(al.HEADER) and "approve: true" not in text[len(al.HEADER):]
    assert cli.main(["aliases", "approve", str(out), "--table", str(table)]) == 0
    assert "the table is unchanged" in capsys.readouterr().out
    assert cli.main(["aliases", "propose", "--out", str(out)]) == 1
