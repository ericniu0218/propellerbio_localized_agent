"""Proposed names for the knowledge table, from files a person downloaded,
approved into `entities.yaml` by a person.

    bedrock-litreview aliases propose --hgnc hgnc_complete_set.txt \\
        [--mesh desc2026.xml] [--symbol TMEM135 ...] --out proposals.yaml
    # a person reads proposals.yaml and sets `approve: true` on each name
    # they have checked
    bedrock-litreview aliases approve proposals.yaml

Why. A paper that names a target only by an older or alternative name is
invisible to the search and to the gate: 160 of the obesity benchmark's
reference targets and 65 of asthma's have no entry in the table at all
(OPEN_ISSUES #10). HGNC lists every gene's alias and previous symbols and
names; MeSH lists the entry terms of every disease heading. Both are free
downloads, so nothing here reaches the network.

Why a person approves, and why nothing is walked. The table's rule is that a
wrong synonym silently lets the wrong papers in, and a missing one only
costs recall. So:

- Only names of the same thing are read: a gene's own HGNC aliases and
  previous names, and the terms of a MeSH heading's *preferred* concept.
  MeSH's narrower and related concepts and its tree are never followed:
  MeSH files bovine RSV under the RSV heading, the exact near-miss
  `entities.yaml` exists to keep out.
- Code refuses what it can prove is wrong, each with its reason: a name the
  table already gives another entity, a near-miss of this one, a name HGNC
  also gives another gene, two characters or fewer, and MeSH's inverted
  index forms ("Arthritis, Rheumatoid"), which papers do not write.
- Everything else is only proposed, `approve: false`. `approve` writes the
  names a person marked into the table as text, so its comments survive,
  re-checks every refusal against the table as it is then, and reloads the
  result to prove it changed only what was approved. The change is then an
  ordinary diff to review and commit.

Approving names changes the search signature of every question that uses
them (`stages/s1_collect.terms_signature`), so a frozen set is collected
again the next time a search is asked for, and a set reused without one says
it was searched with fewer names.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from bedrock.litreview.knowledge import KIND_BY_SECTION, TABLE, Knowledge, KnownEntity, fold

SECTION_BY_KIND = {kind: section for section, kind in KIND_BY_SECTION.items()}
#: HGNC columns read, in the order names are proposed.
HGNC_FIELDS = ("symbol", "alias_symbol", "prev_symbol", "name", "alias_name", "prev_name")
#: A name this short is refused: "AD", "RA", "p1".
MIN_CHARS = 3
#: A symbol this short is proposed with a caution.
SHORT = 4

HEADER = """\
# Proposed names for the knowledge table (entities.yaml). NOTHING HERE IS USED
# until a person approves it. For each name, change `approve: false` to
# `approve: true` only if you have checked that, in papers of this field, the
# name means exactly this entity. A wrong name silently lets the wrong papers
# in; a missing one only costs recall. Then run:
#
#   bedrock-litreview aliases approve <this file>
#
# Names under `refused` were refused by code, with the reason, and cannot be
# approved. Written by `bedrock-litreview aliases propose`.
"""


class AliasError(ValueError):
    """A file that cannot be read as what it was given as, or a proposal
    that cannot be applied. The message says which."""


# --- the downloaded files ------------------------------------------------------------


@dataclass
class Gene:
    hgnc_id: str
    symbol: str
    names: list[tuple[str, str]] = field(default_factory=list)
    """(name, HGNC column) in `HGNC_FIELDS` order, the symbol first."""


def _split(value: str) -> list[str]:
    return [v.strip() for v in (value or "").strip().strip('"').split("|") if v.strip()]


def read_hgnc(path: Path) -> list[Gene]:
    """HGNC's complete set, or a custom download, as tab-separated text with
    a header row naming at least `symbol`. Withdrawn entries are skipped."""
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or "symbol" not in reader.fieldnames:
            raise AliasError(f"{path}: not an HGNC download (no `symbol` column in the header)")
        genes = []
        for row in reader:
            if (row.get("status") or "Approved").strip().lower().startswith("entry withdrawn"):
                continue
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                continue
            gene = Gene(hgnc_id=(row.get("hgnc_id") or "").strip(), symbol=symbol)
            for column in HGNC_FIELDS:
                for name in _split(row.get(column, "")):
                    gene.names.append((name, column))
            genes.append(gene)
    return genes


@dataclass
class Heading:
    ui: str
    name: str
    terms: list[str]
    """The preferred concept's terms only."""


def read_mesh(path: Path) -> list[Heading]:
    """MeSH descriptors from NLM's XML download (descYYYY.xml), read as a
    stream: each heading's name and the terms of its preferred concept."""
    headings = []
    try:
        for _, el in ET.iterparse(str(path), events=("end",)):
            if el.tag != "DescriptorRecord":
                continue
            ui = (el.findtext("DescriptorUI") or "").strip()
            name = (el.findtext("DescriptorName/String") or "").strip()
            terms: list[str] = []
            for concept in el.findall("ConceptList/Concept"):
                if concept.get("PreferredConceptYN") != "Y":
                    continue
                terms += [(t.findtext("String") or "").strip() for t in concept.findall("TermList/Term")]
            headings.append(Heading(ui=ui, name=name, terms=[t for t in terms if t]))
            el.clear()
    except ET.ParseError as exc:
        raise AliasError(f"{path}: not MeSH descriptor XML ({exc})") from exc
    if not headings:
        raise AliasError(f"{path}: no MeSH descriptor records in it")
    return headings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- what code refuses -------------------------------------------------------------


def _owners(kb: Knowledge) -> dict[str, list[tuple[str, str]]]:
    """Folded name -> (entity, "term" | "near-miss") for every name the
    table holds."""
    owners: dict[str, list[tuple[str, str]]] = {}
    for e in kb.entities:
        for term in (e.canonical, *e.terms, *e.context_terms):
            owners.setdefault(fold(term), []).append((e.canonical, "term"))
        for term in e.near_miss:
            owners.setdefault(fold(term), []).append((e.canonical, "near-miss"))
    return owners


def refusal(name: str, entity: str, owners: dict[str, list[tuple[str, str]]],
            other_genes: list[str] | None = None) -> str | None:
    """Why code refuses `name` for `entity`, or None. "already" means the
    name is already one of the entity's own terms: nothing to propose."""
    if len(name.strip()) < MIN_CHARS or not re.search(r"[A-Za-zα-ωΑ-Ω]", name):
        return f"{MIN_CHARS - 1} characters or fewer, or no letters: too short to be unambiguous"
    if ", " in name:
        return "an inverted index form; papers do not write it"
    held = owners.get(fold(name), [])
    if any(owner == entity and how == "term" for owner, how in held):
        return "already"
    if any(owner == entity and how == "near-miss" for owner, how in held):
        return f"a near-miss of {entity} in the table"
    others = sorted({owner for owner, _ in held if owner != entity})
    if others:
        return "the table already gives this name to " + ", ".join(others)
    if other_genes:
        return "HGNC also gives this name to " + ", ".join(other_genes)
    return None


# --- proposals ---------------------------------------------------------------------


@dataclass
class Proposal:
    entity: str
    kind: str
    new_entity: bool
    names: list[dict] = field(default_factory=list)
    """{"name", "from", "approve": False, and "note" when there is one}."""
    refused: list[dict] = field(default_factory=list)
    """{"name", "from", "reason"}."""

    def as_dict(self) -> dict:
        out = {"entity": self.entity, "kind": self.kind, "new_entity": self.new_entity, "names": self.names}
        if self.refused:
            out["refused"] = self.refused
        return out


def _add(proposal: Proposal, name: str, where: str, reason: str | None, note: str = "") -> None:
    if reason == "already" or any(fold(n["name"]) == fold(name) for n in proposal.names + proposal.refused):
        return
    if reason:
        proposal.refused.append({"name": name, "from": where, "reason": reason})
        return
    entry = {"name": name, "from": where, "approve": False}
    if note:
        entry["note"] = note
    proposal.names.append(entry)


def _hgnc_index(genes: list[Gene]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for gene in genes:
        for name, _ in gene.names:
            index.setdefault(fold(name), set()).add(gene.symbol)
    return index


def propose_targets(kb: Knowledge, genes: list[Gene], symbols: list[str] = ()) -> list[Proposal]:
    """Names for every target the table keys by a current HGNC symbol, and a
    new entry for each of `symbols` the table does not know."""
    by_symbol = {g.symbol: g for g in genes}
    index = _hgnc_index(genes)
    owners = _owners(kb)
    wanted: list[tuple[str, Gene, bool]] = []
    for e in kb.entities:
        if e.kind == "target" and e.canonical in by_symbol:
            wanted.append((e.canonical, by_symbol[e.canonical], False))
    for symbol in dict.fromkeys(symbols):
        if kb.get(symbol) is not None or kb.lookup(symbol, ("target",)) is not None:
            continue
        if symbol not in by_symbol:
            raise AliasError(f"{symbol}: not an approved symbol in the HGNC file")
        wanted.append((symbol, by_symbol[symbol], True))
    proposals = []
    for canonical, gene, new in wanted:
        proposal = Proposal(entity=canonical, kind="target", new_entity=new)
        for name, column in gene.names:
            others = sorted(index.get(fold(name), set()) - {gene.symbol})
            reason = refusal(name, canonical, owners, others)
            note = ""
            if column.startswith("prev"):
                note = "a previous HGNC name: older papers use it"
            elif column.endswith("symbol") and len(name) <= SHORT:
                note = f"{SHORT} characters or fewer: check it is not an ordinary word or abbreviation here"
            _add(proposal, name, f"{gene.hgnc_id or 'HGNC'} {column}", reason, note)
        if proposal.names or proposal.refused:
            proposals.append(proposal)
    return proposals


def propose_diseases(kb: Knowledge, headings: list[Heading]) -> list[Proposal]:
    """Names for every disease in the table from the one MeSH heading whose
    name or preferred terms it already holds. A disease two headings claim
    is left alone: which one is meant is a person's call."""
    owners = _owners(kb)
    by_fold: dict[str, list[Heading]] = {}
    for h in headings:
        for name in {h.name, *h.terms}:
            by_fold.setdefault(fold(name), []).append(h)
    proposals = []
    for e in kb.entities:
        if e.kind != "disease":
            continue
        found = {h.ui: h for term in (e.canonical, *e.terms) for h in by_fold.get(fold(term), [])}
        if len(found) != 1:
            continue
        heading = next(iter(found.values()))
        proposal = Proposal(entity=e.canonical, kind="disease", new_entity=False)
        for name in [heading.name, *heading.terms]:
            _add(proposal, name, f"MeSH {heading.ui} preferred concept", refusal(name, e.canonical, owners))
        if proposal.names or proposal.refused:
            proposals.append(proposal)
    return proposals


def write_proposals(proposals: list[Proposal], sources: list[dict], out: Path) -> None:
    body = {"sources": sources, "proposals": [p.as_dict() for p in proposals]}
    text = yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100)
    Path(out).write_text(HEADER + "\n" + text, encoding="utf-8")


# --- approval ----------------------------------------------------------------------


def _yaml_item(name: str) -> str:
    """`name` as one item of a YAML flow list: bare when it reads back as
    itself, else double-quoted."""
    try:
        if yaml.safe_load(f"[{name}]") == [name]:
            return name
    except yaml.YAMLError:
        pass
    return json.dumps(name, ensure_ascii=False)


def _entity_block(lines: list[str], section: str, canonical: str) -> tuple[int, int, int]:
    """(section end, entity start, entity end) line indexes in the table's
    text; entity start is -1 when the section has no such entity."""
    try:
        head = lines.index(f"{section}:")
    except ValueError as exc:
        raise AliasError(f"the table has no `{section}:` section") from exc
    end = next((i for i in range(head + 1, len(lines)) if lines[i] and not lines[i][0].isspace()
                and not lines[i].startswith("#")), len(lines))
    # A comment block just above the next section belongs to it.
    while end > head + 1 and (not lines[end - 1].strip() or lines[end - 1].startswith("#")):
        end -= 1
    for i in range(head + 1, end):
        m = re.match(r"^  ([^\s#].*?):\s*(#.*)?$", lines[i])
        if m and str(yaml.safe_load(m.group(1))) == canonical:
            stop = next((j for j in range(i + 1, end) if re.match(r"^  [^\s#]", lines[j])), end)
            return end, i, stop
    return end, -1, -1


def _add_terms(lines: list[str], start: int, stop: int, names: list[str]) -> None:
    """Append `names` to the entity's `terms: [...]` flow list."""
    at = next((i for i in range(start, stop) if lines[i].lstrip().startswith("terms: [")), None)
    if at is None:
        raise AliasError(f"{lines[start].strip()} has no `terms: [...]` line to add to")
    close = next((i for i in range(at, stop) if "]" in lines[i].split("#", 1)[0]), None)
    if close is None:
        raise AliasError(f"{lines[start].strip()}: its terms list does not close")
    line = lines[close]
    cut = line.rindex("]", 0, len(line.split("#", 1)[0]))
    lines[close] = line[:cut] + "".join(f", {_yaml_item(n)}" for n in names) + line[cut:]


def approve(proposals_path: Path, table: Path = TABLE) -> list[str]:
    """Write the names a person approved into `table`, and say what changed.
    Nothing is written unless every approved name is still acceptable and
    the reloaded table differs from the old one only by those names."""
    body = yaml.safe_load(Path(proposals_path).read_text(encoding="utf-8")) or {}
    before = Knowledge.load(table)
    owners = _owners(before)
    lines = Path(table).read_text(encoding="utf-8").split("\n")
    changes: list[str] = []
    expected: dict[str, set[str]] = {}
    for p in body.get("proposals") or []:
        entity, kind = str(p["entity"]), str(p["kind"])
        names = [str(n["name"]) for n in p.get("names") or [] if n.get("approve") is True]
        names = [n for n in dict.fromkeys(names)]
        if not names:
            continue
        for name in names:
            reason = refusal(name, entity, owners)
            if reason == "already":
                raise AliasError(f"{entity}: {name!r} is already in the table; was this file approved before?")
            if reason:
                raise AliasError(f"{entity}: {name!r} cannot be approved: {reason}")
        section = SECTION_BY_KIND.get(kind)
        if section is None:
            raise AliasError(f"{entity}: unknown kind {kind!r}")
        end, start, stop = _entity_block(lines, section, entity)
        if start >= 0:
            _add_terms(lines, start, stop, names)
            changes.append(f"{entity}: added {', '.join(names)}")
        else:
            if not p.get("new_entity"):
                raise AliasError(f"{entity}: not in the table's `{section}` section")
            items = ", ".join(_yaml_item(n) for n in names)
            lines[end:end] = [f"  {_yaml_item(entity)}:", f"    terms: [{items}]"]
            changes.append(f"{entity}: new {kind}, terms {', '.join(names)}")
        expected[entity] = set(names)
        owners = _owners(_reload("\n".join(lines)))
    if not changes:
        return []
    after = _reload("\n".join(lines))
    _prove(before, after, expected)
    Path(table).write_text("\n".join(lines), encoding="utf-8")
    return changes


def _reload(text: str) -> Knowledge:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "entities.yaml"
        path.write_text(text, encoding="utf-8")
        try:
            return Knowledge.load(path)
        except yaml.YAMLError as exc:
            raise AliasError(f"the edited table does not parse: {exc}") from exc


def _prove(before: Knowledge, after: Knowledge, expected: dict[str, set[str]]) -> None:
    """The edited table is the old one plus exactly the approved names."""

    def shape(e: KnownEntity | None) -> tuple:
        if e is None:
            return ()
        return (e.kind, tuple(e.terms), e.near_miss, e.near_miss_patterns, e.context_terms)

    names = {e.canonical for e in before.entities} | {e.canonical for e in after.entities}
    for canonical in sorted(names):
        old, new = before.get(canonical), after.get(canonical)
        if canonical not in expected:
            if shape(old) != shape(new):
                raise AliasError(f"refusing to write: the edit would also change {canonical}")
            continue
        old_terms = list(old.terms) if old else []
        if new is None or list(new.terms) != old_terms + [n for n in new.terms if n not in old_terms] \
                or set(new.terms) - set(old_terms) != expected[canonical]:
            raise AliasError(f"refusing to write: {canonical} would not read back as approved")
