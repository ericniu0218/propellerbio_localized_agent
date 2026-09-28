"""Which tool a request plainly asks for, when its domain names none.

The Specialist picks a tool by the request's domain (``specialist._DOMAIN_TOOL``),
and five of the eight tools had no domain that led to them, so no chat could
ever reach them. These detectors look for unmistakable evidence in the request
itself and return the tool with its arguments already filled in:

* a record in a configured system of record (``lims://P42``) -> ``eln_lims``;
* a nucleotide or protein sequence **and** the operation asked for (translate,
  transcribe, reverse complement, BLAST, structure) -> ``sequence_structure``;
* SMILES strings RDKit can parse, in a chemistry request -> ``chemistry``;
* "draft/write an SOP / study report / deviation report" -> ``doc_template``.

Deliberately code, not a model: a model choosing among tools is where tool
misuse comes from, and each of these needs specific arguments a guess would get
wrong. No detector ever selects ``data_analysis_sandbox`` - it runs code, and
stays unreachable until it runs in real isolation (docs/prompts/
AGENT_PACK_INTEGRATION.md section 9.8).
"""

from __future__ import annotations

import re

from bedrock.server.schemas.enums import TaskDomain

_REF = re.compile(r"\b([a-z][a-z0-9_-]*)://([^\s,;]+)")
_FASTA = re.compile(r">[^\n]*\n([A-Za-z\s*]+)")
_NUCLEOTIDES = re.compile(r"\b[ACGTUN]{30,}\b")
_PROTEIN = re.compile(r"\b[ACDEFGHIKLMNPQRSTVWY]{20,}\b")
_OPERATIONS = (
    ("reverse_complement", re.compile(r"reverse[- ]complement", re.I)),
    ("transcribe", re.compile(r"\btranscri(be|ption)\b", re.I)),
    ("translate", re.compile(r"\btranslat(e|ion)\b", re.I)),
    ("blastn", re.compile(r"\bblast\b", re.I)),
    ("structure_predict", re.compile(r"\b(fold|structure)\b", re.I)),
)
_CHEMISTRY_WORDS = re.compile(r"\b(smiles|molecule|compound|similarity|logp|molecular weight|properties)\b", re.I)
_SMILES_TOKEN = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$/\\%.]{4,}$")
_TEMPLATES = (
    ("sop", re.compile(r"\b(sop|standard operating procedure)\b", re.I)),
    ("study_report", re.compile(r"\bstudy report\b", re.I)),
    ("deviation_report", re.compile(r"\bdeviation report\b", re.I)),
)
_DRAFT = re.compile(r"\b(draft|write|prepare|create)\b", re.I)


def detect_tool(text: str, domain: TaskDomain, connectors: set[str]) -> tuple[str, dict] | None:
    for detector in (_record, _sequence, _smiles, _template):
        found = detector(text, domain, connectors)
        if found is not None:
            return found
    return None


def _record(text, domain, connectors):
    for match in _REF.finditer(text):
        if match.group(1) in connectors:
            operation = "get_sample" if re.search(r"\bsample\b", text, re.I) else "get_experiment"
            return "eln_lims", {"operation": operation, "source": match.group(0)}
    return None


def _sequence(text, domain, connectors):
    operation = next((name for name, pattern in _OPERATIONS if pattern.search(text)), None)
    if operation is None:
        return None
    fasta = _FASTA.search(text)
    if fasta:
        sequence = re.sub(r"\s", "", fasta.group(1))
    else:
        match = _NUCLEOTIDES.search(text) or _PROTEIN.search(text)
        if match is None:
            return None
        sequence = match.group(0)
    return "sequence_structure", {"operation": operation, "sequence": sequence}


def _smiles(text, domain, connectors):
    if domain != TaskDomain.CHEMISTRY and not _CHEMISTRY_WORDS.search(text):
        return None
    try:
        from rdkit import Chem, RDLogger  # type: ignore
    except ImportError:
        return None
    RDLogger.DisableLog("rdApp.*")
    found = []
    for token in re.split(r"[\s,;]+", text):
        token = token.strip("`'\".")
        if not _SMILES_TOKEN.match(token) or not re.search(r"[=#()\[\]0-9]", token):
            continue
        molecule = Chem.MolFromSmiles(token)
        if molecule is not None and molecule.GetNumHeavyAtoms() >= 2:
            found.append(token)
    if not found:
        return None
    if len(found) >= 2:
        return "chemistry", {"operation": "similarity", "smiles": found[0], "other_smiles": found[1]}
    return "chemistry", {"operation": "properties", "smiles": found[0]}


def _template(text, domain, connectors):
    if not _DRAFT.search(text):
        return None
    for template, pattern in _TEMPLATES:
        if pattern.search(text):
            return "doc_template", {"template": template, "fields": {"title": text.strip()[:120]}}
    return None
