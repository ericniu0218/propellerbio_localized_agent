"""FASTA — sequence data, not prose.

Before this adapter, a FASTA file fell through to ``TextAdapter``, which
treats it as paragraphs and headings. That throws away exactly the
structure a biologist actually cares about: how many sequences are in the
file, whether it's nucleotide or protein, and how long each record is. A
FASTA header line (``>id description``) isn't a Markdown heading and a
sequence line isn't a paragraph, so the generic adapter's output — "N
paragraphs, 0 headings" — is close to useless here.

Same idea as ``TabularAdapter``: compute the exact, structural facts
(record count, per-record length, alphabet) instead of asking a model to
guess at a summary, and keep every full sequence as its own segment for
exact lookup later.

Alphabet detection is a cheap, honest heuristic — a majority-character
vote against the standard IUPAC nucleotide and amino-acid codes — not a
real bioinformatics classifier. It gets typical DNA/RNA/protein files
right and says "ambiguous" rather than guessing when it isn't sure, which
is the right failure mode for a summary nobody should treat as ground
truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bedrock.context.adapters.base import Source
from bedrock.context.types import Extraction, Segment

_HEADER = re.compile(r"^>(\S*)\s*(.*)$")
_MAX_LISTED_RECORDS = 15

# IUPAC codes. Nucleotide alphabets include the ambiguity codes (R, Y, S,
# W, K, M, B, D, H, V, N) because real sequencing data uses them; a strict
# ACGT-only check would misclassify almost every real-world file.
_DNA_CHARS = frozenset("ACGTRYSWKMBDHVN")
_RNA_CHARS = frozenset("ACGURYSWKMBDHVN")
_PROTEIN_ONLY_CHARS = frozenset("EFILPQZX*")  # appear in protein, never in nucleotide codes
_PROTEIN_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYBJZUOX*")
_GAP_CHARS = frozenset("-.\n\r ")


@dataclass(frozen=True)
class FastaRecord:
    header: str
    description: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


class FastaAdapter:
    name = "fasta"

    def can_handle(self, source: Source) -> bool:
        if source.suffix in {".fasta", ".fa", ".fna", ".ffn", ".faa", ".frn", ".mpfa"}:
            return True
        stripped = source.text.lstrip()
        if not stripped.startswith(">"):
            return False
        return len(_parse(source.text)) > 0

    def extract(self, source: Source) -> Extraction:
        records = _parse(source.text)
        if not records:
            return Extraction(source_kind="fasta", overview="Empty or unparsable FASTA file.", segments=[])

        lengths = [r.length for r in records]
        alphabets = [classify_alphabet(r.sequence) for r in records]
        overall = _overall_alphabet(alphabets)

        overview = _render_overview(records, lengths, alphabets, overall)
        segments = [
            Segment(text=overview, kind="schema", label="schema", weight=5.0),
        ]
        for record in records:
            label = record.header or record.description or "unnamed record"
            segments.append(
                Segment(
                    text=f">{record.header} {record.description}\n{record.sequence}".strip(),
                    kind="record",
                    label=label[:80],
                )
            )

        return Extraction(
            source_kind="fasta",
            overview=overview,
            segments=segments,
            metadata={
                "records": len(records),
                "alphabet": overall,
                "total_residues": sum(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
            },
        )


def classify_alphabet(sequence: str) -> str:
    """"dna" | "rna" | "protein" | "ambiguous" for one sequence.

    A majority vote, not a strict membership test: real sequences carry
    the occasional stray character (a masked base, an OCR artifact in a
    pasted file) that shouldn't tip a 500-residue protein into
    "ambiguous". Everything is checked against the DNA/RNA alphabets
    first because those are the smaller, more specific ones — a
    nucleotide sequence is never mistaken for protein, but a short,
    coincidentally ACGT-heavy protein fragment is the one real ambiguity,
    and it's rare enough that this order gets it right in practice.
    """
    letters = [c for c in sequence.upper() if c not in _GAP_CHARS]
    if not letters:
        return "ambiguous"
    total = len(letters)

    has_u = sum(1 for c in letters if c == "U")
    has_t = sum(1 for c in letters if c == "T")
    dna_hits = sum(1 for c in letters if c in _DNA_CHARS)
    rna_hits = sum(1 for c in letters if c in _RNA_CHARS)
    protein_only_hits = sum(1 for c in letters if c in _PROTEIN_ONLY_CHARS)

    if protein_only_hits / total < 0.02:
        if dna_hits / total >= 0.95 and has_t >= has_u:
            return "dna"
        if rna_hits / total >= 0.95 and has_u > has_t:
            return "rna"

    protein_hits = sum(1 for c in letters if c in _PROTEIN_CHARS)
    if protein_hits / total >= 0.95:
        return "protein"
    return "ambiguous"


def gc_content(sequence: str) -> float | None:
    """GC% for a nucleotide sequence, or None if it isn't one worth reporting."""
    letters = [c for c in sequence.upper() if c not in _GAP_CHARS]
    if not letters:
        return None
    gc = sum(1 for c in letters if c in "GC")
    return 100.0 * gc / len(letters)


def _overall_alphabet(alphabets: list[str]) -> str:
    distinct = set(alphabets)
    if len(distinct) == 1:
        return distinct.pop()
    return "mixed"


def _render_overview(
    records: list[FastaRecord], lengths: list[int], alphabets: list[str], overall: str
) -> str:
    total = sum(lengths)
    lines = [
        f"FASTA file: {len(records):,} sequence{'s' if len(records) != 1 else ''}, "
        f"alphabet: {overall}, {total:,} total residues "
        f"(length {min(lengths):,}-{max(lengths):,}, avg {total // len(records):,}).",
    ]
    if overall in ("dna", "rna"):
        gc_values = [g for g in (gc_content(r.sequence) for r in records) if g is not None]
        if gc_values:
            lines.append(f"GC content: {sum(gc_values) / len(gc_values):.1f}% average.")
    lines.append("")
    lines.append("Records:")
    for record, length, alphabet in list(zip(records, lengths, alphabets))[:_MAX_LISTED_RECORDS]:
        label = record.header or "(no id)"
        desc = f" — {record.description}" if record.description else ""
        flag = "" if alphabet == overall or overall == "mixed" else f" [{alphabet}]"
        lines.append(f"- {label}{desc}: {length:,} residues{flag}")
    if len(records) > _MAX_LISTED_RECORDS:
        lines.append(f"  ... and {len(records) - _MAX_LISTED_RECORDS} more records")
    return "\n".join(lines)


def _parse(text: str) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    description = ""
    sequence_lines: list[str] = []

    def flush() -> None:
        if header is not None:
            records.append(FastaRecord(header=header, description=description, sequence="".join(sequence_lines)))

    for line in text.splitlines():
        match = _HEADER.match(line)
        if match:
            flush()
            header, description = match.group(1), match.group(2)
            sequence_lines = []
            continue
        stripped = line.strip()
        if header is not None and stripped:
            sequence_lines.append(stripped)
    flush()
    return records
