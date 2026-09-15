"""Strict parsing for the monomer FASTA accepted by the first AF2 capability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bedrock.biotech.errors import ToolIntegrationError

# AlphaFold maps unsupported residues to unknown internally. Keeping this list
# explicit catches common mistakes (alignment gaps, stop markers, pasted DNA
# punctuation) before an expensive job starts while retaining standard
# ambiguous/rare amino-acid letters.
ALLOWED_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYXBZJUO")


@dataclass(frozen=True)
class FastaRecord:
    description: str
    sequence: str


def read_monomer_fasta(path: Path, max_length: int) -> FastaRecord:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ToolIntegrationError(f"FASTA file does not exist: {path}")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ToolIntegrationError("FASTA file is unexpectedly large (limit: 10 MB).")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolIntegrationError("FASTA must be UTF-8 text.") from exc

    descriptions: list[str] = []
    sequences: list[list[str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            description = line[1:].strip()
            if not description:
                raise ToolIntegrationError(f"FASTA header on line {line_number} is empty.")
            descriptions.append(description)
            sequences.append([])
            continue
        if not sequences:
            raise ToolIntegrationError("FASTA sequence appears before its >header.")
        sequences[-1].append("".join(line.split()).upper())

    if len(sequences) != 1:
        raise ToolIntegrationError(
            "The first integration accepts exactly one FASTA record (a monomer)."
        )
    sequence = "".join(sequences[0])
    if not sequence:
        raise ToolIntegrationError("FASTA record has no sequence.")
    if len(sequence) > max_length:
        raise ToolIntegrationError(
            f"Sequence has {len(sequence)} residues; configured limit is {max_length}."
        )
    invalid = sorted(set(sequence) - ALLOWED_AMINO_ACIDS)
    if invalid:
        shown = " ".join(repr(letter) for letter in invalid[:8])
        raise ToolIntegrationError(f"FASTA contains unsupported residue characters: {shown}")
    return FastaRecord(descriptions[0], sequence)
