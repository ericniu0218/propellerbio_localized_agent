"""Chemistry tools: RDKit for molecule parsing, properties, and similarity.
Stubs a clear result if RDKit isn't installed."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel

from app.tools.base import Tool, ToolResult

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Descriptors  # type: ignore
    from rdkit.DataStructs import TanimotoSimilarity  # type: ignore
    from rdkit.Chem import RDKFingerprint  # type: ignore

    _RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when RDKit is absent
    _RDKIT_AVAILABLE = False


class ChemistryToolArgs(BaseModel):
    operation: Literal["properties", "similarity"]
    smiles: str
    other_smiles: str | None = None  # required for "similarity"


class RDKitTool(Tool):
    name: ClassVar[str] = "chemistry"
    description: ClassVar[str] = (
        "RDKit-based molecule parsing, property calculation, and Tanimoto "
        "similarity between two SMILES strings."
    )
    args_model: ClassVar[type[BaseModel]] = ChemistryToolArgs

    async def run(self, args: ChemistryToolArgs) -> ToolResult:
        if not _RDKIT_AVAILABLE:
            return ToolResult(
                summary="RDKit is not installed; this is a stub result.",
                content="",
                citations=[],
            )
        if args.operation == "properties":
            return self._properties(args.smiles)
        return self._similarity(args.smiles, args.other_smiles)

    def _properties(self, smiles: str) -> ToolResult:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ToolResult(summary=f"Invalid SMILES: {smiles}", content="", citations=[])
        props = {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "h_bond_donors": Descriptors.NumHDonors(mol),
            "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
            "tpsa": Descriptors.TPSA(mol),
        }
        return ToolResult(
            summary=f"Computed {len(props)} properties for {smiles}.",
            content=str(props),
            citations=[],
        )

    def _similarity(self, smiles: str, other_smiles: str | None) -> ToolResult:
        if not other_smiles:
            return ToolResult(
                summary="`other_smiles` is required for similarity.",
                content="",
                citations=[],
            )
        mol_a, mol_b = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(other_smiles)
        if mol_a is None or mol_b is None:
            return ToolResult(summary="One or both SMILES were invalid.", content="", citations=[])
        sim = TanimotoSimilarity(RDKFingerprint(mol_a), RDKFingerprint(mol_b))
        return ToolResult(
            summary=f"Tanimoto similarity: {sim:.3f}", content=str(sim), citations=[]
        )
