from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from bedrock.biotech import ProteinWorkflow, ToolIntegrationError
from bedrock.biotech.alphafold2 import AlphaFold2Adapter, AlphaFold2Config
from bedrock.biotech.fasta import read_monomer_fasta
from bedrock.biotech.pymol import PyMOLAdapter
from bedrock.biotech.process import SubprocessCommandRunner


class FakeToolRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], Path, Mapping[str, str] | None]] = []

    async def run(
        self,
        args: Sequence[str],
        log_path: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.calls.append((tuple(args), log_path, env))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake tool completed\n")
        if env and "BEDROCK_PYMOL_PREVIEW" in env:
            Path(env["BEDROCK_PYMOL_PREVIEW"]).write_bytes(b"png")
            Path(env["BEDROCK_PYMOL_SESSION"]).write_bytes(b"pse")
            return
        output_arg = next(arg for arg in args if arg.startswith("--output_dir="))
        output_dir = Path(output_arg.partition("=")[2])
        target_dir = output_dir / "input"
        target_dir.mkdir()
        (target_dir / "ranked_0.pdb").write_text("ATOM\n")
        (target_dir / "confidence_model_1.json").write_text('{"confidenceScore": [95]}\n')


class RenderFailingRunner(FakeToolRunner):
    async def run(
        self,
        args: Sequence[str],
        log_path: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if env and "BEDROCK_PYMOL_PREVIEW" in env:
            raise ToolIntegrationError("headless renderer unavailable")
        await super().run(args, log_path, env)


def configured_workflow(tmp_path: Path, runner: FakeToolRunner) -> ProteinWorkflow:
    launcher = tmp_path / "AlphaFold source" / "docker" / "run_docker.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fake launcher\n")
    data_dir = tmp_path / "databases"
    data_dir.mkdir()
    alphafold = AlphaFold2Adapter(
        AlphaFold2Config(
            launcher=launcher,
            data_dir=data_dir,
            python_executable=sys.executable,
            max_template_date="2026-01-01",
            docker_image="alphafold@sha256:test",
            database_version="2026-01-01",
        ),
        runner,
    )
    pymol = PyMOLAdapter(sys.executable, runner)
    return ProteinWorkflow(tmp_path / "private runs", 2500, alphafold, pymol)


def test_monomer_fasta_validation(tmp_path: Path):
    fasta = tmp_path / "protein.fasta"
    fasta.write_text(">candidate A\nACDE FGHIK\n")

    record = read_monomer_fasta(fasta, max_length=20)

    assert record.description == "candidate A"
    assert record.sequence == "ACDEFGHIK"


@pytest.mark.parametrize(
    "contents, message",
    [
        (">one\nACD\n>two\nEFG\n", "exactly one"),
        (">one\nACD-*\n", "unsupported residue"),
    ],
)
def test_monomer_fasta_rejects_unsupported_inputs(tmp_path: Path, contents: str, message: str):
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(contents)

    with pytest.raises(ToolIntegrationError, match=message):
        read_monomer_fasta(fasta, max_length=20)


async def test_full_local_workflow_hands_af2_output_to_pymol_and_records_provenance(tmp_path: Path):
    runner = FakeToolRunner()
    workflow = configured_workflow(tmp_path, runner)
    fasta = tmp_path / "company protein.fasta"
    fasta.write_text(">secret candidate\nACDEFGHIKLMNPQRSTVWY\n")

    result = await workflow.predict_and_render(fasta)

    assert result.structure_path.name == "ranked_0.pdb"
    assert result.preview_path and result.preview_path.read_bytes() == b"png"
    assert result.session_path and result.session_path.read_bytes() == b"pse"
    assert len(runner.calls) == 2
    af2_args, _, _ = runner.calls[0]
    assert any(arg.startswith("--fasta_paths=") for arg in af2_args)
    assert "--model_preset=monomer_ptm" in af2_args
    assert "--max_template_date=2026-01-01" in af2_args
    assert "--docker_image_name=alphafold@sha256:test" in af2_args
    assert "--enable_gpu_relax=true" in af2_args
    _, _, pymol_env = runner.calls[1]
    assert pymol_env and pymol_env["BEDROCK_PYMOL_INPUT"] == str(result.structure_path)
    assert pymol_env["BEDROCK_PYMOL_CONFIDENCE"] == "plddt"

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["source"]["residue_count"] == 20
    assert manifest["source"]["description"] == "secret candidate"
    assert manifest["outputs"]["structure"]["sha256"]
    assert workflow.latest_structure() == result.structure_path


def test_status_lists_missing_operator_configuration(tmp_path: Path):
    workflow = ProteinWorkflow(
        tmp_path,
        2500,
        AlphaFold2Adapter(AlphaFold2Config(None, None, "missing-python", None)),
        PyMOLAdapter("missing-pymol"),
    )

    statuses = workflow.status()

    assert statuses[0].ready is False
    assert any("ALPHAFOLD_LAUNCHER" in issue for issue in statuses[0].issues)
    assert statuses[1].ready is False


async def test_render_failure_preserves_successful_prediction(tmp_path: Path):
    workflow = configured_workflow(tmp_path, RenderFailingRunner())
    fasta = tmp_path / "protein.fasta"
    fasta.write_text(">candidate\nACDEFGHIK\n")

    result = await workflow.predict_and_render(fasta)

    assert result.structure_path.is_file()
    assert result.preview_path is None
    assert "headless renderer unavailable" in result.warnings[0]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] == "completed_with_render_error"


async def test_subprocess_runner_does_not_pass_parent_secrets(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BEDROCK_API_KEY", "do-not-pass-this")
    log = tmp_path / "environment.log"

    await SubprocessCommandRunner().run(
        (
            sys.executable,
            "-c",
            "import os; print(os.environ.get('BEDROCK_API_KEY', 'absent'))",
        ),
        log,
    )

    assert log.read_text().strip() == "absent"
