from __future__ import annotations

import json
import sys
import asyncio
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from bedrock.biotech import ProteinWorkflow, ToolIntegrationError
from bedrock.biotech.alphafold2 import AlphaFold2Adapter, AlphaFold2Config
from bedrock.biotech.fasta import read_monomer_fasta
from bedrock.biotech.pymol import PyMOLAdapter
from bedrock.biotech.process import SubprocessCommandRunner
from bedrock.config import Settings
from bedrock.provenance import HistoryError, HistoryStore


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
        fasta_arg = next(arg for arg in args if arg.startswith("--fasta_paths="))
        fasta = Path(fasta_arg.partition("=")[2])
        residue_count = len("".join(
            line.strip() for line in fasta.read_text().splitlines() if not line.startswith(">")
        ))
        (target_dir / "ranking_debug.json").write_text(
            json.dumps({"order": ["model_3_ptm", "model_1_ptm"]})
        )
        (target_dir / "confidence_model_1_ptm.json").write_text(
            json.dumps({"residueNumber": list(range(1, residue_count + 1)),
                        "confidenceScore": [10] * residue_count})
        )
        (target_dir / "confidence_model_3_ptm.json").write_text(
            json.dumps({"residueNumber": list(range(1, residue_count + 1)),
                        "confidenceScore": [95] * residue_count})
        )
        (target_dir / "pae_model_3_ptm.json").write_text(
            json.dumps([{"predicted_aligned_error": [[0] * residue_count for _ in range(residue_count)],
                         "max_predicted_aligned_error": 31.75}])
        )


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


class PredictionFailingRunner(FakeToolRunner):
    async def run(
        self,
        args: Sequence[str],
        log_path: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not env:
            raise ToolIntegrationError("GPU worker failed")
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
    return ProteinWorkflow(
        tmp_path / "private runs",
        2500,
        alphafold,
        pymol,
        history=HistoryStore(tmp_path / "history"),
        actor_id="alice",
    )


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

    result = await workflow.predict_and_render(fasta, project_id="project-X")

    assert result.structure_path.name == "ranked_0.pdb"
    assert result.preview_path and result.preview_path.read_bytes() == b"png"
    assert result.session_path and result.session_path.read_bytes() == b"pse"
    assert result.confidence_path and result.confidence_path.is_file()
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
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "completed"
    assert manifest["project_id"] == "project-X"
    assert manifest["source"]["residue_count"] == 20
    assert manifest["source"]["description"] == "secret candidate"
    assert manifest["outputs"]["structure"]["sha256"]
    assert manifest["outputs"]["input"]["sha256"]
    assert manifest["outputs"]["ranking"]["sha256"]
    assert manifest["outputs"]["confidence_summary"]["sha256"]
    assert manifest["alphafold2"]["selected_model"] == "model_3_ptm"
    summary = json.loads(result.confidence_path.read_text())
    assert summary["model_id"] == "model_3_ptm"
    assert summary["plddt"] == [95.0] * 20
    assert workflow.latest_structure() == result.structure_path
    events = workflow.history.events()
    assert [(item["action"], item["state"]) for item in events] == [
        ("fold", "intent"),
        ("fold", "succeeded"),
    ]
    assert events[1]["details"]["manifest_sha256"]



async def test_manifest_names_the_system_of_record_when_the_sequence_was_pulled(tmp_path: Path):
    workflow = configured_workflow(tmp_path, FakeToolRunner())
    staged = tmp_path / "3f1c02d9b4e5-SEQ-00123.fasta"
    staged.write_text(">SEQ-00123\nACDEFGHIK\n")
    provenance = {"ref": "registry://SEQ-00123", "version": "r7", "sha256": "abc123"}

    result = await workflow.predict_and_render(staged, source_provenance=provenance)

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["source"]["system_of_record"] == provenance
    assert manifest["source"]["residue_count"] == 9

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


def test_from_settings_defers_private_history_io_until_an_operation(tmp_path: Path):
    action_history_dir = tmp_path / "history"

    workflow = ProteinWorkflow.from_settings(
        Settings(
            biotech_work_dir=tmp_path / "runs",
            action_history_dir=str(action_history_dir),
            user_id="alice",
        )
    )

    assert workflow.history is None
    assert not action_history_dir.exists()


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
    assert workflow.history.events()[-1]["state"] == "succeeded"


async def test_prediction_failure_retains_failed_manifest_and_history(tmp_path: Path):
    workflow = configured_workflow(tmp_path, PredictionFailingRunner())
    fasta = tmp_path / "protein.fasta"
    fasta.write_text(">candidate\nACDEFGHIK\n")

    with pytest.raises(ToolIntegrationError, match="GPU worker failed"):
        await workflow.predict_and_render(fasta)

    manifest_path = next((tmp_path / "private runs" / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["outputs"]["input"]["sha256"]
    events = workflow.history.events()
    assert [event["state"] for event in events] == ["intent", "failed"]
    assert events[-1]["details"]["error_code"] == "toolintegrationerror"


class CancelRenderingRunner(FakeToolRunner):
    async def run(
        self,
        args: Sequence[str],
        log_path: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if env and "BEDROCK_PYMOL_PREVIEW" in env:
            raise asyncio.CancelledError
        await super().run(args, log_path, env)


async def test_render_cancellation_records_manifest_and_history(tmp_path: Path):
    workflow = configured_workflow(tmp_path, CancelRenderingRunner())
    fasta = tmp_path / "protein.fasta"
    fasta.write_text(">candidate\nACDEFGHIK\n")

    with pytest.raises(asyncio.CancelledError):
        await workflow.predict_and_render(fasta)

    manifest_path = next((tmp_path / "private runs" / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "cancelled"
    assert workflow.history.events()[-1]["state"] == "cancelled"


class FailingOutcomeHistory:
    def __init__(self, real: HistoryStore) -> None:
        self.real = real

    def append(self, operation_id: str, **kwargs) -> int:
        if kwargs["state"] == "succeeded":
            raise HistoryError("disk full")
        return self.real.append(operation_id, **kwargs)


class FailingIntentHistory:
    def append(self, operation_id: str, **kwargs) -> int:
        raise HistoryError("history unavailable")


async def test_fold_does_not_start_when_intent_cannot_be_committed(tmp_path: Path):
    runner = FakeToolRunner()
    workflow = configured_workflow(tmp_path, runner)
    workflow.history = FailingIntentHistory()
    fasta = tmp_path / "protein.fasta"
    fasta.write_text(">candidate\nACDEFGHIK\n")

    with pytest.raises(ToolIntegrationError, match="local history"):
        await workflow.predict_and_render(fasta)

    assert runner.calls == []
    assert not (tmp_path / "private runs" / "runs").exists()


async def test_success_is_not_repeated_when_terminal_history_write_fails(tmp_path: Path):
    runner = FakeToolRunner()
    workflow = configured_workflow(tmp_path, runner)
    real_history = workflow.history
    workflow.history = FailingOutcomeHistory(real_history)
    fasta = tmp_path / "protein.fasta"
    fasta.write_text(">candidate\nACDEFGHIK\n")

    result = await workflow.predict_and_render(fasta)

    assert result.structure_path.is_file()
    assert len(runner.calls) == 2
    assert any("history" in warning.lower() for warning in result.warnings)
    assert real_history.events()[-1]["state"] == "intent"


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
