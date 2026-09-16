"""Shared test helpers.

``FakeBackend`` implements the InferenceBackend protocol in memory, so the UI
can be tested without any server. This is exactly the swap-ability the
architecture is designed for: the app cannot tell the difference.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import AsyncIterator

import pytest

from bedrock.biotech.confidence import ConfidenceReport, PaeOverview
from bedrock.config import Settings
from bedrock.inference import ModelInfo
from bedrock.inference.base import InferenceError
from bedrock.provenance import HistoryStore, RunCatalog, RunError
from bedrock.ui.services import ComplianceServices


class FakeBackend:
    def __init__(self, models: list[str] | None = None, reply: str = "Hello from the fake model.", connected: bool = True) -> None:
        self.models = models if models is not None else ["test-model", "other-model"]
        self.reply = reply
        self.connected = connected
        self.requests: list[tuple[str, list[dict[str, str]]]] = []
        self.closed = False
        self.fragment_delay = 0.0

    async def check_connection(self) -> bool:
        return self.connected

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=m) for m in self.models]

    async def stream_chat(self, model: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.requests.append((model, list(messages)))
        for word in self.reply.split(" "):
            if self.fragment_delay:
                await asyncio.sleep(self.fragment_delay)
            yield word + " "

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


# ---------------------------------------------------------------------------
# Private directories
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def private_bedrock_directories(tmp_path, monkeypatch):
    """Never let a test touch the directories a real employee's data lives in.

    ``Settings`` defaults point at ``~/.local/share/bedrock/...``. A test that
    builds ``Settings(user_id="alice")`` and then runs `/digest` or `/audit`
    would open the actual vault and the actual history database of whoever ran
    ``pytest``. This fixture gives those defaults a temporary home instead.

    Only *defaults* are redirected. ``load_settings`` passes every directory
    explicitly, so configuration precedence is still tested for real, and a
    test that names its own directory still gets that one.
    """
    root = tmp_path / "private-bedrock"
    defaults = {
        "history_dir": str(root / "history"),
        "context_dir": str(root / "context"),
        "workspace_dir": str(root / "workspaces"),
        "biotech_work_dir": root / "biotech",
    }
    original_init = Settings.__init__

    def redirected_init(self, *args, **kwargs):
        for name, value in defaults.items():
            kwargs.setdefault(name, value)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Settings, "__init__", redirected_init)
    monkeypatch.setenv("BEDROCK_HISTORY_DIR", defaults["history_dir"])
    return root


# ---------------------------------------------------------------------------
# History and run-catalogue doubles
# ---------------------------------------------------------------------------


class FakeHistoryStore:
    """In-memory ``HistoryStore`` with the same intent/outcome contract.

    Used where a test is about what the *interface* does with history -
    refusing to act when an intent cannot be written, keeping a confirmed
    result when an outcome cannot be - rather than about SQLite. Set
    ``fail_states`` to make those writes fail on demand.
    """

    def __init__(self, fail_states: set[str] | None = None) -> None:
        self.rows: list[dict] = []
        self.fail_states = fail_states or set()

    def append(
        self,
        operation_id: str,
        *,
        actor_id: str,
        action: str,
        state: str,
        subject_id: str | None = None,
        details: dict | None = None,
    ) -> int:
        if state in self.fail_states:
            raise RuntimeError(f"history is unwritable for {state}")
        slot = "intent" if state == "intent" else "outcome"
        for row in self.rows:
            if row["operation_id"] == operation_id and row["slot"] == slot:
                return row["sequence"]
        row = {
            "schema_version": 1,
            "sequence": len(self.rows) + 1,
            "operation_id": operation_id,
            "slot": slot,
            "recorded_at": f"2026-09-15T12:00:{len(self.rows):02d}+00:00",
            "actor_id": actor_id,
            "action": action,
            "state": state,
            "subject_id": subject_id,
            "details": dict(details or {}),
        }
        self.rows.append(row)
        return row["sequence"]

    def events(self, *, after: int = 0, limit: int = 50) -> list[dict]:
        return [row for row in self.rows if row["sequence"] > after][:limit]

    def export(self, destination) -> int:
        destination.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in self.rows), encoding="utf-8"
        )
        return len(self.rows)

    def latest_project_assignments(self) -> dict[str, str]:
        return {
            row["subject_id"]: row["details"]["project_id"]
            for row in self.rows
            if row["action"] == "run.tag" and row["state"] == "succeeded"
        }

    # Convenience for assertions.
    def states(self, action: str) -> list[str]:
        return [row["state"] for row in self.rows if row["action"] == action]


class FakeRunCatalog:
    """Canned ``RunCatalog`` answers, for tests about how they are drawn.

    Filtering, pagination and hashing are Agent A's and are exercised against
    the real catalogue in ``test_runs_command.py``; this double exists so a
    rendering test does not have to build a believable run on disk first.
    """

    def __init__(
        self,
        *,
        page: dict | None = None,
        detail: dict | None = None,
        verification: list[dict] | None = None,
        reports: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self.page = page or {"items": [], "total": 0, "next_offset": None, "warnings": []}
        self.detail = detail
        self.verification = verification or []
        self.reports = reports or {}
        self.error = error
        self.calls: list[tuple] = []

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        if self.error:
            raise self.error
        return self.page

    def show(self, run_id: str):
        self.calls.append(("show", run_id))
        if self.error:
            raise self.error
        return self.detail or {"run_id": run_id, "manifest": {}, "warnings": []}

    def verify(self, run_id: str):
        self.calls.append(("verify", run_id))
        if self.error:
            raise self.error
        return self.verification

    def tag(self, run_id: str, project_id: str, *, actor_id: str, operation_id: str) -> None:
        self.calls.append(("tag", run_id, project_id, actor_id, operation_id))
        if self.error:
            raise self.error

    def confidence(self, run_id: str):
        self.calls.append(("confidence", run_id))
        if run_id not in self.reports:
            raise RunError(f"No confidence for {run_id}")
        return self.reports[run_id]


def make_services(settings=None, *, history=None, runs=None) -> ComplianceServices:
    """A services object wired to doubles, never to the real provenance modules."""
    return ComplianceServices(
        settings or Settings(user_id="alice"),
        history=history if history is not None else FakeHistoryStore(),
        runs=runs if runs is not None else FakeRunCatalog(),
    )


def real_services(tmp_path, *, user_id: str = "alice") -> ComplianceServices:
    """Agent A's real ``HistoryStore`` and ``RunCatalog``, in a temporary home."""
    settings = Settings(
        user_id=user_id,
        history_dir=str(tmp_path / "history"),
        biotech_work_dir=tmp_path / "biotech",
    )
    history = HistoryStore(Path(settings.history_dir))
    return ComplianceServices(settings, history=history, runs=RunCatalog(settings.biotech_work_dir, history))


def write_run_manifest(work_dir: Path, run_id: str, **overrides) -> Path:
    """Write a manifest of the shape the workflow really produces.

    Only what a run page and a detail page read; the scientific artifacts
    themselves are written as tiny files so hashes are real.
    """
    run_dir = Path(work_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source = run_dir / "source.fasta"
    source.write_text(">candidate\nACDEFGHIK\n", encoding="utf-8")
    structure = run_dir / "ranked_0.pdb"
    structure.write_text("ATOM\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "completed",
        "created_at": f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}T09:00:00+00:00",
        "source": {
            "original_filename": "candidate.fasta",
            "snapshot": "source.fasta",
            "sha256": _sha256(source),
            "description": "candidate",
            "residue_count": 9,
        },
        "alphafold2": {
            "model_preset": "monomer_ptm",
            "db_preset": "full_dbs",
            "max_template_date": "2026-01-01",
            "selected_model": "model_3",
        },
        "outputs": {"structure": {"path": "ranked_0.pdb", "sha256": _sha256(structure), "bytes": 5}},
        "pymol": {"executable": "pymol", "confidence_metric": "plddt"},
    }
    manifest.update(overrides)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_confidence_report(
    run_id: str = "run_1",
    *,
    model_id: str | None = "model_3",
    plddt: tuple[float, ...] | None = (95.0, 92.0, 80.0, 65.0, 45.0, 40.0, 99.0),
    low_regions: tuple[tuple[int, int], ...] = ((4, 6),),
    very_low_regions: tuple[tuple[int, int], ...] = ((5, 6),),
    pae: PaeOverview | None = None,
    notices: tuple[str, ...] = (),
) -> ConfidenceReport:
    return ConfidenceReport(
        run_id=run_id,
        model_id=model_id,
        residue_count=len(plddt) if plddt else 0,
        plddt=plddt,
        low_regions=low_regions,
        very_low_regions=very_low_regions,
        pae=pae,
        notices=notices,
    )


def make_pae(values, ranges, maximum_angstrom: float = 31.75, source_path=None) -> PaeOverview:
    return PaeOverview(
        values=tuple(tuple(float(cell) for cell in row) for row in values),
        ranges=tuple(ranges),
        maximum_angstrom=maximum_angstrom,
        source_path=Path(source_path or "/tmp/pae_model_3.json"),
    )


class FakeAgentPackClientUnavailable:
    """An agent-pack backend that is simply not there.

    Everything that needs the AI system fails; everything local must not.
    """

    async def check_connection(self) -> bool:
        return False

    async def create_session(self, user_id: str) -> str:
        raise InferenceError("Could not start a session: connection refused.")

    async def send_message(self, session_id: str, text: str) -> str:
        raise InferenceError("Could not send your message: connection refused.")

    async def get_result(self, result_ref: str) -> str:
        raise InferenceError("Could not fetch the answer: connection refused.")

    async def approve(self, task_id: str, decision: str, approved_by: str, comment=None) -> None:
        raise InferenceError("Could not record your decision: connection refused.")

    async def events(self, session_id: str):
        raise InferenceError("Lost the live connection: connection refused.")
        yield  # pragma: no cover - makes this an async generator

    async def close(self) -> None:
        return None
