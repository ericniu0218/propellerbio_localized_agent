"""Shared test helpers.

``FakeBackend`` implements the InferenceBackend protocol in memory, so the UI
can be tested without any server. This is exactly the swap-ability the
architecture is designed for: the app cannot tell the difference.

``isolated_history`` is autouse and matters just as much: without it a test that
builds a real ``Settings`` would save conversations into the developer's own
home directory, and a later test would cheerfully resume them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import AsyncIterator, Callable

import pytest

from bedrock.biotech import ProteinWorkflowResult, ToolStatus
from bedrock.biotech.confidence import ConfidenceReport, PaeOverview
from bedrock.config import PROFILES, Settings
from bedrock.inference import InferenceError, ModelInfo, RequestOptions, Usage, tokens
from bedrock.provenance import HistoryStore, RunCatalog, RunError
from bedrock.storage import ConversationStore
from bedrock.titles import is_title_request, is_topic_check
from bedrock.ui.services import ComplianceServices


class FakeBackend:
    """In-memory stand-in for a model server.

    Title requests and YES/NO topic checks (see ``bedrock/titles.py``) are
    recorded in ``title_requests`` and ``topic_checks`` rather than ``requests``,
    so assertions about chat turns stay exact. By default every conversation has
    a topic (``topic_reply="YES"``) and, like a real model, gets its own title -
    "Re <first question>" - unless a fixed ``title_reply`` is supplied.

    The options each call carried are recorded the same way:
    ``request_options[i]`` belongs to ``requests[i]``, and title and topic
    requests' options go in ``background_options``. Set ``usage`` to have every
    chat turn report those token counts, as a real server does after a reply.
    """

    def __init__(
        self,
        models: list[str] | None = None,
        reply: str = "Hello from the fake model.",
        connected: bool = True,
        title_reply: str | None = None,
        topic_reply: str = "YES",
        model_sizes: dict[str, int] | None = None,
    ) -> None:
        self.models = models if models is not None else ["test-model", "other-model"]
        self.model_sizes = model_sizes or {}
        self.reply = reply
        self.connected = connected
        self.title_reply = title_reply
        self.topic_reply = topic_reply
        self.topic_checks: list[tuple[str, list[dict[str, str]]]] = []
        self.fail_titles = False
        self.requests: list[tuple[str, list[dict[str, str]]]] = []
        self.title_requests: list[tuple[str, list[dict[str, str]]]] = []
        self.closed = False
        self.fragment_delay = 0.0
        self.hold_reply: asyncio.Event | None = None
        self.request_options: list[RequestOptions | None] = []
        self.background_options: list[RequestOptions | None] = []
        self.usage: Usage | None = None

    async def check_connection(self) -> bool:
        return self.connected

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=m, size_bytes=self.model_sizes.get(m)) for m in self.models]

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: RequestOptions | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        if is_topic_check(messages):
            self.topic_checks.append((model, list(messages)))
            self.background_options.append(options)
            if self.fail_titles:
                raise InferenceError("Request failed: connection refused.")
            yield self.topic_reply
            return
        if is_title_request(messages):
            self.title_requests.append((model, list(messages)))
            self.background_options.append(options)
            if self.fail_titles:
                raise InferenceError("Request failed: connection refused.")
            yield self.title_reply if self.title_reply is not None else _title_for(messages)
            return
        self.requests.append((model, list(messages)))
        self.request_options.append(options)
        if self.hold_reply is not None:
            await self.hold_reply.wait()
        for word in self.reply.split(" "):
            if self.fragment_delay:
                await asyncio.sleep(self.fragment_delay)
            yield word + " "
        if on_usage is not None and self.usage is not None:
            on_usage(self.usage)

    async def close(self) -> None:
        self.closed = True


def _title_for(messages: list[dict[str, str]]) -> str:
    """The fake model's title: "Re " plus the first question in the request."""
    first_question = re.search(r"^User: (.+)$", messages[1]["content"], re.MULTILINE)
    return f"Re {first_question.group(1)}" if first_question else "Untitled"


class FakeBiotechWorkflow:
    """In-memory stand-in for ``ProteinWorkflow``: records what it was asked
    to fold and open, and returns a finished run without any tool installed.
    Shared by both apps' tests, which must drive `/fold` and `/pymol` alike."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.folded: list[tuple[Path, str | None]] = []
        self.opened: list[Path | None] = []
        self.provenance: list[dict | None] = []

    def status(self):
        return (
            ToolStatus("AlphaFold 2", True, ()),
            ToolStatus("open-source PyMOL", True, ()),
        )

    async def predict_and_render(
        self,
        fasta_path: Path,
        *,
        project_id: str | None = None,
        source_provenance: dict | None = None,
    ) -> ProteinWorkflowResult:
        self.folded.append((fasta_path, project_id))
        self.provenance.append(source_provenance)
        return ProteinWorkflowResult(
            run_id="run_test",
            run_dir=self.root,
            structure_path=self.root / "ranked_0.pdb",
            preview_path=self.root / "preview.png",
            session_path=self.root / "structure.pse",
            manifest_path=self.root / "manifest.json",
        )

    async def open_in_pymol(self, structure_path: Path | None = None) -> Path:
        self.opened.append(structure_path)
        return structure_path or self.root / "ranked_0.pdb"


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


# ---------------------------------------------------------------------------
# Private directories
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_idle_wait(monkeypatch):
    """Let background work (titles) start as soon as the app is idle.

    The real profiles wait ``idle_seconds`` after a chat reply before a title
    may start. App tests would each sit through that wait for nothing, so it
    is zero here; ``test_scheduler.py`` tests the wait itself with explicit
    profiles.
    """
    for name, profile in list(PROFILES.items()):
        monkeypatch.setitem(PROFILES, name, dataclasses.replace(profile, idle_seconds=0.0))


@pytest.fixture(autouse=True)
def fresh_token_estimate():
    """Forget any characters-per-token ratio a previous test taught ``tokens``."""
    tokens.reset()
    yield
    tokens.reset()


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    """Point the default history directory at a throwaway path, for every test."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "bedrock" / "conversations"


@pytest.fixture
def store(tmp_path) -> ConversationStore:
    """A store in a temporary directory, shared by tests that need one."""
    return ConversationStore(tmp_path / "conversations")


@pytest.fixture(autouse=True)
def private_bedrock_directories(tmp_path, monkeypatch):
    """Never let a test touch the directories a real employee's data lives in.

    ``Settings`` defaults point at ``~/.local/share/bedrock/...``. A test that
    builds ``Settings(user_id="alice")`` and then runs `/digest` or `/audit`
    would open the actual vault and the actual action journal of whoever ran
    ``pytest``. This fixture gives those defaults a temporary home instead.
    Saved conversations are covered by ``isolated_history`` above.

    Only *defaults* are redirected. ``load_settings`` passes every directory
    explicitly, so configuration precedence is still tested for real, and a
    test that names its own directory still gets that one.
    """
    root = tmp_path / "private-bedrock"
    defaults = {
        "action_history_dir": str(root / "history"),
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
    monkeypatch.setenv("BEDROCK_ACTION_HISTORY_DIR", defaults["action_history_dir"])
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
        action_history_dir=str(tmp_path / "history"),
        biotech_work_dir=tmp_path / "biotech",
    )
    history = HistoryStore(Path(settings.action_history_dir))
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
