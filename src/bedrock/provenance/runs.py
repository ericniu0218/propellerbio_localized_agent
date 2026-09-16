"""Read-only run browsing plus append-only project annotations."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

from bedrock.provenance.history import HistoryError, HistoryStore


MAX_MANIFEST_BYTES = 1024 * 1024
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class RunError(RuntimeError):
    """A stored scientific run could not be queried or validated."""


class RunCatalog:
    """Browse local AlphaFold run manifests without mutating their history."""

    def __init__(self, work_dir: Path | str, history: HistoryStore) -> None:
        self.work_dir = Path(work_dir).expanduser()
        self.runs_dir = self.work_dir / "runs"
        self.history = history

    def list(
        self,
        *,
        project_id: str | None = None,
        since: str | None = None,
        before: str | None = None,
        status: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict:
        if project_id is not None:
            validate_project_id(project_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise RunError("Run page size must be between 1 and 100.")
        if type(offset) is not int or offset < 0:
            raise RunError("Run offset must be a non-negative number.")
        start = _date_bound(since, "since")
        end = _date_bound(before, "before")
        if start is not None and end is not None and start >= end:
            raise RunError("The before date must be later than the since date.")

        try:
            assignments = self.history.latest_project_assignments()
        except HistoryError as exc:
            raise RunError("Could not read run project annotations.") from exc

        items: list[dict] = []
        warnings: list[str] = []
        paths = sorted(self.runs_dir.glob("*/manifest.json")) if self.runs_dir.is_dir() else []
        for path in paths:
            try:
                if path.parent.is_symlink():
                    raise RunError("run directory is a symlink")
                path.parent.resolve().relative_to(self.runs_dir.resolve())
                manifest = _read_manifest(path)
                run_id = _manifest_run_id(manifest, path.parent.name)
                created_at = _created_at(manifest)
                run_project = assignments.get(run_id, _optional_project(manifest.get("project_id")))
                run_status = manifest.get("status")
                if not isinstance(run_status, str) or not run_status:
                    raise RunError("manifest has no status")
                residue_count = manifest.get("source", {}).get("residue_count")
                if type(residue_count) is not int or residue_count < 1:
                    residue_count = None
            except (RunError, OSError, ValueError) as exc:
                warnings.append(f"Skipped run `{path.parent.name}`: {exc}")
                continue

            if project_id is not None and run_project != project_id:
                continue
            if status is not None and run_status != status:
                continue
            if start is not None and created_at < start:
                continue
            if end is not None and created_at >= end:
                continue
            items.append(
                {
                    "run_id": run_id,
                    "created_at": created_at.isoformat(),
                    "status": run_status,
                    "project_id": run_project,
                    "residue_count": residue_count,
                }
            )

        items.sort(key=lambda item: (item["created_at"], item["run_id"]), reverse=True)
        total = len(items)
        page = items[offset : offset + limit]
        next_offset = offset + len(page) if offset + len(page) < total else None
        return {
            "items": page,
            "total": total,
            "next_offset": next_offset,
            "warnings": warnings,
        }

    def show(self, run_id: str) -> dict:
        path = self._manifest_path(run_id)
        manifest = _read_manifest(path)
        actual_id = _manifest_run_id(manifest, path.parent.name)
        if actual_id != run_id:
            raise RunError("Run manifest ID does not match its directory.")
        try:
            project_id = self.history.latest_project_assignments().get(
                run_id, _optional_project(manifest.get("project_id"))
            )
        except HistoryError as exc:
            raise RunError("Could not read run project annotations.") from exc
        warnings: list[str] = []
        if manifest.get("schema_version") == 1:
            warnings.append("Legacy v1 manifest; some artifacts may not have recorded hashes.")
        return {
            "run_id": run_id,
            "manifest_path": path.resolve(),
            "project_id": project_id,
            "manifest": manifest,
            "warnings": warnings,
        }

    def verify(self, run_id: str) -> list[dict]:
        shown = self.show(run_id)
        manifest = shown["manifest"]
        run_dir = shown["manifest_path"].parent
        entries: list[tuple[str, str | None]] = []

        source = manifest.get("source")
        if isinstance(source, dict) and isinstance(source.get("snapshot"), str):
            expected = source.get("sha256") if isinstance(source.get("sha256"), str) else None
            entries.append((source["snapshot"], expected))
        for item in _artifact_entries(manifest.get("outputs")):
            entries.append((item["path"], item.get("sha256")))

        recorded = {path for path, _ in entries}
        if "input.fasta" not in recorded and (run_dir / "input.fasta").exists():
            entries.append(("input.fasta", None))
            recorded.add("input.fasta")
        for ranking_path in sorted(run_dir.rglob("ranking_debug.json")):
            try:
                relative = str(ranking_path.resolve().relative_to(run_dir.resolve()))
            except (OSError, ValueError):
                continue
            if relative not in recorded:
                entries.append((relative, None))
                recorded.add(relative)

        results: list[dict] = []
        seen: set[str] = set()
        for relative, expected in entries:
            if relative in seen:
                continue
            seen.add(relative)
            results.append(_verify_artifact(run_dir, relative, expected))
        return results

    def tag(
        self,
        run_id: str,
        project_id: str,
        *,
        actor_id: str,
        operation_id: str,
    ) -> None:
        self.show(run_id)
        validate_project_id(project_id)
        details = {"run_id": run_id, "project_id": project_id, "actor_source": "config"}
        try:
            self.history.append(
                operation_id,
                actor_id=actor_id,
                action="run.tag",
                state="intent",
                subject_id=run_id,
                details=details,
            )
            self.history.append(
                operation_id,
                actor_id=actor_id,
                action="run.tag",
                state="succeeded",
                subject_id=run_id,
                details=details,
            )
        except HistoryError as exc:
            raise RunError("Could not record the run project annotation.") from exc

    def confidence(self, run_id: str):
        from bedrock.biotech.confidence import load_confidence_report

        shown = self.show(run_id)
        configured_limit = shown["manifest"].get("limits", {}).get(
            "max_sequence_length", 2500
        )
        if type(configured_limit) is not int or configured_limit < 1:
            configured_limit = 2500
        return load_confidence_report(
            run_id,
            shown["manifest_path"].parent,
            shown["manifest"],
            max_residue_count=configured_limit,
        )

    def _manifest_path(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise RunError("Run ID is invalid.")
        path = self.runs_dir / run_id / "manifest.json"
        try:
            resolved_parent = path.parent.resolve()
            resolved_parent.relative_to(self.runs_dir.resolve())
        except (OSError, ValueError) as exc:
            raise RunError("Run path escapes the configured run directory.") from exc
        if not path.is_file() or path.is_symlink():
            raise RunError(f"Run not found: {run_id}")
        return path


def validate_project_id(project_id: str) -> str:
    if not isinstance(project_id, str) or not PROJECT_ID_PATTERN.fullmatch(project_id):
        raise RunError(
            "Project ID must be 1–64 letters, numbers, dots, underscores, or hyphens."
        )
    return project_id


def _read_manifest(path: Path) -> dict:
    try:
        if path.is_symlink() or not path.is_file():
            raise RunError("manifest is not a regular file")
        size = path.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise RunError("manifest exceeds the 1 MiB limit")
        data = json.loads(path.read_text(encoding="utf-8"))
    except RunError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunError("manifest is unreadable or invalid JSON") from exc
    if not isinstance(data, dict):
        raise RunError("manifest root must be an object")
    if data.get("schema_version") not in (1, 2):
        raise RunError("manifest schema version is unsupported")
    return data


def _manifest_run_id(manifest: dict, directory_name: str) -> str:
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise RunError("manifest has an invalid run ID")
    if run_id != directory_name:
        raise RunError("manifest run ID does not match its directory")
    return run_id


def _created_at(manifest: dict) -> datetime:
    raw = manifest.get("created_at")
    if not isinstance(raw, str):
        raise RunError("manifest has no UTC creation time")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunError("manifest creation time is invalid") from exc
    if parsed.tzinfo is None:
        raise RunError("manifest creation time has no timezone")
    return parsed.astimezone(timezone.utc)


def _date_bound(value: str | None, label: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise RunError(f"{label} must use YYYY-MM-DD.") from exc
    return datetime.combine(parsed, time.min, timezone.utc)


def _optional_project(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return validate_project_id(value)
    except RunError:
        return None


def _artifact_entries(value: Any) -> Iterable[dict]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            yield value
        else:
            for nested in value.values():
                yield from _artifact_entries(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _artifact_entries(nested)


def _verify_artifact(run_dir: Path, relative: str, expected: str | None) -> dict:
    row = {
        "path": relative,
        "state": "invalid",
        "expected_sha256": expected,
        "actual_sha256": None,
    }
    try:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            return row
        path = run_dir / candidate
        if path.is_symlink() or not path.exists():
            row["state"] = "invalid" if path.is_symlink() else "missing"
            return row
        resolved = path.resolve()
        resolved.relative_to(run_dir.resolve())
        if not resolved.is_file():
            return row
        with resolved.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        row["actual_sha256"] = actual
        if expected is None:
            row["state"] = "unrecorded"
        elif not re.fullmatch(r"[a-f0-9]{64}", expected):
            row["state"] = "invalid"
        else:
            row["state"] = "match" if actual == expected else "mismatch"
        return row
    except (OSError, ValueError):
        return row
