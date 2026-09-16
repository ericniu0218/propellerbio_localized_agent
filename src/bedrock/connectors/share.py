"""Connector for a controlled folder the source system exports into.

Plenty of LIMS and instrument systems have no usable API but do write exports
to a network share on a schedule. This connector treats that share as the
source::

    [connectors.lims]
    type = "share"
    root = "/mnt/lims-exports"
    extensions = [".csv", ".fasta"]     # optional allow-list

Record ids are paths relative to ``root``: ``lims://plates/P42.csv``.

Authorisation here is the share's own file permissions, checked by the
operating system for whoever is running Bedrock. That is real enforcement, not
a stand-in - but it means access is only as fine-grained as IT made the share.

Every id is resolved and checked to still lie inside ``root`` *after* symlinks
are followed, so neither ``../`` nor a link pointing elsewhere can reach files
outside the share.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from bedrock.connectors.base import (
    MAX_RECORD_BYTES,
    AccessDenied,
    ConnectorConfigError,
    ConnectorError,
    FetchedRecord,
    RecordNotFound,
    RecordSummary,
    SourceRef,
    check_record_id,
    safe_filename,
)


class ShareConnector:
    kind = "share"

    def __init__(
        self,
        name: str,
        *,
        root: str,
        extensions: list[str] | str | None = None,
        max_search_files: int = 20_000,
    ) -> None:
        if not root:
            raise ConnectorConfigError(f"[connectors.{name}] needs root.")
        self.name = name
        self.root = Path(root).expanduser()
        # A single extension written as a plain string is the natural thing to
        # put in a config file. Iterating it directly would turn "fasta" into
        # ('.f', '.a', '.s', '.t', '.a'), which reads as configured but refuses
        # every file - so accept it as the one-item list it was meant to be.
        if isinstance(extensions, str):
            extensions = [extensions]
        self.extensions = tuple(
            (ext if ext.startswith(".") else f".{ext}").lower() for ext in (extensions or ())
        )
        self.max_search_files = int(max_search_files)

    def describe(self) -> str:
        return f"folder {self.root}"

    def readiness_issues(self) -> tuple[str, ...]:
        if not self.root.is_dir():
            return (f"{self.root} is not a reachable folder. Is the share mounted?",)
        if not os.access(self.root, os.R_OK | os.X_OK):
            return (f"No read access to {self.root}.",)
        return ()

    async def fetch(self, record_id: str, *, user: str) -> FetchedRecord:
        ref = SourceRef(self.name, record_id)
        path = self._resolve(record_id)
        if self.extensions and path.suffix.lower() not in self.extensions:
            raise ConnectorError(
                f"`{ref}` is not a file type `{self.name}` serves ({', '.join(self.extensions)})."
            )
        try:
            content, modified = await asyncio.to_thread(_read, path)
        except FileNotFoundError as exc:
            raise RecordNotFound(f"No record `{ref}` in `{self.name}`.") from exc
        except PermissionError as exc:
            raise AccessDenied(
                f"No permission to read `{ref}`. Access to `{self.name}` is set by "
                "the share's file permissions."
            ) from exc
        except OSError as exc:
            raise ConnectorError(f"Could not read `{ref}`: {exc.strerror or exc}") from exc

        return FetchedRecord(
            ref=ref,
            filename=safe_filename(path.name),
            content=content,
            connector_kind=self.kind,
            location=str(path),
            fetched_by=user,
            version=f"modified {modified}",
        )

    async def search(self, query: str, *, user: str, limit: int = 10) -> list[RecordSummary]:
        terms = query.casefold().split()
        if not terms:
            return []
        matches = await asyncio.to_thread(self._walk, terms, limit)
        return [RecordSummary(SourceRef(self.name, relative), relative) for relative in matches]

    def _resolve(self, record_id: str) -> Path:
        check_record_id(record_id)
        root = self.root.resolve()
        candidate = (root / record_id).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise AccessDenied(
                f"`{self.name}://{record_id}` points outside the `{self.name}` share."
            ) from None
        return candidate

    def _walk(self, terms: list[str], limit: int) -> list[str]:
        root = self.root.resolve()
        found: list[str] = []
        scanned = 0
        for directory, subdirectories, files in os.walk(root, followlinks=False):
            subdirectories[:] = sorted(d for d in subdirectories if not d.startswith("."))
            for filename in sorted(files):
                if filename.startswith("."):
                    continue
                scanned += 1
                if scanned > self.max_search_files:
                    return found
                relative = (Path(directory) / filename).relative_to(root).as_posix()
                if self.extensions and Path(filename).suffix.lower() not in self.extensions:
                    continue
                if all(term in relative.casefold() for term in terms):
                    found.append(relative)
                    if len(found) >= limit:
                        return found
        return found


def _read(path: Path) -> tuple[bytes, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    if stat.st_size > MAX_RECORD_BYTES:
        raise ConnectorError(
            f"{path.name} is larger than the {MAX_RECORD_BYTES // 1_000_000} MB limit."
        )
    modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")
    return path.read_bytes(), modified
