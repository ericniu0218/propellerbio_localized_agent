"""Builds connectors from configuration and runs the fetches commands need.

The commands (``/digest``, ``/fold``, ``/sources``) and the ``bedrock-biotech``
CLI all go through one ``SourceHub``, so identity, error wording and the
on-disk provenance format cannot drift between them.

A broken ``[connectors.x]`` table never stops the app from starting. Its error
is kept and shown by ``/sources``, and every other connector keeps working -
one typo in one team's config should not take everyone's sources away.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from uuid import uuid4
from typing import Any, Mapping, Sequence

from bedrock.connectors.base import (
    ConnectorConfigError,
    ConnectorError,
    FetchedRecord,
    RecordSummary,
    SourceConnector,
    SourceRef,
    parse_source_ref,
    valid_connector_name,
)
from bedrock.connectors.rest import RestConnector
from bedrock.connectors.share import ShareConnector

CONNECTOR_TYPES = {"rest": RestConnector, "share": ShareConnector}

# Constructor arguments that exist for tests only and must not be settable
# from a config file.
_CODE_ONLY_OPTIONS = frozenset({"env", "transport"})


def build_connectors(config: Any) -> tuple[dict[str, SourceConnector], list[str]]:
    """Turn the ``[connectors]`` config table into connectors plus any errors."""
    if not config:
        return {}, []
    if not isinstance(config, Mapping):
        return {}, ["`connectors` in the config file must be a table of [connectors.<name>] sections."]

    connectors: dict[str, SourceConnector] = {}
    errors: list[str] = []
    for raw_name, table in config.items():
        name = str(raw_name)
        section = f"[connectors.{name}]"
        if not valid_connector_name(name):
            errors.append(
                f"{section}: names must be lowercase letters, digits, - or _, start with a "
                "letter, and not be a URL scheme such as https."
            )
            continue
        if not isinstance(table, Mapping):
            errors.append(f"{section} must be a table.")
            continue
        kind = table.get("type")
        # ``kind`` is whatever the file held, which may be unhashable (a TOML
        # list), so even the lookup has to be guarded. A bad ``type =`` line is
        # a config mistake to report, not an exception to raise at the caller.
        try:
            factory = CONNECTOR_TYPES.get(kind)
        except TypeError:
            factory = None
        if factory is None:
            errors.append(
                f"{section}: type must be one of {', '.join(sorted(CONNECTOR_TYPES))} (got {kind!r})."
            )
            continue
        options = {key: value for key, value in table.items() if key != "type"}
        blocked = sorted(_CODE_ONLY_OPTIONS & options.keys())
        if blocked:
            errors.append(f"{section}: unknown option `{blocked[0]}`.")
            continue
        try:
            connectors[name] = factory(name, **options)
        except ConnectorConfigError as exc:
            errors.append(str(exc))
        except TypeError as exc:
            errors.append(f"{section}: {_explain_type_error(exc)}")
        except Exception as exc:
            # Wrongly-typed values land here: timeout = "30s" raises ValueError,
            # base_url = 123 raises AttributeError, and so on. This module
            # promises that one broken table never stops the app, and that
            # promise is only true if nothing at all escapes.
            errors.append(f"{section}: one of its settings is not usable - {exc}")
    return connectors, errors


class SourceHub:
    def __init__(
        self,
        connectors: Mapping[str, SourceConnector],
        *,
        user: str,
        staging_dir: Path,
        config_errors: Sequence[str] = (),
    ) -> None:
        self._connectors = dict(connectors)
        self.user = user
        self.staging_dir = Path(staging_dir).expanduser()
        self.config_errors = tuple(config_errors)

    @classmethod
    def from_settings(cls, settings) -> "SourceHub":
        connectors, errors = build_connectors(settings.connectors)
        return cls(
            connectors,
            user=settings.user_id,
            # Next to the AlphaFold runs, under the same private (0700) folder.
            staging_dir=Path(settings.biotech_work_dir) / "sources",
            config_errors=errors,
        )

    @property
    def connectors(self) -> list[SourceConnector]:
        return [self._connectors[name] for name in sorted(self._connectors)]

    def connector(self, name: str) -> SourceConnector:
        found = self._connectors.get(name)
        if found is not None:
            return found
        if not self._connectors:
            raise ConnectorError(
                "No company sources are configured yet. Add a [connectors.<name>] table to "
                "~/.config/bedrock/config.toml - see the README's \"Company data sources\" section."
            )
        configured = ", ".join(f"`{n}`" for n in sorted(self._connectors))
        raise ConnectorError(f"No source called `{name}` is configured. Configured: {configured}.")

    async def fetch(self, ref: SourceRef | str) -> FetchedRecord:
        parsed = parse_source_ref(ref) if isinstance(ref, str) else ref
        if parsed is None:
            raise ConnectorError(f"`{ref}` is not a source reference like `registry://SEQ-00123`.")
        return await self.connector(parsed.connector).fetch(parsed.record_id, user=self.user)

    async def search(self, name: str, query: str, *, limit: int = 10) -> list[RecordSummary]:
        return await self.connector(name).search(query, user=self.user, limit=limit)

    async def fetch_to_file(self, ref: SourceRef | str) -> tuple[FetchedRecord, Path]:
        """Fetch a record and stage it on disk for tools that need a real file."""
        record = await self.fetch(ref)
        path = await asyncio.to_thread(self.stage, record)
        return record, path

    def stage(self, record: FetchedRecord) -> Path:
        """Write the record and its provenance sidecar into the private staging folder.

        Files are named by content hash, so fetching an unchanged record twice
        reuses one file, and two different versions can never overwrite each
        other. Both files are readable by their owner only (0600): these are
        company sequences, not scratch output.
        """
        try:
            self.staging_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.staging_dir, 0o700)
            path = self.staging_dir / f"{record.sha256[:12]}-{record.filename}"
            _write_private(path, record.content)
            sidecar = path.with_name(path.name + ".provenance.json")
            _write_private(
                sidecar,
                (json.dumps(record.provenance(), indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        except OSError as exc:
            raise ConnectorError(
                f"Could not save `{record.ref}` into {self.staging_dir}: {exc.strerror or exc}"
            ) from exc
        return path


def _write_private(path: Path, data: bytes) -> None:
    # A unique temporary name per write. Two stages of the same record can run
    # at once - /sources fetch and /fold are separate workers - and a shared
    # <name>.tmp would let one replace() the other's half-written file, leaving
    # content that its own provenance sidecar no longer vouches for.
    temporary = path.with_name(f"{path.name}.{uuid4().hex[:12]}.tmp")
    try:
        temporary.write_bytes(data)
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _explain_type_error(exc: TypeError) -> str:
    """Turn Python's constructor complaints into config-file language."""
    message = str(exc)
    unexpected = re.search(r"unexpected keyword argument '(\w+)'", message)
    if unexpected:
        return f"unknown option `{unexpected.group(1)}`."
    missing = re.search(r"required keyword-only argument: '(\w+)'", message)
    if missing:
        return f"missing required option `{missing.group(1)}`."
    return message
