"""The contract every company data-source connector implements.

Until now data only reached Bedrock as a local file path: somebody exports a
FASTA from the registry, saves it somewhere, and types the path. This package
replaces that manual export with a reference to the record itself::

    /fold registry://SEQ-00123
    /digest lims://plates/P42-results.csv

``registry`` and ``lims`` are names the company picks in its config file, and
``SEQ-00123`` is whatever identifier that system already uses. The part before
``://`` chooses a connector; the part after is handed to it unchanged.

This is the same seam idea as ``InferenceBackend`` and ``SourceAdapter``: the
commands only ever talk to a ``SourceConnector``, so hooking up the company's
actual system (a LIMS, an ELN, an in-house registry) is one class plus one
config table. Nothing in ``ui/``, the digest pipeline or the AlphaFold
workflow changes.

Two decisions shape everything here:

*Authorisation stays with the system of record.* Connectors pass the
employee's identity along and turn the source's own refusal (HTTP 401/403, or
a filesystem permission error) into ``AccessDenied``. They never keep a second
copy of the company's permission rules, because a copy drifts out of date and
then quietly grants access the real system would have refused.

*Everything fetched carries its provenance* - which system, which record,
which version, who fetched it, when, and the SHA-256 of the exact bytes. That
record is saved next to the data and follows it into the AlphaFold run
manifest, so "where did this sequence come from?" still has an answer months
later.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Protocol

MAX_RECORD_BYTES = 64 * 1024 * 1024
MAX_RECORD_ID_LENGTH = 512

# Prefixes that already mean something. "https://..." is a web address someone
# pasted, not a company source, so it must never be mistaken for one - and a
# connector may not be named after one either.
RESERVED_NAMES = frozenset(
    {"http", "https", "ftp", "ftps", "sftp", "file", "ssh", "git", "s3", "mailto", "data"}
)

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_REF = re.compile(r"^([A-Za-z][A-Za-z0-9_-]{0,31})://(\S.*)$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


class ConnectorError(RuntimeError):
    """Any failure getting data out of a company source.

    Messages are written for the employee and shown as they are, so they say
    what went wrong and who can fix it rather than naming exception types.
    """


class RecordNotFound(ConnectorError):
    """The source answered, and it has no such record."""


class AccessDenied(ConnectorError):
    """The source refused this employee, or the reference escaped its bounds."""


class ConnectorConfigError(ConnectorError):
    """A ``[connectors.<name>]`` table in the config file is unusable."""


@dataclass(frozen=True)
class SourceRef:
    """``registry://SEQ-00123`` split into its connector and record id."""

    connector: str
    record_id: str

    def __str__(self) -> str:
        return f"{self.connector}://{self.record_id}"


def parse_source_ref(text: str) -> SourceRef | None:
    """Return a ``SourceRef`` if ``text`` is ``name://id``, otherwise ``None``.

    Only the shape is checked here. Whether the connector exists and whether
    the id is acceptable are answered later, with a message that says which -
    so a typo produces "no source called `regsitry`" instead of silently being
    treated as a file path that does not exist.
    """
    match = _REF.match(text.strip())
    if not match:
        return None
    name = match.group(1).lower()
    if name in RESERVED_NAMES:
        return None
    return SourceRef(name, match.group(2))


def valid_connector_name(name: str) -> bool:
    return bool(_NAME.match(name)) and name not in RESERVED_NAMES


def check_record_id(record_id: str) -> str:
    """Reject identifiers no real system uses and every path trick does."""
    if not record_id or len(record_id) > MAX_RECORD_ID_LENGTH:
        raise ConnectorError(
            f"Record identifiers must be 1 to {MAX_RECORD_ID_LENGTH} characters long."
        )
    if _CONTROL.search(record_id):
        raise ConnectorError("Record identifiers cannot contain control characters.")
    if record_id in {".", ".."}:
        raise ConnectorError(f"`{record_id}` is not a record identifier.")
    return record_id


def safe_filename(name: str, fallback: str = "record") -> str:
    """A name that is safe to create on disk, whatever the source sent.

    The source's own filename is untrusted input: it could be
    ``../../.bashrc``. Only the last path component is kept, and anything
    outside a conservative character set becomes ``_``.
    """
    base = PurePosixPath(name.replace("\\", "/")).name
    cleaned = _UNSAFE_FILENAME.sub("_", base).lstrip(".")[:120]
    return cleaned or fallback


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RecordSummary:
    """One search result: enough to show a person and to fetch it next."""

    ref: SourceRef
    title: str


@dataclass(frozen=True)
class FetchedRecord:
    """The bytes of one record plus the facts needed to trust them later."""

    ref: SourceRef
    filename: str
    content: bytes
    connector_kind: str
    location: str
    """Where the bytes came from (a URL or a path). Never includes credentials."""
    fetched_by: str
    version: str | None = None
    fetched_at: str = field(default_factory=_now)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.filename).suffix.lower()

    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def provenance(self) -> dict:
        """The record that travels with the data into manifests and sidecars."""
        return {
            "ref": str(self.ref),
            "connector": self.ref.connector,
            "connector_type": self.connector_kind,
            "record_id": self.ref.record_id,
            "version": self.version,
            "location": self.location,
            "filename": self.filename,
            "bytes": len(self.content),
            "sha256": self.sha256,
            "fetched_by": self.fetched_by,
            "fetched_at": self.fetched_at,
        }


class SourceConnector(Protocol):
    """Everything the commands need from one company system."""

    name: str
    kind: str

    def describe(self) -> str:
        """Where this connector points, for ``/sources``. No secrets."""
        ...

    def readiness_issues(self) -> tuple[str, ...]:
        """Problems visible without contacting the source. Must not raise."""
        ...

    async def fetch(self, record_id: str, *, user: str) -> FetchedRecord:
        """Retrieve one record as ``user``. Raises ``ConnectorError`` subclasses."""
        ...

    async def search(self, query: str, *, user: str, limit: int = 10) -> list[RecordSummary]:
        """Find records as ``user``, so a reference does not have to be memorised."""
        ...
