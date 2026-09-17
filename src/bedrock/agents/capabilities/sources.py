"""Company systems of record, through the connectors that already exist.

``/digest``, ``/fold`` and ``/sources`` already accept ``registry://SEQ-00123``
and turn it into bytes plus a provenance sidecar. This file gives an agent the
same thing and adds nothing of its own::

    [connectors.registry]
    type = "rest"
    base_url = "https://registry.internal.example.com/api/v1"

There is no ``[agents.capabilities.sources]`` table on purpose. A second place
to configure the registry is a second place for it to be configured *wrongly*,
and an agent that reached a different endpoint from the one ``/fold`` uses
would be a provenance record that quietly means something else.

Both capabilities are ``READ_ONLY``. Authorisation is not re-implemented here
either: the system of record decides what this employee may see, and its
refusal arrives as ``AccessDenied`` with a message written for a person. Until
sign-in reaches the terminal client, ``BEDROCK_USER`` identifies but does not
authenticate - which is a reason to point connectors at an API that checks the
token, not one that trusts the header.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from bedrock.agents.capability_support import (
    bounded,
    done,
    failed,
)
from bedrock.agents.contract import Outcome, Risk
from bedrock.agents.protocol import Capability
from bedrock.connectors.base import ConnectorError, parse_source_ref
from bedrock.connectors.hub import SourceHub

MAX_EXCERPT_CHARS = 20_000


class _SourcesCapability:
    """Shared access to one ``SourceHub``.

    The hub is built the first time it is needed, not at import, so listing
    capabilities never reads the config file or contacts a company system.
    Tests pass a hub in.
    """

    def __init__(
        self,
        hub: SourceHub | None = None,
        *,
        hub_factory: Callable[[], SourceHub] | None = None,
    ) -> None:
        self._hub = hub
        self._hub_factory = hub_factory or _hub_from_settings

    @property
    def hub(self) -> SourceHub:
        if self._hub is None:
            self._hub = self._hub_factory()
        return self._hub


def _hub_from_settings() -> SourceHub:
    from bedrock.config.settings import load_settings

    return SourceHub.from_settings(load_settings())


# ---- sources.fetch -----------------------------------------------------------


class SourcesFetch(_SourcesCapability):
    """Pull one record out of a company system and stage it on disk."""

    name = "sources.fetch"
    risk = Risk.READ_ONLY

    def describe(self, arguments: dict[str, str]) -> str:
        ref = str(arguments.get("ref", "")).strip()
        where = ref or "(no reference given)"
        return f"Fetch {where} from the company system that serves it."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        ref = str(arguments.get("ref", "")).strip()
        if not ref:
            return failed(
                "sources.fetch needs a `ref` such as `registry://SEQ-00123`.",
            )
        if parse_source_ref(ref) is None:
            return failed(
                f"`{ref}` is not a source reference like `registry://SEQ-00123`. "
                "A web address or a file path is a different capability.",
            )

        try:
            record, path = await self.hub.fetch_to_file(ref)
        except ConnectorError as exc:
            # These messages are already written for the employee - who refused,
            # which system, what to do - so they are passed through unchanged.
            return failed(str(exc))

        sidecar = path.with_name(path.name + ".provenance.json")
        provenance = record.provenance()
        lines = [
            f"{record.ref} from `{record.ref.connector}` ({record.connector_kind}): "
            f"{provenance['bytes']} bytes, sha256 {provenance['sha256']}",
            f"Version {provenance['version'] or 'not reported'}, fetched by "
            f"{provenance['fetched_by']} at {provenance['fetched_at']} from "
            f"{provenance['location']}",
            f"Saved to {path}",
        ]
        excerpt = _as_text(record.content)
        if excerpt:
            lines.append(excerpt)
        artifacts: list[Path] = [path]
        if sidecar.exists():
            artifacts.append(sidecar)
        return done(lines, artifacts=artifacts)


def _as_text(content: bytes) -> str:
    """A readable excerpt, or nothing at all if the record is not text.

    A NUL byte near the start is the cheap, reliable signal that these bytes
    are a structure file or an archive. Decoding one produces pages of
    replacement characters that help nobody and cost a model its context.
    """
    if b"\x00" in content[:8192]:
        return ""
    return bounded(content.decode("utf-8", errors="replace"), MAX_EXCERPT_CHARS)


# ---- sources.search ----------------------------------------------------------


class SourcesSearch(_SourcesCapability):
    """Find records in one company system, so a reference need not be memorised."""

    name = "sources.search"
    risk = Risk.READ_ONLY

    def describe(self, arguments: dict[str, str]) -> str:
        source = str(arguments.get("source", "")).strip() or "(no source given)"
        query = str(arguments.get("query", "")).strip()
        return f"Search `{source}` for {query!r}."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        source = str(arguments.get("source", "")).strip()
        query = str(arguments.get("query", "")).strip()
        if not source:
            return failed(
                "sources.search needs a `source`, the name before `://` in a reference.",
            )
        if not query:
            return failed("sources.search needs a `query` to look for.")
        try:
            limit = max(1, min(int(arguments.get("limit", 10)), 50))
        except (TypeError, ValueError):
            limit = 10

        try:
            results = await self.hub.search(source, query, limit=limit)
        except ConnectorError as exc:
            return failed(str(exc))

        if not results:
            return done((f"`{source}` found nothing for {query!r}.",))
        return done(
            (
                f"{len(results)} record(s) in `{source}` for {query!r}:",
                *(f"{summary.ref} - {summary.title}" for summary in results),
            ),
        )


CAPABILITY: tuple[Capability, ...] = (SourcesFetch(), SourcesSearch())
"""What the roster finds in this file.

A tuple, which ``roster.discover_capability`` unpacks by name. One file
per adapter, several verbs per file where they share one set of rules.
"""

CAPABILITIES = CAPABILITY
"""Plural alias, for reading a module's contents rather than the roster's."""
