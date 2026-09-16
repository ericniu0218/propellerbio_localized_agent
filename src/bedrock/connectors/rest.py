"""Connector for any internal system that serves records over HTTP and JSON.

Most registries, LIMS and ELNs have some HTTP API, and they differ mainly in
URL layout and field names. So instead of one class per product, this
connector is shaped by configuration::

    [connectors.registry]
    type = "rest"
    base_url = "https://registry.internal.example.com/api/v1"
    record_path = "/sequences/{id}"         # {id} is replaced, URL-encoded
    content_field = "data.fasta"             # omit to use the whole response body
    filename_field = "data.name"
    version_field = "data.revision"
    suffix = ".fasta"                        # tells /digest what kind of data it is
    search_path = "/sequences"               # optional; enables /sources search
    search_param = "q"
    results_field = "results"
    id_field = "id"
    title_field = "name"
    token_env = "REGISTRY_TOKEN"            # the token itself stays out of the file

Identity. Every request carries the employee's name in ``user_header`` and,
if ``token_env`` is set, a bearer token from that environment variable. The
system of record decides what that person may see, and a 401 or 403 reaches
the employee as a plain "access refused" message. Until real sign-in reaches
this client (see NEXT_STEPS.md) the name comes from ``BEDROCK_USER`` and can be
set by anyone, so it identifies but does not authenticate - point this at an
API that authenticates the token, not one that trusts the header alone.

A fresh HTTP client is opened for each request. Fetches are occasional and
this avoids tying a long-lived connection pool to whichever event loop happened
to create the connector (the terminal app and the CLI each run their own).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote

import httpx

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


@dataclass(frozen=True)
class _Fetched:
    """One HTTP response, already read under the size limit."""

    content: bytes
    url: str
    headers: httpx.Headers


class RestConnector:
    kind = "rest"

    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        record_path: str = "/records/{id}",
        content_field: str | None = None,
        filename_field: str | None = None,
        version_field: str | None = None,
        suffix: str = "",
        search_path: str | None = None,
        search_param: str = "q",
        results_field: str | None = None,
        id_field: str = "id",
        title_field: str = "name",
        token_env: str | None = None,
        user_header: str | None = "X-Bedrock-User",
        timeout: float = 30.0,
        env: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # ``env`` and ``transport`` exist so tests can supply a fake environment
        # and a fake server; the registry refuses them from the config file.
        if not base_url:
            raise ConnectorConfigError(f"[connectors.{name}] needs base_url.")
        if "{id}" not in record_path:
            raise ConnectorConfigError(f"[connectors.{name}] record_path must contain {{id}}.")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.record_path = record_path
        self.content_field = content_field
        self.filename_field = filename_field
        self.version_field = version_field
        self.suffix = suffix if not suffix or suffix.startswith(".") else f".{suffix}"
        self.search_path = search_path
        self.search_param = search_param
        self.results_field = results_field
        self.id_field = id_field
        self.title_field = title_field
        self.token_env = token_env
        self.user_header = user_header
        self.timeout = float(timeout)
        self._env = os.environ if env is None else env
        self._transport = transport

    def describe(self) -> str:
        return f"HTTP API at {self.base_url}"

    def readiness_issues(self) -> tuple[str, ...]:
        if self.token_env and not self._env.get(self.token_env):
            return (f"${self.token_env} is not set, so requests would have no token.",)
        return ()

    async def fetch(self, record_id: str, *, user: str) -> FetchedRecord:
        check_record_id(record_id)
        ref = SourceRef(self.name, record_id)
        # Encoding with safe="" turns "/" into %2F, so an id like "../admin"
        # stays one path segment instead of walking up the API's URL tree.
        path = self.record_path.replace("{id}", quote(record_id, safe=""))
        fetched = await self._get(path, user=user, what=f"`{ref}`")

        body: Any = None
        if self.content_field or self.filename_field or self.version_field:
            body = _json(fetched, self.name)

        if self.content_field:
            value = _field(body, self.content_field)
            if value is None:
                raise ConnectorError(
                    f"`{ref}` came back without a `{self.content_field}` field. "
                    f"Check content_field in [connectors.{self.name}]."
                )
            content = value.encode("utf-8") if isinstance(value, str) else json.dumps(value).encode("utf-8")
        else:
            content = fetched.content

        return FetchedRecord(
            ref=ref,
            filename=self._filename(body, record_id),
            content=content,
            connector_kind=self.kind,
            location=fetched.url,
            fetched_by=user,
            version=self._version(body, fetched),
        )

    async def search(self, query: str, *, user: str, limit: int = 10) -> list[RecordSummary]:
        if not self.search_path:
            raise ConnectorError(f"`{self.name}` is not set up for search (it has no search_path).")
        fetched = await self._get(
            self.search_path,
            user=user,
            what=f"search on `{self.name}`",
            params={self.search_param: query, "limit": str(limit)},
        )
        body = _json(fetched, self.name)
        items = _field(body, self.results_field) if self.results_field else body
        if not isinstance(items, list):
            raise ConnectorError(
                f"`{self.name}` search did not return a list"
                + (f" under `{self.results_field}`." if self.results_field else ".")
            )

        results: list[RecordSummary] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            record_id = item.get(self.id_field)
            if record_id in (None, ""):
                continue
            title = item.get(self.title_field) or record_id
            results.append(RecordSummary(SourceRef(self.name, str(record_id)), str(title)))
            if len(results) >= limit:
                break
        return results

    # ---- Internals ---------------------------------------------------------

    async def _get(
        self,
        path: str,
        *,
        user: str,
        what: str,
        params: dict[str, str] | None = None,
    ) -> _Fetched:
        headers = self._headers(user)
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                transport=self._transport,
            ) as client:
                async with client.stream("GET", path, headers=headers, params=params) as response:
                    self._check_status(response.status_code, what, user)
                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > MAX_RECORD_BYTES:
                        raise ConnectorError(_too_large(what))
                    # Accumulate rather than letting httpx buffer the whole
                    # body: an oversized response has to stop being read, not
                    # be measured once it is already in memory.
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RECORD_BYTES:
                            raise ConnectorError(_too_large(what))
                    return _Fetched(bytes(body), str(response.url), response.headers)
        except httpx.ConnectError as exc:
            raise ConnectorError(
                f"Could not reach `{self.name}` at {self.base_url}. "
                "Check that it is reachable from this network."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectorError(f"`{self.name}` took too long to respond.") from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Request to `{self.name}` failed: {exc}") from exc

    def _check_status(self, status: int, what: str, user: str) -> None:
        if status in (401, 403):
            raise AccessDenied(
                f"`{self.name}` refused {what} for `{user}`. Access is decided by "
                f"{self.name} itself - ask its owner if you should have it."
            )
        if status == 404:
            raise RecordNotFound(f"`{self.name}` found nothing for {what}.")
        if status >= 400:
            raise ConnectorError(f"`{self.name}` returned HTTP {status} for {what}.")

    def _headers(self, user: str) -> dict[str, str]:
        headers = {"Accept": "application/json, text/plain;q=0.9, */*;q=0.5"}
        if self.user_header and user:
            headers[self.user_header] = user
        if self.token_env:
            token = self._env.get(self.token_env)
            if not token:
                raise ConnectorError(
                    f"`{self.name}` needs a token in ${self.token_env}, which is not set."
                )
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _filename(self, body: Any, record_id: str) -> str:
        chosen = _field(body, self.filename_field) if self.filename_field else None
        name = safe_filename(str(chosen) if chosen else record_id, fallback="record")
        # The suffix is how /digest picks an adapter, so a record called
        # "SEQ-00123" from a FASTA registry still gets read as FASTA-shaped text.
        if self.suffix and not name.lower().endswith(self.suffix.lower()):
            name += self.suffix
        return name

    def _version(self, body: Any, fetched: _Fetched) -> str | None:
        if self.version_field:
            value = _field(body, self.version_field)
            if value not in (None, ""):
                return str(value)
        return fetched.headers.get("etag") or fetched.headers.get("last-modified")


def _too_large(what: str) -> str:
    return f"{what} is larger than the {MAX_RECORD_BYTES // 1_000_000} MB limit."


def _json(fetched: _Fetched, name: str) -> Any:
    try:
        return json.loads(fetched.content)
    except ValueError as exc:
        raise ConnectorError(f"`{name}` did not return JSON where JSON was expected.") from exc


def _field(data: Any, dotted: str | None) -> Any:
    """Read ``data.fasta`` style paths; ``None`` when any step is missing."""
    if not dotted:
        return None
    current = data
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current
