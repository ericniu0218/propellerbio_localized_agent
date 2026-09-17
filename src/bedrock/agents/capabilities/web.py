"""The internet, narrowly: one allowlisted address at a time, through a proxy.

This is the first code in the product that transmits anything off the machine,
so it is deliberately the least convenient adapter here::

    [agents.capabilities.web]
    proxy = "http://egress.internal:3128"    # required; empty disables web.fetch
    allowed_domains = ["fda.gov", "clinicaltrials.gov"]
    search_url = "https://searx.internal"    # a self-hosted SearxNG instance
    search_via_proxy = false                 # it is usually on the inside
    store_dir = "~/.local/share/bedrock/agents/web"
    max_bytes = 2000000
    max_excerpt_chars = 20000
    timeout = 30.0

Both capabilities are ``READ_ONLY`` - they change nothing - but read-only is
not the same as harmless, and three rules apply because of that:

*An unconfigured proxy means no fetching, not direct fetching.* If the egress
proxy is missing, ``web.fetch`` refuses. The alternative - quietly going
straight out - is how a privacy promise turns into a privacy incident, and a
capability that silently downgrades its own protection is worse than one that
does not exist.

*The allowlist is checked after parsing, and redirects are not followed.* A
redirect is reported back as an address to ask for, so the second address gets
the same domain check as the first. Following it would let an allowlisted host
choose where this agent goes next.

*The bytes hit disk before a model sees them.* Every fetch is written to
``store_dir`` and hashed, so what the model was shown can be compared with what
the server actually sent, months later.

``web.search`` goes to a SearxNG instance the company runs. Not a commercial
search API: with search, the query text is the leak, and "what is our
sponsor's withdrawal rate" is a sentence that must not leave the building.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from bedrock.agents.capability_support import (
    CapabilityConfigError,
    as_bool,
    as_float,
    as_int,
    as_list,
    as_path,
    as_text,
    bounded,
    capability_config,
    done,
    failed,
)
from bedrock.agents.contract import Outcome, Risk
from bedrock.agents.protocol import Capability
from bedrock.connectors.base import safe_filename

DEFAULT_STORE_DIR = Path.home() / ".local" / "share" / "bedrock" / "agents" / "web"

_TEXTUAL = ("text/", "application/json", "application/xml", "+json", "+xml", "javascript")


# ---- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class WebConfig:
    proxy: str = ""
    allowed_domains: tuple[str, ...] = ()
    search_url: str = ""
    search_via_proxy: bool = False
    store_dir: Path = DEFAULT_STORE_DIR
    max_bytes: int = 2_000_000
    max_excerpt_chars: int = 20_000
    max_results: int = 10
    timeout: float = 30.0

    @classmethod
    def from_settings(cls, settings: Any = None) -> "WebConfig":
        table = capability_config("web", settings)
        return cls(
            proxy=as_text(table, "proxy"),
            allowed_domains=tuple(
                domain.lower().lstrip(".") for domain in as_list(table, "allowed_domains")
            ),
            search_url=as_text(table, "search_url").rstrip("/"),
            search_via_proxy=as_bool(table, "search_via_proxy", False),
            store_dir=as_path(table, "store_dir") or DEFAULT_STORE_DIR,
            max_bytes=as_int(table, "max_bytes", 2_000_000),
            max_excerpt_chars=as_int(table, "max_excerpt_chars", 20_000),
            max_results=as_int(table, "max_results", 10),
            timeout=as_float(table, "timeout", 30.0),
        )


class _WebCapability:
    """Shared configuration and HTTP client construction.

    ``transport`` exists so tests can answer requests without a server. It is a
    code-only seam: no config file can set it, exactly as the company data-source
    connectors do it.
    """

    def __init__(
        self,
        config: WebConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport

    @property
    def config(self) -> WebConfig:
        if self._config is None:
            self._config = WebConfig.from_settings()
        return self._config

    def client_options(self, *, proxy: str) -> dict[str, Any]:
        """Everything the HTTP client is built with.

        Kept as a separate, public method because "did this really go through
        the egress proxy?" is the question this whole file exists to answer,
        and a test should be able to ask it without opening a socket. httpx
        routes through a proxy *instead of* an injected transport, so the two
        are never passed together.
        """
        options: dict[str, Any] = {
            "timeout": httpx.Timeout(self.config.timeout, connect=10.0),
            "follow_redirects": False,
            "headers": {"Accept": "*/*"},
        }
        if self._transport is not None:
            options["transport"] = self._transport
        elif proxy:
            options["proxy"] = proxy
        return options


class AddressRefused(Exception):
    """An address an agent asked for is not one it may have."""


def check_address(url: str, allowed: tuple[str, ...]) -> str:
    """The host of ``url``, proven to be on the allowlist.

    A bare domain covers its subdomains (``fda.gov`` allows
    ``www.fda.gov``) and nothing else: ``notfda.gov`` does not end in
    ``.fda.gov``, so the suffix test cannot be fooled by a lookalike name
    someone registers.
    """
    text = (url or "").strip()
    if not text:
        raise AddressRefused("No address was given.")
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https"):
        raise AddressRefused(f"`{text}` is not an http or https address.")
    if parts.username or parts.password:
        raise AddressRefused("An address with a username or password in it is refused.")
    host = (parts.hostname or "").lower()
    if not host:
        raise AddressRefused(f"`{text}` has no host name.")
    if not allowed:
        raise AddressRefused(
            "No web addresses are allowed yet. Add `allowed_domains` to "
            "[agents.capabilities.web] in ~/.config/bedrock/config.toml."
        )
    if any(host == domain or host.endswith("." + domain) for domain in allowed):
        return host
    raise AddressRefused(
        f"`{host}` is not in the allowed list ({', '.join(allowed)}). "
        "Addresses are configured, never chosen by the agent."
    )


# ---- web.fetch ---------------------------------------------------------------


class WebFetch(_WebCapability):
    """Fetch one allowlisted address through the egress proxy and keep the bytes."""

    name = "web.fetch"
    risk = Risk.READ_ONLY

    def describe(self, arguments: dict[str, str]) -> str:
        return f"Fetch {str(arguments.get('url', '')).strip() or '(no address given)'}."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        try:
            config = self.config
            url = str(arguments.get("url", "")).strip()
            check_address(url, config.allowed_domains)
            if not config.proxy:
                raise AddressRefused(
                    "No egress proxy is configured, so nothing is fetched. Set `proxy` "
                    "in [agents.capabilities.web]; leaving it empty deliberately turns "
                    "web.fetch off rather than sending requests straight out."
                )
        except (CapabilityConfigError, AddressRefused) as exc:
            return failed(str(exc))

        try:
            options = self.client_options(proxy=config.proxy)
            async with httpx.AsyncClient(**options) as client:
                async with client.stream("GET", url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location", "(none given)")
                        return failed(
                            f"{url} redirected to {location}. Nothing was fetched: ask for "
                            "that address instead, so it gets checked against the allowed "
                            "list too.",
                        )
                    if response.status_code >= 400:
                        return failed(
                            f"{url} answered HTTP {response.status_code}."
                        )
                    declared = response.headers.get("content-length", "")
                    if declared.isdigit() and int(declared) > config.max_bytes:
                        return failed(_too_large(url, config.max_bytes))
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > config.max_bytes:
                            return failed(_too_large(url, config.max_bytes))
                    content_type = response.headers.get("content-type", "")
                    final_url = str(response.url)
        except httpx.TimeoutException:
            return failed(f"{url} took longer than {config.timeout:.0f}s to answer.")
        except httpx.HTTPError as exc:
            return failed(f"Could not reach {url}: {exc}")

        content = bytes(body)
        digest = hashlib.sha256(content).hexdigest()
        try:
            path = _store(config.store_dir, final_url, content, digest)
        except OSError as exc:
            return failed(
                f"Fetched {url} but could not save it: {exc.strerror or exc}"
            )

        lines = [
            f"GET {final_url} - {len(content)} bytes, {content_type or 'unknown type'}, "
            f"sha256 {digest}",
            f"Saved to {path}",
        ]
        excerpt = _as_text(content, content_type, config.max_excerpt_chars)
        if excerpt:
            lines.append(excerpt)
        return done(lines, artifacts=(path,))


def _too_large(url: str, limit: int) -> str:
    return f"{url} is larger than the {limit // 1000} kB limit, so it was not read."


def _store(store_dir: Path, url: str, content: bytes, digest: str) -> Path:
    """Write the body where it can be checked later, named by its own hash.

    Private to its owner (0600) because a fetch can be anything - a draft
    protocol, a regulator's letter - and these files sit around.
    """
    store_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = safe_filename(urlsplit(url).path.rsplit("/", 1)[-1], fallback="page")
    path = store_dir / f"{digest[:12]}-{name}"
    # A unique temporary name: two agents fetching the same address at once
    # produce the same final name, and a shared ".part" would let one replace
    # the other's half-written file.
    temporary = path.with_name(f"{path.name}.{uuid4().hex[:12]}.part")
    try:
        temporary.write_bytes(content)
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _as_text(content: bytes, content_type: str, limit: int) -> str:
    lowered = content_type.lower()
    if not any(marker in lowered for marker in _TEXTUAL):
        return ""
    return bounded(content.decode("utf-8", errors="replace"), limit)


# ---- web.search --------------------------------------------------------------


class WebSearch(_WebCapability):
    """Search using the company's own SearxNG instance."""

    name = "web.search"
    risk = Risk.READ_ONLY

    def describe(self, arguments: dict[str, str]) -> str:
        query = str(arguments.get("query", "")).strip()
        return f"Search the company search service for {query!r}."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return failed("web.search needs a `query` to look for.")
        try:
            config = self.config
            if not config.search_url:
                return failed(
                    "No search service is configured. Set `search_url` in "
                    "[agents.capabilities.web] to a SearxNG instance the company runs - "
                    "a commercial search API would send the query text outside.",
                )
            if urlsplit(config.search_url).scheme not in ("http", "https"):
                return failed(
                    "`search_url` must be an http or https address, not "
                    f"{config.search_url!r}."
                )
            limit = min(int(arguments.get("limit", config.max_results)), config.max_results)
        except (CapabilityConfigError, ValueError) as exc:
            return failed(str(exc))

        proxy = config.proxy if config.search_via_proxy else ""
        try:
            async with httpx.AsyncClient(**self.client_options(proxy=proxy)) as client:
                response = await client.get(
                    f"{config.search_url}/search",
                    params={"q": query, "format": "json"},
                )
                if response.status_code >= 400:
                    return failed(
                        f"The search service answered HTTP {response.status_code}. "
                        "SearxNG only serves JSON when `json` is in its `formats` setting.",
                    )
                payload = response.json()
        except httpx.TimeoutException:
            return failed("The search service took too long to answer.")
        except httpx.HTTPError as exc:
            return failed(f"Could not reach the search service: {exc}")
        except ValueError:
            return failed(
                "The search service did not answer with JSON. Check that `json` is in "
                "the SearxNG `search.formats` setting.",
            )

        results = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(results, list) or not results:
            return done((f"Nothing found for {query!r}.",))

        lines = [f"Results for {query!r}:"]
        for item in results[: max(limit, 1)]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or "(untitled)").strip()
            url = str(item.get("url") or "").strip()
            snippet = " ".join(str(item.get("content") or "").split())[:300]
            lines.append(f"{title} - {url}" + (f"\n    {snippet}" if snippet else ""))
        return done(lines)


CAPABILITY: tuple[Capability, ...] = (WebFetch(), WebSearch())
"""What the roster finds in this file.

A tuple, which ``roster.discover_capability`` unpacks by name. One file
per adapter, several verbs per file where they share one set of rules.
"""

CAPABILITIES = CAPABILITY
"""Plural alias, for reading a module's contents rather than the roster's."""
