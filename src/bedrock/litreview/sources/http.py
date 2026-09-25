"""One polite HTTP client for every literature source.

Retries what is worth retrying (429, 5xx, dropped connections) with
exponential backoff, honours `Retry-After`, and spaces requests to stay under
each API's published rate limit. Tests inject an `httpx.MockTransport` and a
no-op `sleep`, so the whole searcher runs offline.

This is the only place in the package that talks to a public server during a
question. It sends search terms and identifiers out; it never sends papers,
findings or reports.

Every request must land on one of `SOURCE_HOSTS`, and so must every redirect it
follows. The URLs here are fixed in code - there is no path by which a model or
a config file chooses one - but a redirect is chosen by the far end, and a
search URL carries the scientist's search terms in its query string. A 302 to
somewhere else would hand those terms to a third party, which is the one thing
this client is careful not to do. So the check is on the request, not the
first one: httpx fires the hook again for each hop, and a redirect to another
host raises `SourceError` before it is sent.

Bedrock's `web.fetch` capability requires a proxy and refuses to follow
redirects at all. That rule was written for URLs a *model* picks. These are not,
so the narrower host allowlist is the right shape here; httpx honours
`HTTPS_PROXY` already, for a site that requires one.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import httpx

# Deliberately anonymous: the client's name next to its search terms in a
# public server's logs would say what it is working on.
USER_AGENT = "litreview/0.2"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

SOURCE_HOSTS = frozenset({"www.ebi.ac.uk", "eutils.ncbi.nlm.nih.gov"})
"""The only hosts a literature search may reach. See the module docstring."""


class SourceError(RuntimeError):
    """A source could not answer, after retries."""


def _only_source_hosts(request: httpx.Request) -> None:
    """Refuse any request, redirect included, to a host that is not a source."""
    host = (request.url.host or "").lower()
    if host not in SOURCE_HOSTS:
        raise SourceError(
            f"refusing to send {request.method} to {host or request.url!r}: "
            f"a literature search may only reach {', '.join(sorted(SOURCE_HOSTS))}"
        )


class SourceHTTP:
    def __init__(
        self,
        *,
        min_interval: float = 0.0,
        timeout: float = 60.0,
        retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.client = httpx.Client(
            timeout=timeout, transport=transport, headers={"User-Agent": user_agent},
            follow_redirects=True,
            event_hooks={"request": [_only_source_hosts]},
        )
        self.min_interval = min_interval
        self.retries = max(1, retries)
        self.sleep = sleep
        self._last = 0.0
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """The response for any status the caller should see (2xx, 404, ...);
        raises `SourceError` once retries are spent."""
        last: str = ""
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self.client.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code not in _RETRY_STATUS:
                    return resp
                last = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit() and attempt < self.retries - 1:
                    self.sleep(min(float(retry_after), 60.0))
                    continue
            if attempt < self.retries - 1:
                self.sleep(min(2.0 ** attempt, 30.0))
        raise SourceError(f"{method} {url} failed after {self.retries} attempts ({last})")

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        self.client.close()
