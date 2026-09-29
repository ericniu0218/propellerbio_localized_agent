"""EPO OPS authentication: OAuth2 client-credentials, confirmed against the
official reference (OPS RESTful Web Services Reference Guide v1.3.19, §2.3.2):

    POST https://ops.epo.org/3.2/auth/accesstoken
    Authorization: Basic base64(consumer_key:consumer_secret)
    Content-Type: application/x-www-form-urlencoded

    grant_type=client_credentials

    -> {"access_token": "...", "expires_in": "1199", ...}   (~20 minutes)

Kept separate from `epo_ops.py` because it has its own thing to get right —
proactive refresh before a ~20-minute token expires mid-query — and its own
narrow thing to test, with no bearing on search or claims parsing.

Refresh happens twice over, deliberately: `header()` refreshes ahead of the
cached expiry with a safety margin, so a long-running search does not start a
request on a token about to lapse; `invalidate()` lets the caller force a
fresh token after a 401, since a request that started just inside the margin
can still land after the real server-side expiry.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Callable

from bedrock.litreview.sources.http import SourceError, SourceHTTP

TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"

# The docs' own example token lives ~1199s (20 minutes, one second short).
# Refreshing this many seconds early is what keeps a request from starting on
# a token that expires while the response is still in flight.
REFRESH_MARGIN_S = 60.0


class AuthError(SourceError):
    """The consumer key/secret were not accepted."""


@dataclass
class TokenStore:
    """Holds one live access token, refreshing it as needed.

    `now` is injectable so a test can move time forward without sleeping;
    `http` likewise, so token issuance is exercised against a mock transport,
    exactly like every other source in this package.
    """

    consumer_key: str
    consumer_secret: str
    http: SourceHTTP = field(default_factory=lambda: SourceHTTP(timeout=30.0))
    now: Callable[[], float] = time.time

    _token: str | None = field(default=None, init=False, repr=False)
    _expires_at: float = field(default=0.0, init=False, repr=False)

    def _basic_auth(self) -> str:
        raw = f"{self.consumer_key}:{self.consumer_secret}".encode("ascii")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _fetch(self) -> None:
        resp = self.http.post(
            TOKEN_URL,
            headers={"Authorization": self._basic_auth(), "Content-Type": "application/x-www-form-urlencoded"},
            content="grant_type=client_credentials",
        )
        if resp.status_code != 200:
            raise AuthError(
                f"EPO OPS rejected the consumer key/secret (HTTP {resp.status_code}): {resp.text[:300]}"
            )
        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise AuthError(f"EPO OPS token response had no access_token: {data}")
        try:
            ttl = float(data.get("expires_in", 1199))
        except (TypeError, ValueError):
            ttl = 1199.0
        self._token = token
        self._expires_at = self.now() + ttl

    def invalidate(self) -> None:
        """Force the next `header()` to fetch a fresh token. Call this on a
        401 from a resource request — the cached token was accepted at issue
        time but the server no longer honours it."""
        self._token = None
        self._expires_at = 0.0

    def header(self) -> dict[str, str]:
        """`{"Authorization": "Bearer ..."}`, refreshing first if the cached
        token is missing or within `REFRESH_MARGIN_S` of expiry."""
        if self._token is None or self.now() >= self._expires_at - REFRESH_MARGIN_S:
            self._fetch()
        return {"Authorization": f"Bearer {self._token}"}
