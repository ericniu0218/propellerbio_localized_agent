"""OIDC login against Keycloak — see docs/server/WORKFLOW.md section 1 for why
Keycloak: employees log in with their existing company account, never a
separate app password.

This is the whole handshake in three functions:
1. `authorize_url` — where to send the browser to log in.
2. `exchange_code` — trade the code Keycloak gave back for tokens
   (server-to-server, the client secret never touches the browser).
3. `verify_id_token` — check the signature is really Keycloak's before
   trusting anything it claims about who logged in.

Only active when `Settings.sso_enabled` is true; see
`src/bedrock/server/api/routes_auth.py` for what happens when it isn't (plain mock login
still works, for tests and GPU-less demos).
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

import httpx
from jose import jwt as jose_jwt

from bedrock.server.config import Settings


@dataclass
class OIDCUser:
    sub: str                       # Keycloak's stable user id — not the username
    email: str | None
    name: str | None
    preferred_username: str | None

    @property
    def display_name(self) -> str:
        return self.name or self.preferred_username or self.email or self.sub


class OIDCClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.sso_enabled:
            raise RuntimeError("OIDCClient constructed without SSO configured")
        self._settings = settings
        self._issuer_base = f"{settings.keycloak_base_url}/realms/{settings.keycloak_realm}"
        self._discovery: dict | None = None
        self._jwks: dict | None = None
        self._jwks_fetched_at: float = 0.0

    @property
    def redirect_uri(self) -> str:
        return f"{self._settings.app_base_url}/auth/callback"

    async def _discover(self) -> dict:
        if self._discovery is None:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._issuer_base}/.well-known/openid-configuration")
                resp.raise_for_status()
                self._discovery = resp.json()
        return self._discovery

    async def _jwks_keys(self, force_refresh: bool = False) -> dict:
        # Cached for the process lifetime with a coarse TTL — Keycloak
        # rotates signing keys rarely, and verify_id_token() already
        # refreshes once on a kid it doesn't recognize.
        stale = (time.monotonic() - self._jwks_fetched_at) > 3600
        if self._jwks is None or force_refresh or stale:
            discovery = await self._discover()
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(discovery["jwks_uri"])
                resp.raise_for_status()
                self._jwks = resp.json()
                self._jwks_fetched_at = time.monotonic()
        return self._jwks

    async def authorize_url(self, state: str) -> str:
        discovery = await self._discover()
        params = {
            "client_id": self._settings.keycloak_client_id,
            "response_type": "code",
            "scope": "openid profile email",
            "redirect_uri": self.redirect_uri,
            "state": state,
        }
        return str(httpx.URL(discovery["authorization_endpoint"], params=params))

    async def exchange_code(self, code: str) -> dict:
        discovery = await self._discover()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                discovery["token_endpoint"],
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self._settings.keycloak_client_id,
                    "client_secret": self._settings.keycloak_client_secret,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def verify_id_token(self, id_token: str, access_token: str) -> OIDCUser:
        discovery = await self._discover()
        header = jose_jwt.get_unverified_header(id_token)

        jwks = await self._jwks_keys()
        key = next((k for k in jwks["keys"] if k["kid"] == header["kid"]), None)
        if key is None:
            jwks = await self._jwks_keys(force_refresh=True)
            key = next((k for k in jwks["keys"] if k["kid"] == header["kid"]), None)
        if key is None:
            raise ValueError("no matching signing key found for this id_token")

        # Keycloak's id_token includes an at_hash claim binding it to the
        # access token issued alongside it — python-jose needs the access
        # token to check that hash, or it refuses to validate at all.
        claims = jose_jwt.decode(
            id_token,
            key,
            algorithms=[header["alg"]],
            audience=self._settings.keycloak_client_id,
            issuer=discovery["issuer"],
            access_token=access_token,
        )
        return OIDCUser(
            sub=claims["sub"],
            email=claims.get("email"),
            name=claims.get("name"),
            preferred_username=claims.get("preferred_username"),
        )


def new_state() -> str:
    """CSRF guard for the login redirect round-trip — see routes_auth.py."""
    return secrets.token_urlsafe(24)
