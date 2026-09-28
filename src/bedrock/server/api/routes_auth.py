"""Login via Keycloak (real SSO) when configured, per
docs/WORKFLOW.md section 1. The whole redirect round-trip:

  browser --GET /auth/login--> backend --302--> Keycloak login page
  browser <--302 (code, state)-- Keycloak
  browser --GET /auth/callback?code&state--> backend
  backend --(code, client secret)--> Keycloak token endpoint   [server-to-server]
  backend <--id_token-- Keycloak
  backend verifies id_token signature, creates an app session,
  browser <--302 to "/?session_id=...&user=..."-- backend

The client secret never reaches the browser — only the backend ever talks
to Keycloak's token endpoint. If SSO isn't configured (no Keycloak in
this environment — e.g. tests, GPU-less mock-mode demos), both routes
return a plain explanatory error instead of crashing, and the frontend's
existing "type a name" login keeps working exactly as before.
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from bedrock.server.auth.oidc import OIDCClient, new_state
from bedrock.server.auth.state_store import get_state_store
from bedrock.server.config import get_settings
from bedrock.server.orchestrator.session import start_session

router = APIRouter(tags=["auth"])


def _require_oidc_client() -> OIDCClient:
    settings = get_settings()
    if not settings.sso_enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "Single sign-on isn't configured on this server "
                "(KEYCLOAK_BASE_URL/REALM/CLIENT_ID/CLIENT_SECRET). "
                "Use the plain sign-in on the login screen instead."
            ),
        )
    return OIDCClient(settings)


@router.get("/auth/login")
async def login():
    client = _require_oidc_client()
    state = new_state()
    get_state_store().issue(state)
    url = await client.authorize_url(state)
    return RedirectResponse(url, status_code=302)


@router.get("/auth/callback")
async def callback(code: str, state: str):
    if not get_state_store().consume(state):
        raise HTTPException(status_code=400, detail="Login expired or was already used. Try again.")

    client = _require_oidc_client()
    tokens = await client.exchange_code(code)
    user = await client.verify_id_token(tokens["id_token"], tokens["access_token"])

    # The Keycloak username is what the rest of the system treats as
    # user_id (RBAC, audit log, "who shared this") — stable and human
    # readable, unlike the opaque `sub`.
    user_id = user.preferred_username or user.email or user.sub
    settings = get_settings()
    session_id = await start_session(user_id)

    redirect_url = (
        f"{settings.app_base_url}/?session_id={quote(session_id)}"
        f"&user_id={quote(user_id)}&user_name={quote(user.display_name)}"
    )
    return RedirectResponse(redirect_url, status_code=302)


@router.get("/auth/config")
async def auth_config():
    """Lets the frontend show "Log in with Keycloak" only when it would
    actually work, instead of a button that 503s."""
    return {"sso_enabled": get_settings().sso_enabled}
