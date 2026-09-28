"""SSO must be opt-in: with no KEYCLOAK_* env vars set (the default for
mock mode, tests, and GPU-less demos), auth/login and auth/config must
still respond sanely rather than crashing or silently requiring Keycloak.
"""

from fastapi.testclient import TestClient

from bedrock.server.main import app


def test_auth_config_reports_disabled_by_default():
    with TestClient(app) as client:
        resp = client.get("/auth/config")
        assert resp.status_code == 200
        assert resp.json() == {"sso_enabled": False}


def test_login_returns_a_clear_error_when_sso_not_configured():
    with TestClient(app) as client:
        resp = client.get("/auth/login", follow_redirects=False)
        assert resp.status_code == 503
        assert "sign-in" in resp.json()["detail"].lower() or "sso" in resp.json()["detail"].lower() or "single sign-on" in resp.json()["detail"].lower()
