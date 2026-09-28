"""Shared set-up for the agent-pack server's tests (``bedrock.server``).

The server is an optional extra. On an install without it these tests are not
collected at all (FastAPI alone is not the test: the literature review's ``api``
extra installs it too) - that install has no server to test, and CI's
``core`` job proves the terminal app does not need one.

Everything below runs before any ``bedrock.server`` module is imported, because
the server builds its settings, database engine and event bus once per process
and caches them (see ``bedrock.server.state``). So:

* the database, workspace and recordings go into one temporary directory for the
  whole session, never the working directory or a real home directory;
* variables a developer's shell may carry (a real Keycloak, a real datacenter,
  a meeting setting) are removed, so the suite behaves the same everywhere;
* bedrock's own ``config.toml`` is not read (``BEDROCK_SERVER_CONFIG_FILE=""``).

The target repository's own autouse fixtures in ``tests/conftest.py`` still
apply here: they redirect ``XDG_DATA_HOME`` and bedrock's default directories
for every test.
"""

import importlib.util

_SERVER_PACKAGES = ("fastapi", "sqlalchemy", "sse_starlette", "multipart")

if any(importlib.util.find_spec(name) is None for name in _SERVER_PACKAGES):
    collect_ignore_glob = ["*"]
else:
    import atexit
    import os
    import shutil
    import tempfile

    _DATA = tempfile.mkdtemp(prefix="bedrock-server-tests-")
    atexit.register(shutil.rmtree, _DATA, True)

    for _name in [
        n for n in os.environ if n.startswith(("KEYCLOAK_", "DATACENTER_", "MEETING_", "BEDROCK_SERVER_"))
    ] + ["DEMO_MODE", "WORKSPACE_DATA_DIR", "DATABASE_URL", "AGENT_PACK_ALLOW_PUBLIC_DATACENTER", "APP_BASE_URL"]:
        os.environ.pop(_name, None)
    os.environ.update(
        AGENT_PACK_MODE="mock",
        AIRGAP_CHECK="false",
        BEDROCK_SERVER_DATA_DIR=_DATA,
        BEDROCK_SERVER_CONFIG_FILE="",
        BEDROCK_SERVER_CONFIG_DIR=os.path.join(_DATA, "config"),
    )

    import pytest

    @pytest.fixture
    async def session_id() -> str:
        from bedrock.server.orchestrator.session import start_session

        return await start_session(f"test-user-{os.urandom(4).hex()}")

    @pytest.fixture
    def db():
        from bedrock.server.db import get_database

        return get_database()
