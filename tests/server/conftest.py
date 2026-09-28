import os
import tempfile

# Must happen before any `app.*` module is imported anywhere in the test
# session: app.config.get_settings() and app.db.base's engine are built
# once from the environment and cached.
os.environ.setdefault("AGENT_PACK_MODE", "mock")
_tmp_db = tempfile.NamedTemporaryFile(prefix="agent_pack_test_", suffix=".db", delete=False)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_db.name}")
os.environ.setdefault("AIRGAP_CHECK", "false")

import pytest

from bedrock.server.db import get_database
from bedrock.server.orchestrator.session import start_session


@pytest.fixture
async def session_id() -> str:
    return await start_session(f"test-user-{os.urandom(4).hex()}")


@pytest.fixture
def db():
    return get_database()
