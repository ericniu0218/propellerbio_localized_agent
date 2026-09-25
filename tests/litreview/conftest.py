"""Isolation for the literature-review tests.

``bedrock.litreview.config.load_settings`` reads bedrock's own settings when it
is not handed any, and bedrock's loader reads ``~/.config/bedrock/config.toml``.
A developer with a real config file there - a different ``api_base``, a
``[litreview]`` table of their own - would otherwise get different results from
``pytest`` than a clean machine does, and a run could be pointed at a real
server. Pointing ``DEFAULT_CONFIG_PATH`` at a file that does not exist makes
every test read the built-in defaults instead.

The data directories are already redirected for the whole suite by
``private_bedrock_directories`` in ``tests/conftest.py``.
"""

from __future__ import annotations

import pytest

from bedrock.config import settings as bedrock_settings


@pytest.fixture(autouse=True)
def no_real_bedrock_config(tmp_path, monkeypatch):
    monkeypatch.setattr(bedrock_settings, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.toml")
