"""`create_backend`: dispatches on `settings.backend`, and refuses what it
cannot build rather than silently substituting the wrong client.

This is deliberately not in `tests/test_openai_backend.py`: that file is
about the HTTP client itself (talking to a fake OpenAI-compatible server),
while this is about the one line of dispatch in front of it.
"""

from __future__ import annotations

import pytest

from bedrock.config import Settings
from bedrock.inference import UnsupportedBackendError, create_backend
from bedrock.inference.openai_compatible import OpenAICompatibleBackend


def test_openai_settings_build_an_openai_compatible_backend():
    settings = Settings(backend="openai", api_base="http://localhost:11434/v1", api_key="secret", request_timeout=42.0)
    backend = create_backend(settings)
    assert isinstance(backend, OpenAICompatibleBackend)


def test_agentpack_is_refused_with_what_to_do_instead():
    """The agent-pack API is not an OpenAI-compatible model server, so
    `create_backend` must say that plainly rather than building a client that
    will talk to the wrong thing."""
    settings = Settings(backend="agentpack")
    with pytest.raises(UnsupportedBackendError) as exc:
        create_backend(settings)
    message = str(exc.value)
    assert "agent-pack" in message
    assert "backend = 'openai'" in message
    assert "api_base" in message


def test_an_unknown_backend_names_the_known_ones():
    """`Settings` can be built directly (tests, `dataclasses.replace`) without
    going through `load_settings`'s validation, so `create_backend` refuses a
    nonsense value itself rather than assuming it is one of the two it knows."""
    settings = Settings(backend="vllm-direct")
    with pytest.raises(UnsupportedBackendError) as exc:
        create_backend(settings)
    message = str(exc.value)
    assert "vllm-direct" in message
    assert "openai" in message and "agentpack" in message
