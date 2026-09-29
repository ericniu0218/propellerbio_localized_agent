"""`website` points chats at the local model the Research tab already uses.

Without this, a laptop run was in mock mode: every chat answered with the
canned "demo analysis" while the Research tab, which talks to bedrock's own
model server, gave real answers from the same Qwen.
"""

from __future__ import annotations

import pytest

from bedrock.server.cli import use_local_model

LOCAL = "http://localhost:11434/v1"


def env(**extra) -> dict[str, str]:
    # BEDROCK_SERVER_CONFIG_FILE="" keeps a developer's own config file out.
    return {"BEDROCK_SERVER_CONFIG_FILE": "", "BEDROCK_API_BASE": LOCAL, "BEDROCK_CONTEXT_WINDOW": "12288", **extra}


def models(*names):
    return lambda base: set(names)


def test_a_pulled_model_turns_on_real_answers_for_every_agent():
    e = env()
    assert use_local_model(e, list_models=models("qwen3:8b", "llama3.2:3b")) == "qwen3:8b"
    assert e["AGENT_PACK_MODE"] == "real"
    for role in ("DIRECTOR", "SPECIALIST", "RUNNER"):
        assert e[f"{role}_LLM_URL"] == LOCAL
        assert e[f"{role}_MODEL_NAME"] == "qwen3:8b"
        assert e[f"{role}_CONTEXT_LENGTH"] == "12288"
    # Still the laptop demo: demo sign-in, and the Research tab may search.
    assert e["DEMO_MODE"] == "true" and e["AIRGAP_CHECK"] == "false"


def test_a_model_named_by_the_person_wins():
    e = env(WEBSITE_MODEL="llama3.1:8b")
    assert use_local_model(e, list_models=models("qwen3:8b", "llama3.1:8b")) == "llama3.1:8b"


def test_only_a_too_small_model_leaves_the_demo_answers(capsys):
    e = env()
    assert use_local_model(e, list_models=models("llama3.2:3b")) is None
    assert "AGENT_PACK_MODE" not in e
    assert "ollama pull qwen3:8b" in capsys.readouterr().out


def test_no_model_server_leaves_the_demo_answers(capsys):
    def down(base):
        raise OSError("connection refused")

    e = env()
    assert use_local_model(e, list_models=down) is None
    assert "AGENT_PACK_MODE" not in e
    assert "No local model server" in capsys.readouterr().out


@pytest.mark.parametrize("chosen", [{"AGENT_PACK_MODE": "mock"}, {"BEDROCK_SERVER_MODE": "real"}])
def test_a_mode_someone_chose_is_left_alone(chosen):
    e = env(**chosen)
    assert use_local_model(e, list_models=models("qwen3:8b")) is None
    assert "DIRECTOR_LLM_URL" not in e


def test_a_model_server_on_another_machine_is_not_used():
    e = env(BEDROCK_API_BASE="http://gpu-box.internal:8000/v1")
    assert use_local_model(e, list_models=models("qwen3:8b")) is None
    assert "AGENT_PACK_MODE" not in e


def test_settings_already_given_are_kept():
    e = env(SPECIALIST_MODEL_NAME="qwen3:14b", DEMO_MODE="false")
    use_local_model(e, list_models=models("qwen3:8b"))
    assert e["SPECIALIST_MODEL_NAME"] == "qwen3:14b"
    assert e["DEMO_MODE"] == "false"


def test_the_agents_read_the_context_length_they_are_given(monkeypatch):
    from bedrock.server.config import load_models_config

    monkeypatch.setenv("DIRECTOR_CONTEXT_LENGTH", "12288")
    load_models_config.cache_clear()
    try:
        assert load_models_config().director.context_length == 12288
    finally:
        monkeypatch.delenv("DIRECTOR_CONTEXT_LENGTH")
        load_models_config.cache_clear()
    assert load_models_config().director.context_length == 131072
