from pathlib import Path

import pytest

from bedrock.config import load_settings
from bedrock.config.settings import DEFAULT_API_BASE


def test_defaults_point_at_local_ollama(tmp_path: Path):
    settings = load_settings(config_path=tmp_path / "missing.toml", env={})
    assert settings.api_base == DEFAULT_API_BASE
    assert settings.model is None
    assert settings.api_key is None


def test_config_file_is_read(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text('api_base = "http://inference.internal:8000/v1/"\nmodel = "qwen"\nfoo = 1\n')
    settings = load_settings(config_path=config, env={})
    assert settings.api_base == "http://inference.internal:8000/v1"  # trailing slash stripped
    assert settings.model == "qwen"
    assert settings.extra == {"foo": 1}


def test_environment_overrides_file(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text('api_base = "http://from-file/v1"\nmodel = "file-model"\n')
    env = {"BEDROCK_API_BASE": "http://from-env/v1", "BEDROCK_MODEL": "env-model", "BEDROCK_API_KEY": "secret"}
    settings = load_settings(config_path=config, env=env)
    assert settings.api_base == "http://from-env/v1"
    assert settings.model == "env-model"
    assert settings.api_key == "secret"


def test_quoted_false_disables_history(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text('save_history = "false"\n')
    assert load_settings(config_path=config, env={}).save_history is False


def test_relative_xdg_data_home_is_ignored(tmp_path: Path):
    settings = load_settings(config_path=tmp_path / "missing", env={"XDG_DATA_HOME": "relative"})
    assert settings.history_dir.is_absolute()


def test_invalid_history_dir_has_a_clear_error(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text("history_dir = 123\n")
    with pytest.raises(ValueError, match="history_dir"):
        load_settings(config_path=config, env={})


def test_workspace_identity_and_private_vault_are_configurable(tmp_path: Path):
    settings = load_settings(
        config_path=tmp_path / "missing.toml",
        env={
            "BEDROCK_USER": "alice-stable-id",
            "BEDROCK_DISPLAY_NAME": "Alice Example",
            "BEDROCK_WORKSPACE_DIR": str(tmp_path / "team-vault"),
        },
    )

    assert settings.user_id == "alice-stable-id"
    assert settings.display_name == "Alice Example"
    assert settings.workspace_dir == str(tmp_path / "team-vault")


def test_action_history_directory_uses_environment_file_default_precedence(tmp_path: Path):
    """The action journal's own setting, which is not ``history_dir``.

    ``history_dir`` holds saved conversations; this one holds the metadata-only
    compliance journal. The test asserts they move independently, because one
    knob for both would relocate an audit trail by accident.
    """
    config = tmp_path / "config.toml"
    config.write_text(f'action_history_dir = "{tmp_path / "file-history"}"\n')

    from_file = load_settings(config_path=config, env={})
    from_environment = load_settings(
        config_path=config,
        env={"BEDROCK_ACTION_HISTORY_DIR": str(tmp_path / "environment-history")},
    )

    assert from_file.action_history_dir == str(tmp_path / "file-history")
    assert from_environment.action_history_dir == str(tmp_path / "environment-history")
    # Moving the journal must not move saved conversations.
    assert "file-history" not in str(from_file.history_dir)


def test_biotech_tool_settings_can_be_provisioned_without_hard_coded_paths(tmp_path: Path):
    settings = load_settings(
        config_path=tmp_path / "missing.toml",
        env={
            "BEDROCK_BIOTECH_WORK_DIR": str(tmp_path / "runs"),
            "ALPHAFOLD_LAUNCHER": str(tmp_path / "alphafold" / "run_docker.py"),
            "ALPHAFOLD_DATA_DIR": str(tmp_path / "data"),
            "ALPHAFOLD_MAX_TEMPLATE_DATE": "2026-01-01",
            "ALPHAFOLD_DOCKER_IMAGE": "alphafold@sha256:test",
            "ALPHAFOLD_DATABASE_VERSION": "2026-01-01",
            "PYMOL_EXECUTABLE": "/opt/pymol/bin/pymol",
        },
    )

    assert settings.biotech_work_dir == tmp_path / "runs"
    assert settings.alphafold_launcher == tmp_path / "alphafold" / "run_docker.py"
    assert settings.alphafold_data_dir == tmp_path / "data"
    assert settings.alphafold_max_template_date == "2026-01-01"
    assert settings.alphafold_docker_image == "alphafold@sha256:test"
    assert settings.alphafold_database_version == "2026-01-01"
    assert settings.pymol_executable == "/opt/pymol/bin/pymol"


def test_automatic_context_is_on_unless_switched_off(tmp_path: Path):
    """On by default, which keeps agent-pack mode behaving as it always has."""
    assert load_settings(config_path=tmp_path / "missing.toml", env={}).auto_context is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_automatic_context_can_be_switched_off_from_the_environment(tmp_path: Path, value: str):
    settings = load_settings(config_path=tmp_path / "missing.toml", env={"BEDROCK_AUTO_CONTEXT": value})
    assert settings.auto_context is False


def test_automatic_context_environment_overrides_file_and_blank_does_not(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text("auto_context = false\n")

    assert load_settings(config_path=config, env={}).auto_context is False
    assert load_settings(config_path=config, env={"BEDROCK_AUTO_CONTEXT": "1"}).auto_context is True
    # An empty variable is "not set", as with BEDROCK_SAVE_HISTORY, so the file still decides.
    assert load_settings(config_path=config, env={"BEDROCK_AUTO_CONTEXT": "  "}).auto_context is False
    # It is a known key, not an unrecognised one kept in `extra`.
    assert "auto_context" not in load_settings(config_path=config, env={}).extra


@pytest.mark.parametrize(
    ("config_text", "env", "name"),
    [
        ('auto_context = "sometimes"\n', {}, "auto_context"),
        ("", {"BEDROCK_AUTO_CONTEXT": "maybe"}, "BEDROCK_AUTO_CONTEXT"),
    ],
)
def test_an_unreadable_automatic_context_value_has_a_clear_error(tmp_path: Path, config_text, env, name):
    config = tmp_path / "config.toml"
    config.write_text(config_text)
    with pytest.raises(ValueError, match=name):
        load_settings(config_path=config, env=env)
