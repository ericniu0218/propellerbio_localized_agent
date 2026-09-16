from pathlib import Path

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


def test_history_directory_uses_environment_file_default_precedence(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text(f'history_dir = "{tmp_path / "file-history"}"\n')

    from_file = load_settings(config_path=config, env={})
    from_environment = load_settings(
        config_path=config,
        env={"BEDROCK_HISTORY_DIR": str(tmp_path / "environment-history")},
    )

    assert from_file.history_dir == str(tmp_path / "file-history")
    assert from_environment.history_dir == str(tmp_path / "environment-history")


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
