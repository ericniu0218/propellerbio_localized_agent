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
