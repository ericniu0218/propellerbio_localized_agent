"""The literature review reads its settings from bedrock.

What is being proved here is the seam between the two: which model set a run
gets and who decides, that a typo in the `[litreview]` table is refused rather
than silently ignored, that a YAML override replaces the packaged file and is
recorded as the file the run read, and that the air-gap checks still refuse a
model server on the public internet.
"""

from __future__ import annotations

import pytest

from bedrock.config import PROFILES, Settings as BedrockSettings
from bedrock.litreview.config import (
    DEFAULTS_DIR,
    ConfigError,
    config_file,
    load_settings,
)


def bedrock(**kwargs) -> BedrockSettings:
    """Bedrock's own settings, built directly so no config file is read."""
    kwargs.setdefault("api_base", "http://localhost:11434/v1")
    return BedrockSettings(**kwargs)


# ---------------------------------------------------------------------------
# Which model set
# ---------------------------------------------------------------------------


def test_the_machine_profile_chooses_the_model_set_when_nothing_else_does():
    assert load_settings(bedrock(), env={}).model_set == "laptop"
    assert load_settings(bedrock(profile=PROFILES["workstation"]), env={}).model_set == "laptop"
    # A machine big enough to be called a server gets the office models.
    assert load_settings(bedrock(profile=PROFILES["server"]), env={}).model_set == "office"


def test_the_table_beats_the_profile_the_env_beats_the_table_the_argument_beats_both():
    table = bedrock(profile=PROFILES["server"], litreview={"model_set": "laptop"})
    assert load_settings(table, env={}).model_set == "laptop"

    env = {"BEDROCK_LITREVIEW_MODELS": "medgemma"}
    assert load_settings(table, env=env).model_set == "medgemma"

    assert load_settings(table, env=env, model_set="office").model_set == "office"


def test_an_unknown_model_set_names_the_ones_that_exist():
    with pytest.raises(ConfigError, match="unknown model set 'tiny'.*laptop"):
        load_settings(bedrock(), model_set="tiny", env={})


def test_the_jobs_table_maps_every_job_onto_a_model_the_set_has():
    settings = load_settings(bedrock(), model_set="medgemma", env={})
    assert settings.model_for("extract") == "medgemma1.5:4b"
    assert settings.model_for("relevance") == "medgemma1.5:4b"
    assert settings.min_context_window == 12288
    # Reply length is litreview's, not the profile's 1,024-token chat limit.
    assert settings.max_tokens == 2048


# ---------------------------------------------------------------------------
# The [litreview] table
# ---------------------------------------------------------------------------


def test_an_unknown_table_key_is_refused_and_the_message_lists_the_allowed_ones():
    with pytest.raises(ConfigError) as exc:
        load_settings(bedrock(litreview={"model-set": "laptop"}), env={})
    assert "model-set" in str(exc.value) and "model_set" in str(exc.value)


def test_the_data_directory_comes_from_the_table_or_the_env_and_derives_the_rest(tmp_path):
    settings = load_settings(bedrock(litreview={"data_dir": str(tmp_path / "from-table")}), env={})
    assert settings.data_dir == tmp_path / "from-table"

    env = {"BEDROCK_LITREVIEW_DIR": str(tmp_path / "from-env")}
    settings = load_settings(bedrock(litreview={"data_dir": str(tmp_path / "from-table")}), env=env)
    assert settings.data_dir == tmp_path / "from-env"
    assert settings.cache_dir == settings.data_dir / "cache" / "llm"
    assert settings.out_dir == settings.data_dir / "reports"
    assert settings.db_path == settings.data_dir / "litreview.sqlite3"


def test_the_context_window_and_server_come_from_bedrock_and_are_not_restated():
    settings = load_settings(
        bedrock(api_base="http://localhost:8000/v1", profile=PROFILES["workstation"]), env={}
    )
    assert settings.base_url == "http://localhost:8000/v1"
    assert settings.context_window == 32768


# ---------------------------------------------------------------------------
# Overriding a packaged YAML file
# ---------------------------------------------------------------------------


def test_an_override_replaces_the_packaged_file_and_the_run_records_which_was_read(tmp_path):
    from bedrock.litreview.runner import input_files

    assert config_file("models.yaml", {}) == DEFAULTS_DIR / "models.yaml"

    (tmp_path / "models.yaml").write_text(
        "min_context_window: 20000\nmodel_sets:\n  house: {medium: {name: house-model}}\n"
        "jobs:\n  extract: medium\ndefaults: {temperature: 0.0, seed: 3, max_tokens: 4096}\n"
    )
    env = {"BEDROCK_LITREVIEW_CONFIG_DIR": str(tmp_path)}

    # Replaced completely, never merged: the packaged sets are simply gone.
    with pytest.raises(ConfigError, match="unknown model set"):
        load_settings(bedrock(), model_set="laptop", env=env)

    settings = load_settings(bedrock(), model_set="house", env=env)
    assert settings.model_for("extract") == "house-model"
    assert settings.min_context_window == 20000
    assert settings.seed == 3 and settings.max_tokens == 4096

    assert config_file("models.yaml", env) == tmp_path / "models.yaml"
    # ... and a run hashes the file it really read, so the report can be matched
    # back to the names it was written under.
    import os

    os.environ["BEDROCK_LITREVIEW_CONFIG_DIR"] = str(tmp_path)
    try:
        assert tmp_path / "models.yaml" in input_files()
    finally:
        del os.environ["BEDROCK_LITREVIEW_CONFIG_DIR"]


def test_the_office_model_names_can_be_corrected_without_a_release(tmp_path):
    """The `office` set's names are guesses until IT says what the vLLM server
    serves (OPEN_ISSUES item 16), and a vLLM name is usually the full path.

    This is the escape hatch for the day that is discovered: one file, no code
    change, no release — and the run still records which file it read, so a
    report can be matched back to the names it was written under.
    """
    (tmp_path / "models.yaml").write_text(
        "min_context_window: 12288\n"
        "model_sets:\n"
        "  office:\n"
        "    small:  {name: 'Qwen/Qwen3-8B', thinking: false}\n"
        "    medium: {name: 'Qwen/Qwen3-30B-A3B-Instruct-2507', thinking: false}\n"
        "jobs:\n  relevance: small\n  extract: medium\n"
        "defaults: {temperature: 0.0, seed: 0, max_tokens: 2048}\n"
    )
    env = {"BEDROCK_LITREVIEW_CONFIG_DIR": str(tmp_path)}
    settings = load_settings(bedrock(), model_set="office", env=env)

    assert settings.model_for("extract") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert settings.model_for("relevance") == "Qwen/Qwen3-8B"
    assert config_file("models.yaml", env) == tmp_path / "models.yaml"


def test_the_knowledge_tables_cannot_be_overridden(tmp_path):
    """They decide what counts as the exact entity, so a change is code review."""
    (tmp_path / "entities.yaml").write_text("entities: {}\n")
    settings = load_settings(bedrock(), env={"BEDROCK_LITREVIEW_CONFIG_DIR": str(tmp_path)})
    assert settings  # loading is unaffected

    from bedrock.litreview.knowledge import TABLE

    assert TABLE.parent.parent == DEFAULTS_DIR.parent
    assert config_file("entities.yaml", {"BEDROCK_LITREVIEW_CONFIG_DIR": str(tmp_path)}) != TABLE


# ---------------------------------------------------------------------------
# The air gap
# ---------------------------------------------------------------------------


def test_a_public_model_endpoint_is_refused_and_a_company_one_is_not():
    with pytest.raises(ConfigError, match="not a local address"):
        load_settings(bedrock(api_base="https://api.openai.com/v1"), env={})

    assert load_settings(bedrock(api_base="http://gpu01.internal:8000/v1"), env={}).base_url
    assert load_settings(bedrock(api_base="http://10.0.0.5:8000/v1"), env={}).base_url


def test_allowed_hosts_lets_one_extra_host_through_from_the_table_or_the_env():
    public = bedrock(api_base="http://gpu-box:8000/v1")
    with pytest.raises(ConfigError):
        load_settings(public, env={})

    from_table = bedrock(api_base="http://gpu-box:8000/v1", litreview={"allowed_hosts": ["gpu-box"]})
    assert load_settings(from_table, env={}).allowed_hosts == ("gpu-box",)

    env = {"BEDROCK_LITREVIEW_ALLOWED_HOSTS": "gpu-box, other-box"}
    assert load_settings(public, env=env).allowed_hosts == ("gpu-box", "other-box")


def test_the_epo_account_comes_from_the_environment_or_the_table_and_is_one_per_machine():
    """A patent search is answered by the machine's own EPO account, so that a
    scientist on the server's Research tab never types a key. Both places exist
    because a server is started differently in different deployments; the
    environment wins, as it does everywhere else here."""
    assert load_settings(bedrock(), env={}).epo_ops_key == ""

    from_table = bedrock(litreview={"epo_ops_key": "from-table", "epo_ops_secret": "secret-from-table"})
    settings = load_settings(from_table, env={})
    assert (settings.epo_ops_key, settings.epo_ops_secret) == ("from-table", "secret-from-table")

    env = {"EPO_OPS_KEY": "from-env", "EPO_OPS_SECRET": "secret-from-env"}
    settings = load_settings(from_table, env=env)
    assert (settings.epo_ops_key, settings.epo_ops_secret) == ("from-env", "secret-from-env")


def test_an_ollama_cloud_model_is_refused_even_though_the_endpoint_is_local(tmp_path):
    (tmp_path / "models.yaml").write_text(
        "min_context_window: 12288\nmodel_sets:\n  cloud: {medium: {name: 'gpt-oss:120b-cloud'}}\n"
        "jobs:\n  extract: medium\ndefaults: {temperature: 0.0}\n"
    )
    env = {"BEDROCK_LITREVIEW_CONFIG_DIR": str(tmp_path)}
    with pytest.raises(ConfigError, match="run off this machine"):
        load_settings(bedrock(), model_set="cloud", env=env)

    settings = load_settings(bedrock(), model_set="cloud", env=env, check_endpoint=False)
    assert settings.model_for("extract") == "gpt-oss:120b-cloud"


def test_a_non_zero_temperature_is_refused_whatever_the_endpoint_check_says(tmp_path):
    (tmp_path / "models.yaml").write_text(
        "min_context_window: 12288\nmodel_sets:\n  x: {medium: {name: m}}\n"
        "jobs:\n  extract: medium\ndefaults: {temperature: 0.7}\n"
    )
    with pytest.raises(ConfigError, match="temperature must be 0.0"):
        load_settings(
            bedrock(), model_set="x", check_endpoint=False,
            env={"BEDROCK_LITREVIEW_CONFIG_DIR": str(tmp_path)},
        )


# ---------------------------------------------------------------------------
# Reading the database without a model set
# ---------------------------------------------------------------------------


def test_the_database_commands_do_not_need_a_model_set_to_exist(tmp_path):
    """`runs`, `verify-run`, `rate` and `/reviews` need `db_path` and nothing else.

    A site whose models.yaml override defines only its own set would otherwise
    be told its machine profile's default set is unknown, and be unable to list
    or re-prove any of its past runs.
    """
    (tmp_path / "models.yaml").write_text(
        "min_context_window: 12288\nmodel_sets:\n  house: {medium: {name: house-model}}\n"
        "jobs:\n  extract: medium\ndefaults: {temperature: 0.0}\n"
    )
    env = {"BEDROCK_LITREVIEW_CONFIG_DIR": str(tmp_path)}

    with pytest.raises(ConfigError, match="unknown model set"):
        load_settings(bedrock(), env=env, check_endpoint=False)

    settings = load_settings(bedrock(), env=env, check_endpoint=False, require_models=False)
    assert settings.db_path.name == "litreview.sqlite3"
    assert settings.models == {}
    # ... and it still refuses to invent a model name.
    with pytest.raises(ConfigError):
        settings.model_for("extract")


def test_rating_a_report_does_not_need_a_model_set_to_exist(tmp_path, monkeypatch):
    """`rate` goes through `rate_report`, which loads its own settings when
    given none. It gets as far as looking the report up, not "unknown model set"."""
    from bedrock.litreview.api import rate_report

    (tmp_path / "models.yaml").write_text(
        "min_context_window: 12288\nmodel_sets:\n  house: {medium: {name: house-model}}\n"
        "jobs:\n  extract: medium\ndefaults: {temperature: 0.0}\n"
    )
    monkeypatch.setenv("BEDROCK_LITREVIEW_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("BEDROCK_LITREVIEW_DIR", str(tmp_path / "data"))
    with pytest.raises(KeyError, match="no report"):
        rate_report("nope", 4, rater="jd")
