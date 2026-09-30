"""First-run diagnostics. Never touches the real network or a real config
file — DEFAULT_CONFIG_PATH and _reachable are monkeypatched throughout.
"""

from pathlib import Path

from bedrock.config.settings import Settings
import bedrock.doctor as doctor


async def _always_reachable(url: str, timeout: float = 3.0) -> bool:
    return True


async def _never_reachable(url: str, timeout: float = 3.0) -> bool:
    return False


def test_no_issues_when_a_config_file_already_exists(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("backend = 'agentpack'\n")
    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", config)

    assert doctor.run_first_run_check(Settings()) == []


def test_flags_default_agentpack_address_when_unconfigured_and_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", tmp_path / "does-not-exist.toml")
    monkeypatch.setattr(doctor, "_reachable", _never_reachable)

    issues = doctor.run_first_run_check(Settings(backend="agentpack"))

    assert any("default address" in issue for issue in issues)
    assert any("Could not reach" in issue for issue in issues)


def test_no_reachability_issue_when_the_default_actually_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", tmp_path / "does-not-exist.toml")
    monkeypatch.setattr(doctor, "_reachable", _always_reachable)

    issues = doctor.run_first_run_check(Settings(backend="agentpack"))

    assert not any("Could not reach" in issue for issue in issues)
    assert any("default address" in issue for issue in issues)  # still notes it's unconfigured


def test_flags_default_openai_address_when_unconfigured_and_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", tmp_path / "does-not-exist.toml")
    monkeypatch.setattr(doctor, "_reachable", _never_reachable)

    issues = doctor.run_first_run_check(Settings(backend="openai"))

    assert any("Ollama" in issue for issue in issues)
    assert any("Could not reach" in issue for issue in issues)


def test_biotech_tools_not_ready_are_noted_but_not_as_a_hard_error(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", tmp_path / "does-not-exist.toml")
    monkeypatch.setattr(doctor, "_reachable", _always_reachable)

    issues = doctor.run_first_run_check(
        Settings(backend="agentpack", biotech_work_dir=tmp_path / "biotech")
    )

    assert any("only needed if you plan to use /fold or /pymol" in issue for issue in issues)


def test_print_first_run_banner_prints_nothing_for_no_issues(capsys):
    doctor.print_first_run_banner([])
    assert capsys.readouterr().out == ""


def test_print_first_run_banner_prints_every_issue(capsys):
    doctor.print_first_run_banner(["issue one", "issue two"])
    out = capsys.readouterr().out
    assert "issue one" in out
    assert "issue two" in out
    assert str(doctor.DEFAULT_CONFIG_PATH) in out


def test_openai_first_run_prints_ollama_settings_matching_the_profile(tmp_path, monkeypatch):
    from bedrock.config import PROFILES

    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", tmp_path / "does-not-exist.toml")
    monkeypatch.setattr(doctor, "_reachable", _always_reachable)

    issues = doctor.run_first_run_check(Settings(backend="openai"))
    [note] = [issue for issue in issues if "OLLAMA_CONTEXT_LENGTH" in issue]
    assert "OLLAMA_CONTEXT_LENGTH=8192" in note  # the laptop default (D4)
    assert "OLLAMA_NUM_PARALLEL=1" in note
    assert "3-4B model at 4-bit" in note

    workstation = doctor.run_first_run_check(Settings(backend="openai", profile=PROFILES["workstation"]))
    assert any("OLLAMA_CONTEXT_LENGTH=32768" in issue for issue in workstation)


def test_ollama_settings_follow_a_context_window_override():
    from dataclasses import replace

    from bedrock.config import PROFILES

    lines = doctor.ollama_settings(replace(PROFILES["laptop"], context_window=4096))
    assert lines[0] == "OLLAMA_CONTEXT_LENGTH=4096"


def test_the_laptop_window_is_flagged_as_too_small_for_a_literature_review():
    """The window lives on the server, so this is worth saying at first run
    rather than only when someone starts a review."""
    from bedrock.config import PROFILES
    from bedrock.doctor import LITREVIEW_MIN_CONTEXT_WINDOW, server_settings_note

    laptop = server_settings_note(PROFILES["laptop"])
    assert "/review" in laptop
    assert f"OLLAMA_CONTEXT_LENGTH={LITREVIEW_MIN_CONTEXT_WINDOW}" in laptop

    # A machine that can already run one is not told to change anything.
    for name in ("workstation", "server"):
        assert "/review" not in server_settings_note(PROFILES[name])


def test_a_live_review_check_replaces_the_static_review_advice(tmp_path, monkeypatch):
    """Once the server answers, the review's own check - which knows the
    window the server really gives - speaks for /review, once."""
    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", tmp_path / "does-not-exist.toml")
    monkeypatch.setattr(doctor, "_reachable", _always_reachable)
    monkeypatch.setattr(doctor, "review_notes",
                        lambda settings: ["Literature reviews (/review) cannot run yet: the window is 4,096"])
    issues = doctor.run_first_run_check(Settings(backend="openai"))
    assert sum("/review" in issue for issue in issues) == 1
    assert any("4,096" in issue for issue in issues)
