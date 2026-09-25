"""Application settings.

Where settings come from, in order of priority (highest wins):

1. Environment variables (``BEDROCK_API_BASE``, ``BEDROCK_MODEL``, ``BEDROCK_API_KEY``,
   ``BEDROCK_HISTORY_DIR``, ``BEDROCK_SAVE_HISTORY``, ``BEDROCK_AUTO_CONTEXT``,
   ``BEDROCK_PROFILE``, ``BEDROCK_CONTEXT_WINDOW``)
2. A config file at ``~/.config/bedrock/config.toml``
3. Built-in defaults, which point at a local Ollama server.

Why this design: employees must never have an inference endpoint hard-coded into
the client. In development the endpoint is a local Ollama instance; in production
IT sets ``BEDROCK_API_BASE`` (or ships a config file) pointing at the company's
private inference server. The rest of the app never needs to change.
"""

from __future__ import annotations

import getpass
import os
import sys
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from bedrock.config.profile import DEFAULT_PROFILE, PROFILES, MachineProfile

DEFAULT_API_BASE = "http://localhost:11434/v1"
DEFAULT_AGENT_PACK_API_BASE = "http://localhost:8000"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "bedrock" / "config.toml"
DEFAULT_CONTEXT_DIR = Path.home() / ".local" / "share" / "bedrock" / "context"
DEFAULT_WORKSPACE_DIR = Path.home() / ".local" / "share" / "bedrock" / "workspaces"
DEFAULT_ACTION_HISTORY_DIR = Path.home() / ".local" / "share" / "bedrock" / "history"
DEFAULT_BIOTECH_WORK_DIR = Path.home() / ".local" / "share" / "bedrock" / "biotech"
DEFAULT_AGENT_ROSTER_PATH = Path.home() / ".config" / "bedrock" / "agents.toml"

AGENT_CAPABILITY_NAMES = (
    "echo",
    "files.search",
    "files.read",
    "sources.fetch",
    "digest.query",
    "web.fetch",
    "web.search",
    "calendar.create",
    "mail.draft",
    "mail.send",
    "files.write",
    "shell.run",
)

FALSE_VALUES = {"0", "false", "no", "off"}
TRUE_VALUES = {"1", "true", "yes", "on"}
"""Spellings of on and off accepted for true/false settings such as
``BEDROCK_SAVE_HISTORY`` and ``BEDROCK_AUTO_CONTEXT``."""


def default_history_dir(env: dict[str, str] | None = None) -> Path:
    """Where saved conversations live: ``~/.local/share/bedrock/conversations``.

    This follows the XDG convention, so an employee whose machine sets
    ``XDG_DATA_HOME`` (or an IT policy that redirects it) gets history in the
    place the rest of their tools use. Data, not configuration: these files are
    written by the app, not edited by hand.
    """
    env = os.environ if env is None else env
    base = env.get("XDG_DATA_HOME")
    root = Path(base) if base and Path(base).is_absolute() else Path.home() / ".local" / "share"
    return root / "bedrock" / "conversations"


def default_agent_settings() -> dict:
    """Return the complete, safe-by-default ``[agents]`` configuration.

    Dotted capability names are quoted table names in TOML, for example
    ``[agents.capabilities."files.read"]``.  Keeping every planned capability
    present but empty gives its adapter a stable place for configuration
    without enabling it; the separate roster decides what an agent may use.
    """
    return {
        "roster_path": str(DEFAULT_AGENT_ROSTER_PATH),
        "max_steps": 8,
        "max_seconds": 30.0,
        "max_bytes": 1_048_576,
        "capabilities": {name: {} for name in AGENT_CAPABILITY_NAMES},
    }


def _agent_settings(value: object) -> dict:
    """Merge a configured ``[agents]`` table over the complete defaults."""
    if value is None:
        return default_agent_settings()
    if not isinstance(value, dict):
        raise ValueError("agents must be a [agents] table")

    merged = default_agent_settings()
    configured = dict(value)
    capabilities = configured.pop("capabilities", None)
    merged.update(configured)
    if capabilities is not None:
        if not isinstance(capabilities, dict):
            raise ValueError("agents.capabilities must contain capability tables")
        merged["capabilities"].update(capabilities)
    return merged


@dataclass(frozen=True)
class Settings:
    """Everything the client needs to know to talk to an inference server."""

    backend: str = "openai"
    """Backend mode: ``openai`` or ``agentpack``."""

    api_base: str = DEFAULT_API_BASE
    """Base URL of an OpenAI-compatible API, e.g. ``http://localhost:11434/v1``."""

    model: str | None = None
    """Model to use at startup. ``None`` means "ask the server and pick the largest
    one that fits ``profile``" (see ``bedrock.inference.model_choice``)."""

    api_key: str | None = None
    """Optional bearer token. Ollama ignores it; vLLM/SGLang can require one."""

    request_timeout: float = 120.0
    """Seconds to wait for a response before giving up."""

    system_prompt: str | None = None
    """Optional instruction sent at the start of every conversation."""

    history_dir: Path = field(default_factory=lambda: default_history_dir())
    """Directory holding saved conversations."""

    save_history: bool = True
    """Whether conversations are written to disk at all.

    Set ``BEDROCK_SAVE_HISTORY=0`` (or ``save_history = false`` in the config
    file) for a shared or kiosk machine where nothing should be left behind."""

    auto_context: bool = True
    """Whether matching ``/digest`` passages are added to outgoing chat messages.

    Applies in both modes. The matching is by keyword, so it can attach a
    passage that merely shares a word with the question (see
    ``bedrock.context.auto_inject``). ``BEDROCK_AUTO_CONTEXT=0``, or
    ``auto_context = false`` in the config file, starts every window with it
    off. ``/digest auto on|off`` changes it for one window."""

    profile: MachineProfile = field(default_factory=lambda: PROFILES[DEFAULT_PROFILE])
    """Limits sized to the machine running the model: context window, reply
    limit, how many calls at once, the largest model to pick by itself. See
    ``bedrock.config.profile``. ``profile = "laptop"`` in the config file or
    ``BEDROCK_PROFILE``; ``context_window = N`` or ``BEDROCK_CONTEXT_WINDOW``
    overrides just the window."""

    agent_pack_api_base: str = DEFAULT_AGENT_PACK_API_BASE
    user_id: str = field(default_factory=getpass.getuser)
    display_name: str | None = None
    context_dir: str = str(DEFAULT_CONTEXT_DIR)
    workspace_dir: str = str(DEFAULT_WORKSPACE_DIR)
    """Directory containing the private SQLite shared-workspace vault."""

    action_history_dir: str = str(DEFAULT_ACTION_HISTORY_DIR)
    """Directory containing the private, metadata-only local action journal.

    Deliberately *not* ``history_dir``: that one holds saved conversations,
    which are the employee's prompts and answers. This one holds the
    compliance journal - identifiers, hashes, decisions and error codes, and
    never any content. Two different kinds of record with two different
    privacy stories, so they get two settings and two directories rather than
    one knob that silently moves both.
    """

    biotech_work_dir: Path = DEFAULT_BIOTECH_WORK_DIR
    alphafold_launcher: Path | None = None
    alphafold_data_dir: Path | None = None
    alphafold_python: str = sys.executable
    alphafold_max_template_date: str | None = None
    alphafold_docker_image: str | None = None
    alphafold_database_version: str | None = None
    alphafold_model_preset: str = "monomer_ptm"
    alphafold_db_preset: str = "full_dbs"
    alphafold_gpu_devices: str = "all"
    alphafold_models_to_relax: str = "best"
    biotech_max_sequence_length: int = 2500
    pymol_executable: str = "pymol"

    connectors: dict = field(default_factory=dict)
    """``[connectors.<name>]`` tables from the config file, one per company
    system of record (see ``bedrock.connectors``). Validated there rather than
    here, so a broken table is reported by ``/sources`` instead of stopping
    the app from starting."""

    litreview: dict = field(default_factory=dict)
    """The ``[litreview]`` table: which model set the literature review uses,
    where its data lives, and which extra hosts count as local. Validated by
    ``bedrock.litreview.config`` when a review runs rather than here, so a typo
    in it is reported by ``/review`` instead of stopping the app from
    starting - the same rule as ``connectors``."""

    agents: dict = field(default_factory=default_agent_settings)
    """The ``[agents]`` block: roster path, execution budgets and capability
    configuration. A configured capability is still unavailable unless an
    agent's roster entry names it explicitly."""

    extra: dict = field(default_factory=dict)
    """Any unrecognised keys from the config file, kept for forward compatibility."""


def _read_config_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _boolean(value: object, setting: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES | FALSE_VALUES:
            return normalized in TRUE_VALUES
    raise ValueError(f"{setting} must be true or false, not {value!r}")


MIN_CONTEXT_WINDOW = 1024
"""Below this there is no room for a conversation at all, so a smaller value
is almost certainly a typo (``8`` for ``8192``) and is refused."""


def _profile(env: dict[str, str], file_values: dict) -> MachineProfile:
    """The configured machine profile, with any ``context_window`` override applied."""
    name = env.get("BEDROCK_PROFILE") or file_values.get("profile") or DEFAULT_PROFILE
    if not isinstance(name, str) or name.strip().lower() not in PROFILES:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"profile must be one of {choices}, not {name!r}")
    profile = PROFILES[name.strip().lower()]

    window = env.get("BEDROCK_CONTEXT_WINDOW") or file_values.get("context_window")
    if window is None or window == "":
        return profile
    try:
        tokens = int(window)
    except (TypeError, ValueError):
        raise ValueError(f"context_window must be a whole number of tokens, not {window!r}") from None
    if isinstance(window, bool) or tokens < MIN_CONTEXT_WINDOW:
        raise ValueError(f"context_window must be at least {MIN_CONTEXT_WINDOW} tokens, not {window!r}")
    # A smaller window than the profile expects also needs a smaller reply
    # limit, or the reply alone could fill the prompt's room.
    return replace(profile, context_window=tokens, reply_limit=min(profile.reply_limit, tokens // 4))


def load_settings(config_path: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Build a :class:`Settings` from defaults, config file, then environment.

    Both arguments exist so tests can supply their own values instead of
    touching the real home directory or process environment.
    """
    env = os.environ if env is None else env
    file_values = _read_config_file(config_path or DEFAULT_CONFIG_PATH)

    known = {
        "backend",
        "api_base",
        "model",
        "api_key",
        "request_timeout",
        "system_prompt",
        "history_dir",
        "save_history",
        "auto_context",
        "action_history_dir",
        "agent_pack_api_base", "user_id", "display_name", "context_dir",
        "workspace_dir", "biotech_work_dir",
        "alphafold_launcher", "alphafold_data_dir", "alphafold_python",
        "alphafold_max_template_date", "alphafold_docker_image",
        "alphafold_database_version", "alphafold_model_preset",
        "alphafold_db_preset", "alphafold_gpu_devices",
        "alphafold_models_to_relax", "biotech_max_sequence_length",
        "pymol_executable", "connectors", "agents", "profile", "context_window",
        "litreview",
    }
    extra = {k: v for k, v in file_values.items() if k not in known}

    backend = env.get("BEDROCK_BACKEND") or file_values.get("backend") or "openai"
    api_base = env.get("BEDROCK_API_BASE") or file_values.get("api_base") or DEFAULT_API_BASE
    model = env.get("BEDROCK_MODEL") or file_values.get("model") or None
    api_key = env.get("BEDROCK_API_KEY") or file_values.get("api_key") or None
    timeout = float(env.get("BEDROCK_TIMEOUT") or file_values.get("request_timeout") or 120.0)
    system_prompt = file_values.get("system_prompt") or None

    history_dir_value = env.get("BEDROCK_HISTORY_DIR") or file_values.get("history_dir")
    if history_dir_value is not None and not isinstance(history_dir_value, (str, os.PathLike)):
        raise ValueError(f"history_dir must be a path, not {history_dir_value!r}")
    history_dir = Path(history_dir_value).expanduser() if history_dir_value else default_history_dir(env)

    action_history_dir = (
        env.get("BEDROCK_ACTION_HISTORY_DIR")
        or file_values.get("action_history_dir")
        or str(DEFAULT_ACTION_HISTORY_DIR)
    )

    save_history_env = env.get("BEDROCK_SAVE_HISTORY")
    if save_history_env is not None and save_history_env.strip():
        save_history = _boolean(save_history_env, "BEDROCK_SAVE_HISTORY")
    else:
        save_history = _boolean(file_values.get("save_history", True), "save_history")

    auto_context_env = env.get("BEDROCK_AUTO_CONTEXT")
    if auto_context_env is not None and auto_context_env.strip():
        auto_context = _boolean(auto_context_env, "BEDROCK_AUTO_CONTEXT")
    else:
        auto_context = _boolean(file_values.get("auto_context", True), "auto_context")

    agent_pack_api_base = env.get("AGENT_PACK_API_BASE") or file_values.get("agent_pack_api_base") or DEFAULT_AGENT_PACK_API_BASE
    user_id = env.get("BEDROCK_USER") or file_values.get("user_id") or getpass.getuser()
    display_name = env.get("BEDROCK_DISPLAY_NAME") or file_values.get("display_name") or user_id
    context_dir = env.get("BEDROCK_CONTEXT_DIR") or file_values.get("context_dir") or str(DEFAULT_CONTEXT_DIR)
    workspace_dir = env.get("BEDROCK_WORKSPACE_DIR") or file_values.get("workspace_dir") or str(DEFAULT_WORKSPACE_DIR)
    biotech_work_dir = Path(env.get("BEDROCK_BIOTECH_WORK_DIR") or file_values.get("biotech_work_dir") or DEFAULT_BIOTECH_WORK_DIR).expanduser()
    launcher_value = env.get("ALPHAFOLD_LAUNCHER") or file_values.get("alphafold_launcher")
    alphafold_launcher = Path(launcher_value).expanduser() if launcher_value else None
    data_value = env.get("ALPHAFOLD_DATA_DIR") or file_values.get("alphafold_data_dir")
    alphafold_data_dir = Path(data_value).expanduser() if data_value else None

    # Nested tables only come from the config file: a whole connector does
    # not fit in one environment variable. Secrets still stay out of the
    # file - a connector names the environment variable that holds its token.
    connectors = file_values.get("connectors") or {}
    litreview = file_values.get("litreview") or {}
    agents = _agent_settings(file_values.get("agents"))
    profile = _profile(env, file_values)

    return Settings(
        backend=backend,
        api_base=api_base.rstrip("/"),
        model=model,
        api_key=api_key,
        request_timeout=timeout,
        system_prompt=system_prompt,
        history_dir=history_dir,
        save_history=save_history,
        auto_context=auto_context,
        profile=profile,
        action_history_dir=action_history_dir,
        agent_pack_api_base=agent_pack_api_base.rstrip("/"),
        user_id=user_id,
        display_name=display_name,
        context_dir=context_dir,
        workspace_dir=workspace_dir,
        biotech_work_dir=biotech_work_dir,
        alphafold_launcher=alphafold_launcher,
        alphafold_data_dir=alphafold_data_dir,
        alphafold_python=env.get("ALPHAFOLD_PYTHON") or file_values.get("alphafold_python") or sys.executable,
        alphafold_max_template_date=env.get("ALPHAFOLD_MAX_TEMPLATE_DATE") or file_values.get("alphafold_max_template_date"),
        alphafold_docker_image=env.get("ALPHAFOLD_DOCKER_IMAGE") or file_values.get("alphafold_docker_image"),
        alphafold_database_version=env.get("ALPHAFOLD_DATABASE_VERSION") or file_values.get("alphafold_database_version"),
        alphafold_model_preset=env.get("ALPHAFOLD_MODEL_PRESET") or file_values.get("alphafold_model_preset") or "monomer_ptm",
        alphafold_db_preset=env.get("ALPHAFOLD_DB_PRESET") or file_values.get("alphafold_db_preset") or "full_dbs",
        alphafold_gpu_devices=env.get("ALPHAFOLD_GPU_DEVICES") or file_values.get("alphafold_gpu_devices") or "all",
        alphafold_models_to_relax=env.get("ALPHAFOLD_MODELS_TO_RELAX") or file_values.get("alphafold_models_to_relax") or "best",
        biotech_max_sequence_length=int(env.get("BEDROCK_MAX_SEQUENCE_LENGTH") or file_values.get("biotech_max_sequence_length") or 2500),
        pymol_executable=env.get("PYMOL_EXECUTABLE") or file_values.get("pymol_executable") or "pymol",
        connectors=connectors,
        litreview=litreview,
        agents=agents,
        extra=extra,
    )
