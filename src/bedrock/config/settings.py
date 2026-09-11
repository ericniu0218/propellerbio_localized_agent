"""Application settings.

Where settings come from, in order of priority (highest wins):

1. Environment variables (``BEDROCK_API_BASE``, ``BEDROCK_MODEL``, ``BEDROCK_API_KEY``, ...)
2. A config file at ``~/.config/bedrock/config.toml``
3. Built-in defaults, which point at a local Ollama server.

Why this design: employees must never have an inference endpoint hard-coded into
the client. In development the endpoint is a local Ollama instance; in production
IT sets ``BEDROCK_API_BASE`` (or ships a config file) pointing at the company's
private inference server. The rest of the app never needs to change.

``backend`` picks which of two very different things this client talks to:

- ``"openai"`` (default, unchanged from the original design): a single
  OpenAI-compatible model server (Ollama/vLLM/SGLang) — one message in, one
  streamed reply out. See ``inference/``.
- ``"agentpack"``: the biotech agent-pack backend — real sessions, multiple
  agents working a request, and human-approval gates. See ``agentpack/``
  and ``agentpack_app.py``. This is what IT points at the company's actual
  AI system; ``agent_pack_api_base`` is that system's URL, and ``user_id``
  identifies the employee until real SSO login is wired in (see
  docs/DEPLOYMENT.md in the agent-pack repo for the plan there).
"""

from __future__ import annotations

import getpass
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_API_BASE = "http://localhost:11434/v1"
DEFAULT_AGENT_PACK_API_BASE = "http://localhost:8000"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "bedrock" / "config.toml"
DEFAULT_BIOTECH_WORK_DIR = Path.home() / ".local" / "share" / "bedrock" / "biotech"


@dataclass(frozen=True)
class Settings:
    """Everything the client needs to know to talk to an inference server."""

    backend: str = "openai"
    """"openai" (a single model server) or "agentpack" (the full agent system)."""

    api_base: str = DEFAULT_API_BASE
    """Base URL of an OpenAI-compatible API, e.g. ``http://localhost:11434/v1``."""

    model: str | None = None
    """Model to use at startup. ``None`` means "ask the server and pick the first"."""

    api_key: str | None = None
    """Optional bearer token. Ollama ignores it; vLLM/SGLang can require one."""

    request_timeout: float = 120.0
    """Seconds to wait for a response before giving up."""

    system_prompt: str | None = None
    """Optional instruction sent at the start of every conversation."""

    agent_pack_api_base: str = DEFAULT_AGENT_PACK_API_BASE
    """Base URL of the agent-pack backend, used when backend="agentpack"."""

    user_id: str = field(default_factory=getpass.getuser)
    """Identifies the employee to the agent-pack backend. A stand-in until
    real SSO login is wired into this client — see the note above."""

    biotech_work_dir: Path = DEFAULT_BIOTECH_WORK_DIR
    """Private working directory for scientific inputs, outputs, logs, and manifests."""

    alphafold_launcher: Path | None = None
    """Path to AlphaFold 2's official ``docker/run_docker.py`` launcher."""

    alphafold_data_dir: Path | None = None
    """Path to the locally provisioned AlphaFold 2 model parameters and databases."""

    alphafold_python: str = sys.executable
    """Python executable belonging to the environment used by the AF2 Docker launcher."""

    alphafold_max_template_date: str | None = None
    """Required template cutoff date, recorded in every prediction manifest."""

    alphafold_docker_image: str | None = None
    """Pinned AlphaFold image tag or digest passed to the official launcher."""

    alphafold_database_version: str | None = None
    """Operator-assigned identifier for the provisioned database snapshot."""

    alphafold_model_preset: str = "monomer_ptm"
    alphafold_db_preset: str = "full_dbs"
    alphafold_gpu_devices: str = "all"
    alphafold_models_to_relax: str = "best"
    biotech_max_sequence_length: int = 2500

    pymol_executable: str = "pymol"
    """Open-source PyMOL executable used for both headless rendering and desktop viewing."""

    extra: dict = field(default_factory=dict)
    """Any unrecognised keys from the config file, kept for forward compatibility."""


def _read_config_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_settings(config_path: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Build a :class:`Settings` from defaults, config file, then environment.

    Both arguments exist so tests can supply their own values instead of
    touching the real home directory or process environment.
    """
    env = os.environ if env is None else env
    file_values = _read_config_file(config_path or DEFAULT_CONFIG_PATH)

    known = {
        "backend", "api_base", "model", "api_key", "request_timeout",
        "system_prompt", "agent_pack_api_base", "user_id", "biotech_work_dir",
        "alphafold_launcher", "alphafold_data_dir", "alphafold_python",
        "alphafold_max_template_date", "alphafold_docker_image",
        "alphafold_database_version", "alphafold_model_preset",
        "alphafold_db_preset", "alphafold_gpu_devices",
        "alphafold_models_to_relax", "biotech_max_sequence_length",
        "pymol_executable",
    }
    extra = {k: v for k, v in file_values.items() if k not in known}

    backend = env.get("BEDROCK_BACKEND") or file_values.get("backend") or "openai"
    api_base = env.get("BEDROCK_API_BASE") or file_values.get("api_base") or DEFAULT_API_BASE
    model = env.get("BEDROCK_MODEL") or file_values.get("model") or None
    api_key = env.get("BEDROCK_API_KEY") or file_values.get("api_key") or None
    timeout = float(env.get("BEDROCK_TIMEOUT") or file_values.get("request_timeout") or 120.0)
    system_prompt = file_values.get("system_prompt") or None
    agent_pack_api_base = (
        env.get("AGENT_PACK_API_BASE")
        or file_values.get("agent_pack_api_base")
        or DEFAULT_AGENT_PACK_API_BASE
    )
    user_id = env.get("BEDROCK_USER") or file_values.get("user_id") or getpass.getuser()
    biotech_work_dir = Path(
        env.get("BEDROCK_BIOTECH_WORK_DIR")
        or file_values.get("biotech_work_dir")
        or DEFAULT_BIOTECH_WORK_DIR
    ).expanduser()

    alphafold_launcher_value = env.get("ALPHAFOLD_LAUNCHER") or file_values.get("alphafold_launcher")
    alphafold_launcher = Path(alphafold_launcher_value).expanduser() if alphafold_launcher_value else None
    alphafold_data_value = env.get("ALPHAFOLD_DATA_DIR") or file_values.get("alphafold_data_dir")
    alphafold_data_dir = Path(alphafold_data_value).expanduser() if alphafold_data_value else None
    alphafold_python = (
        env.get("ALPHAFOLD_PYTHON") or file_values.get("alphafold_python") or sys.executable
    )
    alphafold_max_template_date = (
        env.get("ALPHAFOLD_MAX_TEMPLATE_DATE")
        or file_values.get("alphafold_max_template_date")
        or None
    )
    alphafold_docker_image = (
        env.get("ALPHAFOLD_DOCKER_IMAGE")
        or file_values.get("alphafold_docker_image")
        or None
    )
    alphafold_database_version = (
        env.get("ALPHAFOLD_DATABASE_VERSION")
        or file_values.get("alphafold_database_version")
        or None
    )
    alphafold_model_preset = (
        env.get("ALPHAFOLD_MODEL_PRESET")
        or file_values.get("alphafold_model_preset")
        or "monomer_ptm"
    )
    alphafold_db_preset = (
        env.get("ALPHAFOLD_DB_PRESET")
        or file_values.get("alphafold_db_preset")
        or "full_dbs"
    )
    alphafold_gpu_devices = (
        env.get("ALPHAFOLD_GPU_DEVICES")
        or file_values.get("alphafold_gpu_devices")
        or "all"
    )
    alphafold_models_to_relax = (
        env.get("ALPHAFOLD_MODELS_TO_RELAX")
        or file_values.get("alphafold_models_to_relax")
        or "best"
    )
    biotech_max_sequence_length = int(
        env.get("BEDROCK_MAX_SEQUENCE_LENGTH")
        or file_values.get("biotech_max_sequence_length")
        or 2500
    )
    pymol_executable = (
        env.get("PYMOL_EXECUTABLE") or file_values.get("pymol_executable") or "pymol"
    )

    return Settings(
        backend=backend,
        api_base=api_base.rstrip("/"),
        model=model,
        api_key=api_key,
        request_timeout=timeout,
        system_prompt=system_prompt,
        agent_pack_api_base=agent_pack_api_base.rstrip("/"),
        user_id=user_id,
        biotech_work_dir=biotech_work_dir,
        alphafold_launcher=alphafold_launcher,
        alphafold_data_dir=alphafold_data_dir,
        alphafold_python=alphafold_python,
        alphafold_max_template_date=alphafold_max_template_date,
        alphafold_docker_image=alphafold_docker_image,
        alphafold_database_version=alphafold_database_version,
        alphafold_model_preset=alphafold_model_preset,
        alphafold_db_preset=alphafold_db_preset,
        alphafold_gpu_devices=alphafold_gpu_devices,
        alphafold_models_to_relax=alphafold_models_to_relax,
        biotech_max_sequence_length=biotech_max_sequence_length,
        pymol_executable=pymol_executable,
        extra=extra,
    )
