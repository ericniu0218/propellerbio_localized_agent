"""Settings: which models, which paths, which knobs.

The server itself is not configured here. It is bedrock's own server - the
`api_base`, `api_key` and `request_timeout` in `~/.config/bedrock/config.toml`
or the matching environment variables - and every review call waits its turn in
bedrock's `Scheduler` behind whatever the employee is typing. What this module
adds on top is the part that is specific to a literature review: which model
name does which job, how big a context window a run needs, where its database
and reports live, and the pipeline knobs.

Three things come from bedrock's own settings and are never restated here:
`base_url` (its `api_base`), `context_window` (the machine profile's), and the
API key and timeout the backend uses.

Model names come from `defaults/models.yaml`, shipped inside the package. A
site that needs different names drops a whole replacement file into
`~/.config/bedrock/litreview/models.yaml`; the two are never merged, so there
is always exactly one file to read to know what a run used, and the run records
which one it was. The same applies to `pipeline.yaml` and `report.yaml`. The
knowledge tables cannot be overridden this way: they decide what counts as the
exact entity a question asked about, so a change to them goes through code
review like any other code.

One check here is not a convenience: `assert_local_url` refuses a model server
that is not on the local network. This system is air-gapped, and a
misconfigured endpoint is how that guarantee quietly stops being true - the
pipeline would run perfectly well while sending unpublished research to
somebody else's machine. `assert_local_model` refuses Ollama's cloud models for
the same reason: the address is local, the model is not.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULTS_DIR = PACKAGE_DIR / "defaults"

# `evals/litreview/` sits beside `src/` in a source checkout and is deliberately
# not in the wheel: it is corpora and benchmark questions, not runtime data.
# Only the benchmark reads it, and only from a checkout.
EVALS_DIR = PACKAGE_DIR.parents[2] / "evals" / "litreview"

DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "bedrock" / "litreview"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "bedrock" / "litreview"

# The two directories above are module constants read at call time rather than
# baked into defaults, so tests can point them at a temporary directory and no
# test ever touches a real home directory.

#: Every key the `[litreview]` table in bedrock's config.toml may contain.
KNOWN_KEYS = ("model_set", "data_dir", "allowed_hosts", "contact_email", "entailment_check")

#: Which model set a machine profile gets when nothing says otherwise.
PROFILE_MODEL_SETS = {"server": "office"}
DEFAULT_MODEL_SET = "laptop"


class ConfigError(RuntimeError):
    pass


_PRIVATE_SUFFIXES = (".local", ".internal", ".lan", ".home.arpa")


def assert_local_url(url: str, allowed_hosts: tuple[str, ...] = ()) -> None:
    """Refuse any model endpoint that is not on this machine or the local network."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        raise ConfigError(f"model endpoint has no host: {url!r}")
    if host == "localhost" or host in allowed_hosts or host.endswith(_PRIVATE_SUFFIXES):
        return
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ConfigError(
            f"refusing model endpoint {url!r}: {host!r} is not a local address. "
            "This system is air-gapped. If this host really is on the company "
            "network, add it to `allowed_hosts` in the [litreview] table of "
            "~/.config/bedrock/config.toml, or to BEDROCK_LITREVIEW_ALLOWED_HOSTS."
        ) from None
    if not (ip.is_loopback or ip.is_private or ip.is_link_local):
        raise ConfigError(f"refusing model endpoint {url!r}: {ip} is a public address")


def assert_local_model(name: str, env: Mapping[str, str]) -> None:
    """Refuse Ollama's cloud models. They are served through the local Ollama
    at localhost, so the endpoint check passes — while every prompt, and every
    paper in it, is sent to ollama.com. The address is local; the model is not."""
    if (name.endswith("-cloud") or name.endswith(":cloud")) and env.get("LLM_ALLOW_CLOUD_MODELS") != "1":
        raise ConfigError(
            f"refusing model {name!r}: Ollama cloud models run off this machine. "
            "Set LLM_ALLOW_CLOUD_MODELS=1 only for public, non-client data."
        )


def config_file(name: str, env: Mapping[str, str] | None = None) -> Path:
    """Where one of the three YAML files is actually read from.

    A file of the same name in the override directory replaces the packaged
    default completely. Returning the path rather than the contents is what
    lets a run hash the file it really read (`runner.input_files`).
    """
    override = config_dir(env) / name
    return override if override.is_file() else DEFAULTS_DIR / name


def config_dir(env: Mapping[str, str] | None = None) -> Path:
    import os

    env = os.environ if env is None else env
    return Path(env.get("BEDROCK_LITREVIEW_CONFIG_DIR") or DEFAULT_CONFIG_DIR)


def private_dir(path: Path) -> Path:
    """Create a directory only this account can read.

    Research questions, the papers behind them and the reports written from
    them are sensitive, so the litreview directories follow bedrock's rule for
    conversations: 0700 on the directory, 0600 on the files inside it.
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


@dataclass
class ModelSpec:
    tier: str
    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Settings:
    """Everything one review run needs to know. Not frozen: the benchmark and
    the tests redirect `cache_dir`, `out_dir` and `db_path` after loading."""

    model_set: str
    base_url: str
    models: dict[str, ModelSpec]
    jobs: dict[str, str]
    context_window: int
    min_context_window: int
    data_dir: Path
    cache_dir: Path
    out_dir: Path
    db_path: Path
    pipeline: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 2048
    allowed_hosts: tuple[str, ...] = ()
    ncbi_api_key: str = ""
    contact_email: str = ""
    """Sent to NCBI with PubMed searches, as their terms ask. Use a generic
    mailbox: it travels with every query."""
    entailment_check: bool = False
    """Run the optional local entailment check in Stage 7
    (`report/entailment.py`). Off by default: it is unmeasured, and a false
    drop loses real evidence."""

    def model_for(self, job: str) -> str:
        tier = self.jobs.get(job, "medium")
        if tier not in self.models:
            raise ConfigError(f"job {job!r} uses tier {tier!r}, which model set {self.model_set!r} lacks")
        return self.models[tier].name

    def model_options(self, name: str) -> dict[str, Any]:
        for spec in self.models.values():
            if spec.name == name:
                return spec.options
        return {}

    def knob(self, path: str, default: Any) -> Any:
        """`settings.knob("rank.max_graded", 40)`."""
        node: Any = self.pipeline
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_settings(
    bedrock_settings=None,
    *,
    model_set: str | None = None,
    env: Mapping[str, str] | None = None,
    check_endpoint: bool = True,
    require_models: bool = True,
) -> Settings:
    """Read the settings for one review. Touches no disk beyond the YAML files.

    `bedrock_settings` is bedrock's own `Settings`; when it is None this loads
    them itself, so a shell command and a running app read the same config.toml
    and the same environment.

    `require_models=False` is for the commands that only read the database -
    `runs`, `verify-run`, `rate`, `/reviews`. They need `db_path` and nothing
    about models, and a site whose `models.yaml` override defines only its own
    set should still be able to list and re-prove its past runs rather than
    being told its machine profile's default set does not exist. `models` comes
    back empty there, so `model_for` still refuses to invent a name.
    """
    import os

    from bedrock.config import load_settings as load_bedrock_settings

    env = dict(os.environ if env is None else env)
    if bedrock_settings is None:
        bedrock_settings = load_bedrock_settings(env=env)

    table = _table(bedrock_settings)
    models_path = config_file("models.yaml", env)
    pipeline_path = config_file("pipeline.yaml", env)
    models_cfg = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
    pipeline = (yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) if pipeline_path.exists() else {}) or {}

    # Argument, then environment, then the [litreview] table, then the machine
    # profile. The profile decides last because it describes the machine, not
    # the job: a big server may still be asked for the laptop model set.
    profile_default = PROFILE_MODEL_SETS.get(bedrock_settings.profile.name, DEFAULT_MODEL_SET)
    name = (
        model_set
        or env.get("BEDROCK_LITREVIEW_MODELS")
        or table.get("model_set")
        or profile_default
    )
    sets = models_cfg.get("model_sets", {})
    if name not in sets and require_models:
        raise ConfigError(f"unknown model set {name!r}; known: {sorted(sets)}")

    models = {}
    for tier, body in (sets.get(name) or {}).items():
        body = dict(body or {})
        models[tier] = ModelSpec(tier=tier, name=str(body.pop("name")), options=body)

    defaults = models_cfg.get("defaults", {}) or {}
    temperature = float(defaults.get("temperature", 0.0))
    if temperature != 0.0:
        # Not configurable on purpose. See CLAUDE.md.
        raise ConfigError("temperature must be 0.0 — determinism is a client requirement")

    data_dir = Path(
        env.get("BEDROCK_LITREVIEW_DIR")
        or table.get("data_dir")
        or DEFAULT_DATA_DIR
    ).expanduser()

    allowed = _allowed_hosts(env, table)
    settings = Settings(
        model_set=name,
        base_url=bedrock_settings.api_base,
        models=models,
        jobs=dict(models_cfg.get("jobs", {}) or {}),
        context_window=int(bedrock_settings.profile.context_window),
        min_context_window=int(models_cfg.get("min_context_window", 12288)),
        data_dir=data_dir,
        cache_dir=data_dir / "cache" / "llm",
        out_dir=data_dir / "reports",
        db_path=data_dir / "litreview.sqlite3",
        pipeline=pipeline,
        temperature=temperature,
        seed=int(defaults.get("seed", 0)),
        max_tokens=int(defaults.get("max_tokens", 2048)),
        allowed_hosts=allowed,
        ncbi_api_key=env.get("NCBI_API_KEY", ""),
        contact_email=str(env.get("SOURCES_CONTACT_EMAIL") or table.get("contact_email") or ""),
        entailment_check=_flag(env.get("BEDROCK_LITREVIEW_ENTAILMENT_CHECK"), table.get("entailment_check", False),
                               "entailment_check"),
    )
    if check_endpoint:
        assert_local_url(settings.base_url, allowed)
        for spec in models.values():
            assert_local_model(spec.name, env)
    return settings


def _table(bedrock_settings) -> dict[str, Any]:
    """The `[litreview]` table, checked for typos.

    Validated here rather than when bedrock starts, so a typo in this table
    stops a review and never stops the app - the same way connectors behave.
    """
    table = dict(getattr(bedrock_settings, "litreview", None) or {})
    unknown = sorted(set(table) - set(KNOWN_KEYS))
    if unknown:
        raise ConfigError(
            f"unknown key(s) in the [litreview] table: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(KNOWN_KEYS)}."
        )
    return table


def _flag(env_value: str | None, table_value: Any, key: str) -> bool:
    """An on/off setting: the environment variable, else the table's value.

    Strict on purpose. TOML `entailment_check = "false"` is a string, and a
    string is truthy in Python: read loosely, it would switch the check ON.
    """
    if env_value is not None:
        word = env_value.strip().lower()
        if word in ("1", "true", "yes", "on"):
            return True
        if word in ("", "0", "false", "no", "off"):
            return False
        raise ConfigError(f"{key}: expected 1 or 0 in the environment, got {env_value!r}")
    if not isinstance(table_value, bool):
        raise ConfigError(f"{key} in the [litreview] table must be true or false, got {table_value!r}")
    return table_value


def _allowed_hosts(env: Mapping[str, str], table: Mapping[str, Any]) -> tuple[str, ...]:
    raw = env.get("BEDROCK_LITREVIEW_ALLOWED_HOSTS")
    values = raw.split(",") if raw is not None else (table.get("allowed_hosts") or [])
    if isinstance(values, str):
        values = values.split(",")
    return tuple(str(h).strip().lower() for h in values if str(h).strip())
