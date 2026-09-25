"""Settings: which models, which server, which paths, which knobs.

Read from `config/models.yaml`, `config/pipeline.yaml` and the environment
(`.env` is loaded if present, without overriding variables already set).

One check here is not a convenience: `assert_local_url` refuses a model server
that is not on the local network. This system is air-gapped, and a
misconfigured endpoint is how that guarantee quietly stops being true — the
pipeline would run perfectly well while sending unpublished research to
somebody else's machine.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    pass


_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: env.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, env) for v in value]
    return value


def load_dotenv(path: Path, env: dict[str, str]) -> None:
    """Minimal .env reader. Existing variables win; comments are ignored."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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
            "network, add it to LLM_ALLOWED_HOSTS."
        ) from None
    if not (ip.is_loopback or ip.is_private or ip.is_link_local):
        raise ConfigError(f"refusing model endpoint {url!r}: {ip} is a public address")


@dataclass
class ModelSpec:
    tier: str
    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Settings:
    profile: str
    backend: str
    base_url: str
    structured_output: str
    models: dict[str, ModelSpec]
    jobs: dict[str, str]
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 2048
    data_dir: Path = ROOT / "data"
    cache_dir: Path = ROOT / ".cache" / "llm"
    out_dir: Path = ROOT / "out"
    db_path: Path = ROOT / "data" / "bedrock.litreview.sqlite3"
    pipeline: dict[str, Any] = field(default_factory=dict)
    allowed_hosts: tuple[str, ...] = ()
    ncbi_api_key: str = ""
    contact_email: str = ""
    """Sent to NCBI with PubMed searches, as their terms ask. Use a generic
    mailbox: it travels with every query."""

    def model_for(self, job: str) -> str:
        tier = self.jobs.get(job, "medium")
        if tier not in self.models:
            raise ConfigError(f"job {job!r} uses tier {tier!r}, which profile {self.profile!r} lacks")
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
    root: Path | None = None,
    profile: str | None = None,
    env: dict[str, str] | None = None,
    *,
    check_endpoint: bool = True,
) -> Settings:
    root = Path(root or ROOT)
    env = dict(os.environ if env is None else env)
    load_dotenv(root / ".env", env)

    models_cfg = _expand(yaml.safe_load((root / "config" / "models.yaml").read_text()) or {}, env)
    pipeline_path = root / "config" / "pipeline.yaml"
    pipeline = yaml.safe_load(pipeline_path.read_text()) if pipeline_path.exists() else {}

    name = profile or env.get("LLM_PROFILE") or models_cfg.get("default_profile", "office")
    profiles = models_cfg.get("profiles", {})
    if name not in profiles:
        raise ConfigError(f"unknown LLM_PROFILE {name!r}; known: {sorted(profiles)}")
    prof = profiles[name]

    models = {}
    for tier, body in (prof.get("models") or {}).items():
        body = dict(body or {})
        model_name = body.pop("name")
        models[tier] = ModelSpec(tier=tier, name=str(model_name), options=body)

    defaults = models_cfg.get("defaults", {})
    temperature = float(defaults.get("temperature", 0.0))
    if temperature != 0.0:
        # Not configurable on purpose. See CLAUDE.md.
        raise ConfigError("temperature must be 0.0 — determinism is a client requirement")

    allowed = tuple(h.strip().lower() for h in env.get("LLM_ALLOWED_HOSTS", "").split(",") if h.strip())
    settings = Settings(
        profile=name,
        backend=str(prof.get("backend", "openai")),
        base_url=str(prof.get("base_url", "")),
        structured_output=str(prof.get("structured_output", "response_format")),
        models=models,
        jobs=dict(models_cfg.get("jobs", {})),
        temperature=temperature,
        seed=int(defaults.get("seed", 0)),
        max_tokens=int(defaults.get("max_tokens", 2048)),
        data_dir=Path(env.get("DATA_DIR", root / "data")),
        cache_dir=Path(env.get("CACHE_DIR", root / ".cache")) / "llm",
        out_dir=Path(env.get("OUT_DIR", root / "out")),
        db_path=Path(env.get("DB_PATH", Path(env.get("DATA_DIR", root / "data")) / "bedrock.litreview.sqlite3")),
        pipeline=pipeline or {},
        allowed_hosts=allowed,
        ncbi_api_key=env.get("NCBI_API_KEY", ""),
        contact_email=env.get("SOURCES_CONTACT_EMAIL", ""),
    )
    if check_endpoint and settings.backend != "fake":
        assert_local_url(settings.base_url, allowed)
        for spec in models.values():
            assert_local_model(spec.name, env)
    return settings


def assert_local_model(name: str, env: dict[str, str]) -> None:
    """Refuse Ollama's cloud models. They are served through the local Ollama
    at localhost, so the endpoint check passes — while every prompt, and every
    paper in it, is sent to ollama.com. The address is local; the model is not."""
    if (name.endswith("-cloud") or name.endswith(":cloud")) and env.get("LLM_ALLOW_CLOUD_MODELS") != "1":
        raise ConfigError(
            f"refusing model {name!r}: Ollama cloud models run off this machine. "
            "Set LLM_ALLOW_CLOUD_MODELS=1 only for public, non-client data."
        )
