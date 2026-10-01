"""Builds the right backend for the current settings.

There is one model-server *protocol* today: OpenAI-compatible chat
completions, which Ollama, vLLM and SGLang all speak. `create_backend` builds
an `OpenAICompatibleBackend` when `settings.backend == "openai"`.

The agent-pack backend is not that protocol at all - it has sessions, multiple
agents working one request, and human-approval gates a single "send messages,
get a reply" client cannot represent (see the module docstring of
`bedrock.agentpack_app`). `bedrock` itself talks to it through
`AgentPackClient` (`bedrock.agentpack.client`), never through an
`InferenceBackend`. So `create_backend` refuses `settings.backend ==
"agentpack"` explicitly, with `UnsupportedBackendError`, instead of silently
building an OpenAI client against an address that answers none of the calls
it would send. Any other value is refused the same way, naming the backends
that are actually supported.

If a server that speaks a genuinely different *model* protocol is ever
needed, add a class in this package and pick it here - the UI never imports a
concrete backend.
"""

from __future__ import annotations

from bedrock.config import Settings
from bedrock.inference.base import InferenceBackend
from bedrock.inference.openai_compatible import OpenAICompatibleBackend


class UnsupportedBackendError(ValueError):
    """`settings.backend` names a mode `create_backend` cannot build.

    `ValueError` is the base on purpose, not an accident: callers written
    before this class existed already wrap `load_settings` (and sometimes
    `create_backend`) in `except (ValueError, TypeError, OSError)`, and those
    keep working unchanged - this is just a more specific error inside that
    family, for the one caller that wants to say something more precise than
    "invalid configuration".
    """


def create_backend(settings: Settings) -> InferenceBackend:
    if settings.backend == "openai":
        return OpenAICompatibleBackend(
            base_url=settings.api_base,
            api_key=settings.api_key,
            timeout=settings.request_timeout,
        )
    if settings.backend == "agentpack":
        raise UnsupportedBackendError(
            "create_backend cannot build a model client for backend = 'agentpack': the "
            "agent-pack API is not an OpenAI-compatible model server. It has sessions, "
            "multiple agents and human-approval gates instead of a single chat-completions "
            "endpoint (see the module docstring of bedrock.agentpack_app). bedrock itself "
            "talks to it through AgentPackClient (bedrock.agentpack.client), not through "
            "InferenceBackend. Point this at an OpenAI-compatible model server instead: set "
            "backend = 'openai' and api_base to that server's address, e.g. "
            "api_base = 'http://vllm-host:8000/v1' in ~/.config/bedrock/config.toml "
            "(or BEDROCK_API_BASE)."
        )
    raise UnsupportedBackendError(
        f"create_backend does not know backend {settings.backend!r}; the known backends "
        "are 'openai' and 'agentpack'."
    )
