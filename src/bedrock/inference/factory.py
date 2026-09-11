"""Builds the right backend for the current settings.

Today there is exactly one backend, because Ollama, vLLM and SGLang all speak
the same OpenAI-compatible protocol. If a server with a different protocol is
ever needed, add a class in this package and pick it here - the UI never
imports a concrete backend.
"""

from __future__ import annotations

from bedrock.config import Settings
from bedrock.inference.base import InferenceBackend
from bedrock.inference.openai_compatible import OpenAICompatibleBackend


def create_backend(settings: Settings) -> InferenceBackend:
    return OpenAICompatibleBackend(
        base_url=settings.api_base,
        api_key=settings.api_key,
        timeout=settings.request_timeout,
    )
