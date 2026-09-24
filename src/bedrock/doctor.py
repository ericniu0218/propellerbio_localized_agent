"""First-run diagnostics for a bench scientist launching this for the
first time with no Python/CLI background.

Deliberately never blocks the app from starting — printed as a plain-
language banner, then the TUI launches anyway, since plenty of sessions
(``/digest``, ``/workspace``) need no live model connection at all and a
hard gate would be actively unhelpful for those. Only runs the network
checks when there is no config file yet (see ``settings.py``): once
someone has configured this once, repeating a multi-second reachability
check on every single launch would be a real, ongoing cost for no benefit.

In plain-model mode it also prints the Ollama server settings that match the
machine profile (``bedrock.config.profile``). The client cannot change how the
server was started, and the one setting that decides a laptop's memory - the
context window - lives on the server, so the employee (or IT) has to set it
there to match.
"""

from __future__ import annotations

import asyncio

import httpx

from bedrock.config.profile import MachineProfile
from bedrock.config.settings import (
    DEFAULT_AGENT_PACK_API_BASE,
    DEFAULT_API_BASE,
    DEFAULT_CONFIG_PATH,
    Settings,
)


async def _reachable(url: str, timeout: float = 3.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return response.status_code < 500
    except httpx.HTTPError:
        return False


def run_first_run_check(settings: Settings) -> list[str]:
    """Plain-language issues to show, or `[]` if there's nothing to flag
    (including: not a first run at all — see module docstring)."""
    if DEFAULT_CONFIG_PATH.exists():
        return []

    issues: list[str] = []

    if settings.backend == "agentpack":
        issues.append(
            f"No configuration file yet, so this is using the default address "
            f"({DEFAULT_AGENT_PACK_API_BASE}) for the company AI system — almost "
            f"certainly not the real one."
        )
        if not asyncio.run(_reachable(f"{settings.agent_pack_api_base}/health")):
            issues.append(
                f"Could not reach an AI system at {settings.agent_pack_api_base}. "
                f"Ask your team for the correct address, then add it to "
                f"{DEFAULT_CONFIG_PATH} as:\n"
                f'      agent_pack_api_base = "https://<the real address>"'
            )
        issues.extend(_biotech_tool_notes(settings))
    else:
        issues.append(
            f"No configuration file yet, so this is using the default address "
            f"({DEFAULT_API_BASE}, a local Ollama install) — almost certainly "
            f"not the right one for a company deployment."
        )
        if not asyncio.run(_reachable(f"{settings.api_base}/models")):
            issues.append(
                f"Could not reach a model server at {settings.api_base}. "
                f"Ask your team for the correct address, then add it to "
                f"{DEFAULT_CONFIG_PATH} as:\n"
                f'      api_base = "https://<the real address>"\n'
                f'      api_key = "<if your team gave you one>"'
            )
        issues.append(server_settings_note(settings.profile))

    return issues


def ollama_settings(profile: MachineProfile) -> list[str]:
    """The Ollama environment variables that match ``profile``, one per line.

    ``OLLAMA_CONTEXT_LENGTH`` is the window the server reserves memory for, so
    it must equal the profile's window. One parallel slot and one loaded
    model, because each extra slot is another full KV cache. Flash attention
    and an 8-bit cache roughly halve that cache's memory.
    """
    return [
        f"OLLAMA_CONTEXT_LENGTH={profile.context_window}",
        f"OLLAMA_NUM_PARALLEL={profile.max_parallel}",
        "OLLAMA_MAX_LOADED_MODELS=1",
        "OLLAMA_FLASH_ATTENTION=1",
        "OLLAMA_KV_CACHE_TYPE=q8_0",
    ]


def server_settings_note(profile: MachineProfile) -> str:
    """What to set on the model server so it matches the ``profile`` in use."""
    lines = "\n".join(f"      {line}" for line in ollama_settings(profile))
    advice = (
        f"Using the {profile.name} profile ({profile.context_window:,}-token window). "
        f"If Ollama runs on this computer, start it with these settings so its "
        f"memory matches:\n{lines}"
    )
    if profile.name == "laptop":
        advice += "\n    On an 8 GB laptop use a 3-4B model at 4-bit, e.g. `ollama pull llama3.2:3b`."
    advice += (
        f"\n    For vLLM, start it with --max-model-len {profile.context_window}. "
        'Choose another profile with profile = "laptop" | "workstation" | "server" in the config file.'
    )
    return advice


def _biotech_tool_notes(settings: Settings) -> list[str]:
    from bedrock.biotech import ProteinWorkflow

    notes = []
    for tool in ProteinWorkflow.from_settings(settings).status():
        if not tool.ready:
            reason = "; ".join(tool.issues) or "not configured"
            notes.append(
                f"{tool.name} isn't set up yet ({reason}) — only needed if you "
                f"plan to use /fold or /pymol; everything else works without it."
            )
    return notes


def print_first_run_banner(issues: list[str]) -> None:
    if not issues:
        return
    print("=" * 72)
    print("First run — a couple of things worth checking before you start:")
    print()
    for issue in issues:
        print(f"  - {issue}")
    print()
    print(f"Config file (create it if it doesn't exist yet): {DEFAULT_CONFIG_PATH}")
    print("Continuing anyway — fix this and restart anytime, or ask your team for help.")
    print("=" * 72)
    print()
