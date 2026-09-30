"""First-run diagnostics for a bench scientist launching this for the
first time with no Python/CLI background.

Deliberately never blocks the app from starting — printed as a plain-
language banner, then the TUI launches anyway, since plenty of sessions
(``/digest``, ``/workspace``) need no live model connection at all and a
hard gate would be actively unhelpful for those. Only runs the network
checks when there is no config file yet (see ``settings.py``): once
someone has configured this once, repeating a multi-second reachability
check on every single launch would be a real, ongoing cost for no benefit.

In both modes, once the server answers, it also runs the literature review's
own readiness check (`bedrock.litreview.readiness`: models, window, structured
output, key) and says what stops `/review` - "reachable" is not the same as
"a review can run".

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

LITREVIEW_MIN_CONTEXT_WINDOW = 12288
"""The window a literature review needs. Kept as one number here rather than
imported, so the first-run banner does not load the review package to print a
sentence; `bedrock.litreview.config` reads the real one from
`defaults/models.yaml` when a run actually starts."""


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
        reachable = asyncio.run(_reachable(f"{settings.agent_pack_api_base}/health"))
        if not reachable:
            issues.append(
                f"Could not reach an AI system at {settings.agent_pack_api_base}. "
                f"Ask your team for the correct address, then add it to "
                f"{DEFAULT_CONFIG_PATH} as:\n"
                f'      agent_pack_api_base = "https://<the real address>"'
            )
        else:
            # Reachable is not the same as able to run a literature review:
            # that needs the server's inference API, a key it accepts, the
            # models the review's model set names, and a window large enough.
            # The local machine profile says nothing about any of it here.
            issues.extend(review_notes(settings))
        issues.extend(_biotech_tool_notes(settings))
    else:
        issues.append(
            f"No configuration file yet, so this is using the default address "
            f"({DEFAULT_API_BASE}, a local Ollama install) — almost certainly "
            f"not the right one for a company deployment."
        )
        reachable = asyncio.run(_reachable(f"{settings.api_base}/models"))
        if not reachable:
            issues.append(
                f"Could not reach a model server at {settings.api_base}. "
                f"Ask your team for the correct address, then add it to "
                f"{DEFAULT_CONFIG_PATH} as:\n"
                f'      api_base = "https://<the real address>"\n'
                f'      api_key = "<if your team gave you one>"'
            )
        # With a server to ask, the live review check replaces the profile's
        # static /review advice: it knows the window the server really gives.
        review = review_notes(settings) if reachable else None
        issues.append(server_settings_note(settings.profile, review_advice=review is None))
        issues.extend(review or [])

    return issues


def review_notes(settings: Settings) -> list[str]:  # tests: see tests/conftest.py no_live_review_check
    """What stops `/review` on this machine, from the same checks a review runs
    first (`bedrock.litreview.readiness`). Empty when a review can run.

    Imported here, not at the top: the review package is only loaded on a
    first run, and only once a server has answered."""
    try:
        from bedrock.litreview.readiness import operational, review_readiness
    except ImportError:  # pragma: no cover - an install without the review
        return []
    try:
        checks = asyncio.run(review_readiness(settings))
    except Exception as exc:  # noqa: BLE001 - the banner must never stop the app
        return [f"Could not check literature-review readiness: {exc}"]
    if operational(checks):
        return []
    failed = [c for c in checks if c.ok is False]
    return ["Literature reviews (/review) cannot run yet:\n" + "\n".join(f"      - {c.name}: {c.detail}" for c in failed)
            + "\n      Run `bedrock-litreview doctor` for the full check."]


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


def server_settings_note(profile: MachineProfile, *, review_advice: bool = True) -> str:
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
    if review_advice and profile.context_window < LITREVIEW_MIN_CONTEXT_WINDOW:
        # Worth saying here rather than only when a review is started: the
        # window is a server setting, so changing it means restarting Ollama,
        # and finding that out mid-review is the expensive way to learn it.
        advice += (
            f"\n    `/review` (the literature review) needs context_window >= "
            f"{LITREVIEW_MIN_CONTEXT_WINDOW:,}: its extraction step sends about "
            f"4,300 tokens of prompt and allows a 4,096-token reply. Set "
            f"context_window = {LITREVIEW_MIN_CONTEXT_WINDOW} and "
            f"OLLAMA_CONTEXT_LENGTH={LITREVIEW_MIN_CONTEXT_WINDOW}, or use the "
            "workstation profile. Everything else here works as it is."
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


_review_notes = review_notes
"""The real check, for tests that replace ``review_notes`` and want it back."""
