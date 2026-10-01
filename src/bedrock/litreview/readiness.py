"""Can this machine run a literature review right now? Checked, not assumed.

`bedrock-litreview doctor` prints this; bedrock's first-run banner
(`bedrock/doctor.py`) shows the parts that fail. It answers two questions that
are easy to confuse, separately:

* **Is the server reachable?** In agent-pack mode, the agent server's
  `/health`, and whether it offers the inference API a review needs (an older
  server does not). Otherwise, the model server at `api_base`.
* **Is the literature review operational?** The same checks a review runs
  before its first call (`service.preflight`): the models the model set names
  are offered, each one's context window - as the server reports it, else as
  configured - holds an extraction call, and each can constrain its output to
  a JSON schema. Plus authentication, which fails as a 401 or 403.

Nothing here calls a model or changes anything: two GETs at most, `/health`
and `/v1/models`. The API key is never printed, only whether one is set.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from bedrock.litreview.config import ConfigError, load_settings


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool | None
    """True passed, False failed, None is information."""
    detail: str

    def line(self) -> str:
        mark = {True: "ok  ", False: "FAIL", None: "info"}[self.ok]
        return f"[{mark}] {self.name}: {self.detail}"


def operational(checks: list[Check]) -> bool:
    return bool(checks) and all(c.ok is not False for c in checks) and checks[-1].name == "Literature review" \
        and checks[-1].ok is True


async def _health(url: str) -> tuple[int, set[str]]:
    """(HTTP status, features) of the agent server's `/health`. A seam for tests."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url)
    features = set((response.json() or {}).get("features") or []) if response.status_code == 200 else set()
    return response.status_code, features


async def review_readiness(bedrock_settings=None, *, env=None, backend=None) -> list[Check]:
    """Every check, in order, stopping at the first one that makes the rest moot,
    with advice that does not stop a review set just before the verdict.

    The advice - a `request_timeout` too short for an extraction call, a nearly
    full disk, a company server name that may need `allowed_hosts` - is
    configuration and the disk, so it needs no server and is shown even when the
    server is down, which is when someone is most likely to be reading this. It
    is information, never failure: `operational()` ignores it.

    It goes *before the last check*, not at the top: the checks that matter keep
    the positions callers and tests read them at (the model server, then the
    agent server, then what went wrong), and the verdict stays last, which
    `operational()` requires.
    """
    checks = await _review_readiness(bedrock_settings, env=env, backend=backend)
    try:
        settings = load_settings(bedrock_settings, env=env)
    except ConfigError:
        return checks  # the configuration failure is already the finding
    from bedrock.litreview.service import config_findings

    # A review opens its own connection (and so has its timeout raised) unless
    # it shares the chat's, which only a default-mode review aimed at bedrock's
    # own `api_base` does.
    advice = [
        Check(f.title, True if f.ok else None, f.detail if f.ok else f"worth knowing: {f.detail}")
        for f in config_findings(settings, timeout_floor_applied=settings.endpoint_source != "api_base")
    ]
    return checks[:-1] + advice + checks[-1:]


async def _review_readiness(bedrock_settings=None, *, env=None, backend=None) -> list[Check]:
    """The checks that decide whether a review can run; see `review_readiness`."""
    from bedrock.inference import InferenceError
    from bedrock.litreview.service import preflight, review_backend

    checks: list[Check] = []
    try:
        settings = load_settings(bedrock_settings, env=env)
    except ConfigError as exc:
        return [Check("Configuration", False, str(exc)), Check("Literature review", False, "not operational")]

    where = {"agentpack": "the agent server's inference API", "litreview": "[litreview] api_base",
             "api_base": "bedrock's api_base"}[settings.endpoint_source]
    jobs = ", ".join(f"{job}={settings.model_for(job)}" for job in ("relevance", "extract", "write")
                     if job in settings.jobs)
    checks.append(Check("Model server", None, f"{settings.base_url} ({where}); API key "
                        f"{'set' if settings.api_key else 'not set'}; model set {settings.model_set!r}: {jobs}"))

    if settings.endpoint_source == "agentpack":
        health_url = settings.base_url[: -len("/v1")] + "/health" if settings.base_url.endswith("/v1") \
            else settings.base_url + "/health"
        try:
            status, features = await _health(health_url)
        except (httpx.HTTPError, ValueError):
            checks += [Check("Agent server", False, f"not reachable at {health_url}. Is it running, and is "
                                                    "`agent_pack_api_base` (AGENT_PACK_API_BASE) its address?"),
                       Check("Literature review", False, "not operational: no server to send model calls to")]
            return checks
        if status != 200:
            checks += [Check("Agent server", False, f"{health_url} answered HTTP {status}"),
                       Check("Literature review", False, "not operational")]
            return checks
        if "inference_api" not in features:
            checks += [Check("Agent server", True, "reachable"),
                       Check("Inference API", False, "this agent server does not offer the inference API a "
                                                     "review needs (it predates it). Update the server."),
                       Check("Literature review", False, "not operational: chat works, reviews cannot")]
            return checks
        checks.append(Check("Agent server", True, "reachable, and offers the inference API"))

    own = backend is None
    backend = backend or review_backend(settings, timeout_floor=0)
    try:
        try:
            offered = await backend.list_models()
        except InferenceError as exc:
            text = str(exc)
            if "401" in text or "403" in text:
                fix = ("set the client's api_key (BEDROCK_API_KEY, or BEDROCK_LITREVIEW_API_KEY) to the server's "
                       "BEDROCK_SERVER_INFERENCE_TOKEN" if settings.endpoint_source == "agentpack"
                       else "check api_key (BEDROCK_API_KEY)")
                checks += [Check("Authentication", False, f"the server refused this client: {fix}."),
                           Check("Literature review", False, "not operational")]
            else:
                checks += [Check("Model discovery", False, f"could not list models at {settings.base_url}: {text}"),
                           Check("Literature review", False, "not operational")]
            return checks
        described = ", ".join(
            f"{m.id}" + (f"={m.root}" if m.root else "") + (f" ({m.context_length:,} tokens)" if m.context_length else "")
            for m in offered) or "none"
        checks.append(Check("Model discovery", True, f"offers {described}"))
        try:
            capacity = await preflight(settings, backend)
        except ConfigError as exc:
            checks.append(Check("Literature review", False, f"not operational: {exc}"))
            return checks
        for note in capacity.notes:
            checks.append(Check("Note", None, note))
        unknown = [m for m in capacity.windows if not next((o.context_length for o in offered if o.id == m), None)]
        if unknown:
            checks.append(Check("Context window", None, f"{', '.join(unknown)}: the server does not report a window "
                                "(Ollama reports one only for a loaded model); the configured "
                                f"{settings.context_window:,} is used, and every call is checked as it runs."))
        sizes = ", ".join(f"{m} {w:,} tokens" for m, w in capacity.windows.items())
        checks.append(Check("Literature review", True, f"operational - {sizes}; structured output "
                                                        "and model selection checked"))
        return checks
    finally:
        if own:
            await backend.close()
