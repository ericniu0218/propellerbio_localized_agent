"""Turning an extraction into the small thing a model actually reads.

There are two layers, and the first one always works:

1. **Rules.** The adapter already produced a deterministic ``overview`` - a
   table's schema and column statistics, a log file's repeated patterns, a
   module's function list. This costs nothing, never fails, and for structured
   data is usually a *better* summary than prose would be.
2. **Model.** If a local backend is available it is asked for a one-line gist
   and a few key points, given that overview plus a bounded sample of the
   content. This is what helps most on prose, where there is no structure to
   lean on and rules have nothing to say.

The model pass is strictly an upgrade. If there is no backend, or the server
is down, or the reply comes back unparseable, the rules digest stands and the
ingest still succeeds. Nothing in this pipeline is allowed to fail because a
model was unavailable - that is the difference between a store you can rely on
and one that is only as good as your GPU's uptime.

The model pass is a job the employee asked for (``Priority.USER_JOB``), so it
goes through the ``Scheduler``: it waits while a chat reply is streaming and
never runs alongside one on a laptop. It asks for a short reply
(``DIGEST_REPLY_TOKENS``) with thinking switched off, and its prompt is sized
in tokens against the machine profile's context window
(``DIGEST_WINDOW_SHARE``), so it cannot overflow the window on a small machine.
"""

from __future__ import annotations

from bedrock.config.profile import DEFAULT_PROFILE, PROFILES
from bedrock.context.store import Artifact, ContextStore
from bedrock.context.types import Chunk, Digest, Extraction
from bedrock.inference import Priority, RequestOptions, Scheduler, estimate_tokens
from bedrock.inference import tokens as token_estimate

# Data types where the computed overview is already the best summary there is,
# so the model pass is skipped entirely. Asking a model to describe a table
# whose exact schema and column statistics we just calculated spends real time
# to produce something vaguer than what we already have. Prose is the opposite
# case - there is no structure to read, so a written summary is the only option
# - which is why it is the one kind not listed here.
RULES_SUFFICE = frozenset({"tabular", "structured", "logs", "code", "fasta"})

DIGEST_WINDOW_SHARE = 0.25
"""The fraction of the context window a digest prompt may fill. On the laptop
profile that is about 2,000 tokens - roughly the 6,000 characters this used to
be capped at - and it grows with the window on bigger machines."""

DIGEST_REPLY_TOKENS = 300
"""The reply limit: a one-line gist and up to six short bullets fit easily."""

MIN_CONTENT_TOKENS = 50
"""A chunk trimmed to less than this is not worth sending."""

MAX_SAMPLE_CHUNKS = 6
MAX_KEY_POINTS = 6

PROMPT = """\
You are summarising a document so that it can be read cheaply later.

Reply in exactly this format and nothing else:

GIST: <one sentence, under 20 words, saying what this document is>
- <a specific, useful fact from the content>
- <another>

Give between 2 and {max_points} bullets. Use only what is in the content
below. Do not invent numbers. Do not add any other text.

--- CONTENT ---
{content}
--- END CONTENT ---
"""


def would_benefit_from_model(source_kind: str) -> bool:
    """Whether asking a model to summarise this kind of data is worth it."""
    return source_kind not in RULES_SUFFICE


def build_digest(extraction: Extraction, chunks: list[Chunk]) -> Digest:
    """The rules-only digest. No model, no network, always succeeds.

    The adapter's overview already starts with its own headline sentence, so
    that first line becomes the gist and the rest becomes the body - which
    keeps the two from repeating each other.
    """
    lines = extraction.overview.strip().splitlines()
    gist = lines[0].strip() if lines else f"{extraction.source_kind} document"
    body = "\n".join(lines[1:]).strip()
    return Digest(gist=gist, overview=body, key_points=[], generated_by="rules")


async def summarize(
    artifact: Artifact,
    scheduler,
    model: str,
    *,
    store: ContextStore | None = None,
    max_chunks: int = MAX_SAMPLE_CHUNKS,
) -> Digest:
    """Ask a local model to improve an artifact's digest.

    ``scheduler`` is the app's ``Scheduler``. A plain ``InferenceBackend``
    also works (``ingest_and_summarize`` and scripts pass one); it is given a
    scheduler of its own with the default profile.

    Returns the upgraded digest, or the artifact's existing one unchanged if
    anything at all went wrong. When ``store`` is given the result is written
    back, so the better digest is what everyone gets from then on.
    """
    if not isinstance(scheduler, Scheduler):
        scheduler = Scheduler(scheduler, PROFILES[DEFAULT_PROFILE])
    sample = _sample_text(artifact, max_chunks, content_budget(scheduler.profile.context_window))
    prompt = PROMPT.format(content=sample, max_points=MAX_KEY_POINTS)
    options = RequestOptions(max_tokens=DIGEST_REPLY_TOKENS, reasoning=False)

    try:
        reply = await scheduler.complete(
            model, [{"role": "user", "content": prompt}], priority=Priority.USER_JOB, options=options
        )
    except Exception:
        # Deliberately broad, and the one place in this package where that is
        # right: a better digest is a nice-to-have, and no failure in an
        # optional enrichment step should lose the artifact we already stored
        # successfully. InferenceError is the expected case; a malformed reply
        # or a backend raising something of its own must not be worse.
        return artifact.digest

    gist, key_points = parse_reply(reply)
    if not gist and not key_points:
        return artifact.digest

    upgraded = Digest(
        gist=gist or artifact.digest.gist,
        overview=artifact.digest.overview,
        key_points=key_points,
        generated_by=model,
    )
    if store is not None:
        store.update_digest(artifact.id, upgraded)
    return upgraded


def content_budget(context_window: int) -> int:
    """Tokens of document content a digest prompt may carry for this window."""
    return max(MIN_CONTENT_TOKENS, int(context_window * DIGEST_WINDOW_SHARE) - estimate_tokens(PROMPT))


def parse_reply(reply: str) -> tuple[str, list[str]]:
    """Pull the gist and bullets out of a model reply, leniently.

    Small local models wander off format constantly - they add preambles, use
    ``*`` instead of ``-``, or forget the ``GIST:`` label. So this takes what
    it recognises and ignores the rest rather than insisting on the format.
    """
    gist = ""
    points: list[str] = []
    for raw in reply.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not gist and line.upper().startswith("GIST:"):
            gist = line[len("GIST:") :].strip()
            continue
        if line.startswith(("- ", "* ", "• ")):
            point = line[2:].strip()
            if point and len(points) < MAX_KEY_POINTS:
                points.append(point)
    if not gist:
        # No label, but a single leading sentence is a reasonable stand-in.
        for raw in reply.splitlines():
            line = raw.strip()
            if line and not line.startswith(("- ", "* ", "• ")):
                gist = line
                break
    return gist, points


def _sample_text(artifact: Artifact, max_chunks: int, budget_tokens: int) -> str:
    """Overview plus a bounded slice of content, within ``budget_tokens``.

    The overview goes in first and is only trimmed if it alone would not fit:
    it is the densest, most reliable thing we have, and on structured data it
    is worth more to the model than any number of raw rows.
    """
    overview = artifact.digest.overview or artifact.digest.gist
    parts = [_trim_to_tokens(overview, budget_tokens)]
    used = estimate_tokens(parts[0])
    for index in range(min(max_chunks, artifact.chunk_count)):
        try:
            text = artifact.chunk(index)
        except (KeyError, IndexError, OSError):
            break
        cost = estimate_tokens(text)
        if used + cost > budget_tokens:
            remaining = budget_tokens - used
            if remaining > MIN_CONTENT_TOKENS:
                parts.append(_trim_to_tokens(text, remaining))
            break
        parts.append(text)
        used += cost
    return "\n\n".join(part for part in parts if part)


def _trim_to_tokens(text: str, budget_tokens: int) -> str:
    if estimate_tokens(text) <= budget_tokens:
        return text
    return text[: int(budget_tokens * token_estimate.characters_per_token())]
