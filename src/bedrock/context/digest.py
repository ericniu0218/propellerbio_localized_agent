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
"""

from __future__ import annotations

from bedrock.context.store import Artifact, ContextStore
from bedrock.context.types import Chunk, Digest, Extraction

# Data types where the computed overview is already the best summary there is,
# so the model pass is skipped entirely. Asking a model to describe a table
# whose exact schema and column statistics we just calculated spends real time
# to produce something vaguer than what we already have. Prose is the opposite
# case - there is no structure to read, so a written summary is the only option
# - which is why it is the one kind not listed here.
RULES_SUFFICE = frozenset({"tabular", "structured", "logs", "code", "fasta"})

MAX_PROMPT_CHARS = 6000
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
    backend,
    model: str,
    *,
    store: ContextStore | None = None,
    max_chunks: int = MAX_SAMPLE_CHUNKS,
) -> Digest:
    """Ask a local model to improve an artifact's digest.

    Returns the upgraded digest, or the artifact's existing one unchanged if
    anything at all went wrong. When ``store`` is given the result is written
    back, so the better digest is what everyone gets from then on.
    """
    sample = _sample_text(artifact, max_chunks)
    prompt = PROMPT.format(content=sample, max_points=MAX_KEY_POINTS)

    try:
        reply = ""
        async for fragment in backend.stream_chat(model, [{"role": "user", "content": prompt}]):
            reply += fragment
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


def _sample_text(artifact: Artifact, max_chunks: int) -> str:
    """Overview plus a bounded slice of content, within the prompt budget.

    The overview goes in first and is never trimmed: it is the densest, most
    reliable thing we have, and on structured data it is worth more to the
    model than any number of raw rows.
    """
    parts = [artifact.digest.overview or artifact.digest.gist]
    used = len(parts[0])
    for index in range(min(max_chunks, artifact.chunk_count)):
        try:
            text = artifact.chunk(index)
        except (KeyError, IndexError, OSError):
            break
        if used + len(text) > MAX_PROMPT_CHARS:
            remaining = MAX_PROMPT_CHARS - used
            if remaining > 200:
                parts.append(text[:remaining])
            break
        parts.append(text)
        used += len(text)
    return "\n\n".join(part for part in parts if part)
