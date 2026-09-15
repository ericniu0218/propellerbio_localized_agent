"""Plain-language rendering of store results.

Pure functions returning Markdown, with no Textual imports - the same
convention as ``bedrock.ui.commands`` and ``bedrock.agentpack.friendly``. Both
apps have a ``/digest`` command and both call into here, so the wording stays
identical between the two modes and stays easy to test on its own.
"""

from __future__ import annotations

from bedrock.context.store import Artifact
from bedrock.context.types import ArtifactStats, SearchHit

PREVIEW_CHARS = 320


def format_bytes(count: int) -> str:
    for unit, size in (("GB", 1_000_000_000), ("MB", 1_000_000), ("kB", 1_000)):
        if count >= size:
            return f"{count / size:.1f} {unit}"
    return f"{count} B"


def format_ingested(artifact: Artifact) -> str:
    """The confirmation shown right after a file is compressed."""
    stats = artifact.stats
    return "\n".join(
        [
            f"**Compressed `{artifact.name}`** ({artifact.source_kind}) → `{artifact.id}`",
            "",
            f"- Storage: {format_bytes(stats.source_bytes)} → "
            f"{format_bytes(stats.stored_bytes)} ({stats.storage_ratio:.1f}x smaller)",
            f"- Context: ~{stats.source_tokens:,} tokens → ~{stats.digest_tokens:,} in the digest "
            f"({stats.context_ratio:.0f}x cheaper to read)",
            f"- {stats.chunk_count:,} chunks indexed for exact lookup",
            "",
            f"Use `/digest show {artifact.id}` to read it, or "
            f"`/digest search <words>` to find something specific.",
        ]
    )


def format_listing(artifacts: list[Artifact], totals: ArtifactStats) -> str:
    if not artifacts:
        return (
            "**Nothing stored yet.**\n\n"
            "Use `/digest <path>` to compress a file into the store. "
            "Text, Markdown, CSV/TSV, JSON, source code and log files are all handled."
        )
    lines = [f"**Stored context** ({len(artifacts)} item{'s' if len(artifacts) != 1 else ''})", ""]
    lines.append("| id | name | kind | on disk | digest |")
    lines.append("|---|---|---|---|---|")
    for artifact in artifacts:
        lines.append(
            f"| `{artifact.id}` | {artifact.name} | {artifact.source_kind} "
            f"| {format_bytes(artifact.stats.stored_bytes)} | ~{artifact.stats.digest_tokens:,} tok |"
        )
    lines.append("")
    lines.append(
        f"Total: {format_bytes(totals.source_bytes)} of source held in "
        f"{format_bytes(totals.stored_bytes)} ({totals.storage_ratio:.1f}x smaller), "
        f"readable as ~{totals.digest_tokens:,} tokens instead of ~{totals.source_tokens:,}."
    )
    return "\n".join(lines)


def format_digest(artifact: Artifact) -> str:
    source = (
        "computed from the file's structure"
        if artifact.digest.generated_by == "rules"
        else f"summarised by `{artifact.digest.generated_by}`"
    )
    return "\n".join(
        [
            f"**{artifact.name}** — `{artifact.id}` ({artifact.source_kind}, {source})",
            "",
            artifact.digest_text(),
            "",
            f"*{artifact.chunk_count:,} chunks available for exact lookup.*",
        ]
    )


def format_hits(query: str, hits: list[SearchHit]) -> str:
    if not hits:
        return (
            f"**No matches for “{query}”.**\n\n"
            "Search matches words, not meaning — try the terms that would "
            "literally appear in the data."
        )
    lines = [f"**{len(hits)} match{'es' if len(hits) != 1 else ''} for “{query}”**", ""]
    for hit in hits:
        where = f"{hit.artifact_name} · {hit.label}" if hit.label else hit.artifact_name
        lines.append(f"**{where}** (chunk {hit.chunk_index}, score {hit.score:.1f})")
        lines.append("")
        lines.append("```")
        lines.append(_preview(hit.text))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _preview(text: str) -> str:
    text = text.strip()
    if len(text) <= PREVIEW_CHARS:
        return text
    return text[:PREVIEW_CHARS].rstrip() + "\n…"
