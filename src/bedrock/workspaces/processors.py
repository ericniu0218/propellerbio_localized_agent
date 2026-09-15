"""Deterministic adapters between Bedrock context and shared artifacts."""

from __future__ import annotations

from bedrock.context import ContextStore
from bedrock.workspaces.models import CORPUS_KIND, FOLD_KIND, Artifact, ArtifactInput
from bedrock.workspaces.validation import WorkspaceError, corpus_documents, validate_artifact_input


def corpus_from_context(
    store: ContextStore,
    references: list[str] | tuple[str, ...],
    *,
    title: str | None = None,
) -> ArtifactInput:
    """Build the exact result shown by the TUI's pre-publication preview.

    Each source contributes its compact digest plus a deterministic, bounded
    extract.  The complete compressed chunks remain in the publishing user's
    context store.  This is the native equivalent of the Gmail hub sharing
    reviewed summaries instead of copying private raw messages.
    """
    if not references:
        raise WorkspaceError(400, "Choose at least one stored digest to publish.")
    if len(references) > 200:
        raise WorkspaceError(400, "Publish at most 200 digests at once.")

    artifacts = []
    seen: set[str] = set()
    for reference in references:
        artifact = store.resolve(reference)
        if artifact is None:
            raise WorkspaceError(404, f"No stored digest matches `{reference}`.")
        if artifact.id not in seen:
            seen.add(artifact.id)
            artifacts.append(artifact)

    documents = []
    for artifact in artifacts:
        digest_text = artifact.digest_text().strip()
        if not digest_text:
            raise WorkspaceError(400, f"`{artifact.name}` has an empty digest.")
        extracts: list[str] = []
        remaining = 1_200
        for index in range(artifact.chunk_count):
            if remaining <= 0:
                break
            text = artifact.chunk(index).strip()
            if not text:
                continue
            selected = text[:remaining]
            extracts.append(selected)
            remaining -= len(selected)
        shared_text = digest_text
        if extracts:
            shared_text += "\n\nSelected source extract:\n" + "\n\n".join(extracts)
        documents.append(
            {
                "id": f"digest-{artifact.id}",
                "title": artifact.name[:240],
                "text": shared_text[:16_000],
                "source": f"bedrock digest {artifact.id}",
            }
        )

    display_title = title or (
        artifacts[0].name if len(artifacts) == 1 else f"{len(artifacts)} shared digests"
    )
    stored_bytes = len("\n\n".join(item["text"] for item in documents).encode("utf-8"))
    value = ArtifactInput(
        kind=CORPUS_KIND,
        schema_version=1,
        title=display_title,
        summary=(
            f"{len(documents)} reviewed Bedrock digest "
            f"{'extract' if len(documents) == 1 else 'extracts'}; full source chunks remain private."
        ),
        content={
            "documents": documents,
            "method": "bedrock-digest-extract-v1",
            "originalBytes": sum(artifact.stats.source_bytes for artifact in artifacts),
            "storedTextBytes": stored_bytes,
        },
        provenance={
            "adapter": "bedrock.context",
            "processor": "bedrock-digest-extract-v1",
            "sourceArtifactIds": [artifact.id for artifact in artifacts],
            "lossy": True,
        },
        parent_ids=(),
    )
    return validate_artifact_input(value)


def fold_corpus(parent: Artifact) -> ArtifactInput:
    """Group one stored corpus by source while retaining exact citations."""
    if parent.kind != CORPUS_KIND or parent.schema_version != 1:
        raise WorkspaceError(400, "This processor requires a v1 corpus.")
    documents = corpus_documents(parent.content.get("documents"))
    buckets: dict[str, list[dict[str, str]]] = {}
    for document in documents:
        buckets.setdefault(document["source"], []).append(document)
    groups = []
    for source, items in buckets.items():
        text = "\n".join(f"{item['title']}: {item['text'][:240]}" for item in items)[:12_000]
        groups.append(
            {
                "source": source,
                "documentIds": [item["id"] for item in items],
                "text": text,
            }
        )
    value = ArtifactInput(
        kind=FOLD_KIND,
        schema_version=1,
        title=f"{parent.title[:145]} · Fold",
        summary=(
            f"{len(documents)} documents grouped into {len(groups)} source "
            f"{'group' if len(groups) == 1 else 'groups'}."
        ),
        content={
            "method": "source-fold-v1",
            "groups": groups,
            "documentCount": len(documents),
        },
        provenance={
            "adapter": "bedrock.workspace",
            "processor": "source-fold-v1",
            "parentChecksum": parent.checksum,
            "lossy": True,
        },
        parent_ids=(parent.id,),
    )
    return validate_artifact_input(value)
