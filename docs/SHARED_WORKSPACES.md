# Shared workspaces in the local Bedrock app

## What was ported

The durable-result model from the Gmail web reference is implemented in the
native terminal application:

> A workspace owns artifacts. A user or integration authors them. Every
> authorized member reads the same stored artifact.

An artifact is immutable JSON. `brief.corpus` contains reviewed Bedrock digest
extracts. `brief.fold` groups a corpus by source and cites its parent and
document IDs. Other trusted local tools can publish namespaced kinds such as
`lab.assay_summary` through `WorkspaceStore`.

The local port retains:

- owner, editor, and viewer authorization;
- single-use, expiring, revocable invitations;
- immutable results with same-workspace lineage;
- canonical SHA-256 content checksums and idempotent publication;
- read/write integration keys returned once and hashed at rest;
- deterministic corpus folding and exact document citations;
- metadata-only listing, bounded records, export, and resumable events.

## How the terminal flow differs

There is no React route, Gmail OAuth, browser session, D1 database, scheduled
inbox agent, or cloud deployment in this repository. The employee interface
remains Textual/Rich. A user selects a workspace with `/workspace use`, reads
results with `/workspace items` and `/workspace show`, and manages access with
the remaining `/workspace` commands.

Gmail's select-and-preview behavior maps to local context:

1. `/digest <path>` builds a private digest and compressed exact chunks.
2. `/workspace publish <digest...>` creates an in-memory preview only.
3. The preview contains the complete text that will enter the workspace: each
   digest plus a deterministic extract capped at 1,200 characters.
4. `/workspace confirm` persists that exact artifact. `/workspace cancel` or
   selecting another workspace discards the preview.

The complete source chunks never leave `BEDROCK_CONTEXT_DIR`.

## Storage and identity

`BEDROCK_WORKSPACE_DIR` contains `workspaces.sqlite3`. Migrations in
`src/bedrock/workspaces/migrations/` are applied once when a workspace command
first opens the vault. Artifact creation and its activity event commit in one
`BEGIN IMMEDIATE` transaction. SQLite's rollback journal and a busy timeout
support short concurrent operations from multiple local processes.

The default directory and database are created for the current OS account
only. A team deployment must point every participating process at the same
operator-provisioned private path, preserve real SQLite file locking, and set
filesystem ownership and permissions deliberately. Do not place the database
on a sync product that copies files independently or a filesystem that fakes
POSIX locks.

`BEDROCK_USER` and `BEDROCK_DISPLAY_NAME` stamp the current actor. Until the
planned company OIDC flow replaces `BEDROCK_USER`, a person who can both alter
that setting and open the vault is not cryptographically prevented from
claiming another ID. Treat the current role checks as workflow enforcement;
use host access controls as the privacy boundary. A service-backed workspace
API with verified identity is the correct future seam for mutually untrusted
machines.

## Integration use

An owner creates a secret with:

```text
/workspace token write Lab pipeline 30
```

The token is displayed once. A trusted Python producer opens only its scoped
workspace view:

```python
from bedrock.workspaces import ArtifactInput, WorkspaceStore

store = WorkspaceStore.for_integration_key("/private/bedrock/workspaces", token)
try:
    result = store.publish(workspace_id, artifact_input, idempotency_key)
finally:
    store.close()
```

Read keys can list, read, export through their own code, and consume events;
write keys can also publish. Keys cannot manage members, invitations, or other
keys, cannot cross into another workspace, and stop working immediately after
revocation or expiry.

## Invariants for future changes

- All artifact and parent lookups include `workspace_id`.
- Built-in `brief.*` schemas remain versioned and validated.
- Custom kinds remain namespaced and JSON-compatible.
- A revision is a new artifact with parent IDs; stored artifacts are never
  edited or deleted.
- `provenance` is caller-declared processing metadata. The store stamps the
  actual configured actor separately.
- An idempotency key identifies one logical operation. Reusing it with changed
  content fails rather than silently returning or overwriting a result.
- Invitation and integration tokens remain hashed at rest and are never shown
  again after creation.
- Keep the 256 KiB artifact bound. Add a private blob store before accepting
  larger payloads.

Run `pytest` after changes. `tests/test_workspaces.py` covers cross-user reads,
permissions, lineage, idempotency, keys, events, and validation;
`tests/test_workspace_command.py` covers the TUI preview/confirm boundary.
