CREATE TABLE workspaces (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created INTEGER NOT NULL
);

CREATE TABLE workspace_members (
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    user_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('owner', 'editor', 'viewer')),
    joined INTEGER NOT NULL,
    PRIMARY KEY (workspace_id, user_id)
);
CREATE INDEX members_user ON workspace_members(user_id);

CREATE TABLE workspace_invites (
    id TEXT PRIMARY KEY NOT NULL,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    token_hash TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('editor', 'viewer')),
    created_by TEXT NOT NULL,
    expires INTEGER NOT NULL,
    used_by TEXT,
    revoked INTEGER,
    created INTEGER NOT NULL
);
CREATE INDEX invites_workspace ON workspace_invites(workspace_id);

CREATE TABLE workspace_tokens (
    id TEXT PRIMARY KEY NOT NULL,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    name TEXT NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('read', 'write')),
    created_by TEXT NOT NULL,
    created INTEGER NOT NULL,
    expires INTEGER NOT NULL,
    revoked INTEGER
);
CREATE INDEX tokens_workspace ON workspace_tokens(workspace_id);

CREATE TABLE workspace_artifacts (
    id TEXT PRIMARY KEY NOT NULL,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    kind TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    content TEXT NOT NULL,
    provenance TEXT NOT NULL,
    parent_ids TEXT NOT NULL,
    created_by TEXT NOT NULL,
    author_name TEXT NOT NULL,
    created INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    request_key TEXT NOT NULL,
    request_hash TEXT NOT NULL
);
CREATE UNIQUE INDEX artifacts_idempotency
    ON workspace_artifacts(workspace_id, created_by, request_key);
CREATE INDEX artifacts_workspace_created
    ON workspace_artifacts(workspace_id, created, id);

CREATE TABLE workspace_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    created INTEGER NOT NULL
);
CREATE INDEX events_workspace_sequence
    ON workspace_events(workspace_id, sequence);
