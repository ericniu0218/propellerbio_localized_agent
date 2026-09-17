"""Owner-scoped metadata and separate artifact storage on the company server.

This single-process SQLite adapter is replaceable without changing object IDs
or the HTTP contract. Project membership is metadata in the demo, not an ACL.
"""
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return secrets.token_hex(16)


def demo_mode():
    return os.getenv("DEMO_MODE", "true").lower() == "true"


class WorkspaceStore:
    def __init__(self, root=None):
        environment = "demo" if demo_mode() else "live"
        self.root = Path(root or os.getenv("WORKSPACE_DATA_DIR", "data/workspace")) / environment
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "artifacts"
        self.blobs.mkdir(exist_ok=True)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    kind TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS objects_owner ON objects(owner,kind);
                CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    name TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS audit (id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    created TEXT NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
                    BEGIN SELECT RAISE(ABORT,'append only'); END;
                CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
                    BEGIN SELECT RAISE(ABORT,'append only'); END;
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / "workspace.sqlite", timeout=15)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    def list(self, owner, kind):
        with self.db() as db:
            return [json.loads(r[0]) for r in db.execute(
                "SELECT data FROM objects WHERE owner=? AND kind=?", (owner, kind))]

    def get(self, owner, object_id, kind=None):
        with self.db() as db:
            row = db.execute("SELECT kind,data FROM objects WHERE owner=? AND id=?",
                             (owner, object_id)).fetchone()
            return json.loads(row[1]) if row and (kind is None or kind == row[0]) else None

    def put(self, owner, kind, data):
        data = dict(data)
        data.setdefault("id", uid())
        data.setdefault("createdAt", now())
        data.update(updatedAt=now(), ownerId=owner, organizationId="development" if demo_mode() else "company")
        with self.db() as db:
            # The conflict clause never lets another owner replace this ID.
            db.execute('''INSERT INTO objects VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE
                SET data=excluded.data WHERE objects.owner=excluded.owner AND objects.kind=excluded.kind''',
                (data["id"], owner, kind, json.dumps(data)))
        return data

    def delete(self, owner, object_id):
        with self.db() as db:
            db.execute("DELETE FROM objects WHERE owner=? AND id=?", (owner, object_id))

    def audit(self, owner, action, target):
        with self.db() as db:
            db.execute("INSERT INTO audit VALUES (?,?,?,?,?)", (uid(), owner, now(), action, target))

    def audit_list(self, owner):
        with self.db() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM audit WHERE owner=? ORDER BY created DESC LIMIT 200", (owner,))]

    def seed(self, owner):
        if not demo_mode() or self.list(owner, "preferences"):
            return
        project = self.put(owner, "projects", {"name": "KRAS G12D Program", "description":
            "Evaluate binding selectivity and plan the next assay series. Fictional demonstration project.",
            "members": [{"name": "Maya Chen (you)", "role": "Owner"}, {"name": "Alex Rivera", "role": "Scientist"}],
            "scope": "Project", "demo": True})
        self.put(owner, "memory", {"title": "Assay reporting conventions", "content":
            "Report concentrations in μM and viability as a percentage. Distinguish technical replicates from biological replicates. Flag missing controls.",
            "scope": "Project", "projectId": project["id"], "demo": True})
        meeting = self.put(owner, "meetings", {"title": "KRAS research weekly", "projectId": project["id"],
            "transcript": "[00:00:12] Maya: The control plate needs another run.\n[00:01:04] Alex: Decision: repeat the assay with a wider concentration range.\n[00:02:20] Maya: Action: I will prepare the samples by Friday.\n[00:03:10] Alex: We still need to check whether the binding change is reproducible.",
            "summary": "The team reviewed controls and planned a repeat assay. This is fictional demo content.",
            "decisions": ["Repeat the assay with a wider concentration range."],
            "actions": ["Maya: prepare samples by Friday."], "questions": ["Is the binding change reproducible?"],
            "people": ["Maya", "Alex"], "findings": ["Control plate requires a repeat; no safety conclusion established."],
            "deadlines": ["Friday — confirm calendar date"], "demo": True})
        self.put(owner, "tasks", {"title": "Prepare repeat-assay samples", "projectId": project["id"],
            "sourceId": meeting["id"], "due": "Friday — date to confirm", "status": "open", "demo": True})
        self.put(owner, "chats", {"title": "Planning the next assay", "projectId": project["id"],
            "messages": [{"role": "assistant", "text": "Your project conventions are available here. Attach the assay export to begin; I’ll keep the file with this conversation.", "demo": True}],
            "artifactIds": [], "pinned": True, "archived": False})
        self.put(owner, "preferences", {"theme": "light", "buddy": True, "vocabulary":
            "KRAS\nG12D\nEGFR\nMAPK\nAlphaFold", "retention": "90 days (requested)", "seeded": True})
