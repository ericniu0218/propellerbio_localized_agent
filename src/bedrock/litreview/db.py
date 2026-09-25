"""Storage. Questions, papers, runs, stage artifacts, reports, ratings.

SQLite. Nothing here needs a server, and an air-gapped machine is a bad place
to debug one. Rows hold pydantic models as JSON: the models are the schema,
and a migration is a model change.

This is also why the chat (Stage 9) will not "forget": everything a report was
built from lives here, not in a conversation.

Papers are kept twice. `papers` holds the current body of each paper, and a
question's current set (`question_papers`) points at it; both are replaced
when a question's papers are refreshed. `paper_versions` holds every body
ever saved, keyed by its content hash, and is never updated or deleted. A run
records the (paper id, hash) pairs it read — its manifest — so a past run can
be reopened on exactly the papers it was built from, whatever has happened to
the question's current set since.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from bedrock.litreview.models import Paper, QuestionSpec, Rating, Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (id TEXT PRIMARY KEY, text TEXT NOT NULL, spec TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS papers (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS paper_versions (
    paper_id TEXT NOT NULL, sha256 TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY (paper_id, sha256)
);
CREATE TABLE IF NOT EXISTS question_papers (
    question_id TEXT NOT NULL, paper_id TEXT NOT NULL, position INTEGER NOT NULL,
    PRIMARY KEY (question_id, paper_id)
);
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, question_id TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL, name TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, question_id TEXT NOT NULL, body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ratings (id INTEGER PRIMARY KEY AUTOINCREMENT, report_id TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bench_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, report_id TEXT NOT NULL, question_id TEXT NOT NULL,
    created TEXT NOT NULL, body TEXT NOT NULL
);
"""

Manifest = list[tuple[str, str]]
"""(paper id, sha256 of its body), in the order the run read the papers."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paper_sha256(paper: Paper) -> str:
    """Content hash of a paper: sha256 hex of its JSON body, as stored."""
    return _sha256(paper.model_dump_json())


class Database:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(_SCHEMA)
        self._backfill_paper_versions()

    def _backfill_paper_versions(self) -> None:
        """A database written before `paper_versions` existed: keep each
        current body as a version, so nothing already stored is unaddressable."""
        if self.conn.execute("SELECT 1 FROM paper_versions LIMIT 1").fetchone():
            return
        rows = self.conn.execute("SELECT id, body FROM papers ORDER BY id").fetchall()
        if not rows:
            return
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO paper_versions (paper_id, sha256, body) VALUES (?, ?, ?)",
                [(pid, _sha256(body), body) for pid, body in rows],
            )

    def close(self) -> None:
        self.conn.close()

    # questions and papers -------------------------------------------------------

    def save_question(self, spec: QuestionSpec) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO questions (id, text, spec) VALUES (?, ?, ?)",
                (spec.id, spec.text, spec.model_dump_json()),
            )

    def get_question(self, question_id: str) -> QuestionSpec | None:
        row = self.conn.execute("SELECT spec FROM questions WHERE id = ?", (question_id,)).fetchone()
        return QuestionSpec.model_validate_json(row[0]) if row else None

    def save_papers(self, question_id: str, papers: list[Paper]) -> Manifest:
        """Freeze the paper set for a question. Replaces any earlier set as the
        question's current one; every body is also kept as a version. Returns
        the manifest of what was saved."""
        manifest: Manifest = []
        with self.conn:
            self.conn.execute("DELETE FROM question_papers WHERE question_id = ?", (question_id,))
            for i, paper in enumerate(papers):
                body = paper.model_dump_json()
                sha = _sha256(body)
                self.conn.execute("INSERT OR REPLACE INTO papers (id, body) VALUES (?, ?)", (paper.id, body))
                self.conn.execute(
                    "INSERT OR IGNORE INTO paper_versions (paper_id, sha256, body) VALUES (?, ?, ?)",
                    (paper.id, sha, body),
                )
                self.conn.execute(
                    "INSERT INTO question_papers (question_id, paper_id, position) VALUES (?, ?, ?)",
                    (question_id, paper.id, i),
                )
                manifest.append((paper.id, sha))
        return manifest

    def save_paper_versions(self, papers: Iterable[Paper]) -> Manifest:
        """Keep these bodies as versions without touching any question's set."""
        manifest: Manifest = []
        with self.conn:
            for paper in papers:
                body = paper.model_dump_json()
                sha = _sha256(body)
                self.conn.execute(
                    "INSERT OR IGNORE INTO paper_versions (paper_id, sha256, body) VALUES (?, ?, ?)",
                    (paper.id, sha, body),
                )
                manifest.append((paper.id, sha))
        return manifest

    def get_paper_versions(self, manifest: Sequence[Sequence[str]]) -> list[Paper]:
        """The exact paper bodies a manifest names, in its order."""
        out: list[Paper] = []
        missing: list[tuple[str, str]] = []
        for paper_id, sha in manifest:
            row = self.conn.execute(
                "SELECT body FROM paper_versions WHERE paper_id = ? AND sha256 = ?", (paper_id, sha)
            ).fetchone()
            if row is None:
                missing.append((paper_id, sha))
            else:
                out.append(Paper.model_validate_json(row[0]))
        if missing:
            shown = ", ".join(f"({pid}, {sha})" for pid, sha in missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise ValueError(f"paper versions missing from the database: {shown}{more}")
        return out

    def paper_version_body(self, paper_id: str, sha: str) -> str | None:
        """The stored text of one paper version, exactly as saved."""
        row = self.conn.execute(
            "SELECT body FROM paper_versions WHERE paper_id = ? AND sha256 = ?", (paper_id, sha)
        ).fetchone()
        return row[0] if row else None

    def finished_run_ids(self) -> list[str]:
        """Runs that got as far as writing a report."""
        rows = self.conn.execute("SELECT body FROM runs ORDER BY id").fetchall()
        return [r.id for r in (Run.model_validate_json(b) for (b,) in rows) if r.stage_reached >= 7]

    def get_papers(self, question_id: str) -> list[Paper]:
        rows = self.conn.execute(
            "SELECT p.body FROM question_papers q JOIN papers p ON p.id = q.paper_id "
            "WHERE q.question_id = ? ORDER BY q.position",
            (question_id,),
        ).fetchall()
        return [Paper.model_validate_json(r[0]) for r in rows]

    # runs and artifacts ------------------------------------------------------------

    def save_run(self, run: Run) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs (id, question_id, body) VALUES (?, ?, ?)",
                (run.id, run.question_id, run.model_dump_json()),
            )

    def get_run(self, run_id: str) -> Run | None:
        row = self.conn.execute("SELECT body FROM runs WHERE id = ?", (run_id,)).fetchone()
        return Run.model_validate_json(row[0]) if row else None

    def put_artifact(self, run_id: str, name: str, body: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO artifacts (run_id, name, body) VALUES (?, ?, ?)", (run_id, name, body)
            )

    def get_artifact(self, run_id: str, name: str) -> str | None:
        row = self.conn.execute(
            "SELECT body FROM artifacts WHERE run_id = ? AND name = ?", (run_id, name)
        ).fetchone()
        return row[0] if row else None

    # reports, ratings, benchmarks ---------------------------------------------------

    def save_report(self, report_id: str, run_id: str, question_id: str, body: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO reports (id, run_id, question_id, body) VALUES (?, ?, ?, ?)",
                (report_id, run_id, question_id, body),
            )

    def get_report(self, report_id: str) -> str | None:
        row = self.conn.execute("SELECT body FROM reports WHERE id = ?", (report_id,)).fetchone()
        return row[0] if row else None

    def add_rating(self, rating: Rating) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO ratings (report_id, body) VALUES (?, ?)", (rating.report_id, rating.model_dump_json())
            )

    def ratings(self, report_id: str) -> list[Rating]:
        rows = self.conn.execute("SELECT body FROM ratings WHERE report_id = ? ORDER BY id", (report_id,)).fetchall()
        return [Rating.model_validate_json(r[0]) for r in rows]

    def add_bench_result(self, report_id: str, question_id: str, created: str, body: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO bench_results (report_id, question_id, created, body) VALUES (?, ?, ?, ?)",
                (report_id, question_id, created, body),
            )
