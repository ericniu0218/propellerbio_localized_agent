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
from dataclasses import dataclass
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
CREATE TABLE IF NOT EXISTS integrity_observations (
    observation_id TEXT PRIMARY KEY, provider TEXT NOT NULL, retrieved_at TEXT NOT NULL,
    raw_sha256 TEXT, body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS integrity_subjects (
    subject_id TEXT NOT NULL, observation_id TEXT NOT NULL, PRIMARY KEY (subject_id, observation_id)
);
CREATE INDEX IF NOT EXISTS integrity_by_time ON integrity_observations (provider, retrieved_at);
CREATE TABLE IF NOT EXISTS integrity_raw (sha256 TEXT PRIMARY KEY, body BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS integrity_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id TEXT NOT NULL, decision TEXT NOT NULL,
    decided_by TEXT NOT NULL, decided_at TEXT NOT NULL, note TEXT
);
"""

MAX_RAW_BYTES = 5_000_000
"""The largest provider response kept verbatim. A lookup of a few hundred
identifiers is far below this; a larger body is recorded by hash only."""

Manifest = list[tuple[str, str]]
"""(paper id, sha256 of its body), in the order the run read the papers."""


@dataclass(frozen=True)
class DeletedRun:
    """What `Database.delete_run` actually removed, so a caller can say it
    precisely rather than claiming more than happened."""

    run_id: str
    question_id: str
    artifacts: int
    report_id: str | None
    report_deleted: bool
    """False when the run wrote no report, or when another run shares it."""
    report_shared_with: tuple[str, ...]
    """Other runs that wrote the identical report, and therefore still need it."""
    ratings: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paper_sha256(paper: Paper) -> str:
    """Content hash of a paper: sha256 hex of its JSON body, as stored."""
    return _sha256(paper.model_dump_json())


class Database:
    def __init__(self, path: Path | str) -> None:
        # The database holds research questions, the papers behind them and the
        # reports written from them, so it follows bedrock's rule for anything
        # private: 0700 on the directory, 0600 on the file. Anyone else on a
        # shared machine can then see that a file exists and nothing more.
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.conn = sqlite3.connect(str(path))
        if str(path) != ":memory:":
            Path(path).chmod(0o600)
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

    def recent_runs(self, limit: int = 20) -> list[Run]:
        """The most recent runs, newest first.

        Ordered by `started` rather than by id: a run id is a hash, so the
        rowid and the id both say nothing about when a run happened.
        """
        rows = self.conn.execute("SELECT body FROM runs").fetchall()
        runs = [Run.model_validate_json(b) for (b,) in rows]
        runs.sort(key=lambda r: (r.started, r.id), reverse=True)
        return runs[:limit]

    def report_id_for(self, run_id: str) -> str | None:
        """The report a run wrote, if it got that far.

        Taken from the run's own record, not by searching the reports table for
        this run id. A report id is the hash of the report body, and asking the
        same question over the same papers twice is meant to produce the same
        bytes — so both runs share one row, whose `run_id` is whichever wrote it
        last. Searching by run id therefore reports the earlier of two identical
        runs as unfinished, which is how this was found.
        """
        run = self.get_run(run_id)
        return run.report_sha256[:16] if run and run.report_sha256 else None

    def finished_run_ids(self) -> list[str]:
        """Runs that got as far as writing a report."""
        rows = self.conn.execute("SELECT body FROM runs ORDER BY id").fetchall()
        return [r.id for r in (Run.model_validate_json(b) for (b,) in rows) if r.stage_reached >= 7]

    def runs_for_report(self, report_id: str) -> list[str]:
        """Every run that wrote this report — usually one, sometimes more.

        A report id is the first 16 characters of the hash of the report body,
        and asking the same question over the same papers twice is *designed* to
        produce the same bytes. Both runs then point at one `reports` row. So
        "which runs does this report belong to?" is answered from the runs, the
        same way `report_id_for` answers the reverse, and not from the row's own
        `run_id` — which only names whichever wrote it last.

        This is what stops `delete_run` removing a report another run still
        needs.
        """
        rows = self.conn.execute("SELECT body FROM runs").fetchall()
        return sorted(
            run.id for run in (Run.model_validate_json(b) for (b,) in rows)
            if run.report_sha256 and run.report_sha256[:16] == report_id
        )

    def delete_run(self, run_id: str) -> DeletedRun | None:
        """Delete one run's own output. Returns None if there is no such run.

        What this removes is the run: its record, its stage artifacts, and its
        report — nothing about the question it answered. The question, its frozen
        paper set and every stored paper version stay, because they are not this
        run's to delete: `paper_versions` is append-only by design and is what
        lets *other* runs be re-proved against the exact bytes they read
        (`get_paper_versions`). Deleting a paper version to tidy up after one run
        would quietly break `verify-run` for every other run that read it.

        The report row goes only when no other run points at it. Two identical
        runs share one report row, so deleting it with the first would take away
        the second run's report while leaving that run looking finished. Ratings
        follow the report, since a rating of a report that no longer exists says
        nothing about anything. Benchmark results are left alone: they are
        measurements of the pipeline rather than part of a run's output, and they
        normally live in a separate database.

        The rendered files are not touched here — this class knows about the
        database and not about the reports directory. `api.forget_run` does both.
        """
        run = self.get_run(run_id)
        if run is None:
            return None
        report_id = run.report_sha256[:16] if run.report_sha256 else None
        shared_with = tuple(r for r in self.runs_for_report(report_id) if r != run_id) if report_id else ()

        with self.conn:
            artifacts = self.conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,)).rowcount
            ratings = 0
            if report_id and not shared_with:
                ratings = self.conn.execute("DELETE FROM ratings WHERE report_id = ?", (report_id,)).rowcount
                self.conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
            self.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

        return DeletedRun(
            run_id=run_id,
            question_id=run.question_id,
            artifacts=artifacts,
            report_id=report_id,
            report_deleted=bool(report_id) and not shared_with,
            report_shared_with=shared_with,
            ratings=ratings,
        )

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

    # publication integrity: append-only -----------------------------------------------

    def add_integrity_observation(self, obs, raw: bytes | None = None) -> str:
        """Append one observation (and its raw response, by hash). Never
        updates or deletes: a later observation is a new row, and the current
        status is derived from all of them (`publication_integrity.derive_state`).
        Returns the observation id; adding the same observation twice is a no-op."""
        oid = obs.observation_id
        subjects = set(obs.requested_ids) | set(obs.resolved_ids)
        subjects |= {e.subject_id for e in obs.events if e.subject_id} | {e.notice_id for e in obs.events if e.notice_id}
        subjects |= {alias for e in obs.events for alias in e.subject_aliases}
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO integrity_observations (observation_id, provider, retrieved_at, raw_sha256, body) "
                "VALUES (?, ?, ?, ?, ?)",
                (oid, obs.provider, obs.retrieved_at, obs.raw_sha256, obs.model_dump_json()),
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO integrity_subjects (subject_id, observation_id) VALUES (?, ?)",
                [(sid, oid) for sid in sorted(subjects)],
            )
            if raw is not None and obs.raw_sha256 and len(raw) <= MAX_RAW_BYTES:
                self.conn.execute("INSERT OR IGNORE INTO integrity_raw (sha256, body) VALUES (?, ?)",
                                  (obs.raw_sha256, raw))
        return oid

    def integrity_observations(self, subject_ids: Iterable[str]):
        """Every stored observation that mentions any of these identifiers."""
        from bedrock.litreview.publication_integrity import PublicationIntegrityObservation

        ids = sorted(set(subject_ids))
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT DISTINCT o.body FROM integrity_observations o JOIN integrity_subjects s "
            f"ON s.observation_id = o.observation_id WHERE s.subject_id IN ({marks}) "
            f"ORDER BY o.retrieved_at, o.observation_id", ids,
        ).fetchall()
        return [PublicationIntegrityObservation.model_validate_json(b) for (b,) in rows]

    def add_integrity_decision(self, subject_id: str, decision: str, *, by: str, at: str, note: str | None) -> None:
        """A person's decision about a paper that was retracted and later
        reinstated: `include` adds it back to new reviews, `exclude` takes it
        out again. Appended, never edited; the latest one counts."""
        if decision not in ("include", "exclude"):
            raise ValueError(f"decision must be include or exclude, not {decision!r}")
        with self.conn:
            self.conn.execute(
                "INSERT INTO integrity_decisions (subject_id, decision, decided_by, decided_at, note) "
                "VALUES (?, ?, ?, ?, ?)", (subject_id, decision, by, at, note))

    def integrity_decision(self, subject_ids: Iterable[str]) -> dict | None:
        """The latest decision recorded about any of these identifiers."""
        ids = sorted(set(subject_ids))
        if not ids:
            return None
        row = self.conn.execute(
            f"SELECT subject_id, decision, decided_by, decided_at, note FROM integrity_decisions "
            f"WHERE subject_id IN ({','.join('?' * len(ids))}) ORDER BY id DESC LIMIT 1", ids).fetchone()
        if row is None:
            return None
        return dict(zip(("subject_id", "decision", "decided_by", "decided_at", "note"), row))

    def integrity_observation(self, observation_id: str):
        from bedrock.litreview.publication_integrity import PublicationIntegrityObservation

        row = self.conn.execute("SELECT body FROM integrity_observations WHERE observation_id = ?",
                                (observation_id,)).fetchone()
        return PublicationIntegrityObservation.model_validate_json(row[0]) if row else None

    def add_bench_result(self, report_id: str, question_id: str, created: str, body: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO bench_results (report_id, question_id, created, body) VALUES (?, ?, ?, ?)",
                (report_id, question_id, created, body),
            )
