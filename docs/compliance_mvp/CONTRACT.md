# Shared implementation contract

Read with [the MVP](../COMPLIANCE_MVP.md). These are proposed interfaces, not
features already implemented. Agent A owns their implementation; Agent B uses
test-local fakes with these shapes until the real modules arrive. Neither agent
edits the other's files. Agree on any interface change before depending on it.

## Imports and construction

```python
from bedrock.provenance import HistoryStore, RunCatalog, HistoryError, RunError
from bedrock.biotech.confidence import ConfidenceReport, PaeOverview

history = HistoryStore(Path(settings.history_dir))
runs = RunCatalog(settings.biotech_work_dir, history)
workflow = ProteinWorkflow.from_settings(settings, history=history)
```

`HistoryError` and `RunError` are user-presentable domain exceptions. Underlying
OS/SQLite/parser exceptions are chained, not leaked as raw user notices. No
network work occurs in these modules. Every history call owns/closes its SQLite
connection, so callers may safely use `asyncio.to_thread`. Constructors and other
disk work run off the Textual event loop. App service initialization is lazy,
single-flight, and injectable for tests; repeated concurrent commands reuse it.

Add `Settings.history_dir: str`, TOML `history_dir`, env `BEDROCK_HISTORY_DIR`.
Default: `~/.local/share/bedrock/history`. Preserve existing precedence.

## HistoryStore

Promote/adapt `reference.Journal`, including its replay checks. Public signatures:

```python
append(operation_id: str, *, actor_id: str, action: str, state: str,
       subject_id: str | None = None, details: dict | None = None) -> int
events(*, after: int = 0, limit: int = 50) -> list[dict]
export(destination: Path) -> int  # JSONL, returns event count
```

- `action`: `fold`, `digest.ingest`, `digest.forget`, `approval`, `run.tag`.
- `state`: `intent`, `succeeded`, `failed`, `cancelled`, `unknown`.
- Allocate one UUID per actual invocation. For fold, use `run_id` as the operation
  ID as well. An event ID/sequence is separate from an artifact ID.
- `append` returns the committed sequence. Identical replay returns its original
  sequence; changed content for the same intent/outcome slot fails.
- Result rows contain `schema_version=1`, `sequence`, `operation_id`, UTC
  `recorded_at`, `actor_id`, `action`, `state`, `subject_id`, `details`.
- Metadata allowlist is in `reference.py`; add `actor_source="config"` in
  production. Error codes are bounded identifiers chosen by code, not `str(exc)`.
  `source_sha256` means SHA-256 of original bytes; digest's short content ID is
  stored as `artifact_id` and never labeled as a full byte hash.
- Export a consistent SQLite read snapshot in sequence order; create a new
  owner-only file exclusively, fail if destination exists, remove a partial file
  on failure. Stream rows, close the snapshot, and report the count/path.
- Missing outcome means unconfirmed, possibly still running. Do not automatically
  add a failure on startup. Neither migration nor browsing invents historical acts.
- Read/write APIs offer no history deletion or replacement. Appending and file
  permissions protect normal operation; the local account owner remains trusted.

## RunCatalog

```python
list(*, project_id: str | None = None, since: str | None = None,
     before: str | None = None, status: str | None = None,
     limit: int = 25, offset: int = 0) -> dict
show(run_id: str) -> dict
verify(run_id: str) -> list[dict]
tag(run_id: str, project_id: str, *, actor_id: str, operation_id: str) -> None
confidence(run_id: str) -> ConfidenceReport
```

Exact IDs; no ambiguous prefix resolution. Dates are `YYYY-MM-DD` UTC: `since`
inclusive, `before` exclusive. Reject invalid/reversed ranges. List stable sorts
by `(created_at, run_id)` descending, validates `1 <= limit <= 100`, `offset >= 0`.

List output:

```python
{
    "items": [{"run_id": str, "created_at": str, "status": str,
               "project_id": str | None, "residue_count": int | None}],
    "total": int, "next_offset": int | None, "warnings": list[str]
}
```

`show` returns `run_id`, `manifest_path: Path`, `project_id`, `manifest: dict`,
`warnings: list[str]`. Status is always described as **last recorded**. Reading
history does not prove a local/GPU process is alive.

`verify` returns rows with `path: str`, `state` (`match`, `mismatch`, `missing`,
`unrecorded`, `invalid`), `expected_sha256: str | None`, `actual_sha256: str | None`.
Include original FASTA and every recorded output. A legacy canonical input/ranking
without a recorded hash is unrecorded, not verified. No filesystem mtime substitute
for `created_at`. Bound manifest reads to 1 MiB and skip corrupt entries with warnings.

`tag` first validates a known run and project ID, then writes `run.tag` intent and
success with that project in details. The committed success event IS the new
assignment; no second mutable store is required. The latest success overrides
`manifest.project_id`; otherwise preserve that original value, or None. IDs:
1–64 characters matching `[A-Za-z0-9][A-Za-z0-9._-]*`. Tagging never edits a manifest
and never changes a workspace's access rules. Store/read projections may remain
internal to HistoryStore; B only calls RunCatalog.

## Confidence result

Agent A exports frozen dataclasses with these fields:

```python
@dataclass(frozen=True)
class PaeOverview:
    values: tuple[tuple[float, ...], ...]  # <=32x32, Angstroms
    ranges: tuple[tuple[int, int], ...]   # 1-based inclusive input positions
    maximum_angstrom: float
    source_path: Path
    # Fixed semantics: rows aligned, columns scored; block maximum aggregation.

@dataclass(frozen=True)
class ConfidenceReport:
    run_id: str
    model_id: str | None
    residue_count: int
    plddt: tuple[float, ...] | None
    low_regions: tuple[tuple[int, int], ...]       # <70
    very_low_regions: tuple[tuple[int, int], ...]  # <50
    pae: PaeOverview | None
    notices: tuple[str, ...]  # missing, unsupported, integrity/mapping problems
```

Unavailable metrics are `None`, never zero. One invalid metric need not suppress
the other when its independent inputs are valid. Missing model pairing suppresses
both. Notices tell the employee why. All scientific quantities are deterministic;
the model is not asked to interpret or invent them.

New v2 run manifests retain v1 fields and add `project_id`,
`alphafold2.selected_model`, and `outputs.ranking` / `outputs.confidence_summary`
as `{path, sha256, bytes}` entries. Hash `input.fasta` as an output entry too.
Keep all existing confidence artifacts, and identify the selected source filenames
in the normalized summary. The versioned summary stores scores, overview ranges,
grid, semantics and source hashes; full PAE stays in its original file.

On reads, verify recorded hashes before consuming metrics. Pair legacy `ranked_0`
using only same-directory ranking JSON; if that ranking was not originally hashed,
add an explicit “model pairing derived from unrecorded local ranking” notice.
Do not represent this as historically verified provenance. If pairing is missing
or ambiguous, report unavailable. Experimental B-factors are not a JSON fallback.
Terminal exports use plain text/JSONL; no HTML or browser component is needed.

## Workflow and UI integration

```python
ProteinWorkflow.from_settings(settings, *, history=None) -> ProteinWorkflow
await workflow.predict_and_render(source_fasta, *, project_id=None) -> ProteinWorkflowResult
await workflow.open_in_pymol(structure_path=None) -> Path
await PyMOLAdapter.open_interactive(structure_path, *, confidence_metric=None) -> Path
```

Agent A adds optional injected `history`/`actor_id` to the workflow constructor;
production `from_settings` always provisions history and uses `settings.user_id`.
Keep existing result fields; append optional `confidence_path: Path | None = None`.
Test fakes add the `project_id` keyword. Workflow owns fold journaling exactly once;
neither UI nor CLI wraps it with another fold event.

Agent B adds optional `history`/`actor_id` arguments to the shared
`run_digest_command`. Production app callers always supply them. Journal the
actual ingest/delete branch, never infer success from a rendered notice. Complete
the ingest outcome immediately after the artifact commits, before optional model
summarization. Re-ingesting identical content still records a fresh operation.
If cancelling an `asyncio.to_thread` ingest, the thread may continue: await its
actual completion or record unknown; never claim the underlying ingest stopped.

Agent B owns approval intent/result logging and all presentation. Disable double
submits synchronously before the first await. A duplicate `approval.required`
event must reuse an existing pending prompt. Do not treat task IDs alone as a
globally unique approval request: a later revision may require another decision.
Deduplicate the same pending session/task/preview fingerprint; allocate a new
operation ID for a new explicit decision. Store the fingerprint in memory only.
Do not auto-submit re-delivered/unknown prompts. Preserve the original backend
contract, and do not guess nonexistent status or idempotency endpoints.
The current `approve()` raises a generic `InferenceError` for both HTTP status
errors and network failures. After dispatch, classify it as unknown; do not parse
its text into a supposedly definitive rejection. Before dispatch, local validation
or intent-write failure is known not-sent and may leave submission available.

Agent A owns `workflow.open_in_pymol` selecting verified run coordinates and
explicit semantics; Agent B owns `PyMOLAdapter` accepting that semantic keyword
and using a fixed script for interactive styling. An arbitrary path stays neutral.

All history/manifest reads, hashing and confidence parsing happen in background
workers. Widget mutation stays on the Textual event loop. Fixture paths and
history directories must be temporary; tests must not touch an employee's vault.
