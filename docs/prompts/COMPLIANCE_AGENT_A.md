# Prompt A — data, provenance and scientific correctness

Implement your half of the compliance/provenance MVP in `/Users/li/Bedrock private AI`.
Agent B is concurrently building the terminal UI, PyMOL behavior and release flow.
Write working code and tests; the plan already exists.

## Read and respect

Read `AGENTS.md`, `docs/COMPLIANCE_MVP.md` and
`docs/compliance_mvp/CONTRACT.md`. The contract is the shared source of truth.
Read `docs/SHARED_WORKSPACES.md` for the existing local identity/storage limits;
keep workspace storage and its activity feed intact. Python 3.11+, native Textual,
private endpoints, no telemetry. The baseline is 137 passing tests.

The screenshot is partly stale: per-run manifests, pLDDT-colored previews, `.pse`
output and workspace events already exist. Implement only the remaining gaps.

## Own these files

- New `src/bedrock/provenance/` and `src/bedrock/biotech/confidence.py`.
- `src/bedrock/biotech/alphafold2.py`, `workflow.py`, relevant package exports.
- `src/bedrock/config/settings.py` and config exports.
- New service/normalization tests, `tests/test_biotech.py`, `tests/test_settings.py`.
- Storage/normalization developer notes in `docs/compliance_mvp/`.

Agent B owns both apps, all UI files, PyMOL scripts/adapter, biotech CLI,
`pyproject.toml`, `tests/conftest.py`, UI tests and release/user docs. Communicate
required dependency/data-file additions; let B make those edits. Preserve unrelated
changes and avoid shared Git index operations while using the same checkout.

## Implement in order

1. Export the exact contract interfaces and dataclasses early so B can integrate.
   Build `HistoryStore` using the tested `docs/compliance_mvp/reference.py` journal.
   Add configuration precedence, typed errors, versioned schema, bounded allowlisted
   metadata and snapshot JSONL export. Each call owns its connection; intent/outcome
   commits are durable and identical replay is idempotent. Export files are private
   and never silently overwritten. New user actions always get new operation IDs.
2. Implement `RunCatalog` over existing manifests: list/show/verify/tag/confidence.
   Support v1 unchanged, UTC date bounds, project/status filtering, pagination,
   malformed-entry warnings and contained artifact paths. List metadata without
   hashing big outputs. Verify original FASTA and all recorded outputs explicitly.
   Derive project assignments from successful `run.tag` journal events; never
   rewrite historical manifests or imply a label grants workspace access.
3. Fix workflow provenance: allocate run/operation ID; commit intent; snapshot the
   FASTA; parse and normalize that snapshot, not a separately read changing source.
   Record actor/project and hashes for original input, canonical input, selected
   structure, ranking and confidence outputs. Journal success/failure/cancellation
   across every phase, including rendering. A rendering/confidence failure preserves
   a usable prediction. If a terminal audit write fails, keep the result and expose
   the history gap without rerunning work. An unfinished intent remains unconfirmed.
4. Implement confidence normalization. Use the reference model mapping, pLDDT
   validation/intervals and directional PAE aggregation. Select the JSON belonging
   to the actual coordinate model using its same-directory ranking; verify hashes.
   Handle official PAE list-of-one-object and object wrappers. Validate numeric
   finiteness, shape, N, one-based positions and byte limits before expensive work.
   Preserve PDB/mmCIF support. No client pickle loading or experimental B-factor
   inference. Missing/invalid data returns a precise notice, not fabricated scores.
5. Write a hashed v1 confidence-summary artifact from new runs and reference it
   from a v2 manifest. Old runs may be analyzed read-only, with unrecorded-ranking
   caveats. Add `project_id` and injected history exactly as specified. Update
   workflow interactive opening to pass explicit pLDDT semantics only for a
   verified known AF2 structure; B implements the adapter's new keyword.

## Hard parts already coded

Adapt `docs/compliance_mvp/reference.py` and promote its meaningful tests from
`test_reference.py`: concurrent/idempotent audit writes, conflicting retries,
traversal/symlink escape rejection, ranked-model mapping, input-position intervals,
asymmetric PAE and block maxima. All 27 reference tests currently pass.

Do not mistake these for the finished feature. Add the missing schema migrations,
catalog, data loading, lifecycle integration and error translation. Keep parsing
and hashing off the UI thread. A hash comparison is local integrity evidence;
it does not authenticate the actor or make the journal tamper-proof.

## Prove completion

Use realistic small fake AF2 files: ranking selects model 3 rather than model 1;
confidence has matching `residueNumber`; asymmetric PAE has a sparse high-error
region. Cover same-data re-ingest IDs at the journal seam, concurrent writers,
restart durability, exact quarter boundaries, legacy tagging without byte changes,
corrupt manifests, changed hashes, missing confidence, invalid numbers/shapes,
source snapshot consistency and cancellation during rendering. Update the existing
`ATOM`/single-score fixture so it does not accidentally test unavailable data only.

Run your focused tests and the reference tests. Give B the exported interfaces,
any dependency additions and a short test summary. After B integrates, fix service
defects until the combined suite passes. Finish with files changed, checks run,
and any real-tool verification still needed; do not claim an unrun GPU test passed.
