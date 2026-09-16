# Provenance storage and confidence normalization

## Local history

`bedrock.provenance.HistoryStore` owns `history.sqlite3` in
`Settings.action_history_dir`. The configuration order is
`BEDROCK_ACTION_HISTORY_DIR`, then TOML `action_history_dir`, then
`~/.local/share/bedrock/history`.

The name is deliberate. `Settings.history_dir` is a different directory holding
saved conversations — the employee's prompts and answers. The action journal is
metadata-only, and the two must not share one configuration knob.

The store is metadata-only. It accepts the action/state/detail vocabulary in
`CONTRACT.md`, adds `actor_source=config`, bounds each serialized event to 8 KiB,
and keeps one committed `intent` plus one committed `outcome` per operation ID.
Identical replay returns the original sequence. Changed replay fails. Each method
opens its own SQLite connection and writes with `BEGIN IMMEDIATE` and
`synchronous=FULL`, so separate terminal processes and thread workers share the
same ordering. JSONL export reads one consistent snapshot and creates a new
owner-only file; it never overwrites a destination.

This is a local activity record. `BEDROCK_USER` is still configured identity, and
the OS-account owner can alter or remove the database. Missing outcomes remain
unconfirmed; startup does not invent failures for them.

## Run catalog

`bedrock.provenance.RunCatalog` reads v1 and v2 manifests below
`<biotech_work_dir>/runs`. It never rewrites them. Project changes are successful
`run.tag` events, with the latest successful event overriding the original label.
Labels organize results and do not grant workspace access.

Manifests are capped at 1 MiB. Run IDs resolve exactly. Symlinked run directories,
absolute artifact paths, traversal, and symlinked artifacts are rejected. Listing
reads metadata only; `verify()` performs hashes and reports `match`, `mismatch`,
`missing`, `unrecorded`, or `invalid`. A hash match compares a file with its local
manifest; it does not authenticate the producer.

## Confidence

New folds write a v2 manifest and `confidence-summary.json` schema v1. The workflow
snapshots the submitted FASTA before parsing, then both normalization and AlphaFold
consume files in that run directory. It records hashes for the source snapshot,
canonical input, chosen coordinates, ranking JSON, all confidence JSON, summary,
and PyMOL artifacts.

For ranked coordinates, the normalizer reads the same directory's
`ranking_debug.json` and selects `order[rank]`; it never assumes `ranked_0` came
from model 1. It then reads only `confidence_<selected-model>.json` and
`pae_<selected-model>.json` whose recorded hashes match. Legacy ranking without a
recorded hash is allowed with an explicit notice and never presented as historical
verification. Pickle files and arbitrary coordinate B-factors are not inputs.

pLDDT positions must exactly match one-based input sequence positions, all values
must be finite in `[0,100]`, and low regions are inclusive input positions. PAE
must be an `N × N` finite matrix. Its terminal overview preserves direction—rows
aligned, columns scored—and takes the maximum per block on a grid no larger than
32 × 32. Full PAE remains in the original bounded JSON; the summary stores only
the overview and source hashes. Failure of one metric leaves an independently
valid metric and the coordinate structure available.
