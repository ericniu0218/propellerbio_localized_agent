# Compliance and provenance MVP

## Verified status — 2026-09-15

Inspected the local terminal client at commit `60bf8af`; its working tree was clean.
This assessment covers the latest screenshot's three items. The separate
agent-pack server was not inspected, so its audit behavior is not assumed.

| Screenshot claim | What the code actually does | Remaining work |
|---|---|---|
| Manifests exist, but employees cannot browse runs | Correct. `biotech/workflow.py` writes per-run source/output hashes, settings and status; `latest_structure()` only finds the latest coordinate file. Neither app nor the biotech CLI has a run-history command. | List/detail/verification commands, project labels and date filters. |
| Nothing is retained locally | Too broad. Digest manifests retain creation metadata; fold manifests retain run provenance; `/workspace events` has a durable SQLite activity feed. None is a general history of digest invocations, folds and approval attempts/outcomes. | A separate local action journal, including failures and uncertainty. Preserve workspace events. |
| PyMOL produces only a PNG, with no pLDDT colors | Already partially fixed. `_pymol_render.py:20` applies four pLDDT bands and saves both PNG and `.pse`; the workflow explicitly declares `confidence_metric="plddt"`. | `/pymol` currently opens raw coordinates, bypassing the styled session. Preserve styling in interactive opening; add numeric confidence, low-confidence intervals and PAE in the TUI. |

Evidence: `src/bedrock/biotech/workflow.py:88`,
`src/bedrock/biotech/alphafold2.py:100`, `src/bedrock/biotech/pymol.py:62`,
`src/bedrock/context/store.py:143`, `src/bedrock/ui/workspace_command.py:279`,
`src/bedrock/agentpack_app.py:343`. Existing tests: **137 passed**.
The reference code accompanying this plan: **27 passed**. These tests use
fakes; they do not demonstrate a real AlphaFold or PyMOL installation.

## Employee flow and scope

Keep the native Python/Textual/Rich application and private configured endpoints.
Deliver local traceability for the current OS account. Actor IDs remain
caller-configured; project labels organize results and do not grant access.
Local history is neither an authenticated approval ledger nor tamper-proof storage.

```text
# Agent-pack TUI; retain the existing one-monomer fold workflow
/fold "/approved/candidate.fasta" --project X

# Both TUI modes can inspect existing local results without a model connection
/runs --project X --since 2026-07-01 --before 2026-10-01
/runs show RUN_ID
/runs verify RUN_ID
/runs tag OLD_RUN_ID --project X
/confidence RUN_ID
/audit --after 0
/audit export "/approved/review/actions.jsonl"

# Existing biotech CLI also supports runs, confidence and audit subcommands
bedrock-biotech runs --project X --since 2026-07-01 --before 2026-10-01
```

The default run page has 25 entries; maximum 100. Dates use UTC, inclusive
`--since` and exclusive `--before`. Add `--status`, `--limit`, `--offset`.
Legacy runs remain visible as unassigned until explicitly tagged. The MVP uses
these deterministic commands for project/quarter questions.

## Step-by-step build and ship

1. **Freeze the interface.** Both agents read [the contract](compliance_mvp/CONTRACT.md).
   Use the ownership table below. Agent B begins with injected fakes while Agent A
   builds the implementation; neither needs to wait for a functioning GPU server.
2. **Build durable local history.** Agent A implements the SQLite journal and
   metadata-only export. Events commit before/after operations; stable operation
   IDs deduplicate the same event, while repeated user actions get new IDs.
   Add `BEDROCK_HISTORY_DIR` / `history_dir` with the existing configuration
   precedence, defaulting to `~/.local/share/bedrock/history`.
3. **Expose existing runs.** Agent A reads v1 manifests, adds paginated filtering,
   per-artifact hash checks, and append-only project annotations. Scanning must
   report corrupt/unknown manifests without losing valid results. List operations
   read metadata, not every large structure. Historical imports are observations,
   not fabricated past audit events.
4. **Make new provenance accurate.** Agent A snapshots the FASTA, parses that exact
   snapshot and runs its normalized sequence; currently parsing and copying happen
   separately. Record intent, project, selected model and hashed ranking/confidence
   inputs. Catch failure/cancellation across prediction AND rendering; keep any
   successful structure. A crash leaves an unconfirmed outcome, not an invented
   completion. Preserve original manifests when browsing/tagging old runs.
5. **Normalize scientific confidence.** Agent A pairs the selected structure with
   its own model's JSON, validates residue indexing and PAE, and writes a hashed,
   versioned summary artifact. Existing runs can be analyzed read-only. Missing
   supported JSON becomes a clear unavailable reason, preserving the structure.
6. **Build the employee experience concurrently.** Agent B adds the commands,
   confidence widget, CLI routing, digest/approval journaling and interactive
   PyMOL styling. Both app modes can browse history. Fold execution stays in its
   existing agent-pack/CLI entry points. `/clear` only clears the screen.
7. **Integrate and verify.** Agent A verifies real storage plus fake scientific
   tools; Agent B verifies real Textual screens plus fake model/agent-pack clients.
   Once both patches land, run the entire suite once and the scenarios below.
   Agent B owns integration edits to UI/CLI; Agent A fixes service-level defects.
8. **Build a reviewable release.** Use a clean environment, run the release tests,
   build wheel/sdist with `python -m build`, and install the wheel in a second
   temporary environment. Test installed `bedrock` and `bedrock-biotech`, SQL/data
   inclusion, offline history browsing, and both fake-server modes. Record version,
   commit, artifact SHA-256 and test results in a release note.
9. **Pilot, distribute, recover.** Run one approved test sequence on a provisioned
   private AlphaFold/PyMOL workstation; confirm actual colors, low-confidence
   regions and PAE axis orientation. Distribute the tested wheel and configuration
   through the company's existing private channel. Keep the previous wheel;
   roll back binaries without deleting history or rewriting old manifests. Back up
   SQLite through its backup API or with all clients stopped. GPU databases,
   identity/SSO, standalone installers and central audit retention are separate work.

## Hard parts: decisions already made

### Correct model and residue pairing

`ranked_0` is a rank, not `model_1`. Read `ranking_debug.json` beside the chosen
coordinate file and use `order[0]`; select `confidence_<model>.json` and
`pae_<model>.json` from that same directory. Supported direct relaxed/unrelaxed
filenames identify their model explicitly. Validate paths, hashes, lengths and
one-based input positions. Never select the first confidence JSON from a glob.
The current fake fixture lacks ranking and realistic residues: upgrade it.
[Official AF2 output writer](https://github.com/google-deepmind/alphafold/blob/c77e5d2a8961d1a353632c462914ff0a32a950f6/run_alphafold.py).

Support the official JSON layouts in the reference code; preserve PDB **and**
mmCIF support. Older installations without these JSON exports get an explicit
unavailable result. Do not deserialize prediction pickle files in the client or
silently interpret arbitrary experimental B-factors as pLDDT.

### Honest confidence display

Show the four bands `[0,50)`, `[50,70)`, `[70,90)`, `[90,100]`, counts/percentages,
and contiguous intervals below 70 and below 50. Label these as **input sequence
positions**, not coordinate residue identifiers. Include: “pLDDT estimates local
model confidence; it is not experimental validation or a stability measurement.”
Low scores can also reflect disorder.
[EMBL-EBI pLDDT guidance](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/).

PAE is directional. In the normalized JSON array, index `[i][j]` aligns on residue
`i` and scores residue `j`; display rows as aligned residues and columns as scored
residues. Keep asymmetry. Use a <=32×32 terminal heatmap of **maximum error per
block**, with residue-range labels, numeric Å scale, aggregation label and a path
to the full original JSON. Describe it as an overview: high local confidence
does not establish correct domain placement.
[AlphaFold DB FAQ](https://alphafold.ebi.ac.uk/faq),
[EMBL-EBI PAE guidance](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/pae-a-measure-of-global-confidence-in-alphafold-predictions/).

Bound confidence JSON reads before parsing (128 MiB maximum), validate
`N <= biotech_max_sequence_length`, and serialize expensive confidence reads in a
background worker. Never put the full N×N matrix into a widget or workspace
artifact. Keep full input JSON on disk. Official PAE exports round to one decimal,
so allow a 0.051 Å tolerance above the unrounded declared maximum.
[Official JSON formats](https://github.com/google-deepmind/alphafold/blob/c77e5d2a8961d1a353632c462914ff0a32a950f6/alphafold/common/confidence.py).

### Audit durability without false claims

The reference journal uses `BEGIN IMMEDIATE`, full synchronous commits and one
connection per call. It has separate unique slots for intent and outcome. A
digest content ID identifies data; a fresh operation UUID identifies each ingest
attempt, including re-ingesting identical data. Store only allowlisted metadata,
IDs, hashes, decision and sanitized error codes—not file contents, prompts,
approval previews, credentials or raw exceptions. Record `actor_source=config`.

If intent cannot commit, do not start a tracked mutation. If the operation has
already succeeded but the outcome write fails, retain its result and explicitly
report the history gap; do not repeat the operation. Never infer failed/stopped
from a missing outcome: another process may still be working. Local files, remote
HTTP and SQLite cannot participate in one atomic transaction.

Approval flow: disable both buttons, persist intent, send exactly once, persist
confirmed success, then remove the prompt. A network timeout, cancellation after
dispatch, or ambiguous server error means **unknown**, not rejected. Keep that
prompt visibly unresolved with submission disabled; reconcile with the server
before another send. The current client has neither an authoritative approval
status method nor an idempotency key, so this MVP cannot guarantee exactly-once
remote decisions or safely automate retry. A clearly rejected request can remain
available for a corrected attempt only when the interface can establish that no
decision was recorded. Today `approve()` collapses HTTP errors into `InferenceError`:
treat every such exception after dispatch as unknown, without parsing its message
to infer a rejection. Local validation/intent-write failures are known not-sent.
Fix the existing removal-before-request bug.

### Legacy data, project labels and PyMOL

Run-project changes are successful `run.tag` journal events; the latest assignment
wins over a manifest's original project label. Old manifests remain byte-for-byte
unchanged. A project label never implies a workspace role or permission.

Use a fixed script and environment data to style interactive PyMOL coordinates,
as the existing headless renderer already does. This preserves confidence colors
without loading arbitrary supplied `.pse` files. Explicit user-provided structures
use neutral styling unless tied to a verified AF2 manifest. Hash agreement is an
integrity comparison with the local manifest, not proof of trusted authorship.

## Parallel ownership

| Agent A: data, provenance and confidence computation | Agent B: employee flow, visualization and release |
|---|---|
| New `provenance/`; `biotech/confidence.py`; `biotech/alphafold2.py`; `biotech/workflow.py` | Both apps; `ui/commands.py`; digest/approval UI; new runs/audit/confidence UI modules |
| `config/settings.py`, config exports; service tests; `tests/test_biotech.py`, `tests/test_settings.py` | `biotech/pymol.py`, fixed PyMOL scripts, `biotech/cli.py`; UI/CLI/PyMOL tests and `tests/conftest.py` |
| Package exports for its modules; storage/normalization migration notes | `pyproject.toml`, README, user/release docs, packaging checks |

This splits the persistence/scientific correctness work from the interaction and
delivery work; effort is approximately half each. Work in the same checkout with
exclusive file ownership, or use two worktrees from the same starting revision.
Avoid shared Git index operations in a common checkout. With worktrees, merge
Agent A's committed patch into Agent B's integration branch after both feature
patches are ready; Agent B performs the final combined checks.

## Release acceptance scenarios

- Existing v1 manifests list after restart; project/UTC quarter filters have exact
  boundary tests; corrupt manifests produce warnings alongside valid rows.
- Repeated ingest of the same file creates two history operations and one content
  artifact. Forgetting a digest or clearing chat preserves the action history.
- Fold success, tool failure, rendering failure and cancellation retain accurate
  manifests and journal states; snapshot hashes describe the bytes actually used.
- Duplicate approval clicks send once; ambiguous errors never display “Approved”
  or trigger retry; history failure after acknowledged approval does not resend it.
- `ranked_0` pointing to model 3 displays model 3's values. A tiny asymmetric PAE
  fixture confirms axes; sparse high error survives aggregation. Missing/malformed
  JSON, NaN, length mismatch and changed hashes are visibly unavailable/invalid.
- The headless PNG and interactive PyMOL use identical pLDDT bands; experimental
  coordinate inputs retain neutral styling. Confidence errors leave structures usable.
- Browsing works without a model/network. All automated fixtures stay in temporary
  directories, and both app modes pass headless tests. A clean wheel installation
  includes the fixed scripts. Real-tool smoke results are reported separately.

## Ready-to-run agent prompts and tested starter code

- [Prompt A — data and scientific correctness](prompts/COMPLIANCE_AGENT_A.md)
- [Prompt B — terminal experience and shipping](prompts/COMPLIANCE_AGENT_B.md)
- [Frozen shared contract](compliance_mvp/CONTRACT.md)
- [Starter code](compliance_mvp/reference.py) and [27 tests](compliance_mvp/test_reference.py)

The starter code is intentionally outside `src/`: this deliverable is the checked
implementation plan and prompts. Product behavior has not yet been changed.
