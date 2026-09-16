# Prompt B — terminal experience, visualization and shipping

Implement your half of the compliance/provenance MVP in `/Users/li/Bedrock private AI`.
Agent A is concurrently building durable history, run queries and confidence
normalization. Write working code and tests; start immediately with injected fakes
matching the shared contract while the real implementation is built.

## Read and respect

Read `AGENTS.md`, `docs/COMPLIANCE_MVP.md` and
`docs/compliance_mvp/CONTRACT.md`. Keep the employee interface native Python,
Textual and Rich. Existing per-run manifests, pLDDT-colored PNGs, saved `.pse`
sessions and workspace events already work; extend those capabilities. Preserve
the local identity limits in `docs/SHARED_WORKSPACES.md`. Baseline: 137 tests pass.

## Own these files

- Both app files; `ui/commands.py`, `ui/digest_command.py`, `ui/approval.py`;
  new runs/audit/confidence UI modules and any needed styling/widgets.
- `biotech/pymol.py`, fixed PyMOL scripts and `biotech/cli.py`.
- UI/CLI/PyMOL tests, `tests/conftest.py`; `pyproject.toml`, README and release docs.

Agent A owns provenance, confidence computation, AlphaFold adapter, workflow,
settings and their tests. Import the exact public contract; do not create another
database, run parser or confidence parser. Send interface concerns to A. Keep
edits in your owned files; avoid shared Git index operations in a common checkout.

## Implement in order

1. Build injectable handlers and fake services. Add `/runs` list/show/verify/tag,
   `/confidence RUN_ID`, `/audit` and `/audit export PATH` in both TUI modes and
   matching `bedrock-biotech` subcommands. Add `--project` to existing fold entry
   points. Use `shlex`/argparse for quoted paths and predictable options; report
   bad syntax without exiting the TUI. Keep local browsing usable when the model
   server is offline. Do not send these commands to a language model.
2. Render run pages with project, UTC creation time, last-recorded status and
   pagination. Detail shows settings, source/output lineage and verification
   results. Preserve legacy warnings. Add help examples for project X this quarter.
   All disk work runs off the event loop; only widget updates run on it. Use plain
   escaped text for user filenames/labels and sanitize terminal control characters.
3. Render confidence from `ConfidenceReport`: four numeric pLDDT bands with
   counts/percentages, intervals below 70 and 50, input-position labels, selected
   model and missing/invalid-data notices. Include the exact scientific caveats
   from the plan. Add a bounded Rich/Textual PAE heatmap with numeric Å legend,
   row/column residue ranges and “maximum per block” label. Rows align; columns
   score. Retain asymmetry and link the full local JSON path. Use a textual legend
   so color is not the only signal. Fold completion shows a short summary plus
   `/confidence RUN_ID`; leave coordinates usable if confidence is unavailable.
4. Preserve interactive PyMOL styling. Implement the agreed `confidence_metric`
   keyword using fixed Python script code plus environment data, shared with the
   existing headless palette. Launch with an argv list and filtered child environment.
   Arbitrary experimental structures stay neutral. Do not widen input validation to
   load arbitrary `.pse` files. Keep preview/session outputs working.
5. Wire history lazily and injectably into both apps and CLI. Workflow owns fold
   events. Your shared digest handler records actual ingest/forget intent and
   outcome; identical content may reuse an artifact but gets a new operation ID.
   Record ingest success before optional model summarization. A cancelled thread
   may still run: await it or record unknown, never claim it stopped. `/clear` and
   digest deletion preserve history. Show explicit history-write failures.
6. Fix approval handling and journal it. Disable both buttons before awaiting;
   commit intent, POST once, commit confirmed success, then remove the prompt.
   Keep known success even if its journal outcome write fails; show the gap and
   never resend. Timeout/cancellation-after-dispatch/ambiguous error is unknown:
   retain an unresolved prompt with submission disabled. The existing server
   interface lacks status reconciliation/idempotency, so do not invent an endpoint
   or auto-retry. Its generic `InferenceError` after dispatch is unknown; do not
   parse error text to infer rejection. Deduplicate repeated pending events by the contract's
   fingerprint, while allowing a genuinely revised request to be a new decision.

## Tests and integration

Drive both real Textual apps headlessly with fake clients and temporary storage.
Cover run filters/pagination, quoted filenames, digest repeat/forget/clear,
double approval clicks, not-sent vs unknown requests, acknowledged success followed
by journal failure, repeated approval events, offline browsing, low-confidence
intervals and small asymmetric PAE. Inspect rendered widget content, not only helper
outputs. Test fixed PyMOL script commands with a fake `cmd` object and verify both
interactive and headless paths use the same thresholds. All tests must use a
temporary history directory, including existing digest/fold/approval fixtures.

Integrate A's real modules, then run the full suite. In separate worktrees, merge
A's completed patch into your integration branch; in a shared checkout, wait only
for real-service integration tests, not for starting UI implementation. Ask A to
fix service defects while you fix UI/CLI defects. Keep the established interfaces.

## Ship the smallest supported release

Own the combined release check: update docs, build wheel/sdist in a clean env,
install the wheel in another temporary env, and test installed commands, offline
history, both app modes and bundled PyMOL scripts. Add only dependencies/data files
actually needed; coordinate A's requirements. Record version/commit, wheel hash,
automated results and rollback instructions preserving history and manifests.

Prepare a private-workstation smoke checklist using one approved sequence: real
prediction, correctly paired confidence, PNG/session output, interactive colors,
PAE and restart history. Record actual results if tools are available; otherwise
name the missing prerequisites. Prepare artifacts for the existing private release
channel; do not invent a destination or call untested work shipped. Finish with
artifact paths, passing checks and the precise remaining real-tool/release step.
