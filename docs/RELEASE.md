# Release check — compliance and provenance MVP

What this records: the artifacts that were built, the checks that were run
against the **installed wheel** rather than the source tree, and how to go
back if the release is wrong. It is a record of one build, not a promise
about future ones.

## What was built

| Item | Value |
|---|---|
| Version | `0.1.0` (`pyproject.toml`) |
| Commit | `d286f3f40b75f61285cf915b0d62d7e87a2c53db` — *Merge origin/main into the compliance/provenance branch* |
| Branch | `fix/compliance-provenance-and-biotech`, merged up to `origin/main` at `f012908` |
| Source digest | `f63a490003c3d1b6ed066ddcc304612f946897b8921a40fc58107cba4bf94b04` |
| Built (UTC) | 2026-09-16T02:42:30Z |
| Wheel | `dist/bedrock_private_ai-0.1.0-py3-none-any.whl` |
| Wheel SHA-256 | `c5e6a8ad054a9d02fd2613bc1f295e8266527086c0cfad2a13c0e9bbee96e9d2` |
| Sdist | `dist/bedrock_private_ai-0.1.0.tar.gz` |
| Sdist SHA-256 | `7e3654da2e90ba276eb6dc95c37a5617fb659f3689174754425ca34409c14818` |

The source digest is `sha256` over the sorted per-file `sha256` of every
`.py`, `.tcss` and `.sql` file under `src/`, plus `pyproject.toml`. A wheel's
own bytes depend on zip timestamps, so two builds of identical source will not
have the same wheel hash; the source digest is the value to compare.

## How it was built and checked

```bash
python3 -m venv /tmp/buildenv && /tmp/buildenv/bin/pip install build
/tmp/buildenv/bin/python -m build --outdir dist .

python3 -m venv /tmp/installenv                 # a second, empty environment
/tmp/installenv/bin/pip install dist/bedrock_private_ai-0.1.0-py3-none-any.whl
/tmp/installenv/bin/python dist/installed_smoke.py
```

`dist/installed_smoke.py` is shipped next to the artifacts so the checks can be
repeated on the machine that receives the wheel. It runs against an empty
temporary home, so it also proves a first installation works.

## Results

**Test suite (source tree): 505 passed.** That is this branch merged with
`origin/main`; 137 was the baseline before either half of the work started.

**Installed wheel: 14/14 checks passed.**

| Check | Result |
|---|---|
| Settings read `BEDROCK_ACTION_HISTORY_DIR` | pass |
| Saved conversations use a different directory from the journal | pass |
| Fixed PyMOL scripts are inside the wheel | pass |
| Headless and interactive PyMOL share the pLDDT thresholds | pass |
| An existing run manifest lists after a fresh install | pass |
| History survives being reopened (restart) | pass |
| Metadata-only JSONL export writes and reports its count | pass |
| `bedrock-biotech runs` filters parse | pass |
| `bedrock-biotech runs show` parses | pass |
| `bedrock-biotech confidence` parses | pass |
| `bedrock-biotech audit export` parses | pass |
| `bedrock-biotech fold --project` parses | pass |
| `openai`-mode app browses runs with no model server | pass |
| `agentpack`-mode app browses runs with no AI system | pass |

The installed `bedrock-biotech` console script was also driven by hand in that
environment: `runs`, `runs tag`, `runs --project X --since … --before …`,
`audit`, `audit export` and `confidence` all produced the expected output,
including the "confidence unavailable" path for a run with no confidence JSON.

**Not covered by any of the above:** everything that needs a real AlphaFold 2
installation, a real GPU or a real PyMOL. Those are the
[private-workstation smoke checklist](SMOKE_CHECKLIST.md), and none of it has
been run. Nothing here demonstrates a real prediction.

## Dependencies and data files

No new runtime dependency was added. `rich` was already required and is what
the command line uses to print. `textual`, `httpx` and `zstandard` are
unchanged.

No new package-data entry was needed: the fixed PyMOL scripts
(`_pymol_style.py`, `_pymol_render.py`, `_pymol_interactive.py`) are ordinary
modules inside `bedrock.biotech` and are collected automatically, which the
wheel listing above confirms. `ui/*.tcss` and `workspaces/migrations/*.sql`
continue to ship through the existing `[tool.setuptools.package-data]` entry.

## Before distributing

1. Open a pull request from `fix/compliance-provenance-and-biotech`. `AGENTS.md`
   forbids pushing to `main` directly. This branch is already merged up to
   `origin/main`, so the PR should be conflict-free; if `main` moves again,
   merge it in and rebuild before distributing.
2. Distribute the tested wheel and a configuration file through the company's
   **existing** private channel. This repository does not record what that
   channel is, and a wheel published anywhere else is not this release.

## Rolling back

Roll back binaries only. Nothing in a rollback may delete history or rewrite a
manifest.

```bash
pip install --force-reinstall bedrock_private_ai-<previous>-py3-none-any.whl
```

* **Keep the previous wheel.** It is the rollback.
* **Do not remove `BEDROCK_ACTION_HISTORY_DIR`.** An older client simply does not read
  it; the journal stays valid for when the newer one returns.
* **Do not touch run directories.** Manifests written by this version keep
  every v1 field, so an older client still reads them. The extra v2 fields
  (`project_id`, `alphafold2.selected_model`, `outputs.ranking`,
  `outputs.confidence_summary`) are ignored by older code.
* **Project labels survive a rollback** because they live in the journal, not
  in the manifests — an older client just will not display them.
* **Back up SQLite properly**: use the SQLite backup API, or copy the files
  with every client stopped. Copying a live database with `cp` can capture a
  torn write.
