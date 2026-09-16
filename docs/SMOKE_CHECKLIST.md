# Private-workstation smoke checklist

One approved sequence, run by a person on a provisioned private workstation,
to confirm the things no fake can confirm: that a real AlphaFold 2 prediction
appears, that its confidence is paired with the model that was actually
scored, and that the colours on screen mean what the terminal says they mean.

**Status on the machine that prepared this release: not run.** None of the
prerequisites are present there:

| Prerequisite | Status here |
|---|---|
| Open-source PyMOL (`pymol` executable) | not installed |
| AlphaFold 2 launcher (`ALPHAFOLD_LAUNCHER`) | not configured |
| AlphaFold 2 databases and parameters (`ALPHAFOLD_DATA_DIR`) | not provisioned |
| Container runtime (`docker`) | not installed |
| NVIDIA GPU (`nvidia-smi`) | not present |

Everything below is therefore a checklist to execute, not a result. The
automated checks that *were* run are in [the release note](RELEASE.md).

## Before you start

```bash
export BEDROCK_BACKEND=agentpack
export AGENT_PACK_API_BASE=http://ai.company.internal:8000
export BEDROCK_USER=your-username
export ALPHAFOLD_LAUNCHER=/opt/alphafold/docker/run_docker.py
export ALPHAFOLD_DATA_DIR=/data/alphafold
export ALPHAFOLD_MAX_TEMPLATE_DATE=2026-01-01
export ALPHAFOLD_DOCKER_IMAGE=alphafold@sha256:…      # pinned, not a floating tag
export BEDROCK_ACTION_HISTORY_DIR=~/.local/share/bedrock/history
```

`bedrock-biotech status` must report both tools ready before anything else.
Use an approved test sequence — a short, non-confidential monomer. Do not use
a real candidate for a smoke test.

## The sequence

| # | Do this | Expect |
|---|---|---|
| 1 | `bedrock-biotech status` | both AlphaFold 2 and PyMOL report ready |
| 2 | In the app: `/fold "/approved/test-monomer.fasta" --project SMOKE` | starts, and says the run is preserved if rendering fails |
| 3 | Wait for completion | a run ID, a structure path, a preview path, a session path, and a one-line confidence summary |
| 4 | `/runs --project SMOKE` | the run, with a UTC creation time and last-recorded status `completed` |
| 5 | `/runs show RUN_ID` | the settings actually used (pinned image, template date, database snapshot), the source lineage, and every artifact verifying as **match** |
| 6 | Open `ranking_debug.json` in the run's AlphaFold directory and read `order[0]` | it names a model, e.g. `model_3` |
| 7 | `/confidence RUN_ID` | **the same model as step 6.** `ranked_0` is a rank, not `model_1`: if these disagree, stop and report it |
| 8 | Compare the band counts with the run's own `confidence_<model>.json` | counts and percentages agree; low-confidence intervals are input sequence positions, counted from 1 |
| 9 | Open the PNG the run produced | orange / yellow / light blue / dark blue regions, matching the bands on screen |
| 10 | `/pymol` | a PyMOL window opens showing the same structure, coloured by the same four bands as the PNG |
| 11 | In that window, pick a residue the terminal called "very low" | its colour is the very-low colour, not a confident one |
| 12 | Open a structure that is **not** an AlphaFold result: `/pymol /approved/experimental.pdb` | it opens in neutral grey — an experimental B-factor column must never be read as confidence |
| 13 | In `/confidence RUN_ID`, read the PAE overview | rows labelled aligned residue, columns scored residue, "maximum per block", an ångström legend, and the path to the full JSON |
| 14 | Check the PAE axes against the original JSON: pick `[i][j]` with a high value and a low `[j][i]` | the overview shows the high value in row *i*, column *j*. If it appears transposed, stop and report it |
| 15 | Quit the app, start it again, `/runs --project SMOKE` and `/audit` | the run and the fold's journal entries are still there |
| 16 | `/audit export "/approved/review/smoke.jsonl"` | a JSONL file of identifiers, hashes, decisions and error codes — no sequences, file contents or previews |

## Also worth doing once

* **Rendering failure.** Point `PYMOL_EXECUTABLE` at something that is not
  PyMOL and fold again: the prediction must survive, the manifest must say
  `completed_with_render_error`, and the structure must still be there.
* **Cancellation.** Interrupt a fold. The journal must show the fold's intent
  with no confirmed outcome, and must not claim it failed.
* **Approval, if the agent-pack server is reachable.** Approve a high-stakes
  answer twice in quick succession: exactly one decision reaches the server.

## Reporting

Record, for each numbered step: the run ID, what you saw, and any deviation.
Steps 7, 10, 12 and 14 are the ones that cannot be checked any other way — a
wrong answer to any of them is a release blocker, not a note.
