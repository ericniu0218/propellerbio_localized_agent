# Runbook: measuring fresh-inference repeatability on the office server

For the operator who runs the vLLM server, and for whoever signs off OPEN_ISSUES
item 14. The harness is built and tested offline. It has not yet been run
against a real server.

## What this does and does not answer

It answers one question: *on this deployment, does the same pinned request,
sent again with no cache, get the same reply?* A pass holds only for the
server, model, hardware and settings it was measured on. It is not a promise
across releases or hardware, and vLLM documents no default reproducibility
guarantee. Cached replay (`bench run`'s "same report twice (cached replay…)")
and recorded-output rendering are separate rows, and neither counts toward
this one.

## Before you start (needs approval)

- An agreed compute budget. One cell is 20 payloads × 20 repeats = 400 calls.
  Extraction payloads send about 3,500 prompt tokens and allow a 4,096-token
  reply.
- Permission for any **server restart**. The harness never restarts
  anything; the operator does, between cells.
- A deployment manifest, `deployment.json`, with every key in
  `MANIFEST_KEYS` (`src/bedrock/litreview/bench/determinism.py`): model
  revision and checksums, tokenizer and template, vLLM/Python/Torch/CUDA/driver
  versions, GPU type and count, TP/PP, quantisation and dtype, attention
  backend, compilation and CUDA-graph choices, the launch command, seed and
  sampling defaults, structured-output backend, speculative and prefix-cache
  settings, concurrency and schedule. Write `unknown` rather than guessing. A
  cell with any unknown is reported as *not fully pinned*.
- The model set in `~/.config/bedrock/litreview/models.yaml`, with `revision:`
  declared per model if the server does not report a weights digest (vLLM
  does not).

## Cells

Use one `--out` directory per day, and a new `--condition` name per cell. A
condition directory is never overwritten.

```sh
OUT=evals/litreview/results/determinism/$(date +%Y%m%d)
P=evals/litreview/determinism/payloads.jsonl
run() { python -m bedrock.litreview.bench determinism $P --mode fresh --models office \
          --repeats 20 --manifest deployment.json --out $OUT --condition "$1"; }

run same-process-serial                      # 1. one client process, one call at a time
run new-process-1; run new-process-2         # 2. separate processes, same server
# 3. operator restarts the server (authorised), then:
run after-restart-1                          #    repeat after each of at least three restarts
# 4. concurrency: the harness sends one call at a time, so start 2 (then 4) harness processes at once,
#    each with its own condition name, against a server that serves requests in parallel
run concurrency-2a & run concurrency-2b & wait
# 5. with other clients sending mixed short/long requests in parallel
run mixed-load
# 6. prefix cache cold (fresh server) versus warm (repeat immediately)
run prefix-cold; run prefix-warm
# 7. the production TP/quantisation configuration, if different from the above
run production-config
# 8. optional: VLLM_BATCH_INVARIANT=1 on a supported build (change only this)
run batch-invariant
```

Change **one** setting per cell. Record each change in that cell's manifest.
Before recommending any vLLM flag, check whether the deployed release still
has the issue that motivated it. Issue #51290 (a tensor-parallel reduction
problem under batch invariance) is closed and links #51292; this audit did
not check which release includes the fix.

Controls that need no server repeatability:

```sh
python -m bedrock.litreview.bench determinism $P --mode replay --cache <call-cache dir> --out $OUT --condition cached-replay
python -m bedrock.litreview.bench replay-control evals/litreview/questions/obesity_treatment_targets.yaml \
  --models office --corpora evals/litreview/corpora-searcher --work $OUT/replay-control
```

`replay-control` makes one live pipeline run and records its replies. It then
renders twice from those replies with no server. JSON, Markdown and HTML must
be byte-identical; if they are not, the defect is in the code, not the server.

## Reading the result

Each cell writes three things:

- `attempts.jsonl`, append-only. One line per call: payload and
  generation-context hashes, the wire attempts (including a
  `reasoning_effort` retry), the raw reply hash before thinking is stripped,
  the normalised reply hash, the status and the timing.
- `outputs/`, holding every distinct reply by hash.
- `summary.json`, holding the cell's row.

A cell **fails** in any of these cases:

- any call errored;
- any reply arrived without a completed server response (`no_backend_call`),
  for example because a cache answered;
- any payload had more than one distinct reply.

Publish, per cell:

- the number of repeats, failures and distinct replies per payload;
- the first diverging payloads;
- whether the manifest was fully pinned.

If bytes differ but a predeclared semantic check passes, report both facts.
Decide the semantic fields before looking: targets, modality, direction,
endpoints, numbers and citation support.

Until these cells have run, the product wording is: *deterministic
orchestration over pinned inputs; reproducible cached replay and rendering
from pinned outputs; fresh-inference repeatability is deployment-specific.*
