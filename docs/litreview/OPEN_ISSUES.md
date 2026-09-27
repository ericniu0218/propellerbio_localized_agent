# Literature review: open issues

The issues that were found and **not** fixed during the deploy-readiness audit
of 25–26 September 2026 (branch `litreview/deploy-readiness-audit`). The audit
fixed the small ones: guard and merge holes found by an adversarial probe, an
invented-result hole found by a hand audit, and the scientist-facing bugs and
docs listed in [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md). Everything
here is bigger: a design change, a decision, or work that needs the office
server.

> **Update, 26 September 2026** (branch `litreview/deployment-and-it`): section
> **D** items 17, 18, 19 and 20 are done; 16 is blocked on IT and 21 is advice.
> Each carries a note on what it actually turned out to be — two of them were
> not quite what this list described, and one (the title between review calls)
> was a real scheduler bug now covered by a test that fails without the fix.
> Sections A, B, C and E are untouched.

Each item says what goes wrong, the evidence, where in the code, and a
suggested fix. "Size" is a rough guide: small is a day or less, medium is a
few days, big is a week or more, or needs a decision first.

Evidence labels such as "asthma S41" refer to the hand-audit sheets in
[`evals/litreview/hand_audit/`](../../evals/litreview/hand_audit/). Each
sheet shows a sampled report sentence beside the paper text around its quote,
and `grades.jsonl` holds the grade and a note for every sentence.
`audit_sheet.py` there rebuilds a sheet from any stored report.

## Where the audit left things

- **Hand audit.** 15 random sentences per report, read against the full
  paper text:
  - qwen3:8b: 38 of 45 faithful to their papers.
  - Claude as the model, same pipeline, obesity: 15 of 15.
- **Judge.** The benchmark's strong judge sees only the quote and its context
  sentence. It scored "claims supported by their citations" as follows:
  - qwen3:8b: 0.98 (obesity), 0.87 (asthma), 0.86 (migraine).
  - Claude as the model, obesity: 0.89.
- **What that means.**
  - The qwen3:8b gap between judge and hand audit is items 1–5 below: errors
    the judge cannot see, because the problem is in the paper text around
    the quote.
  - The Claude-model report lost its judge points mostly to item 6. A
    stronger writer attaches each finding's target to the sentence, and the
    quote does not say it.
  - 15 sentences per report is a small sample: the hand audit of that report
    did not happen to draw one of those sentences.
- **Code checks.** Every report passed every one. Quotes were always the
  paper's own words.

## A. Report accuracy

1. **A qualifier in the next sentence is lost.**
   - *Evidence:* obesity S6 reports a 0.6 kg/week setmelanotide difference.
     The paper's next sentence says it was not significant (p = 0.09).
   - *Where:* `extract/verify.py` `sentence_bounds` widens a quote only to its
     own sentence; the writer and the guard see nothing after it.
   - *Fix:* when the sentence after a quote starts with a qualifier
     ("Although", "However", "But", "did not reach significance"), store it
     as a second context sentence (like `context_quote`), show it, and give
     it to the writer and the guard. Add tests built from this case.
   - *Size:* medium.
2. **Other people's results are reported as the paper's own.**
   - *Evidence:* asthma S9. A birth-cohort result that the paper quotes from
     earlier work, with a reference bracket in its Introduction, is cited as
     the paper's finding.
   - *Where:* extraction rule 2 in `llm/prompts/extract.md` asks the model to
     skip these; nothing in code enforces it (`stages/s5_extract.py`, after
     the quote is located).
   - *Fix:* drop a finding whose quote ends in a citation marker (`[]`,
     `[12]`, `(Smith et al., 2019)`) or comes from an Introduction or
     Background section.
   - *Size:* small to medium.
3. **Aims, hypotheses and predictions become findings.**
   - *Evidence:*
     - asthma S32: "In this study, we investigated the role of AhR in…" is
       cited for "AhR regulates…".
     - obesity S56: "cold exposure is therefore predicted to benefit those
       patients", from mouse work, is cited as prevention evidence.
     - migraine S23: a methods sentence stands in for a finding.
   - *Where:* `stages/s5_extract.py`, `report/guard.py`.
   - *Fix:* a code rule for sentence type ("we investigated", "we aimed", "we
     hypothesized", "is predicted to", "may offer"). Such a sentence is never
     a finding, and a result sentence may not cite it.
   - *Size:* medium.
4. **A conclusion sentence can overstate who was studied.**
   - *Evidence:* asthma S3 repeats the authors' "…in patients with asthma";
     the work was in a cell line and mice. The who-was-studied check passes
     because the quote says "patients".
   - *Fix:* when a quote is a conclusion sentence ("In summary", "Our
     findings", "This study revealed") and names humans, check the paper's
     Methods for human subjects. If there are none, show the Methods species
     and do not let a sentence say "patients".
   - *Size:* medium.
5. **Abbreviations and outcome measures become targets.**
   - *Evidence:*
     - asthma S34 and S52: "AHR" (airway hyperresponsiveness) is filed under
       the target AhR (aryl hydrocarbon receptor).
     - migraine S9: "MMDs" (monthly migraine days) is listed as a target.
   - *Where:* target normalisation (`knowledge/__init__.py` `Knowledge.normalize`,
     `looks_like_target`), `stages/s6_score.py`.
   - *Fix:*
     - Read the paper's own definitions ("airway hyperresponsiveness (AHR)")
       before mapping an abbreviation to the knowledge table.
     - Keep a list of outcome measures that are never targets (AHR, MMD,
       PASI, BMI, FEV1…).
   - *Size:* medium.
6. **A target named only elsewhere in the passage can take the quote's
   result.**
   - *Evidence:* the adversarial probe. The quote is about IL-23, the passage
     also names IL-17A, and a sentence saying "Blocking IL-17A reduced ear
     thickness by 60%" passes every check.
   - *Also seen in a real report:* in the Claude-as-model obesity report, the
     writer names the finding's target beside the agent: "tubastatin A,
     acting on HDAC6", "REGN4461, which acts on LEPR", "the MC4R agent
     setmelanotide". Each is true of the paper's passage but not stated in
     the quote the reader sees. The judge marked about seven such sentences
     partially supported.
   - *Where:* `extract/provenance.py` `trace` accepts a target found in the
     chunk; `report/guard.py` `unit_text` then lets a sentence name it.
   - *Fix:* require the target in the quote or in a displayed context
     sentence. Otherwise store the sentence that names it as context, so the
     reader can see it.
   - *Size:* medium.
7. **The same sentence appears in two sections.**
   - *Evidence:* obesity S4/S23, migraine S20/S37, and the Claude-model
     report.
   - *Where:* the repeat check in `stages/s7_write.py` works within one
     section.
   - *Fix:* check repeats across the whole report.
   - *Size:* small.
8. **Filler sentences sit in findings sections.**
   - *Evidence:* migraine S23 (methods) and S32 (demographics); asthma S57
     (background).
   - *What already exists:* the new guard rule drops a sentence that claims
     an effect without a result-bearing quote. It does not catch sentences
     that claim nothing.
   - *Fix:* in findings sections, require every sentence to cite a
     result-bearing quote.
   - *Size:* small.
9. **Model labels are sometimes wrong and are used downstream.**
   - *Evidence:*
     - human tissue labelled `human_genetic` (obesity S34)
     - a cell line labelled `in_vivo` (asthma S13)
     - an association labelled `worsened` (obesity S3)
   - *Fix:* set the evidence type from the quote and Methods words in code,
     as the species already is. See old step 15 in
     [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).
   - *Size:* medium.
10. **Direction and negation checks are word-level, not meaning.**
    - *What exists:* the guard now catches the flips the probe found. Its own
      docstring says it cannot catch every rewording.
    - *Fix, long-term:* a local entailment check between each sentence and
      its quote, measured against the judge and a scientist before it is
      trusted.
    - *Size:* big.

## B. Completeness

11. **Few of the known targets reach the report.**
    - *Evidence:* reference-target recall of 0.11 (obesity) and 0.06
      (asthma), against targets that are named in the collected papers. Only
      40 papers are graded and 16 read per question.
    - *Fix:* old steps 25–27 in `BENCHMARK_RESULTS.md`: pre-select by study
      design, plan by evidence type, put the key human numbers in the Answer.
    - *Size:* big.
12. **Background from reviews is not covered.**
    - *Evidence:* 0 of the obesity review facts were supported by the
      collected papers.
    - *Fix:* this is the client decision in old step 28.

## C. Evidence integrity

13. **Retraction status can be stale or incomplete.**
    - A separate retraction notice is not linked to its original.
    - "Expression of Concern" is not surfaced.
    - Retraction status is frozen at search time.
    - *Fix:* follow Europe PMC's correction links, add a flag that is shown
      (not an exclusion), and add the outside retraction-refresh lane (old
      step 54).
    - *Size:* medium.
14. **`verify-run` does not say the rules changed.**
    - It re-checks an old run under today's code and knowledge tables, and
      does not say when those changed since the run (`report/invariants.py`
      `verify_run`). The run records input hashes, so this is available.
    - *Fix:* print a note when a hash differs.
    - *Size:* small.
15. **Determinism on vLLM is unproven** (old step 36). Every benchmark run here
    re-ran identically, but through the call cache.

## D. Deployment and IT

**Items 17–20 were fixed on branch `litreview/deployment-and-it`.** What each
one turned out to be is recorded below, because two of them were not quite what
this list said. Item 16 is still blocked on IT, and 21 is advice rather than
work.

16. **The `office` model set needs IT's real model names.** *Still open —
    blocked on IT, not on code.*
    - It names two placeholder models at one server address
      (`defaults/models.yaml`).
    - *What was done:* the code half, which is everything except the names.
      `defaults/models.yaml` now says they are guesses and why (a vLLM server
      answers to its `--served-model-name`, usually the full path such as
      `Qwen/Qwen3-30B-A3B-Instruct-2507`). When the server lacks a model,
      preflight and `doctor` print the names the server *does* offer and the
      path of the `models.yaml` override to write the right ones into, so a site
      can fix this without waiting for a release. A test covers that override
      for the `office` set specifically.
    - *Not done, and cannot be:* the names. IT has to say what the server serves.
      Then correct `defaults/models.yaml` and ship it.
    - *Assessed risk, lower than it looks:* nothing goes silently wrong. A wrong
      name fails preflight before a run starts, with the fix written out.
    - *Size:* small, once the decision exists.
17. **No re-runnable setup check.** *Fixed.*
    - `bedrock-litreview doctor` and `/reviews check`. Every check runs, not
      just the first: window, `request_timeout`, free disk, `allowed_hosts`,
      server reachability, whether the server has the models the set names — and
      which `models.yaml`, `pipeline.yaml` and `report.yaml` were actually read,
      because "the model set is wrong" is unactionable until you know which of
      two files it came from. `--no-server` skips the server half, which is what
      a machine looks like before Ollama is started.
    - The checks live once, in `service.config_findings` / `server_findings`.
      `preflight` raises on the first blocking one; `findings` returns all of
      them. A check cannot exist in the doctor and not in a run's preflight.
    - *Honest gap:* the free-disk threshold (10 GB) is a judgement call, not a
      measurement, and the code says so. It is about the model server and swap —
      a review's own files are megabytes. Worth replacing with a measured figure
      after a full run on the office server and on a 32 GB laptop.
18. **`/review` in the app still uses chat's 120-second timeout.** *Fixed.*
    - `/review` now says so before the run starts, not after — the point is to
      go and change it, and a review takes hours.
    - *One thing this list had wrong:* litreview's `Settings` had no
      `request_timeout` and no profile name, so the warning could not be written
      where the issue said. Both are now copied from bedrock's settings (`doctor`
      needed them too).
    - *A false alarm that had to be avoided:* `bedrock-litreview` and the HTTP
      API already raise their own timeout to 600 through `standalone_scheduler`,
      so warning them would be telling them to fix something already fixed.
      `timeout_floor_applied` distinguishes the two cases; `doctor` deliberately
      reports the configured value, because it describes the machine rather than
      one process.
    - *The better fix, not done:* `RequestOptions` has no `timeout`, so a review
      cannot ask for a longer one per call. Adding it is ~10 lines in
      `inference/openai_compatible.py` and would remove the need for the warning
      — but it touches the layer the chat uses, which is why this is a warning
      for now.
19. **A review cannot be deleted.** *Fixed, scoped to the run's own output.*
    - `bedrock-litreview forget <run-id>` and `/reviews forget <run-id>`. The
      run, its stage artifacts and its rendered report. Not reversible.
    - *Deliberately narrow:* the question, its frozen paper set and every stored
      paper version stay. `paper_versions` is append-only and is what lets other
      runs be re-proved against the exact bytes they read, so deleting one to
      tidy up after a run would break `verify-run` for every run that shared it
      — silently, until somebody ran it months later.
    - *A hazard this list did not mention:* a report id is the hash of the report
      body, and an identical re-run is *designed* to produce identical bytes, so
      two runs share one `reports` row and one directory. Deleting it with the
      first run would leave the second finished and reportless. `runs_for_report`
      reference-counts, and the report goes only when nothing else needs it.
    - The CLI confirms and shows the question first (run ids are hashes, and the
      way this goes wrong is deleting a neighbour); the app deletes on the first
      ask, as `/digest forget` does, and says exactly what went.
20. **Smaller integration items.** *All five fixed.*
    - report files are 0644 (#9) — now 0700 directories and 0600 files, via the
      `private_dir` helper that already existed. Not a live leak on the default
      path, whose parent is already 0700; it mattered because `--out` may point
      at a shared drive.
    - a chat title can run between two review calls (#11) — **real, and confirmed
      by a test that fails without the fix.** The scheduler restarted its idle
      clock only for `Priority.CHAT`, and a review is `USER_JOB`, so every gap
      between two review calls looked idle. On a laptop holding one model a title
      then unloads the review's model and loads the chat's. Three lines: any
      non-`BACKGROUND` call now restarts the clock. README:252 said the opposite
      and has been corrected.
    - no test checks that `package-data` covers every runtime file (#13) —
      `tests/test_package_data.py`. The globs did cover all 13 files; the test
      stops the next one drifting, and also fails on a glob that has stopped
      matching anything.
    - no test checks the "only two modules import `bedrock.inference`" rule (#14)
      — `tests/litreview/test_inference_boundary.py`, an `ast` walk. It also
      fails if an allow-listed module stops needing the exemption.
    - the app shows a raw `ModuleNotFoundError` for a stale install (#16) —
      **the location in INTEGRATION_REVIEW was wrong.** It names `app.py:385`,
      but `start_review` calls `parse_review_command` first, and *that* imports
      the pipeline (`review_command.py:96`). Catching it only where the review
      said would still have crashed. Both are now handled, with a notice naming
      the four packages and the reinstall command.
21. **A 16 GB laptop is not a review machine.** *Advice, not work.*
    - qwen3:8b took about two hours per question on a 16 GB M4.
    - With ordinary apps open, its 6.4 GB footprint pushed the Mac into swap
      until the disk filled. The run had to be paused and resumed three
      times.
    - *Fix:* run reviews on the office server, or on machines with 32 GB or
      more. The disk check asked for here is in item 17 and is done.

## E. Measurement

22. **No scientist has checked any accuracy number.** The judge and the hand
    audit were both AI models. A scientist should re-judge at least 40 claims
    blind (old step 23).
23. **The office model is unmeasured.**
    - Qwen3-30B-A3B sits between qwen3:8b and Claude.
    - *Fix:* run the five benchmark questions on it, cold and timed, and
      hand-audit 15 sentences per report as done here (old step 37).
24. **No gold set for extraction** (old step 22).

## Long-term

25. **Semantic retrieval.** Ranking is keyword-based (BM25), so a paper that
    says "pH" is not found for "acidity". This is blocked on a deployed
    embedding model; the plan is in `NEXT_STEPS.md` §6.
