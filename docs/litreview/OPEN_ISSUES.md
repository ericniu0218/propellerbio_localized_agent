# Literature review: open issues

The issues that were found and **not** fixed during the deploy-readiness audit
of 25–26 September 2026 (branch `litreview/deploy-readiness-audit`). The audit
fixed the small ones: guard and merge holes found by an adversarial probe, an
invented-result hole found by a hand audit, and the scientist-facing bugs and
docs listed in [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md). Everything
here is bigger: a design change, a decision, or work that needs the office
server.

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

16. **The `office` model set needs IT's real model names.**
    - It names two placeholder models at one server address
      (`defaults/models.yaml`).
    - *Fix:* IT decides the names, then ship them. See
      INTEGRATION_REVIEW fix #2 and old steps 31 and 33.
    - *Size:* big (decision).
17. **No re-runnable setup check.**
    - *Fix:* a `bedrock-litreview doctor` that runs preflight. It should
      also check free disk space, and say when a company server name needs
      `allowed_hosts`. See INTEGRATION_REVIEW #3 and #10.
    - *Size:* small to medium.
18. **`/review` in the app still uses chat's 120-second timeout.**
    - INSTALL.md now tells laptop users to set `request_timeout = 600`; the
      code does not warn.
    - *Fix:* warn in `/review` preflight on the laptop profile. See
      INTEGRATION_REVIEW #4.
    - *Size:* small.
19. **A review cannot be deleted.**
    - `/forget` is complete for chat, not for reviews.
    - *Fix:* add `/reviews forget` and `bedrock-litreview forget`. See
      INTEGRATION_REVIEW #8.
    - *Size:* medium.
20. **Smaller integration items.** All small; details in INTEGRATION_REVIEW.
    - report files are 0644 (#9)
    - a chat title can run between two review calls (#11)
    - no test checks that `package-data` covers every runtime file (#13)
    - no test checks the "only two modules import `bedrock.inference`" rule
      (#14)
    - the app shows a raw `ModuleNotFoundError` for a stale install (#16)
21. **A 16 GB laptop is not a review machine.**
    - qwen3:8b took about two hours per question on a 16 GB M4.
    - With ordinary apps open, its 6.4 GB footprint pushed the Mac into swap
      until the disk filled. The run had to be paused and resumed three
      times.
    - *Fix:* run reviews on the office server, or on machines with 32 GB or
      more, and add the disk check in item 17.

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
