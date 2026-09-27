# Literature review: open issues

The issues that were found and **not** fixed during the deploy-readiness audit
of 25–26 September 2026 (branch `litreview/deploy-readiness-audit`). The audit
fixed the small ones: guard and merge holes found by an adversarial probe, an
invented-result hole found by a hand audit, and the scientist-facing bugs and
docs listed in [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md). Everything
here is bigger: a design change, a decision, or work that needs the office
server.

**26 September 2026, section A.** Every report-accuracy item in section A has
since been worked through. Five are gone from this list because code now
enforces them and a test built from the real case fails without the fix: a
qualifier in the paper's next sentence travels with the quote; a finding that
reports someone else's result is dropped; an aim, method or prediction is
never a finding; a sentence is never repeated across sections; and every
sentence in a findings section must cite a quote that reports a result. The
rest stayed, smaller and better understood, and are rewritten below with what
was fixed and what is left. Each rule was measured on the three stored
qwen3:8b reports (593 evidence units, 179 kept sentences) before it was
accepted, and the drops were read one by one.

Re-running the benchmark afterwards was worth more than the unit tests: it
found three bugs no test had (items A5, A6 and A7 in the fixed list — a
methods sentence counting as a result because "diet-induced" contains
"induced", the guard trusting the extraction model's own `no_effect` label,
and a hypothesis reported as a finding) and one flaw in the benchmark itself
(item 26). **Run the benchmark after changing an accuracy rule, not only the
tests.**

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

The five items that are gone, and what enforces them now, so a later change
does not quietly undo one:

| Was | Now enforced by |
|---|---|
| A qualifier in the next sentence is lost | `extract/verify.py` `qualifier_after`; stored as `EvidenceUnit.qualifier_quote`, shown to the reader and the writer, and guard check 8 drops a sentence that restates the quote's result without the paper's "not significant" |
| Other people's results reported as the paper's own | `extract/verify.py` `other_work_reason`, applied in `stages/s5_extract.py` (citation markers including Europe PMC's stripped `[]`, reported-other-work lead-ins, a section the parser could only call "other") |
| Aims, hypotheses and predictions become findings | `extract/verify.py` `not_a_result_reason` (aim, hope and hypothesis cues, each scoped to its clause) plus guard checks 6 and 7 |
| The same sentence in two sections | `stages/s7_write.py`, one `already_kept` list threaded through every section |
| Filler in findings sections | a per-section `findings` flag in `defaults/report.yaml`; guard check 6 requires a result-bearing quote |

Three more were fixed while re-running the benchmark: a disease-model
adjective ("Diet-induced obesity … were infused with") no longer counts as a
reported result; a sentence may claim no effect only when the paper's own
displayed text says so, never on the extraction model's `outcome` label,
which had let a report state the opposite of its paper; and an explicit
hypothesis ("cold exposure might induce …") is not a finding.

What is left:

1. **A conclusion can still overstate which experiment backs it.** *Partly
   fixed.* When a quote is the authors' conclusion, names people, and the
   paper's Methods (or the abstract's METHODS part) studied only animals or
   cells, the finding now takes its subjects from the Methods, shows that
   Methods sentence, and the guard drops a sentence citing it that says
   "patients" (`extract/provenance.py` `methods_subjects`). On the three
   stored reports exactly one unit is caught, correctly.
   - *What is left:* the case that prompted it, asthma S3, is **not** caught.
     That paper did recruit 64 patients, for sputum; the mechanism was shown
     in BEAS-2B cells and mice. So the Methods do name people, and the
     overstatement is about *which experiment supports which half of the
     conclusion*. No word rule decides that. It needs the claim matched to
     the experiment it came from — closest to the entailment check (item 5).
   - *Also:* a paper with no Methods text is left alone on purpose (many
     open-access records are abstract-only); nothing is guessed.
   - *Size:* medium.
2. **An abbreviation a paper never defines can still merge with a target.**
   *Mostly fixed.* The paper's own definitions are read first
   (`knowledge/abbreviations.py`, Schwartz–Hearst), an outcome-measure list
   (`knowledge/outcome_measures.yaml`) is never a target, and a flagged
   finding keeps its evidence but gets no target row. Across the five corpora
   (1,040 papers, 8,440 definitions) 20 papers define AHR as airway
   hyperresponsiveness and one as the receptor; all are handled.
   - *What is left:* three papers use "AHR" and never spell it out. One means
     the gene, two mean airway hyperresponsiveness. An undefined "AHR" can
     still merge into the AhR row, as in the old asthma S34.
   - *Fix:* fall back to the disease context (an asthma paper's bare "AHR" is
     the measure), or ask for a scientist's ruling once and store it.
   - *Size:* small.
3. **A displayed naming sentence can carry a whole report sentence.** *Partly
   fixed.* A target or agent found only elsewhere in the passage must now be
   in text the reader sees, or the finding is dropped: the sentence that
   names it is stored, anchored, rendered and given to the writer, and guard
   check 9 stops a sentence giving the quote's result to a name the quote
   does not state (the IL-23/IL-17A probe). 106 of 593 units gain a naming
   sentence; no kept sentence was a misattribution.
   - *What is left:* the naming sentence is citable text, so its numbers are
     citable too, and the writer can build a sentence almost entirely out of
     it. In the new obesity report, S6 reads "In animal studies, shRNA
     sequences targeting Gpr75, sh517 and sh1109, reduced Gpr75 mRNA and
     protein levels in HEK293 cells by approximately 80% and 60%", citing a
     quote about female mice gaining weight; every number comes from the
     naming sentence, and "in animal studies … in HEK293 cells" contradicts
     itself.
   - *Fix:* a sentence's result must come from the quote; a naming sentence
     may supply a name, not a finding. Consider citing the naming sentence as
     its own unit instead.
   - *Size:* medium.
4. **Two evidence labels are still wrong on purpose.** *Mostly fixed.* The
   evidence type now comes from the quote's own words (cells in a dish are
   `in_vitro` even when the context names mice; human tissue is
   `human_clinical` unless genetic words are present), and an observation
   with nothing done to its target is an `association`, not `improved` or
   `worsened`. On the stored reports 43 evidence types and 12 outcomes
   change, each read; every correction is recorded on the unit and counted.
   - *What is left:* two accepted misfits — a background sentence about
     postmenopausal women in a mouse paper becomes `human_clinical` (correct
     for what that sentence says, wrong for the paper), and a finding where
     EP300 inhibition changed PQBP1 lactylation becomes an `association`
     because nothing was done to PQBP1 itself.
   - *Size:* small.
5. **Direction and negation are still word-level.** A local entailment check
   now exists (`report/entailment.py`, prompt `llm/prompts/entail.md`,
   setting `entailment_check` in the `[litreview]` table) and is **off by
   default and unmeasured**. It asks the local model, through the same
   scheduler and at temperature 0, whether the displayed paper text supports
   each kept sentence, and drops the ones it calls unsupported.
   `evals/litreview/entailment_agreement.py` compares it with the strong
   judge.
   - *Fix:* run the agreement tool on the benchmark reports, then have a
     scientist check a sample, before it is turned on anywhere.
   - *Size:* big (measurement, then a decision).
6. **A paper with no headings is filed wholly as "results".** The parser's
   fallback files unlabelled text as `results`
   (`parsing/sections.py` `_Collector.result`), so a paper's Introduction and
   Discussion are read as results, and `from_discussion` is never set. Two
   asthma papers in the corpus are like this (PMID41858165, PMID41596081).
   This is why the section rule in the someone-else's-result check cannot
   catch intro text, and why the citation-marker rule has to do that work.
   - *Fix:* when a paper has no headings, do not claim it is results: mark
     the section "unknown" and treat it as discussion for flagging.
   - *Size:* small to medium.
7. **Background from another study with no citation marker is still taken as
   the paper's own.** The new obesity report opens a target section with
   "Human genetic studies have identified GPR75 loss-of-function variants to
   be strongly protective against obesity" — true, but it is other people's
   human genetics quoted in the background of a mouse paper, and its abstract
   carries no reference brackets for the rule to see.
   - *Fix:* treat "Human genetic studies have identified", "It is well
     established", "have been shown to" and the like as reported-other-work
     lead-ins wherever they start a sentence in an abstract's background, not
     only at the start of the quote.
   - *Size:* small.
8. **The result vocabulary is a word list, and will keep having holes.** It
   decides whether a quote reports a result at all, so a missing word throws
   real evidence away and a word in the wrong place keeps filler. Both
   happened in one day: the findings-section rule falsely dropped seven real
   results ("led to weight loss", "potentiate", "no differences", "results
   in", "highly expressed … compared to", "similar between", "most frequent
   AEs") until the list was widened, and "Diet-induced" made a methods
   sentence look like a result until model adjectives were masked.
   - *Fix:* this is the same need as item 5 — a meaning check, measured
     first. Until then, any change to `RESULT_WORDS` must be re-measured on
     the stored reports with `recheck.py`.
   - *Size:* medium, or big if folded into item 5.

## B. Completeness

9. **Few of the known targets reach the report.**
    - *Evidence:* reference-target recall of 0.11 (obesity) and 0.06
      (asthma), against targets that are named in the collected papers. Only
      40 papers are graded and 16 read per question.
    - *Fix:* old steps 25–27 in `BENCHMARK_RESULTS.md`: pre-select by study
      design, plan by evidence type, put the key human numbers in the Answer.
    - *Size:* big.
10. **Background from reviews is not covered.**
    - *Evidence:* 0 of the obesity review facts were supported by the
      collected papers.
    - *Fix:* this is the client decision in old step 28.

## C. Evidence integrity

11. **Retraction status can be stale or incomplete.**
    - A separate retraction notice is not linked to its original.
    - "Expression of Concern" is not surfaced.
    - Retraction status is frozen at search time.
    - *Fix:* follow Europe PMC's correction links, add a flag that is shown
      (not an exclusion), and add the outside retraction-refresh lane (old
      step 54).
    - *Size:* medium.
12. **`verify-run` does not say the rules changed.**
    - It re-checks an old run under today's code and knowledge tables, and
      does not say when those changed since the run (`report/invariants.py`
      `verify_run`). The run records input hashes, so this is available.
    - *Fix:* print a note when a hash differs.
    - *Size:* small.
13. **Determinism on vLLM is unproven** (old step 36). Every benchmark run here
    re-ran identically, but through the call cache.

## D. Deployment and IT

14. **The `office` model set needs IT's real model names.**
    - It names two placeholder models at one server address
      (`defaults/models.yaml`).
    - *Fix:* IT decides the names, then ship them. See
      INTEGRATION_REVIEW fix #2 and old steps 31 and 33.
    - *Size:* big (decision).
15. **No re-runnable setup check.**
    - *Fix:* a `bedrock-litreview doctor` that runs preflight. It should
      also check free disk space, and say when a company server name needs
      `allowed_hosts`. See INTEGRATION_REVIEW #3 and #10.
    - *Size:* small to medium.
16. **`/review` in the app still uses chat's 120-second timeout.**
    - INSTALL.md now tells laptop users to set `request_timeout = 600`; the
      code does not warn.
    - *Fix:* warn in `/review` preflight on the laptop profile. See
      INTEGRATION_REVIEW #4.
    - *Size:* small.
17. **A review cannot be deleted.**
    - `/forget` is complete for chat, not for reviews.
    - *Fix:* add `/reviews forget` and `bedrock-litreview forget`. See
      INTEGRATION_REVIEW #8.
    - *Size:* medium.
18. **Smaller integration items.** All small; details in INTEGRATION_REVIEW.
    - report files are 0644 (#9)
    - a chat title can run between two review calls (#11)
    - no test checks that `package-data` covers every runtime file (#13)
    - no test checks the "only two modules import `bedrock.inference`" rule
      (#14)
    - the app shows a raw `ModuleNotFoundError` for a stale install (#16)
19. **A 16 GB laptop is not a review machine.**
    - qwen3:8b took about two hours per question on a 16 GB M4.
    - With ordinary apps open, its 6.4 GB footprint pushed the Mac into swap
      until the disk filled. The run had to be paused and resumed three
      times.
    - *Fix:* run reviews on the office server, or on machines with 32 GB or
      more, and add the disk check in item 17.

## E. Measurement

20. **No scientist has checked any accuracy number.** The judge and the hand
    audit were both AI models. A scientist should re-judge at least 40 claims
    blind (old step 23).
21. **The office model is unmeasured.**
    - Qwen3-30B-A3B sits between qwen3:8b and Claude.
    - *Fix:* run the five benchmark questions on it, cold and timed, and
      hand-audit 15 sentences per report as done here (old step 37).
22. **No gold set for extraction** (old step 22).
23. **The judge must be shown exactly what the reader is shown.** *Fixed, and
    worth guarding.* The report grew two new displayed sentences (the
    qualifier, the naming sentence) before `bench/run.py` `_unit_evidence`
    did, so for one round the judge saw less than the reader and marked
    claims unsupported for a drug named directly under the quote. That is
    the benchmark measuring its own blind spot.
    - *Fix, done:* `_unit_evidence` now shows every sentence
      `report/guard.py` `_shown` displays.
    - *Left:* nothing enforces it. A test that fails when a displayed field
      is added to `_shown` but not to the judge's evidence would.
    - *Size:* small.
24. **Judge answers on file come from more than one judge.** The scores below
    mix answers written by an earlier strong judge (the 25–26 September runs)
    with answers written later for the claims this branch changed. Both were
    Claude models, but they are not the same rater, and the earlier ones were
    not re-checked. Before-and-after accuracy numbers are therefore
    indicative, not a controlled comparison.
    - *Fix:* re-judge a fixed sample of both runs with one rater in one
      sitting, or store the judge's identity beside each answer and report
      per-rater agreement.
    - *Size:* small to medium.

## Long-term

25. **Semantic retrieval.** Ranking is keyword-based (BM25), so a paper that
    says "pH" is not found for "acidity". This is blocked on a deployed
    embedding model; the plan is in `NEXT_STEPS.md` §6.
