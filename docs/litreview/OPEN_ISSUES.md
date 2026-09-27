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
(item 25). **Run the benchmark after changing an accuracy rule, not only the
tests.**

Each item says what goes wrong, the evidence, where in the code, and a
suggested fix. "Size" is a rough guide: small is a day or less, medium is a
few days, big is a week or more, or needs a decision first.

Evidence labels such as "asthma S41" refer to the hand-audit sheets in
[`evals/litreview/hand_audit/`](../../evals/litreview/hand_audit/). Each
sheet shows a sampled report sentence beside the paper text around its quote.
`grades.jsonl` holds the grade and a note for every sentence of the first
audit (`qwen_*.md`, `claude_obesity.md`); `grades_after.jsonl` does the same
for the audit of the reports this branch produces (`qwen_*_after.md`). An
"after" label below means the second sheet. `audit_sheet.py` there rebuilds a
sheet from any stored report.

## Where the audit left things

Two measurements, before the section A fixes (25–26 September) and after
them (26 September, same questions, same frozen searcher corpora, same
qwen3:8b at temperature 0, replayed through the model-call cache).

- **Hand audit.** 15 random sentences per report, read against the full paper
  text. Sheets and grades: `qwen_*.md` / `grades.jsonl` for before,
  `qwen_*_after.md` / `grades_after.jsonl` for after.

  | | before | after |
  |---|---|---|
  | obesity | 13 of 15 | 12 of 15 |
  | asthma | 10 of 15 | 14 of 15 |
  | migraine | 15 of 15 | 14 of 15 |
  | **all three** | **38 of 45** | **40 of 45** |

  Claude as the model, same pipeline, obesity, before: 15 of 15.
- **Judge.** "Claims supported by their citations", strong judge through the
  file seam:

  | | before | after |
  |---|---|---|
  | obesity | 0.98 | 0.86 |
  | asthma | 0.87 | 0.95 |
  | migraine | 0.86 | 0.93 |

  Overall score (0–100): 72.2 → 66.6, 64.6 → 68.2, 67.0 → 80.0. Every gate
  passes on all three, including the benchmark-validity gate.
- **Read those numbers carefully.**
  - **They are not a controlled comparison.** The two runs were judged by
    different raters — see item 25 — and the claims themselves changed, so
    most answers are new. Treat them as indicative.
  - **The obesity drop is mostly the rater, not the pipeline.** Today's
    judges mark an added interpretation as unsupported: of the seven
    obesity claims that lost points, five add a clause the paper does not
    state ("suggesting a role for NOX2 in atrial remodeling", "indicating a
    protective effect against hyperglycemia", "to assess its preventive
    effects"). The earlier judge scored that shape *supported* — its own
    obesity run contains "…, suggesting a potential link…" marked supported.
    The same strictness applied to asthma and migraine, and both rose.
  - **The failure classes changed, which is the real result.** The errors
    the fixes targeted did not recur in the new sample: no invented
    efficacy, no dropped "not significant" qualifier, no other study's
    result presented as the paper's, no AhR/AHR collision, and no filler
    sentence in a findings section (the before-audit flagged three). What
    the after-audit found instead is one new class, all five of its
    non-faithful sentences: **the writer builds a sentence out of a
    qualifier or naming sentence while citing a quote that says something
    else** (item 3 below).
  - 15 sentences per report is a small sample, and two AI raters are not a
    scientist (item 21).
- **Code checks.** Every report passed every one, before and after. Quotes
  were always the paper's own words.
- **What the rerun cost.** About 25 minutes per question on a 16 GB M4 with
  a warm extraction cache, and the disk filled through swap once; see
  item 20.

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
3. **A sentence can be built out of the qualifier or the naming sentence
   instead of the quote.** *This is the one new failure class, and it was
   introduced by the fixes themselves.* Giving the reader the qualifying next
   sentence (the old item 1) and the sentence that names the target (the old
   item 6) also made both of them citable, and the writer now mines them: it
   writes a sentence whose content is one of those sentences, while citing a
   unit whose **quote** says something else. The guard cannot see it, because
   every word really is the paper's text.
   - What the naming half already does right: a target or agent found only
     elsewhere in the passage must now be in text the reader sees or the
     finding is dropped; the naming sentence is stored, anchored, rendered
     and given to the writer; and guard check 9 stops a sentence giving the
     quote's result to a name the quote does not state (the IL-23/IL-17A
     probe). 106 of 593 units gain a naming sentence, and no kept sentence
     was a misattribution of that kind.
   - *Evidence:* all five non-faithful sentences in the after-audit.
     - obesity S6 (*misleading*): "In animal studies, shRNA sequences
       targeting Gpr75, sh517 and sh1109, reduced Gpr75 mRNA and protein
       levels in HEK293 cells by approximately 80% and 60%" — cites a quote
       about female mice gaining weight; every number comes from the naming
       sentence, and "in animal studies … in HEK293 cells" contradicts
       itself.
     - obesity S18 and S29, migraine S5 (*weak_citation*): the sentence is
       the qualifier verbatim, and the cited quote is a different result
       (S29's quote is a textbook line about AMPK).
     - asthma S25 (*misleading*): "Leukotriene mediated airway inflammation
       **is associated with** insufficient response…" from a background
       sentence that says "**potentially due to** leukotriene mediated
       airway inflammation" — the hedge is dropped and the speculation is
       filed under mechanism.
   - *Fixed, after the measurement above.* Guard check 10
     (`report/guard.py` `rests_on_quote`) compares what a sentence takes
     from the quote alone with what it takes from the other displayed
     sentences alone, so a word both of them use — usually the target's
     name — decides nothing. A sentence that takes essentially nothing
     distinctive from the quote is dropped: "restates the context,
     qualifier or naming sentence, not the quote it cites". One cited quote
     the sentence really does rest on is enough to keep it, because it may
     be citing the others only for a name.
   - *Measured on the three new reports:* 8 of 133 kept sentences are
     dropped, all read — the four above, the two Ccl2/Ccr2 knockout
     sentences that cite a neutralizing-antibody quote, a MAIT IFN-γ
     sentence taken from its context, and the MEDI0618 binding half the
     judge had already called unsupported. On the three older reports it
     adds two, both genuine mis-citations. Tuning it caught two false drops
     first (a sentence that states the quote's result *and* the qualifier's
     limit, and one that adds the context's population to the quote's
     result); both are kept and both are tests.
   - *What is left:* the benchmark was **not** re-run after this check, so
     the accuracy numbers above predate it — they would rise a little, since
     five of the eight are sentences the judge marked down. Still open from
     the same audit: strip a leading "However,"/"Moreover," from a lifted
     sentence, and never drop a hedge the source sentence carries (asthma
     S25's "potentially due to").
   - *Size:* the rule is done; re-measuring is a benchmark run.
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
9. **Smaller things the after-audit saw, each small on its own.** All are in
   `grades_after.jsonl` with the sentence that shows them.
   - A **corrupted sentence reaches the reader**: obesity S13 begins
     "Hypoth, hypothalamic Sik3 mRNA levels …". Nothing checks a sentence for
     a dangling fragment or a repeated stem.
   - A **structured-abstract label sits inside a quote**: asthma E1's quote
     starts "Background and Purpose: Children with acute asthma…". The
     splitter does not strip `Background and Purpose:` / `Key Results:` /
     `Conclusion and Implications:`, which also defeats any section filter.
   - **Context selection is by proximity, not relevance**: obesity E74's
     context sentence is "Mice were excluded from experiments if they showed
     any sign of morbidity."; E190's is about transgenic overexpressing mice
     attached to a knockout result.
   - **Near-duplicate sentences still reach one report** (obesity S23/S24 on
     one unit; asthma S32/S34). The cross-section repeat check catches an
     identical sentence, not a reworded one.
   - **A drug or a metabolite is still filed as a target**: `targets:AEs`,
     `targets:SAEs`, `targets:eptinezumab`, `targets:acetyl-CoA`,
     `targets:15-keto-PGE2`. The outcome-measure list (item 2) covers the
     measures it knows; it does not cover a drug or a metabolite in the
     target slot.
   - **A direction label can contradict its own quote**: migraine E4 is
     `direction=activate` for MEDI0618, an antibody that blocks PAR2.
   - *Size:* small each; the sentence-sanity check and the abstract-label
     strip are the two worth doing first.

## B. Completeness

10. **Few of the known targets reach the report.**
    - *Evidence:* reference-target recall of 0.11 (obesity) and 0.06
      (asthma), against targets that are named in the collected papers. Only
      40 papers are graded and 16 read per question.
    - *Fix:* old steps 25–27 in `BENCHMARK_RESULTS.md`: pre-select by study
      design, plan by evidence type, put the key human numbers in the Answer.
    - *Size:* big.
11. **Background from reviews is not covered.**
    - *Evidence:* 0 of the obesity review facts were supported by the
      collected papers.
    - *Fix:* this is the client decision in old step 28.

## C. Evidence integrity

12. **Retraction status can be stale or incomplete.**
    - A separate retraction notice is not linked to its original.
    - "Expression of Concern" is not surfaced.
    - Retraction status is frozen at search time.
    - *Fix:* follow Europe PMC's correction links, add a flag that is shown
      (not an exclusion), and add the outside retraction-refresh lane (old
      step 54).
    - *Size:* medium.
13. **`verify-run` does not say the rules changed.**
    - It re-checks an old run under today's code and knowledge tables, and
      does not say when those changed since the run (`report/invariants.py`
      `verify_run`). The run records input hashes, so this is available.
    - *Fix:* print a note when a hash differs.
    - *Size:* small.
14. **Determinism on vLLM is unproven** (old step 36). Every benchmark run here
    re-ran identically, but through the call cache.

## D. Deployment and IT

15. **The `office` model set needs IT's real model names.**
    - It names two placeholder models at one server address
      (`defaults/models.yaml`).
    - *Fix:* IT decides the names, then ship them. See
      INTEGRATION_REVIEW fix #2 and old steps 31 and 33.
    - *Size:* big (decision).
16. **No re-runnable setup check.**
    - *Fix:* a `bedrock-litreview doctor` that runs preflight. It should
      also check free disk space, and say when a company server name needs
      `allowed_hosts`. See INTEGRATION_REVIEW #3 and #10.
    - *Size:* small to medium.
17. **`/review` in the app still uses chat's 120-second timeout.**
    - INSTALL.md now tells laptop users to set `request_timeout = 600`; the
      code does not warn.
    - *Fix:* warn in `/review` preflight on the laptop profile. See
      INTEGRATION_REVIEW #4.
    - *Size:* small.
18. **A review cannot be deleted.**
    - `/forget` is complete for chat, not for reviews.
    - *Fix:* add `/reviews forget` and `bedrock-litreview forget`. See
      INTEGRATION_REVIEW #8.
    - *Size:* medium.
19. **Smaller integration items.** All small; details in INTEGRATION_REVIEW.
    - report files are 0644 (#9)
    - a chat title can run between two review calls (#11)
    - no test checks that `package-data` covers every runtime file (#13)
    - no test checks the "only two modules import `bedrock.inference`" rule
      (#14)
    - the app shows a raw `ModuleNotFoundError` for a stale install (#16)
20. **A 16 GB laptop is not a review machine.**
    - qwen3:8b took about two hours per question on a 16 GB M4.
    - With ordinary apps open, its 6.4 GB footprint pushed the Mac into swap
      until the disk filled. The run had to be paused and resumed three
      times.
    - *Fix:* run reviews on the office server, or on machines with 32 GB or
      more, and add the disk check in item 17.

## E. Measurement

21. **No scientist has checked any accuracy number.** The judge and the hand
    audit were both AI models. A scientist should re-judge at least 40 claims
    blind (old step 23).
22. **The office model is unmeasured.**
    - Qwen3-30B-A3B sits between qwen3:8b and Claude.
    - *Fix:* run the five benchmark questions on it, cold and timed, and
      hand-audit 15 sentences per report as done here (old step 37).
23. **No gold set for extraction** (old step 22).
24. **The judge must be shown exactly what the reader is shown.** *Fixed, and
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
25. **Judge answers on file come from more than one judge.** The scores above
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

26. **Semantic retrieval.** Ranking is keyword-based (BM25), so a paper that
    says "pH" is not found for "acidity". This is blocked on a deployed
    embedding model; the plan is in `NEXT_STEPS.md` §6.
