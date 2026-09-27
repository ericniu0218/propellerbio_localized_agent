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

**27 September 2026, the mechanical fix batch.** Eight more items, all
structural or reusing a mechanism already trusted elsewhere, are done:
commits `5373d99..de1915b` plus the correction in `0a2822f`. Six more rows are
in the section A table below, and items 3, 6, 7, 9, 13 and 24 are rewritten
with what is left. The rule held again, twice over. The benchmark found that
the new hedge rule's only firing was a **false drop** that deleted a faithful
sentence, with every unit test green before and after; and that obesity's "no
distractor leaked" gate had been passing because no distractor was ever built,
so "every gate passes on all three" above was not a measurement. Three new
measurement items (26, 27, 28) record why the scores in this document cannot
be compared across runs: the benchmark never checks its context window, a gate
can pass on absent input, and the file judge is keyed by claim text so
rewording sentences leaves claims unjudged. See
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

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
| A structured-abstract label inside the quote ("Key Results: Among 45 participants ...") | `extract/verify.py` `strip_leading_label`, applied to abstract-section spans in `stages/s5_extract.py` and `extract/provenance.py`, advancing `start` so `section[start:end] == text` still holds; `integrity check passes` and `quote_validity` 1.00 on all three corpora after it |
| A duplicated, truncated first word reaching the reader ("Hypoth, hypothalamic Sik3 mRNA ...") | `stages/s7_write.py` `_garbled_start`, anchored to the sentence opening; counted as `garbled` on the methods "Written:" line. Fired exactly once on the obesity rerun, on exactly that sentence |
| A leftover "However," on a lifted sentence | `stages/s7_write.py` `_strip_leading_connective`, after the guard has judged the model's real words. Confirmed on the obesity rerun's SIK3 sentence |
| Other people's human genetics read as the paper's own finding | `extract/verify.py` `_OTHER_WORK_LEAD_IN`, four more phrases, each with a recorded occurrence. The GPR75 sentence was a claim before the batch and is gone after it |
| Unlabelled text in a paper with no recognised structure filed as "results" | `parsing/sections.py` files it under a new "unknown" bucket when none of methods/results/discussion was recognised, and `extract/verify.py` `_INTRO_LIKE_SECTIONS` treats "unknown" like "other". **Not measurable on any stored corpus** - see item 6 |

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
   - *Both leftovers are now done, and the second one is the cautionary tale
     of this branch.* A kept sentence's leading "However, " is stripped in
     `stages/s7_write.py` after the guard has judged the model's real words.
     A hedge the cited evidence carries may no longer be dropped
     (`negation.py` `drops_hedge`, wired into guard check 5 over the quote,
     context, qualifier and naming sentences together).
   - *"Moreover," is deliberately NOT stripped.* `report/invariants.py`
     `check_sentences` re-runs the guard over the *stored* sentence on every
     `verify-run`, so a post-guard rewrite must be verdict-neutral. "however"
     is in `extract/verify.py`'s `_FILLER` and so never reaches
     `content_words()`; "moreover" is not, and removing it was measured
     flipping guard check 8's overlap ratio and its verdict - which would
     report a correctly written sentence as a violation forever. Making it
     safe means putting "moreover" in `_FILLER`, which moves checks 8 and 10
     for every sentence of every report and needs its own measurement. No
     kept sentence in the three hand audits opens with "Moreover,".
   - *The hedge rule shipped with a false drop, and only the benchmark found
     it.* Its single firing on the obesity rerun deleted a faithful sentence:
     obesity E189's qualifier states a fact and then speculates about its
     cause ("... did not elicit a significant body weight difference between
     the genotypes, presumably because the already rapid diet-induced weight
     gain ..."), and the report sentence repeated only the fact. Every unit
     test was green before and after. Narrowed in `0a2822f`: a word appearing
     before the cue belongs to the stated fact, not the guess, and a claim
     carrying any hedge of its own is left alone. That second narrowing is
     the cautious side of the trade on purpose - a claim hedging one clause
     while unhedging another is now missed, because a missed hedge leaves a
     claim slightly too firm where a false drop deletes a faithful sentence.
   - *Size:* done. Obesity should be re-run once to confirm the false drop is
     gone; the fix has a test from the real sentences either way.
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
   *Fixed in code, and not measurable.* `parsing/sections.py` files the
   leftover text under a new "unknown" bucket when none of methods, results
   or discussion was recognised anywhere, and `extract/verify.py`
   `_INTRO_LIKE_SECTIONS` treats "unknown" like "other".
   - *What the fix taught us:* both named papers DO have headings. They are
     numbered - "1. Introduction", "2. Materials and Methods", "3. Results" -
     and every pattern in `_HEADING_MAP` anchors at `^\s*`, so none of them
     matched and only the unnumbered back matter did. "No heading matched at
     all" is therefore the wrong test: `parts` is not empty for these papers.
     The signal used is that none of methods/results/discussion was
     recognised; "abstract" cannot count (its own XML tag) and neither can
     "other" (it holds a real Introduction heading and pure boilerplate
     indistinguishably).
   - *What is left, and it is the real fix:* **`_HEADING_MAP` never matches a
     numbered heading.** Tolerating a numeral prefix would file both papers
     correctly instead of merely refusing to guess. Until then they are
     abstract-only: `Paper.has_full_text` is now False for them, so the
     planner asks only for their abstracts - less coverage, and no longer
     presenting Introduction prose as findings.
   - *What no stored corpus can measure:* `ingest.prepare()` sets
     `paper.full_text = None` after the one-time split at corpus-build time
     and re-splits only when `sections` is empty. All three corpora were built
     2026-09-25, so all 208 of their papers carry old-parser sections with no
     raw text left. The "unknown" bucket is unreachable through them, for
     every paper, not just these two. An unchanged benchmark number is not
     evidence the fix is inert. Measuring it needs a corpus rebuilt from
     source, and the numbered-heading fix should land first or the rebuild
     will file both papers as "unknown" again.
   - *Size:* small (the numbered headings); medium (a corpus rebuild).
7. **Background from another study with no citation marker is still taken as
   the paper's own.** The new obesity report opens a target section with
   "Human genetic studies have identified GPR75 loss-of-function variants to
   be strongly protective against obesity" — true, but it is other people's
   human genetics quoted in the background of a mouse paper, and its abstract
   carries no reference brackets for the rule to see.
   *Fixed.* `extract/verify.py` `_OTHER_WORK_LEAD_IN` gained "human genetic
   studies have identified", the plural "have been shown to", "genetic
   studies have shown" and "genome-wide association studies have identified",
   each with an occurrence recorded in the comment above the pattern.
   - *Evidence:* the GPR75 sentence was a claim in the pre-batch obesity
     report and is gone from the rerun; `evidence_units` 206 to 205, and no
     other claim was lost to these phrases.
   - *Deliberately not added:* "it is well established". Its only occurrence
     in the corpora is mid-sentence and means something else entirely ("we
     selected the BALB/c mouse model because it is well established in our
     laboratory"). The value of this list is that every entry is evidenced,
     so an unevidenced phrase stays out even when this document named it.
   - *Checked, and not a gap:* `other_work_reason` is called only on
     `finding.quote`, never on the context, qualifier or naming sentences. No
     currently-kept unit in any hand audit has one of those carrying a lead-in
     or a citation bracket, so the check was not widened speculatively.
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
   - ~~A **corrupted sentence reaches the reader**~~ *Fixed.* Obesity S13's
     "Hypoth, hypothalamic Sik3 mRNA levels …" is dropped by
     `stages/s7_write.py` `_garbled_start` and counted as `garbled` on the
     methods "Written:" line. It fired exactly once on the obesity rerun, on
     exactly that sentence, and on no sentence in migraine.
   - ~~A **structured-abstract label sits inside a quote**~~ *Fixed.*
     `extract/verify.py` `strip_leading_label` removes it from abstract-section
     quote, context, qualifier and naming spans, advancing `start` by the same
     number of characters so `section[start:end] == text` still holds - the
     invariant `report/invariants.py` `_anchored()` re-checks on every run.
     `integrity check passes` and `quote_validity` 1.00 on all three corpora
     after it.
     - *Left:* `extract/provenance.py` `methods_subjects()` still stores its
       context sentence with the label attached ("METHODS: ..."), and
       `tests/litreview/test_methods_subjects.py` currently asserts that text.
       Same defect, separate code path, deliberately not changed together with
       an assertion that encodes it.
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
   - *Size:* small each. The two that were worth doing first - the
     sentence-sanity check and the abstract-label strip - are done; the
     remaining four are untouched.

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
13. **`verify-run` does not say the rules changed.** *Fixed.*
    `runner.stale_reasons()` compares a run's recorded `code_version` and
    `input_hashes` against today's and returns plain-English notes ("code
    changed: ... -> ...", "pipeline.yaml changed since this run"). Both
    `verify-run` and `/reviews --verify` print them as indented notes.
    - They are never counted as violations and cannot change the exit code:
      "no violations" means the report is still sound, and that is a
      different question from whether it is still current. A run written
      before `code_version` existed returns no notes rather than a false
      alarm.
    - *Evidence:* `tests/litreview/test_runner_staleness.py` - a fresh run
      reports nothing, and a mutated `code_version` or `input_hashes` entry
      each produce their note.
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
    - *Now enforced:* `tests/litreview/test_judge_parity.py` puts a distinct
      marker in all four displayed fields and asserts every sentence
      `_shown` returns also appears in `_unit_evidence`. It reads the list
      from `_shown`'s own output rather than retyping the field names, so a
      second copy cannot drift; asserting that `_shown` returns exactly four
      is the other half of the tripwire, so adding a fifth displayed field
      forces a deliberate decision about whether the judge sees it.
25. **Judge answers on file come from more than one judge.** The scores above
    mix answers written by an earlier strong judge (the 25–26 September runs)
    with answers written later for the claims this branch changed. Both were
    Claude models, but they are not the same rater, and the earlier ones were
    not re-checked. Before-and-after accuracy numbers are therefore
    indicative, not a controlled comparison.
    - *Fix:* re-judge a fixed sample of both runs with one rater in one
      sitting, or store the judge's identity beside each answer and report
      per-rater agreement.
    - *Measured on 27 September, and it is worse than "indicative".* The same
      asthma report `3367ba2e60f434df` scored 69.4, 69.2, 69.2 across three
      passes, with 9, 8 and 9 of 9 gates. Same report id, so the pipeline
      output was byte-identical and all of that is the judge. `judge catches
      flipped effects` also went FAIL then PASS across two runs of identical
      pre-batch code. The variation is larger than the before/after
      differences anyone has been reading off these tables.
    - *Size:* small to medium.

26. **`bench/run.py` never calls `preflight`, so a run can be silently
    undersized.** `service.run_review` refuses to start below
    `defaults/models.yaml`'s `min_context_window` of 12,288, but the benchmark
    builds its own `Settings` and calls `runner.answer_question` directly with
    no such check. The default `laptop` profile is 8,192, and
    `inference/openai_compatible.py` never sends `num_ctx`, so the real limit
    is whatever `ollama serve` was started with.
    `OLLAMA_CONTEXT_LENGTH` was unset before the 27 September run, which means
    every benchmark number written before it was produced at an unverified
    context window.
    - *Fix:* call `service.preflight` from `bench/run.py`, and have the
      scorecard record the context window it actually ran at.
    - *Size:* small, and it invalidates measurements until it is done.

27. **A gate can pass because its input was never built.** Obesity's "no
    distractor leaked" PASSed on 26 September with `distractors.json` equal to
    `{}` - no distractor papers were made, so there was nothing to leak. The
    27 September run made 5 and 2 leaked (rate 0.40, both graded 3 and about
    the exact entity), so the gate now FAILs on a real, pre-existing weakness
    that the earlier "every gate passes on all three" had never measured.
    Migraine made 5 and leaked 0, so the machinery itself works.
    - *Fix:* a gate with no input must report "not measured", never PASS. Then
      look at the obesity leak on its own merits.
    - *Size:* small for the gate; the leak itself is unknown until read.

28. **The file judge is keyed by claim text, so changing the writer
    invalidates the comparison.** Any change that rewords or replaces
    sentences leaves claims with no answer on file: the 27 September migraine
    run logged 8 cache misses and judged 30, then 40, of 42 pipeline claims,
    and its 70.0 against the earlier 80.0 is substantially unjudged claims
    rather than measured regression.
    - *Fix:* report the judged fraction beside every score and refuse to
      compare two runs whose judged fractions differ materially.
    - *Size:* small.

## Long-term

29. **Semantic retrieval.** Ranking is keyword-based (BM25), so a paper that
    says "pH" is not found for "acidity". This is blocked on a deployed
    embedding model; the plan is in `NEXT_STEPS.md` §6.
