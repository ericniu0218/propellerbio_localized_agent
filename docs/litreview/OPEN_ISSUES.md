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
> Sections A, B, C and E are untouched by that branch. Renumbered 16-21 to
> 15-20 when that branch merged `main`, so section E still starts at 21.

> **Update, 29 September 2026** (same branch): three UX and hygiene items from
> a separate deploy review, none of which had a row in this document.
>
> - **Long silences during a run.** `rank`, `extract` and `write` make one
>   model call per item and said nothing between milestones — about four
>   minutes of silence while papers were graded, against a run of over two
>   hours. They now emit a per-item *tick* (`litreview/progress.py`), which the
>   app shows in the status bar and the shell prints about once per tenth of a
>   stage. Ticks never become notices (40 toasts) and never change the run: the
>   report is byte-identical with and without one, which is tested. Counts and
>   elapsed time only — no ETA, because call durations vary more than tenfold
>   and a confident wrong one is worse than none. Writing the throttle turned
>   up a flaw in its own first draft: with a 15-second floor, a *slower* stage
>   printed *more* lines (40 against 11), which the test now pins.
> - **`SyntaxWarning` on import.** `agentpack_app.py` wrote a Markdown-escaped
>   pipe as `\|` in an ordinary string. Harmless today, a `SyntaxError` in a
>   future Python — the agent-pack app would stop importing over a help table.
>   Fixed on branch `litreview/syntax-warning`, off `main`, with a test that
>   compiles every file under `src/`, `tests/` and `scripts/` with
>   `SyntaxWarning` as an error. It was the only offender, and it first failed
>   on its own docstring.
> - **`/reviews settings` and `/reviews reinstate` were undiscoverable in
>   agent-pack mode.** Both work there; neither was in that mode's help. The
>   cause was three hand-maintained copies of one table, so all three are now
>   generated from `REVIEWS_VERBS` and a test fails if a dispatched verb is
>   missing from any of them. Two nearby claims were wrong and are fixed: those
>   two verbs *write* (a preference, a decision), so "reads local files only"
>   no longer says that; and the agent-pack help ended "see `/review` below"
>   with no such row, while an empty `/reviews` told people to start one with
>   `/review` — the one command that mode refuses.

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

**27 September 2026, the mechanical fix batch.** Seven more items, all
structural or reusing a mechanism already trusted elsewhere, are done. Six more
rows are in the section A table below, and items 3, 6, 7, 9 and 24 are
rewritten with what is left. (An eighth item in that batch was `verify-run`
saying the rules changed — item 13 — which landed on `main` first and far more
thoroughly, as `assess_run`; the duplicate was dropped rather than kept
alongside it.)

The rule held again, twice over. The benchmark found that the new hedge rule's
only firing was a **false drop** that deleted a faithful sentence, with every
unit test green before and after; and that obesity's "no distractor leaked"
gate had been passing because no distractor was ever built, so "every gate
passes on all three" above was not a measurement. Three new measurement items
(26, 27, 28) record why the scores in this document cannot be compared across
runs: the benchmark never checks its context window, a gate can pass on absent
input, and the file judge is keyed by claim text so rewording sentences leaves
claims unjudged. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

**30 September 2026, the measurement batch** (branch `litreview/measurement`,
task A of [FRONTIER_LITREVIEW.md](FRONTIER_LITREVIEW.md)). Items 26, 27 and 28
are done, 25 is recorded and enforced, and the tools for 21 and 23 are built;
both now wait only for a scientist. A scorecard now names the context window
it ran at, what each score rests on and who judged it, a check with no input
says "not measured", and `bench compare` refuses two runs that differ in any
of those. Every number in this document predates all of it, and compare
refuses every pair of them.

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
   - *Confirmed by re-running obesity* (`20260927T082510Z`): the writer drafted
     the same sentence again and the guard now keeps it - identical text citing
     E189, `kept: true`, no drop reason, where before it was dropped for the
     hedge. The hedge reason appears zero times in the report, and
     `sentences_kept` is 32 against 31.
   - *A second look behind check 10* (FRONTIER_LITREVIEW.md task C). Check 10
     is a word count against a tuned threshold, so a sentence that rewords the
     borrowed sentence enough can pass it. The optional entailment check (#5)
     now asks the same question by meaning: prompt version 2 shows the quote
     apart from the sentences beside it and says the claim must be in the
     quote. *Re-audited on the five cases above, it catches none of them on
     qwen3:8b* (BENCHMARK_RESULTS.md, 30 September): all five are
     "supported" or "partial" under both prompts, so the local model does not
     apply the rule. Check 10 remains what catches four of the five; S25 (the
     dropped hedge) is caught by neither. Whether a stronger model applies it
     is the office-model measurement in #5. It stays off by default.
   - *Size:* done.
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
   exists (`report/entailment.py`, prompt `llm/prompts/entail.md`, setting
   `entailment_check` in the `[litreview]` table) and is **off by default**.
   It asks the local model, through the same scheduler and at temperature 0,
   whether the quote a kept sentence cites supports it, and drops the ones it
   calls unsupported.
   - *Narrowed (FRONTIER_LITREVIEW.md task C, prompt version 2).* The model
     sees each quote apart from the context, next and naming sentences, each
     labelled, and is told the sentence's main claim must be in a quote.
     Version 1 read all four as one passage and so would have passed #3.
   - *Measured against the hand audit* (`evals/litreview/entailment_audit.py`,
     105 sentences, qwen3:8b, v1 before and v2 after; BENCHMARK_RESULTS.md,
     30 September): it dropped none of the 93 faithful sentences under either
     prompt, and 1 (v1) or 2 (v2) of the 12 unfaithful ones. Safe on this
     sample, but it catches little on the laptop model, and its
     supported/partial labels are unstable (35 of 105 moved when only the
     prompt changed).
   - *Left:* a scientist marks the two sentences it drops
     (`evals/litreview/entailment/drops_v2_qwen3-8b.md`); the same measurement
     on the office model (`--models office --before
     evals/litreview/entailment/rows_v2_qwen3-8b.jsonl`); and
     `entailment_agreement.py` on the stored benchmark reports, on the machine
     that keeps them. Then decide whether it is on by default.
   - *Size:* medium (a scientist's marks, then a decision).
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

Status after branch `litreview/completeness-integrity-11-15` (from `d3e3342`,
26 September 2026). Everything below marked *done* is built and tested
offline with fake models; nothing here has been measured on a real model yet.

10. **Few of the known targets reach the report.**
    - *Evidence:* reference-target recall of 0.11 (obesity) and 0.06
      (asthma), against targets that are named in the collected papers. Only
      40 papers are graded and 16 read per question. The raw mention ceiling
      is far lower than 171/73: only 19 obesity and 18 asthma reference
      targets are named anywhere in the passing papers, and 160/65 have no
      curated alias entry.
    - *Done - coverage ledger.* Every run writes `coverage.v1` beside (never
      in) the report. It records where each paper stopped, from gate to
      grading, reading, extraction, synthesis and citation, with a named loss
      reason. It also records per-chunk outcomes, planned against realised
      facets, target sets by *cited verified units*, and the search funnel
      for live searches. Counts carry their denominators (`coverage.py`).
      The benchmark adds `cov_*` rows: raw mention, corpus-conditional,
      adjudicated (not available), verified-unit, narrative/Answer, human
      numbers, negatives and facets. `overall` is unchanged.
    - *Done - the report says what was verified.* The gaps and "What the
      answer needed" sections now count papers that gave verified findings.
      Before, they counted papers chosen to be read. The old fixture report
      said a paper "supplied" causal evidence when none of its findings
      survived. This changes new reports' bytes (with the background
      section, fixture report id `0ef23b491fe436c4` → `0b858528433bc13f`
      against `995d193`); stored reports are untouched.
    - *Done, opt-in - evidence-aware selection.* `study_design.py` labels
      each paper's design and document role from its publication type and
      its own design wording, with offsets. `rank.selection: evidence_aware`
      reserves 8/4/4/8 of the 40 grades for human interventional, human
      genetic/observational, prevention and mechanistic papers. The planner
      then counts only grounded human studies toward the clinical need.
      **The default stays `lexical`.** A synthetic control (every chosen paper
      given grade 3; not model output) reproduced the audit's lexical numbers.
      Evidence-aware read 2→4 (obesity) and 2→5 (asthma) human studies at
      40/16, but asthma's reference-target mentions in the papers read fell
      from 9 to 5. Only a matched real-model benchmark can decide the default.
    - *Done - section sentence limits* (`report.yaml` `sentences`) are parsed
      and the maximum is enforced after the citation guard. No section is
      padded to reach its minimum.
    - *Open:*
      - Human-number prioritisation in `best_units`, and the bounded retry
        or fallback for an Answer with no numbers. These need a
        scientist-adjudicated `evals/litreview/goldset.jsonl`, which does
        not exist yet (item 23).
      - Adaptive batch expansion (80/32 caps).
      - Curated aliases. *The tool is built* (`bedrock-litreview aliases
        propose|approve`, `knowledge/aliases.py`): names from HGNC and MeSH
        downloads, refused in code where provably wrong, written into
        `entities.yaml` only when a person marks them. What is left is a
        person's approval, then the before-and-after below.
      - Citation snowballing (FRONTIER_LITREVIEW.md task D). *Built, off*
        (`search.snowball.seeds: 0`): the references and citations of the
        best searched papers, through Europe PMC and PubMed's `elink`,
        passing the same gate. Waits for the before-and-after below.
      - The real-model before/after comparison on the frozen
        obesity/asthma corpora. With aliases approved and
        snowballing on, the corpora are collected once again
        (`build-corpus --searcher`) and target recall compared with
        the same model and settings.
    - *Size:* big.
11. **Background from reviews is not covered.** *Done (owner decision, 26 September 2026: keep all the
    evidence, in a separate area with links).*
    - Every report now has a "Background from reviews (not primary evidence)" section. It lists the review
      articles collected for the exact subject, most relevant first: 12 with up to two quote-checked
      sentences from their own abstracts and a link, then up to 40 more as links.
    - The section is written by code, not a model. `invariants.check_background` holds every quote to the
      review's text.
    - Reviews are counted nowhere else: not in targets, the evidence table or the Answer.
    - Retracted reviews are not shown; flagged ones carry the notice.
    - *Not built:* following each review's citations to fetch the trials it cites (old step 28, option 3).
12. **Retraction status can be stale or incomplete.** *Done, with the owner's decisions of 26 September 2026.*
    - *What is parsed and stored:*
      - `publication_integrity.py` parses Europe PMC `commentCorrectionList`, PubMed `CommentsCorrections`
        and Crossref `update-to`/`updated-by`, keeping direction. "Retraction in" on the paper's own record
        means the paper is retracted; "Retraction of" on a notice means the record *is* a notice.
      - Observations are append-only, with raw responses stored by hash.
      - Status and freshness are derived from them with a clock. A failed lookup erases nothing.
    - *Automatic by default.* Each new review asks PubMed (PMIDs) and Crossref (papers with only a DOI)
      about its papers before reading them:
      - It sends identifiers only, never the question.
      - It skips papers checked in the last `integrity.stale_after_days` (7).
      - It stops at once with no network, and keeps to a 90-second budget.
      - When it finds a concern, retraction or reinstatement, it looks the notice up. The notice's own title
        and opening words come from PubMed, Crossref or Europe PMC open-access full text.
    - *Manual as an option.* Scientists switch with `/reviews settings manual` in the app, which explains
      both choices in plain words. `integrity_checks` in `[litreview]` sets a site default, and
      `BEDROCK_LITREVIEW_INTEGRITY_CHECKS` overrides both. The review's start notice says which mode is on.
      Reports say how many papers read have never been checked, and when the automatic check stopped early.
    - *Crossref may be contacted directly* (owner decision). `api.crossref.org` is on a separate allowlist
      used only by the retraction check (`INTEGRITY_HOSTS`); the literature search cannot reach it.
    - *Retractions* keep a paper out of new runs, as before.
    - *Expressions of concern* (owner decision: "don't recommend it, don't hide it"):
      - The paper stays in.
      - Its findings are ranked after all other evidence.
      - They are never used in the Answer, and the Answer names any target left out for that reason.
      - They are marked ⚠ in the evidence table.
      - Each reference quotes the notice and links to it.
    - *Corrections* are flagged only.
    - *Reinstatement* (owner decision):
      - A paper that was retracted and later reinstated is never added back automatically.
      - The report and the review's progress say so.
      - A scientist adds it back with `/reviews reinstate <paper>`, or
        `bedrock-litreview integrity include <paper> --by <name>`. `--undo` / `exclude` reverses it. The
        decision is recorded append-only with who and when, and shown in later reports.
      - Only a paper with a recorded reinstatement can be added back.
    - *What a run knew is pinned* in its `publication_integrity_snapshot.v1`. A later notice is an advisory
      on `verify-run`; stored reports never change.
    - *Tested against captured public responses only.* No provider was exercised live by this branch.
    - *Open:*
      - PMCID-only papers are not checked (no validated lookup).
      - The seven-day re-check interval is a default, not a guarantee.
13. **`verify-run` does not say the rules changed.** *Done.*
    - `assess_run` answers three separate questions: are the pinned
      artifacts intact, do today's checks pass, and is today's environment
      the one the run recorded?
    - New runs record `verification_context.v1` per invocation. It covers the
      config/knowledge roles and hashes of the bytes actually loaded, prompt
      file hashes, code identity and how it was found, a verifier source
      fingerprint, and schema hashes.
    - A changed environment is reported (exit 0), never counted as a
      violation. `verify-run --require-same-rules` fails on drift or
      incomplete comparability, for CI.
    - Legacy runs compare what they recorded and say what they did not.
      Editing `pipeline.yaml` in this branch means every older run now
      reports that file as changed. That is expected and correct.
14. **Determinism on vLLM is unproven** (old step 36). Every benchmark run here
    re-ran identically, but through the call cache.
    - *Done - cache identity.*
      - The call-cache key now includes a `generation_context.v1`: seed,
        thinking policy, weights digest or `None`, declared revision, template
        and deployment, server, and normalisation.
      - The audit's probe (seed 1 replayed seed 0 with zero calls) now calls
        the server.
      - A client that cannot identify itself is never replayed.
      - Old cache entries are kept; they are replayed only with
        `bench run --legacy-cache-read`. Expect every existing entry to miss
        once.
    - *Done - honest labels.* `bench run`'s gate is now "same report twice
      (cached replay, not fresh inference)" when it went through the cache.
      Historical scorecards keep their old wording.
    - *Done - harness, not run.*
      - `bench determinism` runs a cell. Fresh mode needs the backend's
        request observer and fails any reply that did not reach the server.
        Records are append-only.
      - `bench determinism-payloads` renders the payloads; 20 pinned ones are
        in `evals/litreview/determinism/payloads.jsonl`.
      - `bench replay-control` is the recorded-output rendering control.
    - **No fresh-inference cell has been run.** It needs the deployed vLLM
      server, an approved compute budget, and (for restarts) IT. Until then
      the accurate statement is: deterministic orchestration over pinned
      inputs; reproducible cached replay and rendering from pinned outputs;
      fresh-inference repeatability is deployment-specific and unmeasured.

## D. Deployment and IT

**Items 16–19 were fixed on branch `litreview/deployment-and-it`.** What each
one turned out to be is recorded below, because two of them were not quite what
this list said. Item 15 is still blocked on IT, and 20 is advice rather than
work.

15. **The `office` model set needs IT's real model names.** *Still open —
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
16. **No re-runnable setup check.** *Fixed.*
    - `bedrock-litreview doctor` and `/reviews check`. *Main built its own
      `doctor` (`readiness.py`) while this was in review, and it is the better
      one* - it checks server-reported windows, structured output and
      authentication, and works in agent-pack mode - so this branch's version
      was dropped rather than kept beside it. What survives is what main's did
      not have: the advice that does not stop a review (a short
      `request_timeout`, free disk, whether a company server name needs
      `allowed_hosts`), shown by `doctor` as information before any network
      check so it appears even when the server is down; and `/reviews check`
      in the app, which runs main's check through the chat's own connection.
    - Both go through `service.preflight`, so a check cannot exist in the doctor
      and not in a run's preflight.
    - *Honest gap:* the free-disk threshold (10 GB) is a judgement call, not a
      measurement, and the code says so. It is about the model server and swap —
      a review's own files are megabytes. Worth replacing with a measured figure
      after a full run on the office server and on a 32 GB laptop.
17. **`/review` in the app still uses chat's 120-second timeout.** *Fixed.*
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
18. **A review cannot be deleted.** *Fixed, scoped to the run's own output.*
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
19. **Smaller integration items.** *All five fixed.*
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
20. **A 16 GB laptop is not a review machine.** *Advice, not work.*
    - qwen3:8b took about two hours per question on a 16 GB M4.
    - With ordinary apps open, its 6.4 GB footprint pushed the Mac into swap
      until the disk filled. The run had to be paused and resumed three
      times.
    - *Fix:* run reviews on the office server, or on machines with 32 GB or
      more. The disk check asked for here is in item 17 and is done.

## E. Measurement

21. **No scientist has checked any accuracy number.** *The tool is built; the
    grading is not.* The judge and the hand audit were both AI models. A
    scientist should re-judge at least 40 claims blind (old step 23).
    - *Now built:* `python -m evals.litreview.rejudge_sheet draw <run id> ...`
      draws 40 kept claims (seeded, shared evenly across the runs, shuffled)
      onto a blind sheet: each claim and exactly the evidence the judge is
      shown, through `bench/run._unit_evidence` imported rather than copied,
      so item 24's guard covers the sheet too. No system, question, run,
      paper or verdict appears on it; a separate key maps rows back. `agree`
      prints raw agreement and Cohen's kappa against the file judge, per run
      and overall, with both vocabularies collapsed to supported / not
      supported and the mapping printed beside the numbers.
    - *Now enforced:* `tests/litreview/test_bench_rejudge.py` asserts that no
      identifying field reaches the sheet and checks the agreement against
      hand calculations.
    - *Left:* a scientist grades the 40 claims (roughly half a day) from
      stored runs; none is stored on the machine this was built on. Until
      then every accuracy number here is an AI's.
22. **The office model is unmeasured.**
    - Qwen3-30B-A3B sits between qwen3:8b and Claude.
    - *Fix:* run the five benchmark questions on it, cold and timed, and
      hand-audit 15 sentences per report as done here (old step 37).
23. **No gold set for extraction** (old step 22). *The tools are built; the
    labels are not.*
    - *Now built:* `python -m evals.litreview.goldset_sheet <corpus>` picks
      20 papers from a frozen corpus by the README's quotas (seeded, scarcest
      category first), holds 5 back one per kind, and writes an index, one
      sheet per paper and an empty JSONL template. Design and review
      categories come from `study_design` labels and roles; the negative-result
      and "mentions but not about" categories are only proposed (null-result
      cues from `negation.py`, the gate's own reason) and marked
      `kind_confirmed: false`. A sheet shows the paper as Stage 5 reads it and
      nothing the pipeline produced. `run_extract` skips `_comment` lines and
      refuses to score while a placeholder, an unconfirmed kind, an empty
      entry without `none_expected`, a review with findings, a quote not in
      the paper or an out-of-schema field remains, naming the line, paper and
      field. It now runs each paper through `extract_paper` behind preflight,
      and pairs findings by the sentence they quote: the old five-field match
      key made five of the seven per-field accuracies 1.00 by construction.
      The labelling guide is in `evals/litreview/README.md`.
    - *Now enforced:* `tests/litreview/test_bench_goldset.py` - selection on a
      fixture corpus, the sheet's blindness, its chunks checked against the
      real `extract_paper`, every refusal, and `run_extract` end to end with
      the fake model.
    - *Left:* a scientist labels 20 papers (about two days). The corpora are
      short of human trials: obesity's searcher corpus gives 3 of the 5 at
      seed 0, so the rest come from another corpus. `run_extract` has not
      been run against a real model.
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
    - The scientist's re-judging sheet (item 21) imports `_unit_evidence`
      too, so the same guard keeps it showing what the reader is shown.
25. **Judge answers on file come from more than one judge.** *Recorded and
    enforced; the answers already on file are not re-judged.* The scores above
    mix answers written by an earlier strong judge (the 25–26 September runs)
    with answers written later for the claims this branch changed. Both were
    Claude models, but they are not the same rater, and the earlier ones were
    not re-checked. Before-and-after accuracy numbers are therefore
    indicative, not a controlled comparison.
    - *Measured on 27 September, and it is worse than "indicative".* The same
      asthma report `3367ba2e60f434df` scored 69.4, 69.2, 69.2 across three
      passes, with 9, 8 and 9 of 9 gates. Same report id, so the pipeline
      output was byte-identical and all of that is the judge. `judge catches
      flipped effects` also went FAIL then PASS across two runs of identical
      pre-batch code. The variation is larger than the before/after
      differences anyone has been reading off these tables.
    - *Now recorded:* each answer in `evals/litreview/judge/answers.jsonl`
      names its `judge` (a model id or `scientist:<name>`); rows written
      before load as `"unknown"`. The file judge counts the answers it used
      per judge, a model judge is its own version, verdicts settled in code
      are listed as `code`, and the scorecard and history row list them all.
    - *Now enforced:* `python -m bedrock.litreview.bench compare A B` refuses
      two runs whose raters differ, whose shared raters judged in shares more
      than 5 points apart, or either of which used `"unknown"` answers
      (`tests/litreview/test_bench_compare.py`).
    - *One way the validity gate flipped, now closed:* a perturbation the file
      judge had no answer for counted as *not caught*, so "judge catches
      flipped effects" could fail on missing answers and pass once they were
      written. It is now left out, and `perturbations_judged` says how many
      counted. Whether that explains the 27 September flip is not known.
    - *Left:* the 1,790 answers on file are all `"unknown"`, so compare
      refuses every comparison built on them until the claims a comparison
      rests on are re-judged by one named judge in one sitting. Nothing here
      removes one rater's own noise: the 69.4 / 69.2 / 69.2 spread stands as
      the floor on any difference worth reading.

26. **`bench/run.py` never calls `preflight`, so a run can be silently
    undersized.** *Done.* `service.run_review` refuses to start below
    `defaults/models.yaml`'s `min_context_window` of 12,288, but the benchmark
    builds its own `Settings` and calls `runner.answer_question` directly with
    no such check. The default `laptop` profile is 8,192, and
    `inference/openai_compatible.py` never sends `num_ctx`, so the real limit
    is whatever `ollama serve` was started with.
    `OLLAMA_CONTEXT_LENGTH` was unset before the 27 September run, which means
    every benchmark number written before it was produced at an unverified
    context window.
    - *Now enforced:* `bench run`, `bench determinism` and `bench
      replay-control` call `service.preflight` before their first model call,
      print its message unchanged and exit non-zero; they open the review's
      own endpoint and size calls to the windows preflight found. A stored
      report judged from a file with no baselines calls no model and opens no
      server (`run_reaches_model`; `NoModelClient` fails loudly if that rule
      goes out of date). `tests/litreview/test_bench_preflight.py`: a server
      reporting 4,096 tokens stops `bench run` before any model call, and the
      offline re-score passes with no backend at all.
    - *Now recorded:* the scorecard, its header and the history row carry the
      window of this run's calls and the window the report was written at -
      for `--from-run`, read back from the history row of the run that wrote
      it. `run_extract` preflights too.
    - *Left:* nothing in code. Every report scored before 30 September reads
      "not recorded", and stays that way.

27. **A gate can pass because its input was never built.** *Done for the
    gates; the obesity leak is open.* Obesity's "no
    distractor leaked" PASSed on 26 September with `distractors.json` equal to
    `{}` - no distractor papers were made, so there was nothing to leak. The
    27 September run made 5 and 2 leaked (rate 0.40, both graded 3 and about
    the exact entity), so the gate now FAILs on a real, pre-existing weakness
    that the earlier "every gate passes on all three" had never measured.
    Migraine made 5 and leaked 0, so the machinery itself works.
    - *Why `{}`:* `distractors.json` is `{}` only when the distractor test was
      skipped, which scoring a stored report (`--from-run`) forces. The leak
      rate was then absent, and the gate's `in (None, 0.0)` read absent as no
      leak. A test that makes no distractor returns `None` too, never `0.0`.
    - *What else passed on nothing:* five gates treated `None` as a pass
      (quotes verified, every sentence cited, numbers in evidence, no
      distractor leaked, judge catches flips); "no near-miss claims" passed a
      report with no claims; "claims supported >= 0.85" read `None` as 0 and
      failed; and "same report twice" vanished from a stored report's table.
    - *Now enforced:* every gate is pass, fail or not measured with the
      missing input named ("no distractor papers were built: skipped with
      --skip-distractors", "... a stored report was scored ...", "... no
      paper was graded relevant enough (2 or more) to copy").
      `gates_passed` counts passes, `gates_total` measured gates, and
      `gates_not_measured` lists the rest; the scorecard prints PASS only for
      a gate that measured something, and a `Gate` refuses `bool()`. Old
      scorecards and history rows still load, the rows marked as written
      when a gate with no input counted as passed.
      `tests/litreview/test_bench_gates.py` covers every gate in every state
      it can reach.
    - *Left:* the obesity leak itself (2 of 5 on 27 September, both graded 3
      and about the exact entity) is a relevance-grading question in the
      pipeline, not a benchmark one, and has not been read.

28. **The file judge is keyed by claim text, so changing the writer
    invalidates the comparison.** *Done: the fraction is shown and a
    mismatch refused; the misses themselves remain.* Any change that rewords or replaces
    sentences leaves claims with no answer on file: the 27 September migraine
    run logged 8 cache misses and judged 30, then 40, of 42 pipeline claims,
    and its 70.0 against the earlier 80.0 is substantially unjudged claims
    rather than measured regression.
    - *Now recorded:* every judged score (accuracy, groundedness, review-fact
      recall) carries judged/total on the scorecard as "(j/n)" and in the
      history row; the overall carries its least-judged part's percentage. A
      review fact whose reachability got no verdict now counts as unjudged
      instead of silently leaving the denominator.
    - *Now enforced:* `bench compare` refuses a pair whose judged fractions
      differ by more than `COMPARABLE_WITHIN` (5 points, a starting value in
      `bench/compare.py`), prints the reasons, and prints anyway only with
      `--force` (`tests/litreview/test_bench_compare.py`).
    - *Left:* answers are still keyed by prompt text, so a reworded claim
      still has no answer until someone writes one (`answers.pending.jsonl`
      lists them). This makes the gap visible and stops the comparison; it
      does not close the gap.

## Long-term

29. **Semantic retrieval.** Ranking is keyword-based (BM25), so a paper that
    says "pH" is not found for "acidity". This is blocked on a deployed
    embedding model; the plan is in `NEXT_STEPS.md` §6.
