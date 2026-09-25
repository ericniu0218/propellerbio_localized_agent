> **Moved from Localized-Airgap-LLM-for-literature-review at `dc5c0123d8e1`.** This is a
> record of measurements made in that repository, and the commands in it are
> kept exactly as they were run. In bedrock they are spelled
> `bedrock-litreview ask`, `python -m bedrock.litreview.bench` and `--models`,
> and `evals/` is `evals/litreview/`; the numbers are unaffected. The to-do
> list in § Next steps is still the to-do list.

# Benchmark results — anchored evidence (before/after)

**24 September 2026.** This section measures the "anchored evidence" change on
branch `fix/anchored-evidence` (uncommitted at the time of the run): quotes are
the paper's own sentence, found in the passage the model read and stored with
offsets; the target, agent, species, n and result summary must be found in that
same text or they are blanked (or, for a target, the finding is dropped); the
citation guard reads paper text only (`quote` + `context_quote` + located
names), no longer `effect_detail`, `species_raw` or `n`, and adds a
"who was studied" check; merges keep review and retraction flags; runs pin
paper versions; and an integrity checker runs after every report. Nothing was
tuned for this run — no pipeline code, prompt, knowledge table or config was
changed while measuring.

*Where this section says "I", it means the judge — an AI model (Claude Opus),
not the model under test, and not a person. No scientist has reviewed these
verdicts.*

## Setup

- Model under test: `medgemma1.5:4b@433252621ab1`, `temperature=0`, via Ollama, `--profile medgemma`, same M4 MacBook.
- Judge: strong judge only, answering from `evals/judge/answers.jsonl` (`--judge-file`). 164 answers were already on file from a stopped attempt and were reused; I answered the rest. **975 answers on file**, and all four scored runs finished with an empty `answers.pending.jsonl`.
- A — frozen corpus (`evals/corpora/`): `python -m litreview.bench run evals/questions/*.yaml --profile medgemma --judge-file evals/judge/answers.jsonl`.
- B — searcher corpus: the same, plus `--corpora evals/corpora-searcher`.
- BEFORE = the strong-judged numbers in [the section below](#benchmark-results--searcher-corpus-vs-frozen-corpus-strong-judge) (commit `c5d7765`). AFTER = this branch, same questions, same corpora, same model.
- `pytest` → **216 passed**, no GPU, no network.

| | run id (scored / determinism re-run) | report id | scorecard |
|---|---|---|---|
| A — frozen, psoriasis | `2dc06c292ff9` / `127903c678af` | `131be54949a9ad4e` | `evals/results/psoriasis-prevention-targets/20260924T140212Z/` |
| A — frozen, RSV | `852256287000` / `7faea80b1ea8` | `b3be4f938fd9c6c7` | `evals/results/rsv-infant-prevention-targets/20260924T140217Z/` |
| B — searcher, psoriasis | `d93b0c57e0ae` / `5bbf6721216b` | `e7d2bcb7d2bc91c3` | `evals/results/psoriasis-prevention-targets/20260924T140326Z/` |
| B — searcher, RSV | `fd311fdc2064` / `b4d14693290c` | `9d79c5e4c8857d62` | `evals/results/rsv-infant-prevention-targets/20260924T140333Z/` |

**One caveat before the numbers, because it changes how to read them.** The
judge is an AI model answering from a file. Answers are keyed by the exact
judging prompt, so a claim whose text and evidence did not change reuses the
same verdict — but the reports *did* change (the psoriasis-A report grew from
76 kept sentences to 95), and every new or changed claim was judged fresh, in
this session, by me. So a `cited_support` difference is a mix of the code
change and of one strict judging pass versus another. The counts that come
from code alone — findings kept, sentences dropped and why, quote validity,
integrity, determinism — do not have this problem.

## After this measurement: two fixes, re-scored

The judge's run above found two problems. Both were fixed and all four reports
re-scored on the fixed code. Every model call and every unchanged judging
verdict replayed from cache. 11 coverage-judging tasks were new because the
RSV-A report text changed; they were answered in the same way as the verdicts
above. One of them ("long-acting monoclonal antibodies … a recent advance") was
aligned with the verdict this judge had already given the same sentence on
the earlier RSV-A report.

1. **"In animals" is now checked.** The who-was-studied vocabulary
   (`knowledge/meaning_words.yaml`) gained an `animal` class
   (animal, animals, rodent, rodents, preclinical). Cited text naming animals,
   mice, rats or primates backs it (`verify.covered`). Re-verifying the stored
   RSV-A run flagged exactly the two sentences described under "Remaining
   failures" and nothing else. The re-run drops them.
2. **The benchmark's quote re-check now reads the section each quote came
   from**, as well as `Paper.all_text()`. All four quotes that "failed" were
   verbatim in `sections["abstract"]`, which `all_text()` skips when it
   differs from `Paper.abstract` (the Highlights case, audit step 14). That
   parsing issue itself is not fixed.

| | Pso A | RSV A | Pso B | RSV B |
|---|---|---|---|---|
| Overall | 74.9 | **66.6** (was 65.5) | 73.1 | 70.9 |
| Claims supported by their citations | 0.92 | **0.87** (was 0.84) | 0.96 | 0.97 |
| Quote validity (bench re-check) | 1.00 | 1.00 | **1.00** (was 0.987) | **1.00** (was 0.962) |
| Sentences kept | 95 | **72** (was 74) | 92 | 58 |
| Gates passed | 9/9 | **9/9** | **9/9** | **9/9** |
| Same report twice | yes | yes | yes | yes |
| `verify-run` | OK | OK | OK | OK |
| Report id | `131be54949a9ad4e` | `885779ed2dd7bb60` | `e7d2bcb7d2bc91c3` | `9d79c5e4c8857d62` |

The other three reports are byte-identical to the ones scored above: same
report ids. Only their quote-validity number changed. `pytest` → 217 passed.

## Before and after, per report

| Metric | Pso A before | Pso A after | RSV A before | RSV A after | Pso B before | Pso B after | RSV B before | RSV B after |
|---|---|---|---|---|---|---|---|---|
| **Overall (0–100)** | 77.5 | **74.9** | 67.9 | **65.5** | 72.9 | **73.1** | 67.9 | **70.9** |
| Claims judged | 76 | 95 | 67 | 74 | 78 | 92 | 53 | 58 |
| Claims supported by their citations | 1.00 | 0.92 | 0.94 | **0.84** | 0.94 | 0.96 | 0.96 | 0.97 |
| Claims supported by the collected papers | 1.00 | 0.95 | 1.00 | 0.91 | 0.96 | 0.97 | 0.96 | 1.00 |
| Numbers present in the evidence | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Quote validity (bench re-check) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.987** | 0.963 | **0.962** |
| Claims naming a near-miss entity | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Review facts covered | 0.00 | 0.00 | 0.00 | **0.09** | 0.00 | 0.00 | 0.00 | **0.07** |
| Vital review facts covered | 0.00 | 0.00 | 0.00 | **0.13** | 0.00 | 0.00 | 0.00 | **0.13** |
| Known drug targets named (reachable) | 0.50 | **0.57** | – | – | 0.39 | 0.33 | – | – |
| Verified findings kept | 223 | 223 | 99 | 96 | 155 | 154 | 54 | 52 |
| Sentences kept / dropped | 76 / 2 | 95 / 4 | 67 / 1 | 74 / 5 | 78 / 1 | 92 / 1 | 53 / 8 | 58 / 12 |
| Supported claims carrying a number | 26 | 21 | 35 | 36 | 6 | 8 | 15 | 19 |
| Distinct sources cited | 49 | 48 | 41 | 37 | 54 | 51 | 32 | 29 |
| Same report twice | yes | yes | yes | yes | yes | yes | yes | yes |
| Gates passed | 8/8 | **8/9** | 8/8 | **8/9** | 7/8 | **8/9** | 7/8 | **8/9** |

Gate detail, after (the ninth gate, "integrity check passes", is new):

| Gate | Pso A | RSV A | Pso B | RSV B |
|---|---|---|---|---|
| quotes all verified (1.00) | PASS | PASS | **FAIL** (0.987) | **FAIL** (0.962) |
| every sentence cited | PASS | PASS | PASS | PASS |
| no near-miss claims | PASS | PASS | PASS | PASS |
| numbers all in cited evidence | PASS | PASS | PASS | PASS |
| no distractor leaked | PASS | PASS | PASS | PASS |
| same report twice | PASS | PASS | PASS | PASS |
| **integrity check passes** | PASS | PASS | PASS | PASS |
| claims supported by citations ≥ 0.85 | PASS | **FAIL** (0.84) | PASS | PASS |
| judge catches flipped effects ≥ 0.7 | PASS | PASS | PASS | PASS |

## What the reports now say about themselves

The "How this answer was assembled" section is new, and these counts are code's
own, not mine:

| | Pso A | RSV A | Pso B | RSV B |
|---|---|---|---|---|
| Findings proposed | 384 | 224 | 313 | 347 |
| Quote not in the paper → discarded | 19 | 13 | 24 | 16 |
| …of those, the model's copy **changed meaning** (number, negation, Greek letter, meaning word) | 7 | 8 | 15 | 6 |
| Discarded as about another entity | 0 | 3 | 3 | 32 |
| **Target not found in the text the model read** → finding dropped | 1 | 4 | 2 | 2 |
| Who was studied **blanked** ("not stated") | 27 | 23 | 34 | 39 |
| Sample size blanked | 1 | 8 | 0 | 0 |
| Result summary blanked | 51 | 23 | 54 | 37 |
| Agent blanked | 0 | 0 | 3 | 2 |
| Verified findings kept | 223 | 96 | 154 | 52 |

Sentences the citation guard dropped, with its reasons:

- **Pso A (4):** "claims no effect, which the cited evidence does not" ×2; "number 90 is not in the cited evidence"; "names 'patients', which the cited evidence does not".
- **RSV A (5):** "names 'RSV infection', which the cited evidence does not" ×2; "names 'influenza', which is not what the question asks about"; "number 290.6 is not in the cited evidence"; "number 1000 is not in the cited evidence".
- **Pso B (1):** "names 'IL-1β', which the cited evidence does not".
- **RSV B (12):** "names 'hMPV', which is not what the question asks about" ×8; "names 'RSV F', which the cited evidence does not"; "names 'RSV infection', which the cited evidence does not" ×2.

The new "who was studied" check dropped **no** sentence in any of the four
reports. Its effect appears earlier instead: 27–39 findings per report now show
"not stated" rather than a species the paper never gave.

## The integrity gate

`Report.integrity` is empty on all four reports, so no warning banner was
printed and the new gate passes four times out of four. `verify-run` was run on
all eight run ids (the four scored runs and their four determinism re-runs):

```
2dc06c292ff9: OK   127903c678af: OK   852256287000: OK   7faea80b1ea8: OK
d93b0c57e0ae: OK   5bbf6721216b: OK   fd311fdc2064: OK   b4d14693290c: OK
```

## What improved

- **Every one of the audit's named quote/field failures is gone.**
  - *"…in human tissues"* (ALOX5, PMID 42409798): the string does not occur anywhere in the new psoriasis-B report.
  - *"IL-17 promotes IL-6 production in mice"* (PMID 40430037): gone. What is printed now is the paper's own sentence, "IL-17 can also promote the production of IL-6, which…", with no species attached.
  - *FABP5 "deletion of"* (PMID 37967009): the kept sentence is now "Similar to the Fapb5 global knockout, **specific deletion of Fabp5 in keratinocytes** reduced the infiltration of neutrophils in the ear, dermis, and peripheral blood" — the modifier the old report dropped is back, because the stored quote is the paper's sentence.
  - *RSV-A `E96`*: no unit in the new frozen-RSV report carries the mismatched `effect_detail` ("cost-effectiveness results were most sensitive to changes in the price…") against the LMIC cost-saving quote. On that paper the units now have a result summary that is a span of their own quote, or an empty one (7 of 96 findings have a blank result summary).
- **Fuzzy quotes no longer hide a meaning change.** 25 fuzzy quotes across the four reports. I diffed the model's copy against the stored quote for every one: all differences are punctuation, whitespace, a figure reference, a hyphen ("HIF-1-α" → "HIF-1α", "UBE2L3" → "Ube2l3"), or the widening of a fragment to the paper's whole sentence. None changes a number, a negation or a direction. In the previous round, 13 fuzzy quotes were found in the two searcher reports and 4 of them were unchecked.
- **Review-fact recall moved off zero for the first time**, on RSV: 0.00 → 0.09 (frozen) and 0.00 → 0.07 (searcher), vital facts 0.00 → 0.13 on both. It is still 0.00 on both psoriasis reports.
- **The searcher corpus now scores better than the frozen one on RSV** (70.9 vs 65.5) and roughly level on psoriasis (73.1 vs 74.9), which reverses part of the earlier verdict.
- Citation accuracy on the searcher corpora is up (psoriasis 0.94 → 0.96, RSV 0.96 → 0.97), corpus grounding on RSV-B is 1.00, no near-miss claim appeared in any report, no distractor leaked, numeric fidelity is 1.00 everywhere, and every report reproduced byte-identically on the determinism re-run.

## What got worse

- **The frozen-corpus scores fell** (psoriasis 77.5 → 74.9, RSV 67.9 → 65.5) and RSV-A's "claims supported by citations" gate now **fails at 0.84**. This is not the anchoring doing damage to existing sentences; it is that both frozen reports now write *more* sentences (76 → 95 and 67 → 74) and the added ones are weaker, plus a fresh strict judging pass. Of RSV-A's 12 non-supported claims, 2 are sentences that claim animal work for a human cost-effectiveness model (see below), 3 are statements the cited unit simply does not make ("nirsevimab is a monoclonal antibody used to prevent RSV in infants" cited to a sentence about clinical outcomes), 2 were marked supported by me but overruled in code because most of the claim's words are not in the evidence, and 1 was called contradicted in code on a wording difference ("lack of provider recommendation" vs the paper's "no provider immunization offer").
- **Findings lost to the stricter checks.** The new counters say exactly how many and why: 7 / 8 / 15 / 6 findings per report were destroyed because the model's copy of the quote changed a number, a negation, a Greek letter or another meaning word — these are the ones that used to pass as "fuzzy" — and 1 / 4 / 2 / 2 more were dropped because the target was not in the text the model read. Net, kept findings fell by 0 (psoriasis-A), 3 (RSV-A), 1 (psoriasis-B) and 2 (RSV-B). The cost is small because most discarded proposals were duplicates of kept ones.
- **Fields blanked rather than shown.** Between 23 and 39 findings per report now say "not stated" for who was studied, and 23–54 have no result summary. That is the intended trade, but it is a visible loss of detail: the frozen-RSV evidence table, for instance, now shows no sample size on 8 findings it used to fill in.
- **Sentences dropped by the guard rose** on three of the four reports (2 → 4, 1 → 5, 8 → 12; psoriasis-B unchanged at 1). Eight of RSV-B's twelve drops are the near-miss rule removing the RSV/hMPV cross-neutralising antibody paper, which is the same product decision the audit flagged, not a regression.
- **Quote validity now fails on both searcher reports** (0.987 and 0.962). I checked every failing unit: there are 4, all with `quote_status: exact`, and every one is found verbatim in `Paper.sections["abstract"]` — the text the extractor read — but not in `Paper.all_text()`, which the benchmark re-checks against and which skips `sections["abstract"]` (`litreview/models.py:336-339`). The two papers are PMID 42106072 (narrative abstract 1774 chars, stored abstract section 846 chars: the Highlights list from the audit's step 14) and PMID 40694426 (1297 vs 1303 chars). **This is the known pre-existing `all_text()` issue, not a fabrication**; psoriasis-B is newly exposed to it because anchored quotes are widened to the whole sentence inside the section the model read. I did not touch this code.
- Distinct sources cited fell slightly on all four reports (49 → 48, 41 → 37, 54 → 51, 32 → 29).

## Remaining failures

- **Misattribution is untouched, as expected.** I read 10 kept sentences per searcher report against the papers. In psoriasis-B all 10 were faithful to the paper's own wording and direction (one, "FABP5 was significantly upregulated in the psoriatic skin", carries a second, padding citation to a Fabp5-deletion finding it does not need). In RSV-B, 8 of 10 were faithful and **2 are true sentences pointing at the wrong finding**: "RSVpreF elicited robust neutralizing responses with efficient transplacental transfer" is the paper's conclusion but is cited to a unit whose quote is about when titers peaked; "Older adults may also develop severe RSV disease and need hospitalization" is cited to a unit about asthma risk after RSV infection. Both were judged unsupported, so the score reflects them.
- **Fixed after this run — see "After this measurement" above.** **"In animals" passes the who-was-studied check (new finding, a bug).** The frozen-RSV report keeps "The study used animal models to assess the program's impact." (cites `E19`) and "The authors studied the effect of vaccination against RSV in animals." (cites `E47`). Both cited findings are human cost-effectiveness / transmission modelling in children and LMICs, typed `evidence_type: in_vivo`, and the writer turned that label into prose about animals. The guard cannot catch it because "animal" is not in the who-was-studied vocabulary: `studied_in("The authors studied the effect of vaccination against RSV in animals.")` returns `{}` (`litreview/extract/verify.py:249`), so the check never fires. These two sentences alone are a sixth of RSV-A's failed-support claims.
- **Truncation at `DraftSentence.text`'s 450-character cap is unchanged**: 5 kept sentences end mid-word in psoriasis-B ("…378 genes were,.", "…and SARM1 , a T.", "…Compared with the wild-type BMDM, the.", "…lesional skin of psoriasis patient.") and 1 in psoriasis-A ("…as the induction time by tamoxifen (TAM) was [ ].").
- **Review-fact recall is still 0.00 on both psoriasis reports**, for the structural reason the earlier section gives.
- **Label pollution and mis-filing persist**: RSV-B files a structural-biology sentence about 5B11 binding under "Negative and conflicting results", and psoriasis-B still puts hepatitis-B reactivation under "Prevention-specific evidence".

## Verdict

The anchoring did what it was meant to do and nothing more. Every quote a
reader sees is now the paper's own sentence; the four specific fabrication-like
failures the audit named are gone; meaning-changing fuzzy matches are gone
(7–15 findings per report were destroyed for exactly that reason); fields the
paper never stated are blanked instead of asserted; and the integrity gate and
`verify-run` pass on every run. The price is visible and small: a handful of
findings lost per report, more "not stated" cells, and a quote-validity gate
that now fails on the searcher corpus for a **benchmark-side** reason
(`all_text()` skips the stored abstract section) that predates this change.
What it did not fix, it did not pretend to fix: a true sentence can still point
at the wrong finding, and one label (`in_vivo`) can still become a claim about
animals that no check catches. Headline scores moved in both directions —
down on the frozen corpora, up on the searcher corpora — and, since the judging
is a strict AI pass rather than a scientist's, those two-to-three-point moves
should not be read as more than noise around a pipeline that is now
demonstrably harder to make say something the paper does not.

*Update:* the benchmark-side quote-validity failure and the "in animals" gap
were both fixed after this run. All four reports now pass 9/9 gates; see
"After this measurement" at the top of this section.

---

# Benchmark results — searcher corpus vs frozen corpus (strong judge)

**23 September 2026.** The paper searcher (`litreview/sources/literature.py`:
Europe PMC including preprints, and PubMed E-utilities, open-access full text)
merged into `main` at commit `c5d7765`. This section measures whether reports
built from what the searcher collects are better than reports built from the
frozen corpora in `evals/corpora/` (hand-written Europe PMC queries) — same
pipeline code, same model, same judge, only the corpus changed. It also
replaces the judge: every score below, for both corpora, comes from
**Claude Sonnet — an AI model, not the model under test — answering from a file**
— not a person; no scientist has reviewed these verdicts yet
(`--judge-file evals/judge/answers.jsonl`), not from MedGemma grading its own
work. The §[first run](#benchmark-results--first-real-model-run) below this one
used the MedGemma judge throughout; its numbers are not comparable to these and
are kept only as a record of that earlier round.

*Where this section says "I", it means the judge — Claude Sonnet — which also ran the benchmark and wrote this section.*

## Setup

- Model: `medgemma1.5:4b@433252621ab1`, `temperature=0`, via Ollama, `--profile medgemma`, on the same M4 MacBook, 16 GB.
- Judge: Claude Sonnet (an AI model, run at development time on public paper text — never part of the air-gapped deployment), answering the benchmark's judging questions from the report text and the paper text actually collected, in the compact format `scripts/judge_answers.py` expands. Code still checks every sentence the judge cites — it cannot mark something supported by asserting it; the cited sentence must literally be in the evidence, and the claim's names and numbers must be too.
- Commit under test: `c5d7765` (paper-searcher merge).
- **A — frozen corpus**: `evals/corpora/`, built once from the hand-written Europe PMC queries in `evals/questions/*.yaml`. Report ids `4527ccdd9b1af4a0` (psoriasis) and `1a968bd0ec2529f5` (RSV).
- **B — searcher corpus**: `evals/corpora-searcher/`, built with `python -m litreview.bench build-corpus evals/questions/*.yaml --searcher --out evals/corpora-searcher` (network: Europe PMC, preprints, PubMed only). Report ids `dd8aa94928477944` (psoriasis) and `f27c5b46768c915b` (RSV).
- Judging: 250 answers were already on file from an earlier session. The judge answered the 109 pending tasks left over from the last frozen-corpus runs, then the 183 (+44 follow-on) tasks the searcher-corpus runs generated, for **586 answers total** in `evals/judge/answers.jsonl`. Every `run` below finished with an empty `answers.pending.jsonl`.

## Papers collected, gated, read

| | A — frozen, psoriasis | A — frozen, RSV | B — searcher, psoriasis | B — searcher, RSV |
|---|---|---|---|---|
| Collected | 108 | 82 | 208 | 208 |
| Full text | 45 | 39 | 73 | 71 |
| Passed the exact-entity gate | 84 | 62 | 184 | 179 |
| Read (planner) | 16 | 16 | 16 | 16 |
| Verified findings extracted | 223 | 99 | 155 | 54 |
| Report sentences kept / dropped by guard | 76 / 2 | 67 / 1 | 78 / 1 | 53 / 8 |
| Review references | 5 | 5 | 5 | 5 |
| Open Targets reference targets | 37 | 0 (none for RSV) | 37 | 0 |

The searcher roughly doubles the candidate pool (108→208, 82→208 — RSV hit the
`max_papers` ceiling from the searcher rather than running out of hits) and
passes more papers through the exact-entity gate in absolute terms, but the
**planner still reads 16 papers either way** (`config/pipeline.yaml`'s
`plan.max_papers`), so a bigger corpus does not mean a more complete report —
it only changes *which* 16 papers get read, and the RSV searcher corpus
produced barely a quarter of the verified findings the frozen RSV corpus did
(54 vs 99) from the same 16-paper budget, because more of what it collected was
abstract-only or tangential (mutation-surveillance papers, antiviral compound
screens) rather than the prevention trials the hand-written RSV queries
targeted directly.

## Headline

*Superseded for the pipeline rows: the anchored-evidence section at the top of this file
re-ran all four reports and reports the current numbers.*

| Metric | Psoriasis A (frozen) | Psoriasis B (searcher) | RSV A (frozen) | RSV B (searcher) |
|---|---|---|---|---|
| **Overall (0–100)** | **77.5** | 72.9 | 67.9 | 67.9 |
| Claims | 76 | 78 | 67 | 53 |
| Claims supported by their citations | **1.00** | 0.94 | 0.94 | **0.96** |
| Claims supported by the collected papers | **1.00** | 0.96 | **1.00** | 0.96 |
| Numbers present in the evidence | 1.00 | 1.00 | 1.00 | 1.00 |
| Claims naming a near-miss entity | 0 | 0 | 0 | 0 |
| Review facts covered | 0.00 | 0.00 | 0.00 | 0.00 |
| Vital review facts covered | 0.00 | 0.00 | 0.00 | 0.00 |
| Known drug targets named (reachable) | 0.50 | 0.39 | – | – |
| Supported claims carrying a number | 26 | 6 | 35 | 15 |
| Distinct sources cited | 49 | 54 | 41 | 32 |
| Gates passed | 8/8 | 7/8 | 8/8 | 7/8 |

Psoriasis got *worse* on the searcher corpus (77.5 → 72.9); RSV was a wash
(67.9 → 67.9, to one decimal — corpus_grounded and one citation-accuracy claim
moved oppositely and cancelled out). On neither question did the searcher
corpus produce a better report. The full per-question scorecards are in
`evals/results/*/2026*/scorecard.md`; copies of both B reports and scorecards
are in `evals/samples/psoriasis-prevention-targets-searcher/` and
`evals/samples/rsv-infant-prevention-targets-searcher/`.

One gate result needs a caveat, not a defence: psoriasis-B's
"judge catches flipped effects ≥ 0.7" gate reads **0.67, FAIL** — not because
the pipeline produced a bad claim, but because *I* chose too narrow an
evidence sentence for one deliberately-corrupted multi-drug claim (it names
five drugs; my cited sentence named one), so code's same-topic overlap check
didn't fire and the corrupted claim read as "supported" instead of
"contradicted" (2 of the 3 flip perturbations attempted on that report were
caught correctly; the other two reports had no flip-kind perturbation drawn
from their sampled claims at all). Every entity and number perturbation across
all four scorecards — the other eight kind-checks — was caught at 1.00.
I'm reporting this against myself because
the instruction was to be a strict, literal judge, and a missed catch is
exactly the kind of thing that instrument is supposed to surface.

## Accuracy against the papers

Both corpora hit **0.94–1.00 citation accuracy** — the guard and the model are
about equally careful regardless of which papers they're fed. But strong
judging (mine, checking the actual quoted evidence unit rather than trusting
that a citation exists) found the same two categories of real error in *both*
corpora, at similar rates:

- **Subject/object swaps that the numbers-and-names guard cannot catch.**
  Psoriasis-B: *"IL-17 effectively suppressed core inflammatory pathways:
  IL-17, neutrophil extracellular trap formation, cytokine-cytokine receptor,
  and chemokine signaling in mice."* The cited paper's own sentence is *"all
  three treatments suppressed core inflammatory pathways: IL-17, [...]"* — the
  three treatments did the suppressing; IL-17 is one of the things suppressed,
  not the suppressor. Every name and number in the claim is in the evidence
  (`IL-17` appears twice, `mice` is the right species), so `report/guard.py`'s
  number/name check passes it. The claim's subject is simply wrong.
- **Citation misattribution: a true, verbatim sentence cited to the wrong
  `EvidenceUnit`.** This is the more serious pattern, and it showed up in
  *both* corpora independently:
  - RSV-A: the report states *"The clinical severity patterns differed across
    seasons... Both nirsevimab recipients experienced markedly milder
    illness..."* — all three sentences are word-for-word from the paper
    (PMID40872923), but the report cites them to `E76` (a sentence about
    statistical power, p=0.40) and `E78` (an unrelated incidence-rate
    sentence). The unit whose own quote actually *is* this text, `E77`, was
    extracted correctly and exists in the same report — it just isn't the one
    these three sentences point to.
  - Psoriasis-B: *"Following treatment with montelukast cream... reduced the
    PASI scores and delayed the disease progression..."* is cited to a unit
    whose own quote is about reduced immune-cell frequencies, not PASI scores;
    the correct sentence is in the same paper, cited by a different, correctly
    quoted unit elsewhere in the same report.
  - A related version of the same bug: RSV-A's `E96` has a `quote` field about
    LMIC cost-saving ($3.50/dose) but an `effect_detail` field ("cost-effectiveness
    results were most sensitive to changes in price, efficacy and duration")
    that summarises a *different* sentence from the same paper. Because
    `report/guard.py`'s `unit_text()` checks a report sentence's numbers and
    names against `quote + effect_detail` together, a sentence built from the
    unverified `effect_detail` half can pass the guard even when it isn't in
    the verbatim `quote`.
  In every case, `verify_quote` — checked per-unit — is not lying: the *cited
  unit's own text* really was found somewhere in the paper. The problem is one
  level up, in which unit a written sentence gets attached to, or in treating
  `effect_detail` (never required to be verbatim) as equally citable to
  `quote` (which is). This is a real, narrow gap in the citation guard as
  currently written, present before the searcher was added and not changed by
  it — it showed up in both corpora because I checked both closely, not
  because the searcher made it worse.
- **A quote/paraphrase mismatch caused by the searcher's own data.** In
  RSV-B, two `EvidenceUnit`s (`ae224002d9c9c635`, `3f781dfc497e6beb` on
  PMID42106072) were extracted with `quote_status: exact` and read correctly
  at extraction time, but the benchmark's independent re-verification against
  `Paper.all_text()` now reports `quote_validity = 0.96` (**FAIL**) for that
  corpus. The cause: the searcher's Paper record for this article carries two
  different abstract texts — `Paper.abstract` (1774 chars, no "Highlights")
  and `Paper.sections["abstract"]` (846 chars, *with* a "Highlights" bullet
  list that has slightly different wording — "in infants" vs. the narrative
  abstract's "in young children"). `all_text()` uses only `Paper.abstract` and
  explicitly skips `sections["abstract"]` to avoid duplicating it, so a quote
  copied from the Highlights list — genuinely present in the paper, and valid
  when it was extracted — cannot be found by a second verification pass run
  against a different assembly of the same paper's text. This is not a
  fabrication (I checked; the sentences are real, verbatim, just in a section
  of the paper `all_text()` doesn't include), but it is a real, reproducible
  discrepancy specific to full-text papers the searcher builds this way, and
  it is new: the frozen corpora's `quote_validity` is 1.00 on both questions.
  **I did not touch this code — flagging it as a blocker, not fixing it.**
  *Still open, and now also seen on psoriasis-B: see the anchored-evidence section.*

Beyond citation misattribution, I read the actual paper text behind more than
20 individual report sentences per corpus while judging (not a separate
"spot check" pass — the same act of finding the cited evidence *is* checking
it against the paper), across both drug-mechanism findings (DGAT2, c-Rel,
UBE2L3, nirsevimab pharmacokinetics, F-protein vs. G-protein targeting) and
numeric findings (PASI scores, coverage percentages, IC₅₀ values, half-lives).
Outside the two error classes above, every other sentence I checked matched
its source paper's actual wording and direction. Both chatbot baselines
repeated the *same* documented hallucinations from the first run verbatim on
the new corpus — "Palivizumab (Synagis): An older humanized mAb targeting G
protein" (it targets F), "bapineuzumab" as a VEGF inhibitor (it's an
anti-amyloid-β Alzheimer's antibody), FLG/tasquinimod (FLG is the atopic-
dermatitis gene; tasquinimod isn't an FLG drug) — because the closed-book
prompt and the model are unchanged; only the corpus is.

## How completely the question was answered

Neither corpus moved the pipeline's biggest known weakness: **review-fact
recall was 0.00 on all four reports**, frozen and searcher alike. This isn't a
new failure — it's structural. The pipeline writes report sentences only from
verified, specific `EvidenceUnit` quotes; the review-derived reference facts
are deliberately broad framing statements ("psoriasis involves dysregulated
cytokine signalling and altered keratinocyte function") that no single paper
states as a specific, quotable finding, so no report sentence can ever satisfy
them by construction. The searcher corpus doesn't change this because the
constraint is in `report/guard.py` and `stages/s7_write.py`, not in what
papers are available — even where I found near-matches in the searcher
corpus's *other* papers during reachability checks (a sentence describing
"disrupted skin barrier function, aberrant proliferation of keratinocytes, and
dysregulated immune response" that's close to one fact), the report itself
never wrote anything close, because its facet coverage is judged "complete"
once mechanism/causal/clinical/prevention/negative/safety facets each have a
supplying paper, not once the reviews' talking points are covered.

Where the corpora genuinely differ is in **target/detail breadth**:
distinct sources cited went up for psoriasis (49 → 54) but the reachable
Open Targets recall went *down* (0.50 → 0.39, against a bigger denominator —
14 reachable targets in the frozen corpus's 84 gated papers vs. 18 reachable
in the searcher's 184), because the extra papers the searcher pulled in
skewed toward mechanism/cell-biology studies (TOPK, CYSLTR1, FABP5, DGAT2,
PGGT1B) rather than the late-phase drug trials that put a target on Open
Targets' phase-3+ list. The psoriasis-B report actually names *more* distinct
molecular targets in absolute terms, but fewer of them are ones with an
approved or late-stage drug — a scientist reading it gets a wider mechanistic
survey and a narrower therapeutic one. For RSV, the reference-target metric
doesn't apply to either corpus (Open Targets has no phase-3+ human-gene
targets for a viral-fusion-protein prevention question), so completeness there
is judged only by review-fact recall (0.00 both) and facet coverage (all six
facets "covered" in every report) — on that measure the two corpora are tied,
but the searcher RSV corpus supplied barely half the verified findings the
frozen one did (54 vs. 99) from the same 16 papers, meaning what *did* make it
into facets like "safety" was thinner: RSV-B's Safety section is effectively
carried by fewer, more topically scattered findings (nirsevimab
pharmacokinetics, an unrelated CypA inhibitor's cytotoxicity assay) than
RSV-A's, which stays focused on nirsevimab and maternal-vaccine trial safety
data throughout.

Near-miss handling held on both corpora: 0 claims named a near-miss entity in
any of the four reports, and the RSV searcher corpus's own near-miss test
(swap "RSV" for "human metapneumovirus" in a real report sentence) was
correctly caught as unsupported by the missing-entity check, same as it was
for the frozen corpus. The exact-entity gate passed 179/208 and 184/208 of the
larger searcher pools — proportionally similar to 62/82 and 84/108 for the
smaller frozen pools — so the gate scales fine; it's what happens after the
gate (planner still capped at 16 papers) that determines completeness, not
gate accuracy.

## What broke

- **The extraction writer still truncates multi-sentence quotes at
  `DraftSentence.text`'s 450-character cap**, landing mid-word: psoriasis-A
  had 4 truncated sentences ("...Compared with the wild-type BMDM, the
  expressed." / "...tamoxifen (TAM) was the." / "...no such significant
  correlation was observed with 1." / "...the absolute concentration of
  DGAT2 in the."), psoriasis-B had 4 different ones (one repeats — "...the
  expressed." — the rest cut off at different points: "...concentration of
  DG.", "...paired non-lesional and lesional skin of psoriasis patient.",
  "...skin lesions of patients with psoriasis (F."). RSV had none in either
  corpus — this appears to be specific to psoriasis papers' denser, longer
  result sentences. This is a different bug from the "replies were being cut
  off" fix in the first round (that was about whole chunks/sections being
  lost to the token budget; this is one kept sentence ending ungrammatically
  inside the budget, present before the searcher and not changed by it). I did
  not touch the code; flagging `litreview/models.py`'s
  `DraftSentence.text: str = Field(max_length=450)` and the writer prompt as
  the place to look.
- **Citation misattribution** (a true sentence pointing at the wrong
  `EvidenceUnit`) and **the `effect_detail`-vs-`quote` guard loophole**,
  detailed above under "Accuracy against the papers" — present in both
  corpora, so not introduced by the searcher, but newly surfaced by strong
  judging rather than the numbers/names guard.
- **A quote valid at extraction time fails re-verification** against a
  differently-assembled version of the same paper's text, specific to the
  searcher's full-text papers (`Paper.abstract` vs. `Paper.sections["abstract"]`
  disagreeing) — detailed above. This one **is** new to the searcher corpus.
- **My own judging missed one deliberately-corrupted claim** (flip
  perturbation, psoriasis-B), for the reason given above under "Headline".
  Not a pipeline bug, but worth being honest about since I am the instrument
  being relied on for every other verdict in this document.

## Verdict

*Partly superseded: after the anchored-evidence change the searcher corpus scores
higher than the frozen one on RSV (70.9 vs 65.5) — see the section at the top of this file.*

**No — on this evidence, the searcher does not improve the pipeline, and on
psoriasis it makes the report worse.** Overall score fell on psoriasis
(77.5 → 72.9) and was flat on RSV (67.9 → 67.9); citation accuracy, review-fact
recall and near-miss safety were statistically indistinguishable between the
two corpora; Open Targets target recall fell (0.50 → 0.39); and the searcher
corpus introduced one new, reproducible defect (the `Paper.abstract` /
`sections["abstract"]` mismatch) that the frozen corpora do not have. Doubling
the number of collected papers did not help, because completeness is capped
by the planner's fixed 16-paper read budget regardless of pool size, and the
extra papers the searcher found skewed toward tangential mechanism studies
(cell lines, cyclophilin inhibitors, structural biology) rather than the
late-phase trials and prevention-specific evidence the hand-written queries
were built to surface directly. The one place the searcher corpus looks
better on paper — more distinct sources cited (54 vs. 49 for psoriasis) — is a
breadth-of-mention effect, not a completeness or accuracy gain: it comes with
*fewer* verified findings on RSV (54 vs. 99) and *lower* reachable target
recall on psoriasis. Before trusting the searcher corpus for a real scientist's
question, the two bugs specific to it (`quote_validity` FAIL from the
double-abstract mismatch, and the searcher's tendency to surface tangential
papers before the prevention/trial evidence a scientist actually wants) need
to be understood, not just noticed.

## Unanswered / not independently re-verified

- I did not re-derive or dispute the code-level checks (`verify_quote`,
  `numbers_in`, `opposite()`) themselves — I trusted them as specified and
  only judged where the model's/report's sentences and citations point.
- The 0.67 flip-catch gate for psoriasis-B was not corrected by re-answering;
  fixing it would mean re-running the benchmark a further time, which risked
  not finishing this comparison. It's reported as a known miss instead.
- I did not build or inspect `evals/goldset.jsonl` (still doesn't exist, same
  as the earlier round) — this document only grades whole reports, not
  per-field extraction accuracy.
- Wall-clock runtime wasn't captured precisely (the harness doesn't log it and
  the model-call cache made reruns fast); corpus build for both questions
  together took a few minutes over network, and each scored pipeline run
  completed in well under 10 minutes with a warm cache.

## Tests

`pytest` → **127 passed**, with no GPU and no network, in under a second (up
from 100 in the first round — the paper-searcher merge added its own tests).

## Files

- Frozen-corpus scorecards used for A: `evals/results/psoriasis-prevention-targets/20260924T020438Z/scorecard.md`, `evals/results/rsv-infant-prevention-targets/20260924T020441Z/scorecard.md`
- Searcher-corpus scorecards used for B: `evals/results/psoriasis-prevention-targets/20260924T035919Z/scorecard.md`, `evals/results/rsv-infant-prevention-targets/20260924T035923Z/scorecard.md`
- Searcher-corpus sample reports: [`evals/samples/psoriasis-prevention-targets-searcher/`](evals/samples/psoriasis-prevention-targets-searcher/), [`evals/samples/rsv-infant-prevention-targets-searcher/`](evals/samples/rsv-infant-prevention-targets-searcher/)
- Judge answers: `evals/judge/answers.jsonl` (586 answers)

---

# Audit — benchmark findings and deployment readiness

**23 September 2026.** Written by an audit agent (Claude Opus). It covers the code
at commit `c5d7765` plus the uncommitted benchmark files named in the section
above. This was a read-only audit: no pipeline code, prompt, knowledge table or
config was changed, and no network call was made. The benchmark scores quoted
here are the ones in the section above. Claude Sonnet judged them at development
time, and **no scientist has reviewed those verdicts or this audit.**

What I ran:

- `pytest`: 127 passed.
- Small offline Python checks against the stored reports, the corpora, the
  run artifacts in `data/bench.sqlite3` and the code. Each check is described
  where its result is used.

I did not start a new benchmark run.

## Benchmark summary

**Did the searcher help? Not yet.**

- Psoriasis got worse on the searcher corpus (77.5 → 72.9).
- RSV was flat (67.9 → 67.9).
- Target recall fell (0.50 → 0.39).
- Supported claims that carry a number fell sharply (psoriasis 26 → 6, RSV 35 → 15).

The searcher did collect papers a scientist would want. Examples: a real-world
effectiveness study of maternal RSVpreF vaccination, palivizumab prophylaxis
studies, a five-year phase 3 extension of deucravacitinib, and long-term
risankizumab data. The pipeline never read them, for reasons in the pipeline
rather than the searcher (see "Completeness" below). So the searcher has not yet
been given a fair test.

**Accuracy: good, but not as good as the scores say.**

- **Citation accuracy is 0.94 (psoriasis) and 0.96 (RSV)** on the searcher
  corpora. Every quote was verified at extraction time. There were no
  near-miss claims and no distractor leaks.
- **What went wrong**, all of which the numbers-and-names guard lets through:
  - **Misattributed citations**: a true sentence pointed at the wrong finding
    (RSV-A three times, psoriasis-B once).
  - **Subject swaps**: "IL-17 suppressed … IL-17 …".
  - **Unsupported claims**: "CYSLTR1 was elevated expression of CYSLTR1 … and
    healthy controls with pigmented nevi".
  - **Truncation**: 4 psoriasis-B sentences were cut mid-word at the 450-character
    cap. Three of them were printed ("…the expressed."), because code adds a full
    stop after the guard has run.
- **The guard loophole is wider than reported.** The guard accepts names and
  numbers from `effect_detail`, from `species_raw` and from `n`. The model
  writes all three fields, and none of them is checked against the paper. The
  writer prompt tells the model to write from "Quote and Result". The judge is
  shown the same fields as evidence.
- **The RSV-B `quote_validity` FAIL (0.96) has a Collector root cause.** The JATS
  parser keeps the first `<abstract>` element (`litreview/parsing/sections.py:153-158`).
  - For 5 searcher papers, that stored section is a different text from
    `Paper.abstract`. Three of them are Highlights bullet lists, for example
    PMID 42106072, which caused the FAIL.
  - Extraction reads the stored section, while re-verification reads
    `Paper.abstract`, so the two disagree.
  - For the Highlights papers, extraction never read the real abstract.
  - The frozen corpora have none of these mismatches (checked).
- **My spot check disagreed with the judge more often than the scores
  suggest.** I read 12 report sentences against the paper text, 6 from each
  searcher report.
  - I agreed on 6 and partly agreed on 1.
  - I disagreed on 5. In 4 of these the judge was too lenient (for example,
    "FABP5 in keratinocytes reduced the infiltration of neutrophils", where the
    paper says *deletion of* FABP5 did). In 1 it was too harsh.
  - Three of the lenient verdicts trace back to the benchmark code:
    - Two were settled by its verbatim-substring shortcut, without the judge.
    - In the third, the judge was shown the model's own "subjects: mice" as
      if it were evidence.
  - So the true citation accuracy is below 0.94–0.96.
  - Details are in [Reliability](#where-the-project-is-lacking-2-reliability).

**Completeness: the reports do not yet answer the question a scientist asked.**

- **Review-fact recall was 0.00 on all four reports.** On the same facts the
  "chatbot, no papers" baseline scored 0.23–0.42.
- **Psoriasis-B missed 11 of 18 reachable targets**, among them PDE4B/D, JAK1–3
  and IL12B.
- **The prevention facet has 0 verified findings on both searcher reports.**
  Safety has 0 on psoriasis-B. Yet the report's own "What the answer needed"
  table marks both "covered".
- **What a scientist would still be missing: RSV**
  - The F protein appears once, in a structural sentence about nirsevimab
    escape mutants. No sentence says that RSV F (prefusion F) is the target to
    act on.
  - The maternal RSVpreF trial's efficacy (75% [−11%, 94%] and 83% [−48%, 99%])
    was extracted and cited but never written.
  - Palivizumab and clesrovimab are absent.
  - The Answer leads with a coverage rate and a cyclophilin inhibitor.
  - "Candidate targets" lists GA (gestational age), LIBRA-seq (a method) and
    SARS-CoV-2.
  - "Negative results" includes a chemistry purification step.
- **What a scientist would still be missing: psoriasis**
  - The Answer leads with TOPK, which comes from one mechanism paper.
  - IL-23 and IL-17 appear, but without trial effect sizes.
  - TNF and JAK inhibitors appear only as drugs that lowered TOPK in patients,
    never as targets with evidence of their own. PDE4 does not appear at all.
  - The one prevention-relevant human study, early secukinumab in new-onset
    psoriasis (STEPIn), is not in the Prevention section. That section cites
    hepatitis B "prophylaxis" instead.
- **Three causes this audit found**, none of them in the section above:
  1. **Only 40 papers per question are ever eligible to be read.** Only the 40
     top papers by BM25 get a model grade, and the planner reads only graded
     papers. So 144 of 184 passing psoriasis papers and 139 of 179 passing RSV
     papers could never be read.
  2. **The RSV population filter drops most of the evidence.** "Infants"
     normalises to the species `other`, so the filter drops findings in
     "patients", "participants", "pregnant women", mice and cotton rats. It
     keeps findings whose species reads "null" or "unclear". The filter dropped
     **226 of 347** proposed RSV-B findings, and 84 of 224 on RSV-A. The report's
     method line never mentions this: 347 − 10 − 33 − 24 ≠ 54. This explains
     RSV-B's 54 findings (against RSV-A's 99) better than the corpus does.
  3. **"Distinct sources cited" counts evidence ids, not papers.**

**Bottom line.** The pipeline is careful: it verifies quotes, blocks near-misses
and is deterministic on the laptop. But it is not yet complete or trustworthy
enough for a scientist to act on without re-reading the papers. The fixes are
mostly narrow code changes, and they come first in the plan below.

## Where the project is lacking: 1. Functionality

What exists in code today, and what must exist before Propeller can run this as a
service on their network. "Verified" means I checked it in code or with an
offline run during this audit.

| Area | State today (evidence) | Needed before deployment |
|---|---|---|
| **Model serving on the office GPU** | The `office` profile has never run. Every measured result is MedGemma 4B via Ollama on a laptop. The profile sends two model names to **one** `base_url` (`config/models.yaml:17-31`, `litreview/config.py` `Settings.base_url`). A vLLM OpenAI server serves one base model per process (not verified against the client's vLLM version). The guided-decoding field is marked "confirm against the deployed version" (`config/models.yaml:20-23`). **The vLLM client reports the last model called, not the model asked for** (verified offline). It has no `version_of` (`litreview/llm/client.py:139-140`), so `run.model_versions`, `EvidenceUnit.extracted_by_model` and the cache keys can name the wrong model. No weight revision is pinned for vLLM. The GPU spec is still unknown (README, open question 1). | GPU spec. A model choice that fits. A per-tier endpoint (or one model). Version pinning. A structured-output smoke test. A timed re-benchmark on the office model. |
| **Model licences** | No file in the repo records a licence for any model (`grep -i licen` finds nothing outside `evals/`). MedGemma (the measured model) and Qwen3 (the planned model) come under different terms: reportedly Google's Health AI Developer Foundations terms and Apache-2.0 respectively. Not verified offline. | Legal sign-off per model file, and a licence register next to `config/models.yaml`. |
| **Installing into an air-gapped network** | `pyproject.toml:6-12` gives lower bounds only. There is no lock file, wheelhouse, container, systemd unit or install runbook. `scripts/dev_setup.sh` installs from PyPI. The dev venv runs Python 3.13, while the project requires ≥3.11. | A pinned lock file with hashes, an offline wheelhouse, images (app, vLLM, GROBID), model weights moved across the gap with checksums, and an install runbook. |
| **Getting papers in** | `litreview search` must run on a machine that can reach the internet, and the JSONL file is carried across by hand. `write_papers` writes no manifest or hash (`litreview/sources/corpus.py:26-31`). `CorpusSearcher` accepts any file for any question. If a question was asked before, a new file is silently ignored unless `--refresh` is given (`litreview/runner.py:149-152`). When no entity is recognised, the **full question text** is sent to Europe PMC and NCBI (`litreview/sources/query.py:76-78`). `"search": true` on POST /ask makes the server itself go online (`litreview/api.py:126`). Full text is open access only: 73 of 208 and 71 of 208 searcher papers, and 7 of the 16 psoriasis-B papers read were abstract-only. The drop folder and GROBID are TODO docstrings (`litreview/dropfolder.py`, `litreview/parsing/grobid.py`). | An agreed transfer procedure with a manifest and hash. No question text leaves the network. Internet paths disabled on the server. Drop folder plus GROBID for the PDFs Propeller already pays for. |
| **API and CLI: auth, users, concurrency** | There is no authentication. `rater` is free text in the request body (`litreview/api.py:49-52, 151-160`). POST /ask runs the whole pipeline inside the HTTP request (`api.py:124-140`), and a cold run took 40–55 min on the laptop. There is no job id, status, cancel or listing. The docstring's `uvicorn litreview.api:app` fails because there is no module-level `app` (`api.py:9`; README's `--factory` form works). Concurrent runs share SQLite with default settings (`litreview/db.py:44`). | Background jobs with status. Sign-in, with identity taken from the session. A bounded GPU worker queue. |
| **UI** | There is none. No setup form: the API accepts `approved_terms`, but nothing shows the parsed terms for approval. The report is static HTML with hover quotes (`litreview/stages/s8_render.py`). There is no review screen: `EvidenceUnit.review_status`, `reviewed_by` and `corrected_fields` are declared (`litreview/models.py:487-489`) but never written by any code. Ratings come in only through POST or the CLI. | Setup form, report view, review and rating screen. |
| **Unbuilt parts** | Stage 9 chat (`litreview/stages/s9_chat.py`). PPTX (no renderer, no `python-pptx`). Patents (`litreview/sources/patents.py`). Drop folder. GROBID PDF rung. OpenAlex (`litreview/sources/openalex.py`). Alias lookup. The embedding model is configured but unused (`config/models.yaml:72-75`). Figures and tables are not read. Retraction status comes only from publication types at search time (`litreview/ingest.py:18-19, 51-55`). | See steps 48–54 below. |
| **Storage, cache, backups** | SQLite, plus one JSON file per model call under `.cache/llm`, plus `out/`. There is no backup, restore or retention. The cache stores every prompt, including the question and the paper text, in plaintext (`litreview/llm/cache.py:66-74`). `save_papers` replaces a question's paper set, and `INSERT OR REPLACE` overwrites paper bodies (`litreview/db.py:63-75`). So an older report cannot be re-derived after a refresh. | Paper sets frozen per run. Backups that include the cache, which is what makes vLLM re-runs identical. Encryption at rest per client policy. |
| **Audit trail, logging, monitoring** | Runs record model and prompt versions, but not who asked. Logs go to stderr only (`litreview/cli.py:74`). There are no log files, metrics, health check or alerts. A quote-failure rate above 30% only logs a warning (`litreview/stages/s5_extract.py:446-450`). | Per-run user, time, corpus hash and config hash. Structured logs. A health endpoint. Alerts. |
| **Secrets and config** | `.env.example` lists `UNPAYWALL_EMAIL`, `EPO_OPS_KEY/SECRET`, `LENS_API_TOKEN`, `GROBID_URL` and `DROP_FOLDER`, which no code reads. `DB_PATH` is read (`litreview/config.py`) but not listed. There is no API-key support for vLLM. On the good side, the config refuses non-local model endpoints and any temperature other than 0 (`config.py:159`). | A production `.env` template containing only what is used. |
| **CI** | None: no CI configuration exists in the repo. | Tests, lint and a no-model end-to-end run on every push, plus a release build. |
| **Gold set** | `evals/goldset.jsonl` exists but holds only a comment line and three `EXAMPLE` rows. `evals/run_extract.py` is not wired to extraction (TODO at lines 120-123). | 20 hand-labelled papers (5 held out) and a working per-field scoreboard. |
| **The client's report format** | The layout is config (`config/report.yaml`), but the client's decks have not arrived. The user cannot pick sections. There is no background, confidence-in-conclusion, competitor-landscape or unmet-need section, all of which the client listed (README, "What the client actually asked for"). There is no PPTX. | The format taken from the old decks, sections the user can pick, and a PPTX renderer. |
| **Performance on the target hardware** | Unknown. On the laptop a cold run took 40–55 min for 11 papers (first run). The latest runs were not timed: the cache was warm, and `run.json` records the same second for `started` and `finished`. | A timed, cold, concurrent run on the office GPU against an agreed time budget. |
| **Acceptance, runbooks, handover** | None exist. | An acceptance plan agreed with Propeller's scientists, a pilot, runbooks and a handover. |

## Where the project is lacking: 2. Reliability

### Accuracy and fabrication risk

- **Fuzzy quote verification can accept a quote whose meaning has changed.** I
  checked this: `verify_quote` returns `fuzzy` for "…did significantly
  reduce…" against a source reading "…did not significantly reduce…". It also
  returns `fuzzy` for "rarely" changed to "often" (`litreview/extract/verify.py:32,
  134`).
  - The report displays the model's version of the quote.
  - The report also says every quote "was matched word for word"
    (`s8_render.py:66-72`, `s7_write.py:410-411`).
  - The two searcher reports hold 13 fuzzy quotes:
    - the 5 I aligned against the paper were harmless word swaps
      ("Furthermore" for "Moreover");
    - 4 are quotes cut off at the 700-character field limit;
    - the remaining 4 were not checked.
  - This is the fabrication guard's one soft spot, and it should be closed
    before anything else.
- **Fields the model writes are treated as evidence** at three points:
  - the guard (`litreview/report/guard.py:45-52`, which checks `effect_detail`,
    `species_raw` and `n`);
  - the writer (`litreview/llm/prompts/write_section.md:47-48` says write from
    "Quote and Result"; rule 4 says say who was studied);
  - the judge (`litreview/bench/run.py:66-68`).

  Report sentences built this way include "…in human tissues" (the paper says
  "across skin and spleen tissues") and "IL-17 promotes IL-6 production in
  mice". The second is a background sentence with literature citations; the
  paper does not say "in mice".
- **The guard checks names and numbers, not meaning.** It cannot catch:
  - subject swaps (IL-17 as the suppressor);
  - a lost modifier ("deletion of Fabp5 … reduced" became "FABP5 … reduced");
  - misattribution (the RSV-A E76/E78 case, montelukast);
  - padded citations (the brodalumab relapse sentence also cites a CRISPR
    mouse paper).
- **Truncation** at `DraftSentence.text` max 450 (`litreview/models.py:250`) is
  hidden, because a full stop is appended after the guard runs
  (`s7_write.py:202-206`).
- **Label pollution feeds the wrong sections:**
  - 25 of 54 RSV-B findings are `no_effect`. At least 12 of those 25 are
    synthesis steps, procedures or modelling choices, not results (E31,
    E35–E42, E45, E50 and E53 in the report's evidence table). A purification
    step prints under "Negative and conflicting results".
  - Demographics ("Of the 219 psoriasis vulgaris patients, 102 (46.6%)
    females") are labelled `no_effect`.
  - Hepatitis B "prophylactic measures" set `setting: prevention`.
  - All six findings from the HBV-reactivation study (E146–E151) were filed
    under clinical, so the Safety section is empty.
- **A review got through as primary evidence.** PMID 41416024 ("a PubMed
  literature search … five unique randomized controlled trials") is typed
  `research-article`. It ranked #1 and is cited as human clinical evidence.
  The review check reads only publication types (`litreview/ingest.py:18`).
- **The species filter is wrong, and it drops findings silently**
  (`litreview/extract/normalize.py:26-39`, `litreview/stages/s5_extract.py:369,
  409-411`, `litreview/stages/s7_write.py:404-406`). Details are in the summary
  above. The report's own caveat says findings "were not filtered by
  population" (`s7_write.py:318-322`, printed when a population is parsed).
- **The near-miss rule silenced a cross-reactive paper.** It removed all 8
  sentences from an RSV/hMPV cross-neutralising antibody paper. That is correct
  by the rule, but for a company building multi-pathway biologics it is a
  product decision to take with the client, not a default to leave unexamined.

### Determinism

- **The mechanism is in place, but proven only on the laptop.** Temperature 0
  is enforced (`config.py:159`), calls replay from the cache, and "same report
  twice" passed on all four scorecards. That was on the laptop, with a warm
  cache.
- **On vLLM the cache key uses whichever model was called last** (verified
  offline). Keys therefore depend on call order: a resumed or reordered run
  misses the cache and asks the model again. Nothing pins the weights.
  Whether vLLM gives identical output under concurrent batching was not
  measured (not verified).
- **A failed chunk or section is skipped silently and not cached**
  (`s5_extract.py:376-379`, `s7_write.py:179-181`). The vLLM client has no
  retry (`litreview/llm/client.py:125-128`). One transient server error
  therefore changes a report, with no alarm.
- **`--refresh` replaces the frozen paper set for every user of that question**,
  and paper bodies are overwritten in place (`db.py:63-75`). The
  "byte-identical rerun" acceptance test then holds only until someone
  refreshes.

### The judge and evaluation validity

The judge is Claude Sonnet, answering from a file at development time. It
cannot run inside the air gap, and no scientist has checked its verdicts. The
code checks it applies ("being a better judge earns no more trust") are sound
in design, but three of them have holes. My spot check read each claim, its
cited quote, and the paper's own text in `evals/corpora-searcher/*/papers.jsonl`:

| Report · claim | Judge | Paper text | My verdict |
|---|---|---|---|
| Pso-B: "FABP5 in keratinocytes reduced the infiltration of neutrophils in the ear, dermis, and peripheral blood" | supported (settled in code as a verbatim passage) | "specific **deletion of** Fabp5 in keratinocytes reduced the infiltration of neutrophils…" (PMID 37967009) | **Disagree.** The meaning is inverted. `verbatim_in` accepts any substring of ≥40 characters (`litreview/bench/judge.py:102-108`). |
| Pso-B: "ALOX5 and IL17RA expression correlated positively (r=0.89) in human tissues" | supported (verbatim, checked in code) | "…( r = 0.89) across skin and spleen tissues" from public datasets. No species is named (PMID 42409798). | **Disagree** on "human". It matched the model's `effect_detail`, which the judge sees as evidence. |
| Pso-B: "IL-17 promotes IL-6 production in mice" | supported | A background sentence citing other literature. The paper does not say "in mice" (PMID 40430037). | **Disagree** (at best partially supported). |
| Pso-B: "Brodalumab … quickest time to relapse, whereas bimekizumab … longest … in patients" | supported | Stated by PMID 41416024. The second citation (E70, a CRISPR mouse paper) says nothing of it, and 41416024 is itself a literature review. | **Partial.** Padded citation plus a review leak. |
| Pso-B: "TOPK levels decreased significantly after tofacitinib treatment" | supported | "after 12 weeks of tofacitinib treatment, TOPK levels decreased significantly" (PMID 39090602) | Agree |
| Pso-B: "BMS-986165 decreases IL17A + TCRγ + T cells in mice" | supported | Flow cytometry in the BMS-986165-treated group of IMQ mice (PMID 40038877) | Agree |
| RSV-B: "nirsevimab was found to cause two types of amino acid substitutions within the nirsevimab-binding site" | supported | "two nirsevimab **resistant mutants emerged**" after 12 rounds of in-vitro selection, and sequencing found the substitutions (PMID 39572535) | **Disagree.** Selection, not causation, and the in-vitro context is lost. |
| RSV-B: "Maternal smoking exposure was reported in 27.0% of mothers" | **contradicted** | "395 mothers (27.0%) reported any exposure to tobacco smoke during pregnancy" (PMID 42654750) | **Disagree: it is supported.** Code's `opposite()` fires on "had no exposure" (verified; `judge.py:174-214`). |
| RSV-B: "nirsevimab was administered to 1462 newborns, representing 94.0% of the eligible population" | supported | "Among 1554 eligible newborns, 1462 received nirsevimab … 94.0%" | Agree (but it is a coverage rate, not target evidence) |
| RSV-B: "Structural analysis revealed that … N208 … makes polar contacts with nirsevimab…" | supported | Verbatim in PMID 39572535 | Agree |
| RSV-B: "RSVpreF resulted in 50% neutralizing titers peaking 2 weeks post-vaccination" | supported | "combined RSV-A/-B 50% neutralizing titers peaked 2 weeks after vaccination" (PMID 42486480) | Agree. The same citation holds the trial's efficacy numbers, which were never written. |
| RSV-B: "This indicates a favorable selectivity index." | supported | Part of the F832/33 cytotoxicity sentence (PMID 42467054) | Agree on the text. As a standalone "Safety" sentence it has no subject. |

Other validity problems:

- **The completeness metric depends on the judge.** Frozen-corpus psoriasis
  review-fact recall was 0.83 under MedGemma and 0.00 under Sonnet. (Those
  were different runs, 11 and 16 papers read, so the comparison is only
  indicative.) Two questions and one run each give no estimate of variance.
  Psoriasis-B failed the flip-catch gate (0.67).
- **"Distinct sources cited" counts evidence ids, not papers**
  (`bench/run.py:147`). The "49 → 54 sources" in the section above is 54
  findings from 16 papers.
- **Several documents still show self-judged numbers:**
  - README "What it does" and "Results" (90.0 / 90.6) are MedGemma-judged.
  - `evals/samples/README.md` says both sample sets were strong-judged. The
    frozen samples' scorecards say "The judge is medgemma1.5:4b" (report
    `8be152cae817d2e5`).
  - `CLAUDE.md` says 97 tests; 127 pass.

### Test coverage

- **127 tests pass** offline in under a second. They cover:
  - quote verification, the guard, cache and config, dedupe, normalisation,
    exactness and relevance grading;
  - section parsing and the searcher (through a mock transport);
  - the benchmark, and the pipeline with `FakeLLMClient`;
  - one HTTP test.
- **Nothing tests:**
  - the vLLM client or the `office` profile (no test mentions `OpenAICompatibleClient`);
  - the species filter (no test mentions `species_filter` or `species_mismatch`);
  - a fuzzy quote with a dropped negation;
  - a JATS record with a Highlights abstract;
  - the judge's verbatim-substring shortcut;
  - concurrent runs or crash-and-resume against a real model server;
  - extraction against the gold set (not wired).

### Operational reliability

- **There is no job queue.** A long pipeline runs inside an HTTP request.
- **SQLite runs on defaults**: no WAL, no busy timeout, no migrations.
- **Nothing is backed up.** The cache grows without limit, and there is no
  retention.
- **Failures are silent**: skipped chunks and "Not written" sections do not
  alert anyone.
- **A new corpus file is silently ignored** for a question already asked.
- **Nothing monitors** the model server, disk space or the quote-failure and
  guard-drop rates.

## Next steps, step by step

Steps are numbered in the order to start them, and "Depends on" refers to step
numbers. Owners are the roles in `CLAUDE.md`: **Collector** (sources, dedupe,
acquisition, GROBID, drop folder), **Reader** (screening, extraction, gold set,
accuracy), **Presenter** (setup form, report, review screen) and **Foreman**
(orchestration, cache, determinism, storage, chat tools).

The order follows three rules:

- Phase 0 is requests to Propeller, which take the longest to come back.
- Accuracy and reliability fixes (phases 1–5) come before new features (phase 6).
- Nothing is installed at the client (phase 7) until it has been measured on
  the office GPU (phase 4).

Every change to a prompt bumps its `version`. Every accuracy fix is measured
before and after with `python -m litreview.bench run evals/questions/*.yaml
--corpora evals/corpora-searcher --profile medgemma --judge-file
evals/judge/answers.jsonl`, or with `--from-run <id>` for a stored report.

### Phase 0 — Inputs from Propeller (request now; nothing in phase 1 waits for them)

1. **Get the office GPU and host spec.**
   - *What:* GPU model and count, VRAM, driver/CUDA version, CPU, RAM, disk, OS, and whether containers are allowed.
   - *Why:* It decides which models fit, the vLLM settings, context length and throughput. It is README open question 1.
   - *Where:* Record it in the README "Open questions" section and in a comment on the `office` profile in `config/models.yaml`.
   - *Done when:* The spec is written down and the team has console access to the machine.
   - *Owner:* Foreman · *Depends on:* —
2. **Get licence and data-terms sign-off.**
   - *What:* Client counsel approves each candidate model file (the Qwen3 family, MedGemma, EmbeddingGemma if used). They also approve the terms for storing Europe PMC / PubMed open-access full text and the NCBI usage terms.
   - *Why:* No licence is recorded anywhere in the repo, and MedGemma's terms differ from Qwen3's.
   - *Where:* A licence register next to `config/models.yaml`.
   - *Done when:* Written approval exists for each model and revision that will be installed.
   - *Owner:* Foreman · *Depends on:* —
3. **Agree the search-and-transfer procedure with client IT.**
   - *What:* Settle:
     - which machine may reach Europe PMC and NCBI, and who runs `litreview search`;
     - how the JSONL crosses into the air-gapped network (approved media or a one-way transfer);
     - what may leave: search terms only, never the question text.
   - *Why:* The searcher needs the internet and the server must not. Today the carry-across is informal.
   - *Where:* A procedure document, plus the flags in step 40.
   - *Done when:* IT has signed the procedure off.
   - *Owner:* Collector · *Depends on:* —
4. **Collect the report format.**
   - *What:* Get 5–10 old decks. Get the sections the scientists want, including background, confidence, competitor landscape and unmet need, and which of them they want to switch on or off.
   - *Why:* No gold-standard report exists. The layout is config precisely so that it can follow their format.
   - *Where:* `config/report.yaml` and `litreview/stages/s8_render.py` (step 52).
   - *Done when:* A marked-up example deck is agreed.
   - *Owner:* Presenter · *Depends on:* —
5. **Get real questions and Propeller's entity list.**
   - *What:* Two real target/indication pairs (one single-target, one pairwise). The diseases and targets they work on. Their journal subscriptions.
   - *Why:* The benchmark has two questions and has never exercised `combo_with`. The knowledge table decides what "exact" means.
   - *Where:* `evals/questions/`, `litreview/knowledge/entities.yaml`.
   - *Done when:* The list has been received.
   - *Owner:* Reader · *Depends on:* —
6. **Book scientist time.**
   - *What:* Name the scientists and the hours they can give to the gold set (about 2 days), to reviewing judge verdicts, and to acceptance and the pilot.
   - *Why:* No scientist has reviewed any verdict so far.
   - *Where:* Project plan.
   - *Done when:* Named people and dates are agreed.
   - *Owner:* Reader · *Depends on:* —
7. **Agree users and sign-in.**
   - *What:* Who may use the system, whether SSO/LDAP is available, and whether projects must be private per user or group.
   - *Why:* The API has no authentication today.
   - *Where:* Input to step 39.
   - *Done when:* The access model is written down.
   - *Owner:* Foreman · *Depends on:* —

### Phase 1 — Stop wrong statements reaching the page (accuracy fixes from the benchmark and this audit)

8. **Store the paper's own words for fuzzy quotes, and reject fuzzy matches that change meaning.**
   - *What:* On a fuzzy match, replace the model's quote with the aligned span from the source. Reject any fuzzy match that differs in a negation, a number or an effect word.
   - *Why:* A quote with "not" removed verifies as `fuzzy` today, and the report shows the model's text as if it were verbatim.
   - *Where:* `litreview/extract/verify.py:112-136`, `litreview/stages/s5_extract.py:392`, `tests/test_verify.py`.
   - *Done when:*
     - Tests show that dropped-negation and changed-number quotes fail.
     - Every `quote` in a new report is an exact substring of its paper.
   - *Owner:* Reader · *Depends on:* —
9. **Make the verbatim quote the only evidence the guard and the writer may draw on.**
   - *What:*
     - The guard stops accepting `effect_detail`, `species_raw` and `n` unless the value is itself in the quote.
     - The writer prompt stops offering "Result" as a source of wording.
     - Code checks `effect_detail` numbers against the quote at extraction time.
   - *Why:* This is the loophole reported for RSV-A E96, and the source of "in human tissues" and "in mice" above.
   - *Where:* `litreview/report/guard.py:45-52`, `litreview/stages/s7_write.py:136-153`, `litreview/llm/prompts/write_section.md` (bump `version`), `litreview/stages/s5_extract.py:327-328`.
   - *Done when:*
     - A test shows that a sentence whose number or name appears only in `effect_detail` is dropped.
     - A benchmark run shows no such sentence kept.
   - *Owner:* Reader · *Depends on:* —
10. **Require every citation to carry its sentence.**
    - *What:* Each cited unit must individually share the sentence's content words (a written overlap threshold). Otherwise the citation is removed, and the sentence is dropped if no citation is left. When a sentence is a near-verbatim copy of a different unit's quote from the same paper, re-point the citation to that unit.
    - *Why:* This covers the misattributions in RSV-A (E76/E78 against E77) and psoriasis-B (montelukast), and the padded [E70] citation.
    - *Where:* `CitationGuard.check` in `litreview/report/guard.py:98-158`.
    - *Done when:* Tests built from those three cases pass, and a re-run judged by a strong judge shows no misattributed citations.
    - *Owner:* Reader · *Depends on:* 9
11. **Catch lost modifiers and swapped subjects.**
    - *What:* When a unit's copied `change` words ("deletion", "knockout", "inhibition") are not in a sentence that names its target, drop the sentence. When a sentence is a substring of its quote, reject it if the words just before the substring are an intervention or a negation.
    - *Why:* "Deletion of Fabp5 … reduced" became "FABP5 … reduced", and "all three treatments suppressed … IL-17" became "IL-17 suppressed".
    - *Where:* `litreview/report/guard.py`, plus tests.
    - *Done when:* The FABP5 and IL-17 sentences are dropped in tests.
    - *Owner:* Reader · *Depends on:* 9
12. **Stop printing truncated sentences.**
    - *What:* Drop a sentence that reaches the length cap or has no terminal punctuation, instead of appending "." to it. Tell the writer to write one sentence per item rather than paste multi-sentence quotes.
    - *Why:* 4 psoriasis-B sentences were cut mid-word, and 3 were printed.
    - *Where:* `litreview/models.py:250`, `litreview/stages/s7_write.py:202-206`, `write_section.md` (bump `version`).
    - *Done when:* No kept sentence ends mid-word in the four benchmark reports after re-running them.
    - *Owner:* Reader · *Depends on:* —
13. **Fix the species filter and show what it removes.**
    - *What:*
      - Map infants, neonates, newborns, children, adults and mothers to `human`.
      - Decide, with a scientist, what a population in the question should do. The recommendation: label animal and cell findings rather than drop them, and drop only human findings from a different population.
      - Print the count of removed findings in "How this answer was assembled".
    - *Why:* The filter removed 226 of 347 RSV-B findings, kept methods sentences, and the method line does not add up.
    - *Where:* `litreview/extract/normalize.py:26-39`, `litreview/stages/s5_extract.py:369, 409-411`, `litreview/stages/s7_write.py:394-413`.
    - *Done when:*
      - Unit tests pass.
      - A re-run of RSV-B keeps the maternal-trial and animal prophylaxis findings.
      - The method line sums.
    - *Owner:* Reader · *Depends on:* 6 (for the rule only)
14. **Read the real abstract, not the Highlights.**
    - *What:* Choose the main `<abstract>` (untyped or `abstract`) and keep highlights out. Make extraction chunks, `Paper.abstract` and `Paper.all_text()` use the same abstract text.
    - *Why:* For 5 searcher papers the stored abstract section differs from `Paper.abstract`, and 3 of them hold Highlights lists (PMIDs 42106072, 42055251 and 42698634). This caused the RSV-B `quote_validity` FAIL, and for those papers the real abstract was never read.
    - *Where:* `litreview/parsing/sections.py:153-158`, `litreview/ingest.py:44-47`, `litreview/models.py:326-329`.
    - *Done when:*
      - A JATS fixture with highlights parses correctly.
      - RSV-B `quote_validity` is 1.00 on re-run.
    - *Owner:* Collector · *Depends on:* —
15. **Stop methods and background sentences becoming findings, and file setting and safety correctly.**
    - *What:*
      - Code allows `no_effect` only when the quote contains a null-result phrase (the guard's `_NULL_CLAIM`); otherwise the label becomes `association`, or the finding is dropped when the quote states no result.
      - `prevention` requires the prevention word to apply to the question's subject.
      - Early-intervention and new-onset wording counts as prevention.
      - Adverse-event, reactivation and toxicity words route a finding to the `safety` facet.
    - *Why:*
      - RSV-B's negative facet holds chemistry steps.
      - Psoriasis-B's prevention section cites HBV prophylaxis, while STEPIn early intervention is labelled `unclear`.
      - The Safety section is empty although six HBV findings were extracted.
    - *Where:* `litreview/stages/s5_extract.py:229-280`, `litreview/extract/normalize.py` (`classify_setting`), `litreview/llm/prompts/extract.md` (bump `version`).
    - *Done when:* Tests built from these sentences pass, and the Negative and Safety sections of a re-run contain only matching findings.
    - *Owner:* Reader · *Depends on:* —
16. **Catch reviews that are not typed as reviews.**
    - *What:* Treat an abstract as a review when it describes a literature search or pooled trials ("literature search", "we identified N trials", "systematic", "pooled"), whatever its publication type says.
    - *Why:* PMID 41416024 ranked #1 and was used as clinical evidence, although "reviews are not evidence" (CLAUDE.md).
    - *Where:* `litreview/ingest.py:18`, `litreview/stages/s3_rank.py` (`grade_from`).
    - *Done when:* A test shows it is excluded, and the psoriasis-B re-run no longer ranks it.
    - *Owner:* Collector · *Depends on:* —
17. **Fix target rows and target order.**
    - *What:* Keep only actionable rows: no demographic variables (GA), methods (LIBRA-seq), other pathogens or chemistry words. In target ordering, count "human clinical evidence of benefit" only when the target itself was modulated, not when another drug lowered it.
    - *Why:* RSV-B lists GA, LIBRA-seq and SARS-CoV-2 as candidate targets. TOPK leads psoriasis-B because JAK/TNF/IL-17 drugs lowered it in patients.
    - *Where:* `litreview/stages/s6_score.py:61-75, 155-163`, `looks_like_target` in `litreview/knowledge/__init__.py`.
    - *Done when:* Re-runs list no non-actionable targets, and IL-23/IL-17 lead psoriasis when the evidence supports it.
    - *Owner:* Reader · *Depends on:* 15
18. **Make the report agree with itself.**
    - *What:*
      - The plan table shows "gap (no finding extracted)" when a facet has 0 findings.
      - The Gaps section and the Prevention section use the same definition of prevention evidence.
      - "Most relevant source" is either the rank-1 paper or is renamed to say what it is (today it is the first full-text paper of the top grade; `s4_plan.py:63, 79`, which contradicts `s3_rank.py:22-23`).
    - *Why:* Psoriasis-B prints "covered" beside 0 findings. It says no prevention finding was extracted and then prints a Prevention section.
    - *Where:* `litreview/stages/s7_write.py:261-343`, `litreview/stages/s4_plan.py:60-79`.
    - *Done when:* Tests cover each inconsistency.
    - *Owner:* Presenter · *Depends on:* 13, 15

### Phase 2 — Make the measurement trustworthy

19. **Fix the judge's code shortcuts.**
    - *What:*
      - Show the judge only verbatim quotes, dropping "(Result: …; subjects: …)".
      - Make `verbatim_in` require a whole evidence sentence, or check that the words just before the matched span are not a modifier or negation.
      - Stop `opposite()` treating a descriptive "had no exposure" as a contradiction.
      - Count distinct **papers** cited, and report evidence ids separately.
    - *Why:* Three lenient verdicts and one false "contradicted" in the spot check come from these shortcuts, and the "sources" metric is mislabelled.
    - *Where:* `litreview/bench/run.py:66-68, 147-156`, `litreview/bench/judge.py:99-108, 174-214`, `tests/test_bench.py`.
    - *Done when:* The spot-check rows above become tests and pass.
    - *Owner:* Reader · *Depends on:* —
20. **Re-measure.**
    - *What:*
      - (a) Re-score the four stored reports with the fixed judge (`--from-run <id> --judge-file evals/judge/answers.jsonl`) and answer any new pending tasks.
      - (b) Run the full benchmark on both corpora with the phase 1 code.
      - Write both results up as a new section in this file.
    - *Why:* It separates the effect of the judge fix from the effect of the pipeline fixes, before any tuning.
    - *Where:* `evals/results/`, `evals/judge/answers.jsonl`, this file.
    - *Done when:* The new section is written, showing old and new numbers side by side.
    - *Owner:* Reader · *Depends on:* 8–19
21. **Correct the documents that show self-judged numbers.**
    - *What:*
      - Replace the MedGemma-judged headline figures (90.0 / 90.6) in README "What it does" and "Results".
      - Fix `evals/samples/README.md`'s claim that the frozen samples were strong-judged, or refresh those samples.
      - Fix the test count in `CLAUDE.md`.
      - Fix the `uvicorn litreview.api:app` docstring (`litreview/api.py:9`).
    - *Why:* A reader should not see a better score than the instrument supports.
    - *Where:* The files named above.
    - *Done when:* Every score in the docs names its judge.
    - *Owner:* Reader · *Depends on:* 20
22. **Build the gold set and wire the extraction scoreboard.**
    - *What:*
      - A scientist labels 20 papers following `evals/README.md`, holding 5 out.
      - Remove the template rows.
      - Wire `evals/run_extract.py` to `litreview/stages/s5_extract.extract_paper`.
    - *Why:* Stage 5's per-field accuracy is unknown, and CLAUDE.md says to build this before tuning extraction.
    - *Where:* `evals/goldset.jsonl`, `evals/run_extract.py:114-125`.
    - *Done when:* `python -m evals.run_extract` prints per-field precision and recall with quote validity 1.00. Extraction precision is > 0.85 (README target), or the gap to it is written down.
    - *Owner:* Reader · *Depends on:* 6, 8, 15
23. **Have a scientist check the judge.**
    - *What:* A scientist re-judges a random sample of at least 40 pipeline claims across the four reports (half of them "supported" verdicts settled by code), blind to the judge's verdict.
    - *Why:* All current accuracy numbers rest on an AI judge, and this audit disagreed with it on 5 of 12 claims.
    - *Where:* A file next to `evals/judge/answers.jsonl`, with the agreement rate recorded in this file.
    - *Done when:* The agreement rate is published, and every disagreement is either a test or an explained exception.
    - *Owner:* Reader · *Depends on:* 6, 20
24. **Widen the benchmark.**
    - *What:* Add at least 4 questions from Propeller's areas (I&I, obesity, CNS), including one pairwise (`combo_with`) question, with frozen searcher corpora and reviews.
    - *Why:* Two questions are not a sample, and the pairwise path is untested on real papers.
    - *Where:* `evals/questions/*.yaml`, `evals/corpora-searcher/` (build them on the outside machine).
    - *Done when:* At least 6 questions are scored in `evals/results/history.jsonl`.
    - *Owner:* Reader · *Depends on:* 5

### Phase 3 — Completeness: read the right papers, and write what was read

25. **Let more than the BM25 top 40 compete to be read.**
    - *What:* Pre-select papers for grading by study design as well as by words: trial and phase words, prevention words, and drug names from the knowledge table. Reserve a quota for human-clinical and prevention hints. Raise `rank.max_graded` within a measured time budget.
    - *Why:* The planner reads only graded papers (`s4_plan.py:36-45`), and only 40 are graded (`s3_rank.py:222-228`). The maternal RSVpreF effectiveness paper (rank 113) and the deucravacitinib phase 3 extension (rank 42) never had a chance.
    - *Where:* `litreview/stages/s3_rank.py`, `config/pipeline.yaml` (`rank`).
    - *Done when:* Benchmark re-runs read those kinds of papers, and target recall and supported numeric claims rise without accuracy falling.
    - *Owner:* Reader · *Depends on:* 20
26. **Plan by evidence type, not just full-text-first.**
    - *What:* Guarantee human-clinical papers for the clinical facet and prevention-tagged papers for the prevention facet before topping up with anything else.
    - *Why:* All 16 papers read in each searcher run were grade 3, and among equal grades full text wins. That favours mechanism studies.
    - *Where:* `litreview/stages/s4_plan.py:60-110`.
    - *Done when:* The plan table shows at least the minimum number of papers per facet with at least one finding each, wherever the corpus has them.
    - *Owner:* Reader · *Depends on:* 25
27. **Put the key numbers in the Answer.**
    - *What:* Feed the Answer and Clinical slices the human-clinical findings with effect sizes first. Enforce each section's sentence count in code (the Answer asks for 3–5 and got 10). Have the Answer name each target together with its kind of evidence.
    - *Why:* The RSVpreF efficacy finding was cited but never written, and the RSV Answer does not name the F protein.
    - *Where:* `litreview/stages/s7_write.py:233-255`, `best_units` in `litreview/stages/s6_score.py`, `config/report.yaml`.
    - *Done when:* The re-run RSV Answer names prefusion F, nirsevimab and maternal RSVpreF with their numbers, and every sentence passes the guard.
    - *Owner:* Presenter · *Depends on:* 9, 15, 17
28. **Decide about background from reviews (client decision).**
    - *What:* If Propeller wants the "background" they listed, add a section that quotes review abstracts verbatim. Label it "background from reviews — not evidence". Never cite it as evidence, and never count it in target order. If they do not want it, stop counting review-fact recall in the overall score.
    - *Why:* Review-fact recall is 0.00 by construction (the section above), while the chatbot covers these facts.
    - *Where:* `config/report.yaml`, `litreview/stages/s7_write.py`, `litreview/bench/run.py` (`overall`).
    - *Done when:* The client has decided and the change is measured.
    - *Owner:* Presenter · *Depends on:* 4
29. **Decide how to treat papers that name both the subject and a near-miss (client decision).**
    - *What:* Decide whether a sentence about an RSV/hMPV cross-reactive antibody may appear in an RSV report when it states an RSV result.
    - *Why:* The rule removed all 8 sentences from such a paper. For a multi-pathway company this is a product choice, not a bug.
    - *Where:* `litreview/report/guard.py:107-113`, `litreview/stages/s5_extract.py` (`entity_check`).
    - *Done when:* The decision is recorded in `CLAUDE.md` and covered by a test.
    - *Owner:* Reader · *Depends on:* 6
30. **Grow the knowledge table.**
    - *What:* Add Propeller's targets and diseases, with exact terms and near-misses.
    - *Why:* Every entity added sharpens the gate, the guard and normalisation. An unknown entity falls back to verbatim matching.
    - *Where:* `litreview/knowledge/entities.yaml`, `tests/test_exactness.py`.
    - *Done when:* Every entity in the step 5 list resolves from the table.
    - *Owner:* Reader · *Depends on:* 5

### Phase 4 — Serve on the office GPU

31. **Choose and pin the models.**
    - *What:* Choose the medium and small models that fit the GPU, and the quantisation. Record the Hugging Face repo, revision SHA and file checksums.
    - *Why:* Only MedGemma 4B has ever been measured, and nothing pins vLLM weights.
    - *Where:* The `office` profile in `config/models.yaml`, and the licence register.
    - *Done when:* The choice is recorded and licence-approved.
    - *Owner:* Foreman · *Depends on:* 1, 2
32. **Record the right model version on the vLLM path.**
    - *What:* Add `version_of(model)` to `OpenAICompatibleClient`, returning name@revision from config or `/v1/models`.
    - *Why:* It reports the last model called (verified offline), which corrupts provenance and makes cache keys depend on call order.
    - *Where:* `litreview/llm/client.py:79-140`, `litreview/llm/__init__.py:54-56`, `litreview/runner.py:136`, a new test using `httpx.MockTransport`.
    - *Done when:* A test shows two models get two versions in any call order.
    - *Owner:* Foreman · *Depends on:* —
33. **Give each tier its own endpoint.**
    - *What:* Allow a `base_url` per model tier (with the settings and client factory following), or run one model for all jobs.
    - *Why:* The profile sends two model names to one vLLM URL.
    - *Where:* `config/models.yaml:17-31`, `litreview/config.py` (`Settings`), `litreview/llm/__init__.py` (`make_client`).
    - *Done when:* Both tiers answer on the office machine.
    - *Owner:* Foreman · *Depends on:* 31
34. **Smoke-test guided decoding on the deployed vLLM.**
    - *What:* A script sends each schema once (question, relevance, extraction, writing, judge). It confirms that `response_format` or `guided_json` is honoured, and that `maxLength`, `enum` and `maxItems` hold. It also sets `--max-model-len` for the largest prompt.
    - *Why:* The profile says to "confirm against the deployed version". Malformed output must be impossible, not unlikely.
    - *Where:* `scripts/`, `config/models.yaml:20-23`.
    - *Done when:* The script passes on the office box.
    - *Owner:* Foreman · *Depends on:* 33
35. **Retry model-server errors, and never skip silently.**
    - *What:* Add bounded retries with backoff for transient errors. A chunk or section that still fails marks the run incomplete and says so at the top of the report.
    - *Why:* Today a failure is logged, skipped and not cached, so reports differ between runs and nobody is told.
    - *Where:* `litreview/llm/client.py:125-133`, `litreview/stages/s5_extract.py:376-379`, `litreview/stages/s7_write.py:179-181`.
    - *Done when:* Tests with a flaky mock server pass.
    - *Owner:* Foreman · *Depends on:* —
36. **Prove determinism on vLLM.**
    - *What:* Show byte-identical reports with a warm cache, a cold cache with serial requests, and a cold cache under concurrent load. If the concurrent case fails, serialise model calls or document that determinism comes from cache replay.
    - *Why:* The client's first acceptance test so far has passed only on the laptop with a warm cache.
    - *Where:* The benchmark's "same report twice" gate, run on the office box.
    - *Done when:* The results are recorded in this file.
    - *Owner:* Foreman · *Depends on:* 32–35
37. **Re-benchmark on the office model, timed, and choose an on-site judge.**
    - *What:* Run all benchmark questions cold, recording wall-clock time per stage. Set `plan.max_papers` and `rank.max_graded` to an agreed time budget. Choose the largest local model as the judge tier, check it catches at least 0.9 of perturbations, and calibrate it against step 23.
    - *Why:* MedGemma results do not carry over to another model, and the Sonnet judge cannot run inside the air gap.
    - *Where:* `evals/results/`, this file, and `jobs.judge` in `config/models.yaml`.
    - *Done when:* A results section for the office model is written, with timings.
    - *Owner:* Reader and Foreman · *Depends on:* 20, 23, 25–27, 34, 36

### Phase 5 — Make it a multi-user service

38. **Run questions as background jobs.**
    - *What:* POST /ask returns a job id at once. One GPU-bound worker runs the pipeline. Add `GET /jobs/{id}` (status and progress), cancel, and resume after a restart through `resume_run_id`.
    - *Why:* Runs take tens of minutes, and today they run inside the HTTP request.
    - *Where:* `litreview/api.py:124-140`, `litreview/runner.py`, a new worker module.
    - *Done when:* Two concurrent submissions both finish, and the API answers in under 1 s.
    - *Owner:* Foreman · *Depends on:* —
39. **Add sign-in and identity.**
    - *What:* Authenticate every endpoint with the method from step 7. Take the rater and the requester from the session, not the request body. Apply per-project visibility if required.
    - *Why:* Today anyone on the network can run jobs, read every report and rate as anyone.
    - *Where:* `litreview/api.py:49-52, 151-160`.
    - *Done when:* Unauthenticated requests get 401, and tests cover roles.
    - *Owner:* Foreman · *Depends on:* 7, 38
40. **Keep the server off the internet.**
    - *What:* A production setting that refuses `"search": true` and `--search`. A host firewall with no egress. Never send the question text as a query: ask for an approved term instead.
    - *Why:* The server can currently go online on request, and the query fallback discloses the question itself.
    - *Where:* `litreview/api.py:126`, `litreview/cli.py`, `litreview/sources/query.py:76-78`.
    - *Done when:* Tests pass, and a firewall check from the server fails to reach Europe PMC.
    - *Owner:* Foreman, with the Collector for the query change · *Depends on:* 3
41. **Freeze the paper set per run.**
    - *What:* Store each run's paper ids and content hashes on the `Run`. Version paper bodies instead of overwriting them. Let a refresh create a new set rather than replace the one other users rely on.
    - *Why:* After a refresh, older reports cannot be re-derived or audited.
    - *Where:* `litreview/db.py:63-86`, `Run` in `litreview/models.py`, `litreview/runner.py:149-156`.
    - *Done when:* A test refreshes the papers and then `load_result` reproduces the old report bytes.
    - *Owner:* Foreman · *Depends on:* —
42. **Set up the database for concurrent use.**
    - *What:* WAL mode, a busy timeout, and a schema-version table with a migration procedure. Move to Postgres only if the load test (step 60) shows lock contention.
    - *Why:* SQLite runs on defaults today, with several writers expected.
    - *Where:* `litreview/db.py:43-46`.
    - *Done when:* A concurrent test passes without "database is locked" errors.
    - *Owner:* Foreman · *Depends on:* 38
43. **Keep an audit trail.**
    - *What:* For every run, record the user, time, question, corpus file hash, model versions, prompt versions and config hash. Keep an append-only log of ratings and review decisions.
    - *Why:* "Which version did you read, and who asked?" must always have an answer.
    - *Where:* `Run` in `litreview/models.py`, `litreview/runner.py`, `litreview/db.py`.
    - *Done when:* The report metadata shows these fields, and a test covers them.
    - *Owner:* Foreman · *Depends on:* 39, 41
44. **Add logging, health checks and alerts.**
    - *What:* Structured log files with rotation. A health endpoint covering model-server reachability, disk space and queue length. Alerts when the quote-failure rate exceeds 30%, when chunks fail, or when the guard drops an unusual share of sentences.
    - *Why:* Today these are warnings on stderr.
    - *Where:* `litreview/cli.py:74`, `litreview/api.py`, `litreview/stages/s5_extract.py:446-450`.
    - *Done when:* An operator can see the last 24 h of runs and their health on one page, or in one log query.
    - *Owner:* Foreman · *Depends on:* 38
45. **Back up, and encrypt at rest.**
    - *What:* Nightly backups of `data/`, `.cache/llm`, `out/`, `config/` and the knowledge table, with a restore drill. Encrypt at rest as client policy requires, and set a retention period.
    - *Why:* The cache is what makes vLLM re-runs identical, and it holds the questions and paper text in plaintext.
    - *Where:* Ops scripts and the runbook (step 63).
    - *Done when:* A restore onto a clean machine reproduces a stored report byte for byte.
    - *Owner:* Foreman · *Depends on:* 1
46. **Write the production configuration.**
    - *What:* A production `.env` template containing only variables the code reads (add `DB_PATH`; drop `UNPAYWALL_EMAIL`, `EPO_OPS_*`, `LENS_API_TOKEN`, `GROBID_URL` and `DROP_FOLDER` until code reads them). Add optional vLLM API-key support. Record a config hash on each run.
    - *Why:* The current template lists keys that nothing uses.
    - *Where:* `.env.example`, `litreview/config.py`.
    - *Done when:* Every variable in the template is read by code, and a test enforces it.
    - *Owner:* Foreman · *Depends on:* 40
47. **Set up CI and a release build.**
    - *What:* On every push, run `pytest`, `ruff` and a `--no-model` end-to-end run. A tagged release builds the wheelhouse, images and checksums (steps 55–56).
    - *Why:* There is no CI today.
    - *Where:* A new CI config in the repo.
    - *Done when:* CI is green on `main`, and a release artefact is produced from a tag.
    - *Owner:* Foreman · *Depends on:* —

### Phase 6 — Features the client asked for (only after phases 1–5 hold)

48. **Build the drop folder and the GROBID PDF rung.**
    - *What:* A watched directory: hash each PDF, match it to a DOI, run GROBID to get TEI, split it into sections (TEI splitting already exists in `litreview/parsing/sections.py`), and ingest it with `text_source: drop_folder`. Rerun the question with the new full text.
    - *Why:* Only open-access full text reaches the pipeline today. Propeller already pays for access, and Sci-Hub is ruled out.
    - *Where:* `litreview/dropfolder.py`, `litreview/parsing/grobid.py`, `litreview/ingest.py`.
    - *Done when:* A dropped PDF appears as full text in the next report, and a fixture test passes.
    - *Owner:* Collector · *Depends on:* 3, 5, 14
49. **Build the setup form.**
    - *What:* Question entry. Show the parsed entities, exact terms and near-misses for approval (the `approved_terms` API already exists). Pick a corpus file or import a search, choose sections, submit a job.
    - *Why:* Scientists will not use a CLI.
    - *Where:* A new front end on `litreview/api.py`.
    - *Done when:* A scientist submits a question and approves a term without help.
    - *Owner:* Presenter · *Depends on:* 38, 39
50. **Build the report view.**
    - *What:* A served HTML report with navigation, evidence-table filters, and a paper view that highlights the quoted sentence in its context (the papers are already in SQLite).
    - *Why:* Hover titles on static HTML are not a way to review 150 findings.
    - *Where:* `litreview/stages/s8_render.py`, `litreview/api.py`.
    - *Done when:* The README's Presenter bar is met: a junior scientist reads a report end to end without asking what anything means.
    - *Owner:* Presenter · *Depends on:* 38, 39
51. **Build the review screen.**
    - *What:* Confirm, correct or reject each finding (writing `review_status`, `reviewed_by` and `corrected_fields`), flag sentences, and rate the report. Corrections feed the gold set and the tests.
    - *Why:* The fields exist, but nothing writes them.
    - *Where:* `litreview/models.py:487-489`, `litreview/api.py`, `litreview/db.py`.
    - *Done when:* A reviewed finding shows its status in the report, and corrections export to `evals/goldset.jsonl` format.
    - *Owner:* Presenter, with the Reader · *Depends on:* 22, 39, 50
52. **Deliver the client's report format and PPTX.**
    - *What:* Implement the step 4 layout. Let the user choose sections. Add the sections they asked for, with confidence stated by a written code rule (never by a model grade of quality). Add a PPTX renderer from `report.json`.
    - *Why:* Their workflow ends in a deck.
    - *Where:* `config/report.yaml`, `litreview/stages/s8_render.py`, `pyproject.toml` (`python-pptx`).
    - *Done when:* A scientist accepts a generated deck for a real question.
    - *Owner:* Presenter · *Depends on:* 4, 27
53. **Build Stage 9 chat.**
    - *What:* Tools over the frozen evidence (`search_evidence`, `get_paper`, `compare_targets`, `list_conflicts`), scoped to one project. Every answer sentence passes the same citation guard.
    - *Why:* It is in the README contract and answers the client's "it can't remember" complaint.
    - *Where:* `litreview/stages/s9_chat.py`.
    - *Done when:* Chat answers cite evidence, and a benchmark of chat answers passes the same gates.
    - *Owner:* Foreman · *Depends on:* 36, 38
54. **Add the outside-network lanes.**
    - *What:* Patents for the competitor landscape only: title, abstract, assignee, priority date and claim 1. Retraction and citation refresh for papers cited in stored reports. Both run on the outside machine and are carried in like the search file.
    - *Why:* The client uses patent databases, and retraction status is frozen at search time today.
    - *Where:* `litreview/sources/patents.py`, `PatentRecord` in `litreview/models.py`, `litreview/sources/openalex.py`.
    - *Done when:* The patent section renders as its own class, and a report shows a "retracted since" banner in a test.
    - *Owner:* Collector · *Depends on:* 2, 3

### Phase 7 — Air-gapped packaging and production install

55. **Lock the dependencies and build an offline wheelhouse.**
    - *What:* A lock file with hashes for Python ≥ 3.11, a wheelhouse built on the release machine, and an SBOM.
    - *Why:* `pyproject.toml` has lower bounds only, and the office machine cannot reach PyPI.
    - *Where:* `pyproject.toml`, a new lock file.
    - *Done when:* A clean install on an offline VM passes `pytest`.
    - *Owner:* Foreman · *Depends on:* 47
56. **Build the container images.**
    - *What:* Images for the app and worker, for vLLM (pinned version, CUDA matching the office driver) and for GROBID. Export them as tarballs with checksums, and add compose or systemd definitions.
    - *Why:* These make the install reproducible without the internet.
    - *Where:* New deployment files in the repo.
    - *Done when:* The images start on an offline test host.
    - *Owner:* Foreman · *Depends on:* 31, 48, 55
57. **Move the model weights across the gap.**
    - *What:* Download the pinned revisions outside, checksum them, transfer them by the approved route, and verify them inside.
    - *Why:* The weights are the largest thing crossing the gap, and they must match what was benchmarked.
    - *Where:* The step 3 procedure and the licence register.
    - *Done when:* The checksums match on the office machine.
    - *Owner:* Foreman · *Depends on:* 2, 3, 31
58. **Set up the outside search machine.**
    - *What:* Install `litreview search` with an NCBI key and a generic contact mailbox. Make it write a manifest (question id, search time, per-source counts, file hash), and have the inside check the hash on import.
    - *Why:* Today the transfer carries no provenance, and the wrong file can be used silently.
    - *Where:* `litreview/sources/corpus.py:26-47`, `litreview/cli.py`, `litreview/runner.py:149-156`.
    - *Done when:* An import with a mismatched hash is refused.
    - *Owner:* Collector · *Depends on:* 3, 40
59. **Install on the office machine.**
    - *What:* Driver, container runtime, directories (`DATA_DIR`, `CACHE_DIR`, `OUT_DIR`, backups), no-egress firewall, the API reachable on the LAN only behind TLS, service accounts. Follow the install runbook.
    - *Why:* This is the production system.
    - *Where:* Runbook (step 63).
    - *Done when:* Someone who did not write the runbook installs from scratch, `pytest` and the smoke test (step 34) pass, and one real question completes.
    - *Owner:* Foreman · *Depends on:* 1, 45, 56, 57
60. **Load-test on the target hardware.**
    - *What:* Run N questions back to back and several concurrent submissions. Measure time per question, GPU memory, and growth of the cache and database.
    - *Why:* Performance on the office hardware is unknown.
    - *Where:* Results in this file.
    - *Done when:* The agreed time budget is met, or the knobs are reset and re-measured.
    - *Owner:* Foreman · *Depends on:* 37, 59

### Phase 8 — Acceptance and handover

61. **Agree the acceptance test plan with Propeller.**
    - *What:* The pass criteria, each with a number:
      - (a) a byte-identical re-run;
      - (b) quote validity 1.00;
      - (c) scientist-graded citation accuracy of at least an agreed target on K reports;
      - (d) zero near-miss claims;
      - (e) completeness against the scientists' own answer for the two real pairs from step 5;
      - (f) the time budget;
      - (g) no egress from the server.
    - *Why:* "Done" must be something the client can check.
    - *Where:* An acceptance document, with the results appended to this file.
    - *Done when:* It is signed by Propeller.
    - *Owner:* Reader (lead) · *Depends on:* 6
62. **Run a pilot with the scientists.**
    - *What:* Scientists use the system on real questions for 2–4 weeks. Collect ratings and flagged sentences, and triage them weekly into tests or prompt changes (bumping versions, measured with `--from-run`).
    - *Why:* This is the only route to knowing whether reports are useful, not just careful.
    - *Where:* The `ratings` table and review screen, then `evals/`.
    - *Done when:* Every step 61 criterion passes on the pilot questions.
    - *Owner:* Presenter and Reader · *Depends on:* 49–51, 59, 61
63. **Write the runbooks.**
    - *What:* Install and upgrade, search and transfer, backup and restore, swapping a model (with a re-benchmark), changing a prompt (version bump plus before/after), responding to a wrong citation, cache and database growth, and adding knowledge-table entries.
    - *Why:* After installation there is no vendor support.
    - *Where:* A `docs/` folder, or the client's wiki.
    - *Done when:* Each runbook has been executed once by a Propeller staff member.
    - *Owner:* Foreman, with each role for its own area · *Depends on:* 59
64. **Hand over and sign off.**
    - *What:* Train the scientists and IT. Name owners at Propeller. Agree the support window. Deliver the final results section for the office model.
    - *Why:* Someone at Propeller has to own the system after we leave.
    - *Where:* A handover document.
    - *Done when:* Propeller signs off acceptance.
    - *Owner:* Foreman · *Depends on:* 60, 62, 63

---

# Benchmark results — first real-model run

**23 September 2026.** The first end-to-end runs of this pipeline with a real
model, on two sample questions, against two chatbot baselines. Everything here
was produced on a laptop with no network access: MedGemma 1.5 4B (Google's
medical Gemma 3) through Ollama, on an M4 MacBook with 16 GB.

**Note (added later):** every score in this section was graded by MedGemma
itself acting as judge, which the section below discovered to be the weakest
part of the instrument (see "The judge was the least reliable part of the
benchmark"). The strong-judge section above this one re-scores the frozen
corpora and is not comparable to the numbers below — read `overall` here as
"what the model that wrote the report thought of its own writing," not as an
independent accuracy figure.

The scores matter less than the failures underneath them. §[What broke](#what-broke-and-what-fixed-it)
is the part to read: every fix in this round came from a specific wrong
sentence in a report, and those failures say more about building on a small
local model than the totals do.

## What was measured

Two questions, each with a frozen corpus fetched once from Europe PMC and then
used offline:

| Question | Papers | With full text | Reviews (yardstick) |
|---|---|---|---|
| If I want to prevent psoriasis, what areas should I target? | 108 | 45 | 5 |
| What should I target to prevent RSV infection in infants? | 82 | 39 | 5 |

Three systems, same model, same temperature (0), same papers available:

- **This pipeline** — gate, rank, plan, extract, verify, write under the citation guard
- **Chatbot, no papers** — the question alone, as a scientist would ask ChatGPT
- **Chatbot + pasted abstracts** — the ten best-matching abstracts pasted in with
  the question, as a scientist would paste papers into a chat window

Reproduce with:

```
python -m litreview.bench run evals/questions/*.yaml --profile medgemma
```

## Headline

| Metric | Psoriasis | | | RSV | | |
|---|---|---|---|---|---|---|
| | **Pipeline** | No papers | Pasted | **Pipeline** | No papers | Pasted |
| **Overall (0–100)** | **90.0** | 69.1 | 54.5 | **90.6** | 70.3 | 84.4 |
| Claims made | 58 | 19 | 22 | 53 | 15 | 13 |
| Claims supported by their citations | **1.00** | – | 0.45 | **0.98** | – | 0.77 |
| Claims supported by the collected papers | **1.00** | 0.79 | 0.93 | **1.00** | 0.73 | 1.00 |
| Numbers present in the evidence | **1.00** | – | – | **1.00** | 0.33 | 1.00 |
| Claims naming a near-miss entity | **0** | 0 | 0 | **0** | 0 | 0 |
| Review facts covered | **0.83** | 0.50 | 0.33 | 0.71 | 0.50 | **0.75** |
| Vital review facts covered | **0.83** | 0.50 | 0.33 | **0.90** | 0.40 | **0.90** |
| Known drug targets named | **0.50** | 0.21 | 0.00 | – | – | – |
| Supported claims carrying a number | **22** | 0 | 0 | **25** | 1 | 2 |
| Distinct sources cited | **40** | 0 | 9 | **34** | 0 | 9 |

"Known drug targets" is Open Targets' list of targets of drugs at phase 3 or
later for the disease, scored only over the ones actually reachable in the
corpus. It is not measured for RSV: Open Targets returns none for it, because
RSV prophylaxis targets a viral protein rather than a human gene.

Every safety check passes on both questions:

| Check | Psoriasis | RSV |
|---|---|---|
| Every quote verified against its paper | PASS (1.00) | PASS (1.00) |
| Every sentence cites evidence | PASS (1.00) | PASS (1.00) |
| No claim names a near-miss entity | PASS | PASS |
| Every number is in the cited evidence | PASS | PASS |
| No planted distractor paper leaked through | PASS (0.00) | PASS (0.00) |
| Claims supported by citations ≥ 0.85 | PASS | PASS |
| Judge catches flipped effects ≥ 0.7 | PASS (1.00) | PASS (1.00) |
| Same question, byte-identical report | PASS | PASS |

Run shape: 11 papers read per question, 130 verified findings (psoriasis) and
74 (RSV), 58 and 53 report sentences kept, 3 and 1 dropped by the citation
guard. A cold run takes about 40 minutes (psoriasis) to 55 minutes (RSV) on the
laptop, including scoring; with the call cache warm it is minutes.

## Before and after

The same psoriasis report pipeline, before and after this round of fixes,
scored by the identical judge (`--from-run` scores a stored report, so the
comparison is not confounded by judge changes):

| | Before | After |
|---|---|---|
| Overall | 69.3 | **90.0** |
| Claims supported by their citations | 0.91 | **1.00** |
| Review facts covered | 0.08 | **0.83** |
| Known drug targets named | 0.07 | **0.50** |
| Supported claims carrying a number | 6 | **22** |
| Papers read | 6 | **11** |
| Verified findings extracted | 37 | **130** |
| Distinct sources cited | 16 | **40** |

The before-run was not bad at being careful — it already grounded every claim
in the corpus. It was bad at being *complete*: it read six niche mechanism
papers and never opened the IL-23, IL-17 or TYK2 trials sitting in the same
corpus, because the relevance grader called everything equally relevant.

## What broke, and what fixed it

### The small model said yes to everything

Asked for a 0–3 relevance grade, MedGemma gave **39 of 40 papers a 3**,
including a Mendelian-randomisation study of *breast cancer risk in people who
have psoriasis*. Asked which of six information needs each paper could supply,
it ticked **all six on 36 of 40**. The planner, believing every need was
covered by the first few papers, read six papers and declared the answer
complete.

The model is no longer asked for either. It answers narrow questions about one
abstract — is the disease what this paper studies, or a risk factor for
something else? is it a review? which molecule did its own experiments change? —
and code turns those answers into a grade by a written rule, after checking
the named molecule actually appears in the abstract. Which needs a paper can
supply is decided by code from study-design words. Grades now spread 22 × 3,
14 × 2, 4 × 1, the planner reads 11 papers, and the trials come in.

### A knockout that made the disease worse was recorded as an improvement

The report said blocking PGGT1B helped, from a paper whose own sentence reads
"Bone marrow PGGT1B deficiency **aggravated** the psoriasis-like lesions". The
model had labelled it `direction: unclear, outcome: improved` — the exact
opposite of the finding, in a report a scientist would act on.

The model now copies the words it based each label on ("deficiency",
"aggravated") and code classifies those words, overruling the label when they
unambiguously disagree and recording the overrule on the finding. Negation is
read first, so "did not significantly reduce" is a null result, and words that
flip meaning ("increased survival") leave the model's label alone.

### The paper's own text was the last thing the model read — until it wasn't

Extraction put the worked examples after the paper text. For an abstract about
a HuR-targeting peptide, MedGemma returned *the guselkumab example from the
prompt* as its finding. Quote verification destroyed it, correctly, and the
paper produced **zero** findings. Moving the paper text to the end of the
prompt turned the same abstract into six verified findings.

### Repetition, fragments, and labels masquerading as claims

One section repeated a single sentence four times; another contained the bare
sentence "A."; a third said "Blocking participants improved psoriasis", which
is a data-model label with English wrapped around it. The writer now sees only
the quote, the result and who was studied — never the labels — and code drops
repeats, fragments under 25 characters, and sentences that merely restate a
label. Repeats fell from 40 in a run to 6.

### "Dropout" and "discontinuation" were listed as drug targets

A statistics-heavy trial paper contributed target rows named `dropout`,
`discontinuation` and `participants`; elsewhere `HaCaT cells` (a cell line) and
long measurement phrases became targets. Code now requires a target row to look
like something one could act on — a gene, protein, drug, cell type or pathway —
and rejects cell lines, measurements ("…frequencies", "…titer ratios") and
clauses. Targets are also grouped case-insensitively, so "Nirsevimab" and
"nirsevimab" stopped being two independent lines of evidence.

### A prevention question with an empty prevention section

MedGemma labelled **all 74** RSV findings `treatment` or `unclear`, including
"Nirsevimab … has shown high efficacy in RSV **prophylaxis** in newborns". The
prevention section of a prevention report read "No evidence for this section".
Code now reads the setting from the words the model already copied —
prophylaxis, vaccination, immunisation, "before challenge" — and the section
fills with the right evidence, including a 0.41% vs 1.09% infection rate in
immunised versus unimmunised neonates.

(Psoriasis still reports no prevention-specific evidence. That one is true: 1
of its 130 findings mentions prevention at all. The report says so, in the gaps
section, instead of inventing something.)

### The replies were being cut off

Seven extraction chunks and one whole report section were lost to replies that
hit the token cap. Schemas now bound what a reply may contain (at most 8
findings, 10 sentences, bounded field lengths), with a larger budget. No
truncation since.

### The judge was the least reliable part of the benchmark

A benchmark that cannot be trusted is worse than none, and the first two judge
designs were both wrong:

- asked for a **verdict**, MedGemma wrote "The evidence states that primidone is
  recommended…" and then answered `unsupported`. It penalised paraphrase, which
  punished the chatbot baselines far more than the pipeline — flattering us.
- asked "**is the main point stated?**", it said yes to nearly everything: 97% of
  closed-book claims "grounded", and **0 of the flipped-effect claims caught**.

The judge is now split the way the pipeline is. The model only copies the
evidence sentence that states the claim, plus details it cannot find. Code
decides: the sentence must really be in the evidence (if the model echoes the
claim back, code finds the closest evidence sentence itself), the claim's names
and numbers must be in the evidence, and the claim must not say the effect went
the other way. Claims that are verbatim passages of their evidence are settled
in code without asking the model at all — 47 of them on psoriasis.

Measured on deliberately corrupted claims, the judge now catches **100%** of
flipped effects, numbers and swapped entities on both questions.

### The guard let a near-miss entity through

Swapping "RSV" for "human metapneumovirus" in a finished sentence passed the
citation guard, because the knowledge table lists hMPV only as RSV's near-miss,
not as an entity in its own right. For a system whose first promise is "exact
means exact", that was the most serious bug found. The guard now checks the
question's own near-miss terms directly and drops any sentence naming one.

## What the chatbots got wrong

The benchmark only measures whether a claim is supported by the collected
papers. Read by hand, the closed-book answers are worse than that number
suggests — these are the errors a scientist would have to catch:

- **"Mutations in the FLG gene (e.g., FLG\*null) are common in psoriasis…
  Targeting FLG (e.g., tasquinimod) is a promising approach."** Filaggrin
  loss-of-function is the classic *atopic dermatitis* gene — one of the
  near-misses this system exists to exclude — and tasquinimod is a quinoline-3-
  carboxamide studied in prostate cancer, not an FLG-targeting drug.
- **"Blocking the IL-23 receptor (IL-23R) (e.g., ustekinumab)"** — ustekinumab
  binds the p40 subunit shared by IL-12 and IL-23, not the receptor.
- **"Inhibiting Vascular Endothelial Growth Factor (VEGF) (e.g., bapineuzumab…)"**
  — bapineuzumab is an anti-amyloid-β antibody for Alzheimer's disease.
- On RSV: **"The main targets are the RSV G proteins (G1 and G2)… Palivizumab
  (Synagis): An older humanized mAb targeting G protein."** Palivizumab and
  nirsevimab both target the **F** (fusion) glycoprotein. The pipeline's RSV
  report names prefusion F seven times and the G protein never.

The "pasted abstracts" baseline is much better grounded — it reads real papers —
but its citations do not hold up: **0.45** of its psoriasis claims were
supported by the papers it cited, against **1.00** for the pipeline. It
attributes iron-metabolism findings to the RIPK1 paper, and mixes claims from
papers it never cites. On RSV it scores well (84.4) because that corpus is
dominated by clinical prophylaxis abstracts that answer the question directly —
which is the honest result: when the top ten abstracts happen to contain the
answer, pasting them into a chat window works.

## What is still wrong

- **The guard checks that numbers are present, not what they mean.** The RSV
  report says years of life lost "exceeded 268.8 … and 271.8 … in the upper
  range of 95% uncertainty"; the quote reads "280.2 (95% UR: 268.8–290.6) …
  311.5 (95% UR: 271.8–377.9)". Every number is genuinely in the cited evidence,
  so the guard passed it — the model picked the *lower* bounds and called them
  the upper range. The judge caught it; code did not. This is the one remaining
  unsupported claim across both reports.
- **Target rows are still only as good as the copied name.** "YLL" (years of
  life lost) gets a row on RSV because it is shaped like a gene symbol.
- **Completeness is capped by what the planner reads.** 11 papers of 84 that
  pass the gate; the psoriasis report still misses IL17RA, PDE4B/D, TNF and
  JAK2, all present in the corpus.
- **The judge is the same 4B model that wrote the reports.** Its sensitivity is
  measured (100% on corrupted claims) rather than assumed, but a larger,
  different judge would be a better instrument.
- **Two questions is not a sample.** These numbers say the pipeline works on
  two biology questions with this model, nothing more.
- **`evals/goldset.jsonl` does not exist yet.** The benchmark grades reports;
  only a hand-labelled gold set grades extraction field by field, and until it
  exists Stage 5's per-field accuracy is unknown. See
  [`evals/README.md`](evals/README.md).

## Tests

`pytest` → **100 passed**, with no GPU and no network, in under a second.

Of these, 31 were added in this round and each one pins down a failure above:
grading rules and the tested-molecule check, direction/outcome reconciliation
from copied words, facet refiling, cell lines and non-targets, case-insensitive
target grouping, prophylaxis settings, repeats and fragments, the near-miss
guard rule, thinking-trace stripping, the judge's code-side checks, de-negation
detection, and reopening a finished run.

## Files

- Sample reports and their scorecards: [`evals/samples/`](evals/samples/)
  (`report.html` is self-contained — hover a citation to see its quote)
- Score history across prompt versions: `evals/results/history.jsonl`
- Full run artifacts (claims, nuggets, perturbations, baselines) stay in
  `evals/results/<question>/<timestamp>/`, which is git-ignored
- How the benchmark works: [`evals/README.md`](evals/README.md)

Model: `medgemma1.5:4b@433252621ab1` for every job, including the judge.
Prompt versions: question 1, relevance 3, extract 4, write_section 3.
