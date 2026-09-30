# Evaluation

> Moved from Localized-Airgap-LLM-for-literature-review with the rest of the
> literature review. This directory is a source checkout only: it is not in the
> wheel, so the benchmark is the one part of `bedrock.litreview` that does not
> run from an install.

Two instruments, measuring different things:

- **The benchmark** (`src/bedrock/litreview/bench/`) grades whole *reports* against
  references, with no scientist needed. Use it every time a prompt, a model or
  a stage changes.
- **The gold set** (below) grades *extraction*, field by field, against papers a
  scientist labelled by hand. It is the only way to know per-field accuracy,
  and it still has to be built.

## The benchmark

```
python -m bedrock.litreview.bench build-corpus evals/litreview/questions/*.yaml   # once, needs network
python -m bedrock.litreview.bench run evals/litreview/questions/*.yaml --models laptop
```

`questions/*.yaml` — the question, the Europe PMC queries that stand in for the
searcher (including a few deliberately near-miss papers), and the references:
review articles and an Open Targets disease id.

`corpora/<id>/` — frozen: `papers.jsonl` (the searcher's handoff format),
`references.jsonl` (review abstracts — never evidence, only a yardstick),
`reference_targets.json` (targets of drugs at phase 3 or later), `manifest.json`.

`results/<id>/<time>/scorecard.md` — pipeline vs "chatbot, no papers" vs
"chatbot + pasted abstracts", same model, same papers: accuracy against
citations, groundedness in the corpus, near-miss claims, distractor leaks,
completeness against review facts and reference targets, detail, determinism,
and how often the judge catches deliberately corrupted claims. The section
"where the pipeline lost points" is the to-do list. `results/history.jsonl`
tracks every run against its prompt versions.

Read the judge-sensitivity rows before trusting the rest: if the judge does not
catch flipped effects, its "supported" verdicts mean little.

**How the judge decides.** A small model cannot be trusted with a verdict: on
the first MedGemma 4B runs, asked for one, it rejected claims its own reasoning
said were stated; asked "is the main point stated?", it said yes to nearly
everything, including claims with the effect reversed. So the judge is split
the way the pipeline is ([`bench/judge.py`](../../src/bedrock/litreview/bench/judge.py)):

- the **model** copies the evidence sentence that states the claim, and lists
  details it cannot find
- **code** decides: a claim that is a verbatim passage of its evidence is
  supported outright; otherwise the copied sentence must be in the evidence
  (if the model echoed the claim instead, code takes the evidence sentence
  sharing most of its words), most of the claim's words and at least one of
  its names must be in the evidence, every name and number must be for full
  support, and the claim must not say the effect went the other way from the
  sentence

The same rules apply to every system. Nugget coverage works the same way: the
model copies the answer's sentence, code checks it.

`--from-run <run id>` scores a stored report instead of producing a new one —
how a report written by earlier code is put beside one written by today's.

### What a score rests on

Every scorecard says, above its table, the context window the report was
written at (checked by `service.preflight` before the first model call; "not
recorded" for anything scored before 30 September 2026) and who supplied the
verdicts. Every judged score carries **(j/n)**: it rests on j of the n claims
or review facts it should, and the rest had no verdict. A check whose input
was never built says **not measured** and why, and counts neither as passed
nor as measured. `python -m bedrock.litreview.bench history` lists past runs
the same way.

A stored report judged from a file with no baselines calls no model, so it
needs no server:

```
python -m bedrock.litreview.bench run evals/litreview/questions/<q>.yaml --from-run <run id> \
    --judge-file evals/litreview/judge/answers.jsonl --skip-baselines --skip-perturb
```

Every row of a judge file names its judge: `{"key": ..., "answer": {...},
"judge": "claude-opus-4-5"}`, or `"judge": "scientist:<name>"`. Rows written
before that field existed read as `"unknown"`.

### Comparing two runs

```
python -m bedrock.litreview.bench compare results/<q>/<before> results/<q>/<after>
```

It prints B minus A for every system, score and check, or **refuses** (and
exits non-zero) when the difference could be something other than the
pipeline:

- the two runs answer different questions;
- any score's judged fraction differs by more than 5 percentage points
  (`COMPARABLE_WITHIN` in `bench/compare.py`, a starting value) - 30 of 42
  claims judged against 42 of 42 is mostly the missing twelve;
- the raters differ, or the same raters judged in different shares
  (verdicts settled in code do not count);
- either run used answers whose judge is `"unknown"` - the answers on file
  before 30 September came from more than one judge, so the fix is to
  re-judge those claims with one named judge;
- either run was scored before any of this was recorded.

`--force` prints the comparison anyway, with the reasons at the top.

**Compare does not remove one rater's own noise.** The same asthma report,
byte for byte, scored 69.4, 69.2 and 69.2 across three passes of one judge
(OPEN_ISSUES #25). A difference inside that spread is not a change, however
cleanly compare prints it.

## The gold set

**Build this before writing a single prompt.** Everything in Stage 5 is
guesswork until it exists.

Without it, the best anyone can say is "the extraction looks pretty good."
Three weeks later a scientist finds a mouse study reported as a human trial,
and nobody can tell whether that is rare or everywhere.

### How to build it (2 days)

Pick **20 papers** from a real run. Not the first 20 — choose deliberately:

| Count | Kind | Why this category exists |
|---|---|---|
| 5 | Human clinical trials | The highest-value evidence |
| 5 | Mouse / animal studies | Most-confused category by a wide margin |
| 4 | In vitro / cell studies | Weakest evidence, must not be inflated |
| 3 | **Negative or no-effect results** | Models skip these unless told not to |
| 2 | Review articles | Not evidence — must produce zero findings |
| 1 | Mentions the target but is not about it | Tests false positives |

Then read all 20 as a human and write down every finding that *should* be
extracted, in the schema from `src/bedrock/litreview/models.py`. Get a scientist to check
it — ideally the junior scientist Person 3 is already shadowing.

The picking and the paperwork are done by code; the reading is not:

```
python -m evals.litreview.goldset_sheet corpora-searcher/obesity-treatment-targets
```

picks the 20 by these quotas from a frozen corpus (seeded, `--seed N`),
holds 5 back, and writes `evals/litreview/goldset/<corpus>/`: an `index.md`
saying why each paper was picked, one sheet per paper, and
`goldset.template.jsonl` with an empty entry per paper. Design and review
categories come from what the paper says it is (`study_design.py`). The
negative-result and "mentions but not about" categories code can only
*propose* - a null-result cue in the abstract, or the exact-entity gate's own
reason - so their entries say `"kind_confirmed": false` until a person
confirms them. A corpus short of a category says so in the index; pick those
from another corpus.

Those last three categories are the whole point. Models are eager to please:
given a paper showing a drug did not work, a naive prompt returns nothing,
because "no finding" feels like failure. Given a review, it happily extracts
the review's summary of someone else's trial as primary evidence. Neither
failure is visible unless it is in here.

### Labelling the gold set

For the scientist doing the labelling. Open a paper's sheet and its entry in
`goldset.template.jsonl` side by side. The sheet shows the paper and nothing
else: exactly the parts the pipeline sends the model, and the methods
paragraphs sent beside them. It shows nothing the pipeline produced, on
purpose - seeing the model's answer first makes a labeller agree with it.

**What counts as a finding.** A result *this paper* reports from its own
experiments or data, about a named molecule, gene or pathway: something was
changed or measured, and something happened (or did not). One entry per
sentence and target. For each, fill in:

| Field | What to write |
|---|---|
| `quote` | The sentence the result is in, copied exactly from the sheet. It is checked against the paper, so copy, do not retype |
| `target` | The molecule the paper changed or measured, as the paper names it ("IL-23", "MC4R") |
| `direction` | `block`, `activate`, or `unclear` |
| `disease` | The disease or model the result is about |
| `evidence_type` | `human_clinical`, `human_genetic`, `in_vivo` (animals) or `in_vitro` (cells) |
| `species` | `human`, `mouse`, `rat`, `cyno`, `cell_line` or `other` |
| `outcome` | `improved`, `no_effect`, `worsened`, `mixed`, or `association` (a measurement with no intervention) |
| `n` | The number studied, only if the text shown states it; otherwise `null` |
| `effect_detail` | A few words on the size of the effect. Not scored |

**Not a finding:** background ("previous studies showed..."), someone else's
result, an aim or a method ("we investigated..."), a hypothesis or the
authors' speculation ("may", "could suggest"), and anything in a review.
**A negative result is a finding**: "did not reduce" is `no_effect`, and
leaving it out is exactly the mistake the gold set is there to catch. Label
only what the sheet shows: a part the pipeline does not read cannot be
extracted.

**No finding expected.** When a paper should yield nothing - every review, a
paper that is not really about the subject, a paper whose shown text reports
no result of its own - leave `"findings": []` and set `"none_expected":
true`. An empty entry without it is read as "not labelled yet", and scoring
refuses.

**Confirm the proposed kinds.** An entry with `"kind_confirmed": false` was
proposed by code (`kind_proposed_because` says from what). While reading,
decide whether it is right; if not, change `kind` to what the paper is. Then
set `"kind_confirmed": true`.

**Held-back papers.** Five entries say `"holdout": true`, and their sheets
say "Held back". Label them exactly like the others and do not change the
flag. Whoever tunes prompts must not open those sheets or entries until the
end; see below.

When every entry is done, delete the `_comment` line and put the entries in
`evals/litreview/goldset.jsonl` (replacing the `EXAMPLE-` lines), or point
`run_extract --goldset` at the file. `run_extract` refuses a set with a
placeholder, an unconfirmed kind, an empty entry without `none_expected`, a
review with findings, a quote that is not in the paper, or any value outside
the schema, and names the line, the paper and the field.

### Hold five back

Mark 5 papers `"holdout": true` and do not look at them while tuning. Without
a holdout you will write a prompt that is excellent at fifteen specific papers
and mediocre at everything else, and you will not find out until a client demo.
The sheet generator marks one each from trials, animal, in vitro, negative
and review papers.

### Running it

```
python -m evals.litreview.run_extract [--goldset PATH] [--models laptop]
python -m evals.litreview.run_extract --holdout      # once, at the end
```

Each paper goes through Stage 5's own `extract_paper`, through the scheduler
after preflight. Expected and extracted findings are paired by the sentence
they quote, so a mouse study extracted as a human trial shows up as an
`evidence_type` error rather than as a miss.

Read the per-field table, not the headline score. "0.84 overall" is not
actionable; "evidence_type 0.78" tells you the model is confusing in vitro with
in vivo, which is a ten-minute prompt fix you can re-measure immediately.

### The loop

1. Run the eval
2. Open the failures
3. Find the **pattern** — one odd paper is noise, five failing the same way is a prompt fix
4. Fix the prompt, or add an example of that exact mistake done correctly
5. Re-run

Expect roughly 60% to 85% over four to six rounds. Each round takes minutes.

### Targets before Stage 5 is done

- Quote validity **1.00**. Not 0.98
- Per-field accuracy above **0.85** on the held-back papers
- A negative-result paper produces `no_effect` units, not silence
- A review article produces zero evidence units
