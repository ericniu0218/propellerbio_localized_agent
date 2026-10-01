# Localized Air-Gapped LLM for Literature Review

> **Moved from Localized-Airgap-LLM-for-literature-review at `dc5c0123d8e1`.** In
> bedrock, `litreview` is `bedrock-litreview` (or `/review` in the app),
> `--profile` is `--models`, `config/*.yaml` is
> `src/bedrock/litreview/defaults/*.yaml`, `evals/` is `evals/litreview/`,
> and the HTTP API is `bedrock.litreview.api` (install the `api` extra).
> See `docs/prompts/LITREVIEW_INTEGRATION.md` §3.
>
> This document is the pipeline's own README: what it does, why it is a
> pipeline rather than an agent, and what each stage owes the next. All of that
> is unchanged by the move. The rules for working on it are in
> `src/bedrock/litreview/CLAUDE.md`.

> **Open issues (September 2026):** what the deploy-readiness audit found and
> did not fix — report-accuracy gaps, completeness, retraction freshness,
> deployment and measurement — is listed in [OPEN_ISSUES.md](OPEN_ISSUES.md).
> The integration check and what the move gained are in
> [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md).

A local AI system that researches a drug target and writes a cited report —
running entirely on Propeller Bio's own in-office GPU, so no data ever leaves
their network.

**Client:** [Propeller Bio](https://propeller.bio/) — Bay Area biotech developing
biologic therapeutics (antibodies, peptides, siRNAs) that target multiple
disease-driving pathways at once for synergistic effects. Focus areas:
immunology & inflammation (I&I), obesity, and CNS disease.

**Built by:** Bedrock Technical Consulting, 4 engineers.

---

## Contents

- [What it does](#what-it-does)
- [Why it has to be local](#why-it-has-to-be-local)
- [The core decision: pipeline, not free-form agent](#the-core-decision-pipeline-not-free-form-agent)
- [The data model](#the-data-model)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [Models](#models)
- [Fixing the client's three complaints](#fixing-the-clients-three-complaints)
- [The paper searcher](#the-paper-searcher)
- [Paywalls, and what we will not do](#paywalls-and-what-we-will-not-do)
- [Patents](#patents)
- [Repository layout](#repository-layout)
- [Current state](#current-state)
- [Getting started](#getting-started)
- [The team](#the-team)
- [Timeline](#timeline)
- [How we measure it](#how-we-measure-it)
- [Results](#results)
- [Next steps](#next-steps)
- [Open questions for the client](#open-questions-for-the-client)
- [Risks](#risks)
- [A note on hallucination](#a-note-on-hallucination)

---

## What it does

A scientist types a question:

```
If I want to prevent psoriasis, what areas should I target?
```

The paper searcher ([`sources/literature.py`](../../src/bedrock/litreview/sources/literature.py))
collects candidate papers from Europe PMC, bioRxiv/medRxiv and PubMed, with
open-access full text. This system then:

1. **reads the question** — what it is about, exactly, and which near-misses
   do not count (psoriatic arthritis is not psoriasis; hMPV is not RSV)
2. **ranks every collected paper** by how directly it answers that exact question
3. **plans the evidence** — which information the answer needs, from how many
   papers, and which ones — and says where the papers cannot supply it
4. **reads the planned papers**, extracting findings with verbatim quotes
   checked against the source
5. **writes a report** in which every sentence cites the findings it rests on,
   and code removes any sentence whose numbers or names are not in its citations

The report is one file the scientist can act on: the direct answer, the single
most relevant source, one section per candidate target, the human and animal
evidence separately, negative and conflicting results, safety, and — stated
plainly rather than hidden — what the collected papers could **not** supply.
Under it sit the full ranking of every paper with the reason for its place, an
evidence table of every finding with its verbatim quote, and the list of papers
excluded for being about something else. It is written to `report.md`,
`report.html` (self-contained; hover a citation to see its quote) and
`report.json`.

Today a junior scientist does this by hand over several days and turns it into a
PowerPoint. We are automating the searching and reading, and leaving the
judgement to them.

On the two sample questions this beats both chatbot baselines on the same local
model and the same papers — **90.0 / 90.6 out of 100**, against 69.1 / 70.3 for
a chatbot asked the question directly. See [Results](#results).

Because Propeller develops drugs that hit two pathways at once, **pairwise
questions are supported** — "is there evidence for blocking target A *and*
target B in disease Y?" — through `combo_with` on every finding.

### What the client actually asked for

From our 15-question intake:

- **Sources they use:** Google, Google Scholar, PubMed, Europe PMC, bioRxiv,
  and patent databases (Espacenet, Lens, WIPO, Google Patents). Also paid
  databases (Tabs, GlobalData), press releases, investor decks, SEC filings.
  Any source is acceptable *as long as it is cited*; humans judge reliability.
- **Their current workflow:** junior scientists start with a Google or chatbot
  search, read papers, and build a PPT — not a written report.
- **What the deck covers:** the target's biological mechanism, the disease's
  unmet need, competitor landscape, supporting evidence (in vitro, in vivo,
  human clinical), strengths and weaknesses, reference links.
- **The key information they want:** evidence that modulating the target —
  blocking or activating it — improves the disease indication.
- **Report elements desired:** background, summary/conclusion, source
  literature, confidence in the conclusion, caveats. Users should be able to
  pick which sections to include.
- **Papers:** usually under 20 pages, with text, images and tables. Many are
  paywalled. **Patents are much harder for AI than papers.**
- **Knowledge storage today:** old PPTs on company servers, unstandardised,
  not formally shared.
- **No gold-standard report exists.** They will tweak once we start.

---

## Why it has to be local

Propeller's research is their entire value. Unpublished target work, competitive
intelligence, and internal assessments of what to pursue are exactly the things
that cannot be pasted into a chatbot run by someone else. A cloud AI service
reads whatever you send it, on their hardware, under their terms.

So the model comes to the data instead. Open-weight models — where the model file
itself is published and you may run it yourself — make this possible. The whole
system runs on a GPU in Propeller's office. Nothing leaves.

Everything awkward about this project follows from that one constraint: we
cannot use any cloud AI API, we must build the surrounding software ourselves,
and we get no vendor support once it is installed.

---

## The core decision: pipeline, not free-form agent

There are two ways to build this.

**Free-form agent.** Tell the model to "research IL-23" and let it decide each
next step itself — which search to run, what to read, when it is finished.

**Pipeline.** We write the steps in ordinary Python. At each stage we hand the
model one narrow job and take the answer back. **The model never decides what
happens next.**

This repository is the pipeline. Three reasons:

**1. Their three complaints about existing chatbots are all failure modes of
free-form agent loops.** Inconsistent answers, lost context, and collapse on
complex questions are what agent loops do when the planning model is not strong
enough. Removing the loop removes the failure mode.

**2. A local ~30B model is weaker at long multi-step planning** than the
frontier models those complaints were formed against. A free-form loop here
would be *worse* than what they already have, not better.

**3. Their process is identical every time.** Every deck has the same sections.
There is nothing to figure out, so we do not pay a model to figure it out.

### Where agent-style behaviour *is* appropriate

The follow-up chat at the end (Stage 9). By then the evidence is frozen and the
tools are narrow — `search_evidence`, `get_paper`, `compare_targets`,
`list_conflicts` — all scoped to one project's saved data. The model cannot
wander because there is nowhere to wander to.

So: **rigid pipeline for the research, flexible chat on top of the results.**

---

## The data model

See [`src/bedrock/litreview/models.py`](../../src/bedrock/litreview/models.py). Everything is built around one
record:

```python
EvidenceUnit
  target          IL23A              normalized gene symbol
  direction       block | activate | unclear
  modality        antibody | peptide | siRNA | small_molecule | vaccine | genetic | other
  intervention    guselkumab         the agent used, as written
  disease         psoriasis
  evidence_type   in_vitro | in_vivo | human_clinical | human_genetic
  species         human | mouse | cyno | cell_line | ...
  outcome         improved | no_effect | worsened | mixed | association
  setting         prevention | treatment | unclear
  effect_detail   "PASI-75 in 78% vs 8% placebo at wk12"
  n               507
  facet           clinical           which part of the answer it supplies
  quote           the paper's own sentence, at quote_start:quote_end of its section
  quote_status    exact | fuzzy
  provenance      where each model-written field was found: quote | chunk | methods | none
  context_quote   the paper's sentence naming the subjects / sample size, if not the quote
  entity_match    exact | paper_context
  combo_with      IL17A | null       for multi-pathway claims
```

Around it: `QuestionSpec` (what was asked, with exact terms and near-misses),
`RankedPaper` (a paper's place in the ranking, with its reasons),
`EvidencePlan` (which information, from which papers) and `Report` (sentences,
each with the evidence ids it cites).

The report, the slide deck, and the chat are all **views of a table of these**.
Build the table well and the rest is rendering.

### Rules enforced in code

**1. Every unit carries a verbatim quote, and code verifies it.**
The model must copy the exact sentence it took the finding from. Then
[`extract/verify.py`](../../src/bedrock/litreview/extract/verify.py) string-matches that sentence
against the source text. No match, the unit is destroyed. A fuzzy match may
differ only in typography: a changed number, negation or other meaning word
fails it. The unit stores the paper's own sentence and its position, not the
model's copy, and every other field the model filled in (target, agent,
subjects, sample size, result) must be found in the text it read or is blanked
([`extract/provenance.py`](../../src/bedrock/litreview/extract/provenance.py)).

This is how you stop a model inventing things: you do not ask it to be honest,
you check. It is plain string matching that anyone can read and nobody has to
trust.

**2. No model ever rates paper quality.**
The client told us they do not trust AI judgement here and will check sources
themselves. So code looks up objective facts — journal, citation count, preprint
status, human vs animal, sample size, retraction status — and displays them. The
scientist decides what they mean.

**3. `combo_with` exists from day one.**
Propeller's whole thesis is hitting two pathways at once. Retrofitting pairwise
support later would mean reworking every stage.

**4. Exact means exact.**
A question about RSV is answered from RSV papers — not human metapneumovirus
papers that look close enough. Exact terms and near-misses come from a table
in code, never from the model, and are shown at the top of every report.

**5. Every sentence of a report is checked against what it cites.**
The writing model can still invent a number or name a drug from memory while
citing real evidence. [`report/guard.py`](../../src/bedrock/litreview/report/guard.py) removes any
sentence whose numbers, names or subjects ("in mice") do not appear in the
paper text it cites. Then [`report/invariants.py`](../../src/bedrock/litreview/report/invariants.py)
re-checks the finished report against the exact paper versions the run pinned;
a violation puts a warning banner on the report. `bedrock-litreview verify-run <id>`
repeats the check on any stored run. It also says whether today's code,
knowledge tables, word lists and prompts are the ones the run recorded
(`report/verification_context.py`). A changed environment is reported beside
a passing report, never counted as a violation, and
`verify-run --require-same-rules` turns it into a failure for CI. It is a
comparison of recorded hashes, not a signed attestation: someone able to
rewrite the local database could rewrite both.

### Why raw and normalised are separate fields

`ExtractedFinding` is what the model returns. `EvidenceUnit` is what we store.
The model's job is to **copy** ("C57BL/6 mice"); code's job is to **translate**
("mouse").

Keeping those separate is deliberate. Copying is something a small model does
reliably. Translating is something code does *deterministically* — and
determinism is a feature the client asked for by name.

---

## The pipeline, stage by stage

Python owns the control flow ([`runner.py`](../../src/bedrock/litreview/runner.py)); every stage
writes its output to SQLite before the next begins, so a crashed run resumes
where it stopped. The model is called only for narrow jobs.

| # | Stage | Who | What it owes the next stage |
|---|---|---|---|
| 0 | [question](../../src/bedrock/litreview/stages/s0_question.py) | model copies, code resolves | `QuestionSpec`: type, goal, each entity with its exact terms and near-misses |
| 1 | [collect](../../src/bedrock/litreview/stages/s1_collect.py) | the searcher, then code | papers with sections, reviews and retractions flagged, duplicates removed; frozen per question |
| 2 | [gate](../../src/bedrock/litreview/stages/s2_gate.py) | code | pass / near-miss / absent / review / retracted, with the reason |
| 3 | [rank](../../src/bedrock/litreview/stages/s3_rank.py) | code + small model | every paper ranked; top ones graded 0–3 by a written rule from the model's answers, with a verified justification sentence |
| 4 | [plan](../../src/bedrock/litreview/stages/s4_plan.py) | code | facets the answer needs, the papers that supply each, gaps, extraction jobs |
| 5 | [extract](../../src/bedrock/litreview/stages/s5_extract.py) | model copies, code checks | verified, normalised `EvidenceUnit`s |
| 6 | [score](../../src/bedrock/litreview/stages/s6_score.py) | code, never a model | short ids, per-target summaries, conflicts, facet slices |
| 7 | [write](../../src/bedrock/litreview/stages/s7_write.py) | model per section, code guard | `Report`: sentences with citations; generated sections |
| 8 | [render](../../src/bedrock/litreview/stages/s8_render.py) | code | `report.json`, `report.md`, self-contained `report.html` |
| 9 | chat | not built | tools over the frozen evidence |

### Stage 0 — Read the question

The model copies entity names out of the question; code verifies each one is
really in the question (the fabrication guard, applied to the question), then
resolves it against [`knowledge/entities.yaml`](../../src/bedrock/litreview/knowledge/entities.yaml):
synonyms that mean exactly this entity (RSV = hRSV = human orthopneumovirus)
and near-misses that do not (hMPV, bovine RSV). Names the table does not know
are matched verbatim and nothing else. The model never adds a search term.

### Stage 2 — The exact-entity gate · *pure code*

A paper counts only if it names the entity by an exact term in its title or
abstract, or repeatedly in its full text. Matching respects boundaries —
"SARS-CoV" does not match inside "SARS-CoV-2", "IL-2" not inside "IL-23" — and
the longest match wins, so "psoriatic arthritis" swallows "psoriatic". A
paper whose title is about a sibling, with the sibling outnumbering the
subject, is excluded as a near-miss. Reviews are excluded: a review citing a
trial is not a trial. Every excluded paper is listed with its reason.

Retracted papers are excluded too. A paper counts as retracted when its own
record says so (a "Retracted Publication" type, or a "Retraction in" link),
or when this machine has *recorded* a retraction notice for it.

- **When notices are recorded.** By default, each new review first asks PubMed
  and Crossref about its papers. It sends ID numbers only, never the question,
  and skips papers checked in the last week. Scientists can make this manual
  with `/reviews settings`; the app explains both choices. They can also be
  recorded by `bedrock-litreview integrity refresh` or by importing an
  offline bundle.
- **Expressions of concern.** The paper is kept and shown, never recommended.
  Its findings are ranked after all other evidence and left out of the
  Answer, and the Answer says which targets were left out for that reason.
  They are marked ⚠ in the evidence table, and the reference quotes the
  notice's own title and opening words with a link.
- **Corrections** are marked only.
- **Reinstated papers.** A paper that was retracted and later reinstated
  stays out until a scientist adds it back (`/reviews reinstate <paper>`); the
  report tells them.
- **History is never rewritten.** What the run knew is pinned in its
  `publication_integrity_snapshot.v1`. A notice recorded later never rewrites
  an old report: `verify-run` reports it as an advisory.

See [`publication_integrity.py`](../../src/bedrock/litreview/publication_integrity.py).

Review articles are not evidence, but they are not thrown away. A separate
report section, "Background from reviews", lists them with links and a
couple of sentences quoted from each abstract. Nothing in it counts toward
the targets, the evidence table or the Answer.

### Stage 3 — Rank · *code, plus one small-model call per paper*

BM25 over title and abstract (the question's words plus the vocabulary of its
type and goal) picks which papers get a model grade. The grade measures how
directly each paper's own results answer the question — **relevance, never
quality**. The model is not asked for a number: it answers narrow questions
about the abstract (is the subject what the paper studies, or only a risk
factor for something else? is it a review? which molecule, pathway or drug did
the paper's own experiments change?), quotes a sentence, and code applies a
written rule:

| Grade | Rule |
|---|---|
| 3 | its own experiments changed a named molecule, pathway or drug (name found in the abstract by code) and measured the effect |
| 2 | it studies the subject with its own data, but changed nothing |
| 1 | the subject is a risk factor or a passing mention, the paper is a review, or the quote does not verify |
| 0 | not about the exact entity |

Which information needs a paper can supply is assigned by code from the
abstract's study-design words, not by the model — a small model asked for
either a grade or a list of needs gives every paper the maximum. The top
usable paper is reported as the *most relevant source*.

Which papers get a grade is `rank.selection` in `pipeline.yaml`. The default,
`lexical`, grades the top `max_graded` by BM25. The opt-in `evidence_aware`
policy keeps the same budget but first reserves lanes for papers whose own
publication type or design wording says human trial, human cohort or
genetics, prevention (prevention questions only) and animal or cell
experiment ([`study_design.py`](../../src/bedrock/litreview/study_design.py)).
Reviews and notices get no grade. It is not the default until a matched
real-model benchmark shows it helps: in a synthetic check it read more human
studies but named fewer reference targets on the asthma corpus.

### Stage 4 — Plan · *pure code*

Facet templates per question type ([`facets.py`](../../src/bedrock/litreview/facets.py)) say what
the answer needs — for "what should I target to prevent X": disease biology,
targets with causal evidence, human clinical evidence, prevention-specific
evidence, negative results, safety. The planner fills every facet's minimum
before topping any up, within a paper budget, preferring full text among
equally relevant papers. A facet no paper supplies is a **gap**, stated in
the report, never filled from elsewhere. The report's gaps and "What the
answer needed" count papers that gave *verified findings* for a need. A
paper chosen because its abstract looked right does not count until one of
its findings survives verification.

Every run also writes a coverage ledger (`coverage.v1`) beside the report:
the stage at which each collected paper stopped, and why, with counts that
reconcile. It is how "the report does not mention X" is told apart from "no
paper we had supports X". See [`coverage.py`](../../src/bedrock/litreview/coverage.py).

### Stage 5 — Extract · *model copies, code checks*

```
model extracts  →  code verifies quotes  →  code checks the entity
                →  code checks direction and outcome  →  code normalises  →  code dedupes
```

One chunk of one section per call, with the Methods paragraphs that describe
subjects passed as context. The finding's own field (disease, drug, target)
is checked against the question's entity: a finding about atopic dermatitis
is dropped even when its quote mentions psoriasis. The model also copies the
words saying what was done to the target ("deficiency") and what happened to
the disease ("aggravated"); code classifies those words and overrules the
model's `direction` / `outcome` labels when they unambiguously disagree, and
records every overrule on the finding. A mouse result filed as clinical
evidence is refiled by code. IDs are content hashes, so the same run produces
the same bytes.

### Stage 6 — Score and group · *pure code, no model, ever*

Targets are ordered by a written rule — human clinical evidence of benefit,
then human genetics, then animal evidence of benefit, then independent papers
— which counts kinds of evidence and never judges a paper. Disagreements are
surfaced as conflicts.

### Stage 7 — Write · *model, one call per section, under the citation guard*

Each section gets its own slice of evidence. The model lists the evidence ids
a sentence relies on *before* writing it (guided decoding restricts the ids to
that slice), and [`report/guard.py`](../../src/bedrock/litreview/report/guard.py) then checks every
sentence: a number or a drug/target/disease name that is not in the cited
evidence removes the sentence. The layout lives in
[`src/bedrock/litreview/defaults/report.yaml`](../../src/bedrock/litreview/defaults/report.yaml), so the client's format is a config change.

The reading view starts with the Answer, Evidence at a glance and How the
studies line up. The study profiles and evidence basis come from verified
fields and the papers' own records, by a written rule in
[`report/synthesis.py`](../../src/bedrock/litreview/report/synthesis.py).
Publications from the same trial count as one independent study.

**Where the studies disagree** follows the synthesis when there is a
disagreement: each entry names the synthesis line or target conflict it
explains, then shows the studies' design, trial name, subjects and sample
size, what was read, reported direction, and whole verbatim quotes with
their qualifying sentences. It covers disagreement within human or
preclinical studies and target conflicts across systems. Code displays the
records; the scientist decides what explains the difference. A dedicated
integrity check binds every displayed quote to evidence from its cited
paper. When the studies agree, the section has no heading.

The same comparison rows supply a gaps bullet naming every measure (or
target for target identification) with no tested effect in humans among the
papers read, with references. This includes animal or cell evidence and
associations only. It also appears among the Answer slide's key limitations.
Other sections generated by code include Key results in context,
Abbreviations, most relevant source, gaps and caveats, the evidence plan,
the relevance ranking, the evidence table, the method, and excluded papers.
References follow the reading view; the evidence table and audit trail are
in the Appendix. These additions make no model calls and add no fields to
stored report or evidence models; existing runs keep their recorded bytes.

#### Optional: the entailment check (off by default, not yet checked by a scientist)

The guard's direction and negation checks are word-level: they cannot tell
whether a reworded sentence means what its quote means
([OPEN_ISSUES.md](OPEN_ISSUES.md) #5).
[`report/entailment.py`](../../src/bedrock/litreview/report/entailment.py)
adds one narrow question for the local model, asked of every sentence the
guard kept: does the quote it cites support it fully, partially, not at all,
or say the opposite? The model is shown each cited quote apart from the
sentences shown beside it (context, next sentence, naming sentence), each
labelled with its role, and told that the sentence's main claim must be in a
quote: the others may add a name, who was studied, a sample size or a limit.
Prompt version 1 showed all four as one passage, so a sentence copied from a
finding's next or naming sentence while citing a quote that says something
else ([OPEN_ISSUES.md](OPEN_ISSUES.md) #3) was "supported"; guard check 10
catches that shape by counting words, and this is the meaning-level look
behind it. `not_supported` and `contradicted` drop the sentence ("the
entailment check found the cited text does not support it: ..."); `partial`
keeps it and is counted in the report's stats; a failed call keeps it. Like
every other call it is structured, at temperature 0, cached, through the
scheduler at `USER_JOB`, on the model pinned for the `entail` job in
`models.yaml`, with its own prompt (`llm/prompts/entail.md`). A run with it
on records `entail` in its model and prompt versions.

```toml
# ~/.config/bedrock/config.toml
[litreview]
entailment_check = true        # or BEDROCK_LITREVIEW_ENTAILMENT_CHECK=1
```

**Leave it off.** A check that drops a sentence the paper really supports
loses real evidence. Two tools measure it, and
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) has their numbers:

- `python -m evals.litreview.entailment_audit --models <set> [--rows OUT.jsonl]
  [--before EARLIER.jsonl] [--sheet OUT.md]` compares it with the hand audits
  (105 graded sentences, every OPEN_ISSUES #3 case among them). It needs only
  this repository and a model. `--before` lists every sentence whose verdict
  changed since an earlier run; `--sheet` writes what the check would drop,
  for a scientist to mark right or wrong.
- `python -m evals.litreview.entailment_agreement <report.json> ... --models
  <set>` compares it with the strong judge on stored benchmark reports. The
  judge sees a quote and the sentences beside it as one passage, so it cannot
  see the #3 class; that is what the hand audit is for.

Measured on the laptop model (qwen3:8b, 30 September 2026) it dropped none of
the hand audit's 93 faithful sentences and 2 of its 12 unfaithful ones, and
none of the five #3 cases: safe on that sample, but it adds little there. It
stays off until a scientist has marked the sheet and it has been measured on
the office model. With it off, a report is byte-identical to one written
before the check existed.

### Stage 8 — Render

JSON → Markdown → self-contained HTML (no external assets; hovering a citation
shows the exact sentence it rests on; a table of contents and links from each
claim to its evidence and from each reference to the sentences that cite it,
with no scripts).

**One status per finding.** Before anything is written, `report/claims.assess`
decides how each verified finding may be used, and the report stores it
(`Report.evidence_status`): **counted** (the requested treatment's own result),
**context** (true, but not that: a subgroup comparison, arms that all got the
treatment, a disease model, another drug) or **excluded** (not a usable result:
it describes the study, is background or the authors' view, was damaged when
the paper was read, points at an unread table, refers to "these markers"
without naming them, or prints an impossible statistic). A label the paper's
own design contradicts (a mouse finding labelled human) is corrected, not
dropped. The writer never sees an excluded finding; the Answer is written only
from counted ones; the synthesis, Key results, glance table and the deck count
only what is counted; the evidence table keeps every finding with its status
and reasons.

The report is structured data, so **the PowerPoint deck is a renderer, not a rewrite**: [report/slides.py](../../src/bedrock/litreview/report/slides.py) renders a finished report into presentation slides drawn in ArcDeck format (layouts, visual elements, native charts) with python-pptx, following Propeller's target-proposal template: single-target, bispecific when the question names two targets, or disease-level with an overview plus one slide per candidate target. Slide content comes from verified sentences with their [n], code-generated lines (counts, comparisons), paper captions and quotes. A slide shows at most 6 of its sentences (4 beside a chart); the rest go to "Appendix: further findings" slides, and text that does not fit a page continues on a "(cont.)" slide, never shrunk. Every page footer lists its references with links. Paper figures are chosen by code from the paper's own xref in the paragraph where a verified quote came from; only open-access papers with a license; fetched only during runs with both `deck` and `search` enabled, at most 8 papers and 2 figures each, cached from www.ebi.ac.uk. Competitive landscape and SWOT are assembled by written rule and marked "Assembled by rule from the report - review before presenting"; company, stage and unmet need say "fill in".

Every report also downloads as a **PDF** ([report/pdf.py](../../src/bedrock/litreview/report/pdf.py)): laid out by code from the report's own Markdown, so it carries the same sentences, citations, tables, quotes and linked references, in a font shipped with the package (DejaVu Sans) because the server is air-gapped and reports use ≥ ± · − η² β. The deck downloads as a PDF too, when LibreOffice is installed on the server ([report/convert.py](../../src/bedrock/litreview/report/convert.py) converts the real .pptx, so the two cannot differ). In the Research tab the report's section list is a collapsed "Contents" panel, and the report can be read full screen or in its own tab.

A Research-tab checkbox "Also make a target-proposal PowerPoint" sets `deck` on the review; without it, only the text report is written. `bedrock-litreview ask --deck` also makes the deck. The download button (`/research/runs/{id}/report?format=pptx`) is hidden for reviews without a deck but available for older runs that predate the option. Layouts and text rules are configurable at `~/.config/bedrock/litreview/deck.yaml`, defaulting to `src/bedrock/litreview/defaults/deck.yaml`.

### Stage 9 — Chat

Not built. Agent-style, and safe when built: evidence is frozen, tools are
narrow and question-scoped.

## Models

**Two model files. Seven jobs. Zero training.**

| Job | Tier | What the model is asked for |
|---|---|---|
| `question` | medium | copy the entity names out of the question — code resolves them |
| `relevance` | **small** | narrow questions about one abstract — code grades it |
| `extract` | medium | copy findings out of one chunk of one section, with their quotes |
| `write` | medium | one report section from one slice of evidence |
| `entail` | medium | *optional, off by default* — does a kept sentence's cited text support it |
| `judge` | medium | *benchmark only* — find the evidence sentence for a claim |
| `baseline` | medium | *benchmark only* — the chatbot answers we compare against |

The split is by **cost, not skill**. Relevance runs once per collected paper so
it gets the cheap model; everything else runs tens of times per question.

Notice what is *not* in that table: nothing asks the model for a score, a
ranking, a plan, or a synonym. Those are code's, because a lookup table returns
the same answer every time and a 4B model does not — see
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for what happened when we asked it
for a grade.

- **small** — 7–8B instruct
- **medium** — Qwen3-30B-A3B-Instruct class. Mixture-of-experts: ~30B total but
  only ~3B active per token, so it follows instructions like a big model and
  runs like a small one, with long context for whole sections
- **embeddings** — a small (~300M) model. Genuinely separate, because it does
  different maths. **Pick it once** — changing it invalidates every stored
  vector and forces a full re-index
- Served with **vLLM**

All model names live in [`src/bedrock/litreview/defaults/models.yaml`](../../src/bedrock/litreview/defaults/models.yaml) and nowhere
else, so swapping one is a one-line change.

### Why we are not fine-tuning

Specialisation comes from the **instructions**, not the weights. One model with
five well-built prompts is five specialists.

1. **No training data.** Fine-tuning extraction would need ~2,000 hand-labelled
   findings. We have zero, and producing them is months of a scientist's time.
2. **The target moves.** The schema will change weekly in weeks 5–8 as
   scientists react to real output. A prompt edit takes 30 seconds; a retrain
   takes hours and invalidates every prior result.
3. **You cannot debug it.** A bad prompt can be read. A bad weight cannot.
4. **Mistakes bake in permanently.** A habit learned in fine-tuning lives in the
   weights; you cannot reach in and remove it.

**What actually makes each job good:** a narrow prompt · a forced output shape ·
3–5 worked examples in the prompt · a checker in code.

**Revisit after months of real use, and only for job 2 (screening).** By then,
scientists' corrections become free training data. Note the order — you use the
system to generate the data that makes training possible.

---

## Fixing the client's three complaints

These are the acceptance criteria that matter to them.

### "I ask the same question twice and get different answers"

- `temperature=0.0` everywhere, pinned model versions, fixed prompts
- **Every model call is cached**, keyed on the prompt, the model, the call's
  parameters and the client's *generation context*: seed, thinking policy,
  weights where the server reports them, declared revision, and which server.
  A re-run replays the cache. See [`llm/cache.py`](../../src/bedrock/litreview/llm/cache.py).
  Editing a prompt changes the key, so a prompt change invalidates exactly the
  entries it should and nothing else
- **What is and is not promised.** The pipeline's own choices are
  deterministic over pinned inputs. Replaying cached replies, or rendering
  from recorded ones, reproduces the report byte for byte. Whether the
  *model server* repeats itself with no cache is a property of one
  deployment: vLLM documents no default guarantee, and a seed alone does not
  make one. `python -m bedrock.litreview.bench determinism --mode fresh`
  measures it; it has not been run on the office server yet
- **The paper set is frozen.** Re-running a report uses the papers already
  gathered; going back out to the sources is a separate, explicit action
- Every run has a `run_id`; reports are immutable

> **The acceptance test: run the same project twice, get byte-identical
> reports.** One sentence the client can verify themselves.
>
> It holds while the inputs hold. If a retraction, expression of concern or
> correction is recorded on the machine between the two runs (`bedrock-litreview
> integrity refresh` or `import`), the second report differs on purpose - it
> marks or excludes that paper - and each run's
> `publication_integrity_snapshot.v1` shows what it knew.

### "It can't remember earlier conversation"

Not a memory problem — a storage problem. Chatbots hold everything in the
conversation and the conversation fills up. Everything here lives in the
project's database, and chat queries that database. Close the laptop for a
month; the project is untouched.

### "It breaks down on complex questions, just stops working"

The model is never asked a complex question. Python decomposes the work, and
every call is "classify this abstract" or "write two paragraphs from these nine
quotes."

---

## The paper searcher

[`src/bedrock/litreview/sources/literature.py`](../../src/bedrock/litreview/sources/literature.py) finds the
papers. It plugs in through the one seam every searcher uses,
[`sources/base.py`](../../src/bedrock/litreview/sources/base.py) — `search(SearchRequest) ->
list[Paper]` — and, like the rest of the system, it is code from end to end:

- **Queries come from the question's exact terms**, never from a model. Each
  required entity contributes an OR-group of the names the knowledge table
  accepts for it; groups are ANDed. One more query per question type and goal
  adds focus words from `src/bedrock/litreview/defaults/pipeline.yaml` (`search.focus`: knockout,
  blockade, trial, prophylaxis …) — added to the plain query, never instead of it.
- **Three sources**, in parallel: Europe PMC (the workhorse, with citation
  counts and open-access flags), a preprint lane through Europe PMC's `SRC:PPR`
  index (bioRxiv has no keyword search of its own), and PubMed E-utilities
  (publication types, PMCIDs). Every record keeps its publication types, so
  reviews and retractions are recognised downstream.
- **Merged in a fixed order**: duplicates collapse on DOI, then PMID, then
  normalised title, keeping Europe PMC's record and every query that found the
  paper; lists are interleaved in proportion to their length, so the same
  responses always give the same list, and cut to `max_results` (200).
- **Open-access full text** (Europe PMC JATS, the first rung of the
  acquisition ladder) for up to `search.full_text_max` papers — only where
  the record says it is open access, or PubMed does not say.
- **Loud failure.** One source down is logged and reported, and the others
  carry on. All sources down raises `SearchFailed` instead of returning
  nothing — an empty list would be frozen as the question's paper set and read
  as "no literature found".
- **Citation snowballing, off by default**
  ([`sources/snowball.py`](../../src/bedrock/litreview/sources/snowball.py),
  `search.snowball` in `pipeline.yaml`). With `seeds` above 0, the first
  searched papers that name every required entity (chosen by code, before any
  model sees a paper) have their references and citations followed through
  Europe PMC and PubMed's `elink`, on the same two hosts, sending only
  identifiers. Linked papers the search did not already keep are added after
  the searched ones, each saying how it was found ("snowball: cited by
  PMID…"), and pass the same gate as any other. The policy is part of the
  frozen set's search signature, so turning it on collects a question's
  papers again the next time a search is asked for. It stays off until a
  before-and-after on the benchmark corpora says what it does to target recall
  ([FRONTIER_LITREVIEW.md](FRONTIER_LITREVIEW.md) task D).

**More names for the search, approved by a person.** A paper that names a
target only by an alias the knowledge table lacks is never found and never
passes the gate. `bedrock-litreview aliases propose --hgnc
hgnc_complete_set.txt [--mesh descYYYY.xml] [--symbol TMEM135 ...] --out
proposals.yaml` reads files a person downloaded from HGNC and NLM (no network)
and writes proposed names, every one `approve: false`; code refuses, with the
reason, names another entity holds, near-misses, names HGNC gives another
gene, names of two characters or fewer and MeSH's inverted forms. Only a MeSH
heading's preferred concept is read, never its narrower concepts or its tree.
A person marks what they have checked and runs `bedrock-litreview aliases
approve proposals.yaml`, which edits `knowledge/entities.yaml` as text (its
comments survive), re-checks every refusal, and proves the reloaded table
changed only by the approved names. Review and commit the diff like any other
change to the table ([`knowledge/aliases.py`](../../src/bedrock/litreview/knowledge/aliases.py)).

**What leaves the network**: the search terms and paper identifiers, sent to
Europe PMC and NCBI, plus the contact address in `SOURCES_CONTACT_EMAIL` to
NCBI (use a generic mailbox). Requests identify only as `litreview`, never
by the client's name. Never papers, findings, reports or anything a model
wrote. The searcher runs only when asked — `bedrock-litreview search`, `ask
--search`, or `"search": true` over HTTP — never as a default. On an
air-gapped deployment run `bedrock-litreview search` on a machine that is allowed out
and carry the JSON-lines file across; `ask --corpus` reads it offline.

It was ported from a standalone retrieval agent, and some of that agent was
deliberately left behind: its **model-scored "credibility"** (no model scores
paper quality here, and that one was a cloud model), its **web, news, SEC and
Google Scholar** results (not papers; a snippet is not an abstract), and its
**patents** (a separate lane that is never evidence — see [Patents](#patents)).

## Paywalls, and what we will not do

Many key papers are paid. The client mentioned Sci-Hub.

**We will not integrate it.** It is illegal and Propeller is a real company with
real lawyers and real IP to protect.

But here is the thing: **Propeller already pays for journal access.** Their
scientists can legally download these papers right now.

So we build a **drop folder** ([`src/bedrock/litreview/dropfolder.py`](../../src/bedrock/litreview/dropfolder.py)).
A watched directory on the shared drive. A scientist downloads a PDF the normal
legal way, drops it in, and the pipeline hashes it, matches it to a DOI, and
ingests it like any other full text.

About 100 lines of code, and it probably doubles usable full-text coverage. We
never fetch anything we should not; we just stop throwing away the access they
already pay for.

**Build it in week 1.**

---

## Patents

The client told us patents are much harder for AI than papers. They are right —
long, dense legal language, odd structure. We do not fight that in v1.

- **EPO OPS** — family resolution and legal status. The only reliable source for
  INPADOC families, but XML and fiddly
- **Lens.org** — search across ~140M records, free for research use
- **PatentsView** — clean, structured US data

**Extract only title, abstract, assignee, priority date and claim 1.** Never the
full body. Patents feed the **competitor landscape section only** — *who else is
working on this* — never the biological evidence sections, and they are marked
as a distinct class in the UI.

---

## Repository layout

Inside bedrock, the package lives at `src/bedrock/litreview/`; the paths below
are relative to it.

```
models.py              ★ the data model. Read this first
config.py                settings; takes the server from bedrock, refuses non-local ones
service.py               preflight, run_review: the one way into a review that uses a model
db.py                    SQLite: questions, papers, runs, artifacts, reports, ratings
runner.py                orchestrator, resumable
api.py  cli.py           front doors: ask(), optional FastAPI app, `bedrock-litreview ask`
ingest.py                sections, review/retraction flags, dedupe
facets.py                what each question type needs (code, not model)
lexical.py               BM25
knowledge/
  entities.yaml        ★ exact terms and near-misses, per entity
  __init__.py            the matcher (boundaries, longest match, case)
llm/
  client.py              LLMClient protocol + the fake, strip_thinking, parse_json_reply
  scheduled.py         ★ SchedulerClient: the one live client, through bedrock's Scheduler
  cache.py             ★ the call cache (determinism)
  prompts/*.md           versioned prompts
stages/s0_question.py … s8_render.py   one file per stage, numeric order
report/
  guard.py             ★ the citation guard
  layout.py              reads defaults/report.yaml, or an override
extract/
  verify.py            ★ quote verification — the fabrication guard
  normalize.py  dedupe.py
parsing/sections.py      JATS, GROBID TEI and flat text → sections
sources/
  base.py              ★ the seam for the paper searcher
  literature.py          the paper searcher: query, merge, full text
  query.py               queries from exact terms, in code
  europepmc.py  pubmed.py  biorxiv.py   one client per source
  http.py                retries, backoff, rate limits, and the host allowlist
  corpus.py              JSON-lines handoff format
bench/                   the benchmark (see "How we measure it")
defaults/  models.yaml  pipeline.yaml  report.yaml
```

And elsewhere in bedrock:

```
src/bedrock/ui/review_command.py    /review and /reviews in the terminal app
tests/litreview/                    the ported tests; `pytest tests/litreview`
evals/litreview/                    questions/*.yaml  corpora/<id>/  results/  goldset.jsonl
                                    (a source checkout only — not in the wheel)
docs/litreview/                     this file, BENCHMARK_RESULTS.md, sources.yaml
scripts/litreview/judge_answers.py  the file-judge helper
```

## Current state

| Part | State |
|---|---|
| Question → report pipeline (stages 0–8) | **Built and tested.** End-to-end with a scripted fake model; no-model runs on the real sample corpora |
| Paper searcher | **Built and tested** (offline, hand-written fixtures through a mock transport) and run live: Europe PMC, preprints, PubMed, open-access full text. `bedrock-litreview search`, `ask --search` |
| Benchmark (`bedrock.litreview.bench`) | **Built and tested** with a fake model; sample corpora for two questions frozen in `evals/litreview/corpora/` |
| Benchmark results with a real model | **Run** on MedGemma 1.5 4B for both sample questions — [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) |
| In bedrock | **Built.** `/review` and `/reviews` in the terminal app, `bedrock-litreview` in a shell, the FastAPI app as the optional `api` extra. Every model call goes through bedrock's `Scheduler` at `USER_JOB` |
| Local audit trail | **Built.** `/review`, `bedrock-litreview ask` and the HTTP API record a `litreview.review` intent/outcome in bedrock's local history, plus one `litreview.search` per outside host a run is allowed to contact (the literature search, the automatic retraction check) - metadata only, never the question. `/audit` reads it back. See `bedrock.litreview.journal` |
| Alias import, citation snowballing | **Built and tested offline.** Aliases wait for a person's approval; snowballing is off until measured |
| Target-proposal deck (PPTX) | **Built.** Code-rendered from the report; Research tab checkbox and `ask --deck` toggle it. Layouts in `defaults/deck.yaml`, figures from www.ebi.ac.uk only during search runs |
| Stage 9 (chat), patents, drop folder, GROBID PDF rung | Not started |

`pytest tests/litreview` runs the pipeline's own tests with no GPU and no
network; `pytest` runs the whole of bedrock. Measured quality on two sample
questions is in [Results](#results); what to do next is in
[Next steps](#next-steps), and what was deliberately left out of the move is
`docs/prompts/LITREVIEW_INTEGRATION.md` §8.

## Getting started

The literature review installs with bedrock; there is nothing separate to set
up.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/litreview
```

No GPU needed for the tests. To search the literature for a question (needs
the network) and save what was found:

```bash
bedrock-litreview search "If I want to prevent psoriasis, what should I target?" --out papers.jsonl
```

To ask a question against a saved corpus — the searcher's, or a benchmark one:

```bash
bedrock-litreview ask "If I want to prevent psoriasis, what should I target?" \
    --corpus papers.jsonl --models laptop
```

Or both in one go on a machine that is allowed out: `bedrock-litreview ask "..." --search`.
A question's papers are frozen once collected; `--refresh` searches again.
`NCBI_API_KEY` and `SOURCES_CONTACT_EMAIL` in the environment are optional but
recommended (PubMed's rate limit, and NCBI's terms).

The same review runs inside the terminal app, beside the chat:

```
/review If I want to prevent psoriasis, what should I target? --corpus papers.jsonl
/reviews
/reviews verify <run-id>
/reviews check
/reviews forget <run-id>
```

**Before the first run with a model,** run `bedrock-litreview doctor`. It
checks, without sending a prompt, that the model server answers, accepts this
client's key, offers the models the review's model set names, gives each a
context window of at least 12,288 tokens, and can constrain output to a JSON
schema - and says what to change if not.

The window matters most. Extraction sends about 4,300 tokens of prompt and
allows a 4,096-token reply, and a server with a smaller window does not refuse
it: it drops the start of the prompt - the instructions - and the model answers
without them. **The Ollama app's default is 4,096 tokens on a 16 GB Mac**
(measured on 28 September 2026: a 14,000-token prompt was read as 2,050
tokens). Set `OLLAMA_CONTEXT_LENGTH=12288` where Ollama is started (or the
app's Settings -> Context length) and `context_window = 12288` in
`~/.config/bedrock/config.toml`. The review checks the window the server
reports (vLLM and the agent server always; Ollama once a model is loaded)
before it starts, and every call's reported prompt size as it runs: a
truncated prompt stops the run with the fix, rather than producing a report
built on answers to questions the model never saw.

`/reviews check` in the app runs the same check as `bedrock-litreview doctor`,
and both are the preflight a run does (`service.preflight`), so a check cannot
exist in one and not the other. Beside what stops a review, they show advice
that does not: a `request_timeout` too short for an extraction call, free disk
(a review keeps a model resident for hours, and on a 16 GB machine that has
filled a disk through swap), and whether a company server name needs
`allowed_hosts`. When the server lacks a model, the message says what it does
offer and which `models.yaml` to write the right names into.

`bedrock-litreview forget <run-id>` (`/reviews forget` in the app) deletes one
run: its record, its stage artifacts and its rendered report. Not reversible,
and deliberately narrow — the question, its frozen paper set and every stored
paper version stay, because `paper_versions` is append-only and is what lets
*other* runs be re-proved against the exact bytes they read. A report that a
second identical run also produced is kept for that run.

Where the model server is: bedrock's `api_base` (`BEDROCK_API_BASE`); in
agent-pack mode (`backend = "agentpack"`) the agent server's inference API at
`agent_pack_api_base` + `/v1`, with the `agentpack` model set (roles `runner`
and `specialist`) and the window each role reports; or `[litreview] api_base`
to point reviews somewhere else. Only the model *names* live here, in
`src/bedrock/litreview/defaults/models.yaml`. `--models office` names the vLLM
model set; `--models laptop` a local Ollama (`ollama pull qwen3:8b`);
`--models medgemma` MedGemma 1.5 4B (`ollama pull medgemma1.5:4b`), which is
what produced [the measured results](#results). `--no-model` runs the code-only
stages (gate, lexical ranking, plan) and writes the report skeleton.

From Python, with a client of your own: `bedrock.litreview.api.ask(question,
papers=..., client=...)`. Over HTTP:

```bash
pip install -e ".[api]"
uvicorn bedrock.litreview.api:create_app --factory --host 127.0.0.1
```

The HTTP app has no authentication yet, so keep it on localhost or the office
network. It runs its reviews through the same `service.run_review` and the same
scheduler as the terminal and the CLI.

---

## The team

Roles 1 → 2 → 3 follow a paper's journey from the internet to a scientist's
screen. Role 4 wraps around all three.

### 1. The Collector — finding and fetching papers
Owns every connection to the outside world. Everything downstream is blocked on
this, so it starts day one.

Builds: source connectors · alias import · citation snowballing · deduplication · the acquisition
ladder · GROBID parsing · **the drop folder**.

**Done when:** type in a target, get 200 deduplicated papers with full text
wherever legally available, plus a count of abstract-only ones.

### 2. The Reader — turning papers into findings
Owns the part that decides whether the product is any good.

Builds: screening · the `EvidenceUnit` schema · extraction · **the quote
checker** · the weekly accuracy tests.

**Done when:** 50 findings from 10 papers, hand-checked and correct. 100% of
quotes verified.

*Hardest job. One invented finding destroys trust in all of them.*

### 3. The Presenter — what the scientist sees
The only part anyone outside the team will look at.

Builds: the project setup form · the report · **the review screen** (click a
finding, see the sentence it came from) · the excluded-papers list · the PPTX deck.

**Done when:** a junior scientist reads a report start to finish without asking
what anything means.

*Should sit next to a real scientist in week 1 and watch them work by hand.*

### 4. The Foreman — running the line
Builds: the runner · the model-call cache · the frozen paper set · storage · run
versioning · restart logic (a crash at stage 6 must not discard stages 1–5) ·
the chat tools.

**Done when:** the same project run twice produces identical reports.

### Working in parallel

Nobody waits. Two data shapes, agreed in writing in week 1:

- **what a `Paper` looks like** (Collector → Reader)
- **what an `EvidenceUnit` looks like** (Reader → Presenter)

Both are in [`src/bedrock/litreview/models.py`](../../src/bedrock/litreview/models.py). Once fixed, everyone
builds against fake data and plugs in later — the Presenter can design the whole
report from twenty made-up findings.

---

## Timeline

| Weeks | Goal |
|---|---|
| 1–2 | **Walking skeleton.** One hardcoded target, Europe PMC only, no screening, ugly Markdown out. End-to-end and ugly beats one beautiful stage |
| 3–4 | All sources, dedupe, GROBID, drop folder. **Report the full-text coverage number** |
| **5–6** | **Screening and extraction. Build the gold set here** ← *we are here* |
| 7–8 | Scoring, sections, first report a scientist reviews properly. Expect the schema to change |
| 9–10 | Determinism, chat, patents |
| 11–12 | Polish, PPTX, handoff |

---

## How we measure it

Three layers, from cheapest to most expensive:

**1. Mechanical checks, on every run.** Quote validity (1.00, not 0.98), every
sentence cited, every number and name in its cited evidence (the guard), and
byte-identical reports for the same question over the same papers.

**2. The benchmark — no scientist in the loop.** `python -m bedrock.litreview.bench run
evals/litreview/questions/*.yaml --models laptop` runs the pipeline and two chatbot-style baselines on
the same local model and the same frozen papers — *closed book* (the question
alone) and *naive RAG* (top abstracts pasted in) — and scores all three:

| Measure | How |
|---|---|
| Accuracy | each claim judged against the evidence it cites (local model, entailment only) |
| Groundedness | each claim judged against the best-matching passages of the whole corpus — no support means outside information |
| Exactness | near-miss entities named in the answer; **distractor papers** (real abstracts with the entity swapped for a near-miss) must not reach the evidence |
| Completeness | facts ("nuggets") from published reviews, and Open Targets' clinically tested targets — counted only where the corpus supports them |
| Detail | supported claims carrying numbers; distinct sources |
| Benchmark validity | supported claims are corrupted (number, effect direction, entity) and the judge and guard must catch them |

Each run writes a scorecard (`evals/litreview/results/<id>/<time>/scorecard.md`) listing
the unsupported claims, missed facts and missed targets — the list to fix —
and appends to `evals/litreview/results/history.jsonl` to track scores across prompt
versions. Scientist ratings are stored beside benchmark scores (`ratings`
table) so the two can be compared once they exist.

**3. The gold set and the old PowerPoints.** Per-field extraction accuracy
against hand-labelled papers, and recall of the papers the client's own decks
cited. Neither exists yet; both need the client.

| Metric | Target |
|---|---|
| Quote validity | **1.00** |
| Claims supported by their citations | ≥ 0.85 |
| Distractor leak rate | **0** |
| Extraction precision (gold set) | > 0.85 |

## Results

First real-model run, 23 September 2026: **MedGemma 1.5 4B** (Google's medical
Gemma 3) through Ollama on a 16 GB laptop, no network. Two questions, two
chatbot baselines, same model and same papers for all three.

| Metric | Psoriasis | | | RSV in infants | | |
|---|---|---|---|---|---|---|
| | **Pipeline** | No papers | Pasted | **Pipeline** | No papers | Pasted |
| **Overall (0–100)** | **90.0** | 69.1 | 54.5 | **90.6** | 70.3 | 84.4 |
| Claims made | 58 | 19 | 22 | 53 | 15 | 13 |
| Claims supported by their citations | **1.00** | – | 0.45 | **0.98** | – | 0.77 |
| Claims supported by the collected papers | **1.00** | 0.79 | 0.93 | **1.00** | 0.73 | 1.00 |
| Review facts covered (vital) | **0.83** | 0.50 | 0.33 | **0.90** | 0.40 | 0.90 |
| Known drug targets named | **0.50** | 0.21 | 0.00 | – | – | – |
| Supported claims carrying a number | **22** | 0 | 0 | **25** | 1 | 2 |
| Distinct sources cited | **40** | 0 | 9 | **34** | 0 | 9 |

Every mechanical check passed on both questions: quote validity 1.00, every
sentence cited, every number in its cited evidence, **zero** claims naming a
near-miss entity, **zero** distractor papers leaked, and byte-identical reports
on re-run. The judge caught 100% of deliberately corrupted claims, so those
verdicts are worth something.

The same pipeline before this round of fixes scored **69.3** on psoriasis with
the identical judge — it was careful but incomplete, reading 6 papers where it
now reads 11 and extracting 37 findings where it now extracts 130.

The failures behind those numbers, each with the fix it produced, are in
**[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)** — the small model graded 39 of
40 papers 3/3, recorded a knockout that *worsened* disease as an improvement,
and left a prevention question's prevention section empty. Sample reports:
[`evals/samples/`](../../evals/litreview/samples/).

`pytest tests/litreview` → the pipeline's own tests, no GPU and no network.

## Next steps

**The to-do list is [BENCHMARK_RESULTS.md § Next steps, step by step](BENCHMARK_RESULTS.md#next-steps-step-by-step).**
It has 64 numbered steps, grouped in phases, from the current state to
production install, acceptance and handover. Each step names its owner and
dependencies. It is the only to-do list. This section only records what has
finished since the first real-model run, and where each earlier item went, so
the two lists cannot disagree.

The same file's [audit](BENCHMARK_RESULTS.md#audit--benchmark-findings-and-deployment-readiness)
explains why that order was chosen. It covers accuracy and completeness in the
latest benchmark, and what is missing for a deployment.

**Done**

- ~~Benchmark the paper searcher's corpora.~~ **Done.** See
  [searcher corpus vs frozen corpus](BENCHMARK_RESULTS.md#benchmark-results--searcher-corpus-vs-frozen-corpus-strong-judge).
  The searcher did not improve the reports. The audit traces most of that to
  the pipeline, not to the searcher: see steps 13 and 25.
- ~~Teach the citation guard about ranges and bounds.~~ **Done.** This is
  `bound_misuse` in [`report/guard.py`](../../src/bedrock/litreview/report/guard.py). On the
  frozen-corpus re-run it removed the RSV "upper range" sentence ("268.8 is a
  bound of the opposite end").
- ~~Raise `plan.max_papers`.~~ **Done.** It went from 12 to 16 in
  `src/bedrock/litreview/defaults/pipeline.yaml`, and the planner now spends any leftover budget on the
  best unread papers (`stages/s4_plan.py`). What is still open is which papers
  can be read at all: only 40 per question get a relevance grade (step 25).
- ~~Use a different, larger model for judging.~~ **The seam is done.**
  `--judge-file` (`src/bedrock/litreview/bench/filejudge.py`) lets another judge answer.
  The latest scores were judged by Claude Sonnet at development time. Still
  open: a judge that runs inside the air gap (step 37), and a scientist checking
  the judge's verdicts (step 23).

**Where the earlier items went**

- Office GPU spec, and a benchmark on the larger model: steps 1 and 31–37.
- The gold set (the file exists but holds only template rows): step 22.
- The client's report format and PPTX: steps 4 and 52.
- Growing the knowledge table: step 30.
- Stage 9 chat, patents and the drop folder: steps 48, 53 and 54.
- Scientist ratings: the `ratings` table and `POST /reports/{id}/ratings`
  exist, but no review screen does yet. The human-review fields on
  `EvidenceUnit` are never written. See step 51.

## Open questions for the client

1. **How much VRAM is on the office GPU?** Decides the model tier, which decides
   the quality ceiling. **Blocking — get this first.**
2. **5–10 old PowerPoints.** The only route to a gold standard.
3. **Which journal subscriptions do they have?** Determines what the drop folder
   buys us.
4. **Tabs Database and GlobalData — API access, or manual export only?**
   Probably manual; plan a CSV import path.
5. **Two real target/indication pairs to build against** — one single-target,
   one pairwise, so multi-pathway is exercised from the start.

---

## Risks

**The paywall ceiling is real.** If the pivotal papers for their targets sit
behind Cell Press and Nature paywalls, abstract-only extraction will look thin,
and no amount of engineering fixes it. The drop folder plus OA coverage is the
mitigation. **Measure the ratio in week 3 and tell them the number early** — do
not let them discover it at a demo.

**Figures are invisible to us.** GROBID does not read them, and in biology a
great deal of the result is a picture of a graph. Flag early.

**Verify OpenAlex pricing.** Usage-based pricing was introduced in early 2026.
The bulk snapshot is the fallback. Confirm before building a dependency on the
live API.

**Patents may stay hard.** We have scoped them to competitor landscape only. If
the client wants real patent claim analysis, that is a separate project.

---

## A note on hallucination

A model making things up is not a bug someone forgot to fix. It is how these
models work: they predict what *sounds* right, one token at a time. Usually that
overlaps with what is true. When it does not, you get a confident, fluent,
entirely invented finding.

That is why the quote-verification rule matters more than any other decision in
this repository.

We are not building a system that is never wrong. We are building one that
**shows its work** — so a scientist can check any claim in seconds, and catch
the wrong ones themselves.
