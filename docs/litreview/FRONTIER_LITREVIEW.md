# Frontier techniques for the literature review: triage and plan

Reviewed on 29 September 2026 against the code on `main` (`ca2f240`). The
catalogue further down came first: 21 techniques from published frontier
research systems (OpenAI Deep Research, Perplexity, Anthropic's multi-agent
research system, Stanford STORM and Co-STORM), mapped onto the `litreview`
pipeline. This top part is the decision: which of them the literature review
needs, and in what order to build them.

## The decision

Five of the 21 are worth building, most in a narrower form than proposed:

| # | Technique | What gets built | Task |
|---|---|---|---|
| 19 | Evaluation harness | Scores that say what they rest on, and a gold set labelled by a scientist | A |
| 12 | Discrepancy matrix | A code-written "Where the studies disagree" section | B |
| 9 | Entailment guard | Measure the check that already exists, and narrow what it reads | C |
| 1 | Term expansion | Synonym import that a person approves; never a walk down an ontology tree | D |
| 2 | Citation snowballing | Citing and cited papers, fetched from the two hosts already approved | D |

The other sixteen:

- **Already built (5):** #3 span-level reading, #6 evidence hierarchy, #7
  retraction checks, #16 evidence table, #18 gaps section. Task B adds one
  bullet to #18.
- **Already planned (1):** #8 neural re-ranking is the existing
  semantic-retrieval item, blocked on a deployed embedding model (task E).
- **Parked (4):** #4 ClinicalTrials.gov, #5 tables, #11 subgroups, #20 DSPy.
  Each waits on a decision or a prerequisite, listed in the plan.
- **Rejected (6):** #10 effect sizes, #13 risk-of-bias ratings, #14 shared
  mind map, #15 agent swarm, #17 streaming badges, #21 DPO. Each item below
  says why; most would break a rule in `src/bedrock/litreview/CLAUDE.md`.

**Why not all at once.** Accuracy changes interact. The 26 September fixes
created a new failure class (OPEN_ISSUES #3), and the benchmark found bugs that
every unit test had missed. When several changes land together, nobody can
tell which one moved a score, and until task A is done a score cannot be
trusted to have moved at all.

## Implementation plan

### First: tasks A and B, in parallel

| Task | Items | Prompt | Why first |
|---|---|---|---|
| **A. Measurement you can trust** | #19; OPEN_ISSUES 21, 23, 25–28 | [LITREVIEW_AGENT_A.md](../prompts/LITREVIEW_AGENT_A.md) | Every later accuracy change is accepted or rejected on these numbers |
| **B. Where the studies disagree, and what has no human evidence** | #12; one bullet for #18 | [LITREVIEW_AGENT_B.md](../prompts/LITREVIEW_AGENT_B.md) | Code only, no model call and no accuracy risk, so it does not wait for A |

The two agents own separate files, named in each prompt. B cannot move A's
numbers: the benchmark judges only the model-written sentences, never the
code-generated sections.

*Status, 30 September 2026:* both merged, A as PR #32 and B as PR #31. A's
scientist half (the gold set and the blind re-judging) is still waiting for
a scientist.

Task A's code is only half of it. The gold set and the blind re-judging need a
scientist: about two days of labelling, and about 40 claims to re-judge. Ask
for that time when A starts; it is the longest wait in the plan.

### Then: the rest, one at a time

**C. Sentence faithfulness (#9).** After A is merged.

- Run `evals/litreview/entailment_agreement.py` on the benchmark reports.
- Narrow what the check reads. Today it reads `guard.source_text`: the quote,
  the context sentence, the qualifier and the naming sentence, joined. The
  result a sentence states must come from the quote; the other sentences may
  supply names and qualifiers only. As it stands, the check would pass
  OPEN_ISSUES #3: a sentence built from the naming sentence while citing a
  quote that says something else.
- Re-audit the #3 cases, have a scientist check a sample of what it drops,
  then decide whether it is on by default.
- Alone, because a false drop loses real evidence, so it needs a clean before
  and after.

*Status, 30 September 2026: done as far as it can go without a scientist; the
check stays off.* Branch `litreview/entailment-faithfulness`, not yet
committed.

- Measured, before and after the narrowing, against the hand audits instead
  of the benchmark reports: `evals/litreview/entailment_audit.py` (new). The
  strong judge cannot see #3, and no stored benchmark report is on this
  machine. The record is in `evals/litreview/entailment/`; the numbers are in
  BENCHMARK_RESULTS.md.
- Narrowed: the check reads each quote apart from the context, next and
  naming sentences, labelled, and prompt v2 says the claim must be in a quote.
- Result on qwen3:8b: 0 of 93 faithful sentences dropped under either prompt,
  and 2 of 12 unfaithful ones dropped under v2. None of the five #3 cases is
  caught, so on this model the narrowing does not catch what it was aimed at;
  check 10 does, for four of the five.
- Left: a scientist marks `evals/litreview/entailment/drops_v2_qwen3-8b.md`
  (2 sentences); the same measurement on the office model; and
  `entailment_agreement.py` on the stored reports, where they are kept. Then
  decide whether it is on by default.

#### Task C work log

A running log, written as the work happens, so that a session that stops
(a crash, a usage limit) can be picked up exactly where it left off. Newest
entry last. **To resume: read "Resume here", then the last entries.**

**Resume here** (kept current):

- Step now: Done for tonight. C (main tree,
  `litreview/entailment-faithfulness`) and D (`/Users/li/bedrock-recall`,
  `litreview/recall`) are complete and tested but not committed or pushed;
  that waits for the user. What is left in each is in the status paragraphs
  under tasks C and D above.
- Branches, neither committed nor pushed: task C on
  `litreview/entailment-faithfulness` in the main tree; task D on
  `litreview/recall` in the worktree `/Users/li/bedrock-recall`. The worktree
  `/Users/li/bedrock-v1` is `main` plus `evals/litreview/entailment_audit.py`,
  kept only so the v1 measurement runs the unchanged v1 code; remove it with
  `git worktree remove ../bedrock-v1` when step 1 is done.
- Scratch directory `S=/private/tmp/claude-501/-Users-li-Bedrock-private-AI/9723e28c-8f85-4d0e-b24e-4abc131b5896/scratchpad/`.
- **The measurement runs on its own**, detached from the Claude session:
  `$S/measure.sh` (check with `pgrep -fl measure.sh`; progress in
  `$S/measure.log`, errors in `$S/v1.log` and `$S/v2.log`). It runs v1 until
  `$S/rows_v1.jsonl` exists, then v2 (`--before rows_v1.jsonl --sheet
  drops_v2.md`), retrying each; it writes `ALL DONE` to `measure.log` at the
  end. If it is not running and `ALL DONE` is missing, start it again with
  `nohup $S/measure.sh >/dev/null 2>&1 & disown`: it skips what is done, and the
  call cache (`~/.local/share/bedrock/litreview/cache/llm`) keeps every answer
  already received. qwen3:8b takes about 1.5 minutes a sentence on this 8 GB
  M2, so v1 needs about 2.5 hours and v2 the same (started 02:26, restarted
  02:57; expect v1 near 05:00 and v2 near 07:30).
- Task C steps: 1 v1 numbers (running); 2 narrowing (done); 3 v2 numbers
  (queued after 1); 4 tests and wording (done); 5 docs: OPEN_ISSUES #3 and #5,
  BENCHMARK_RESULTS, `docs/litreview/README.md` (the entailment section),
  the header of `report/entailment.py` ("AND UNMEASURED") and this plan's
  status, written from `$S/table_v1.md`, `$S/table_v2.md` (which ends with
  the v1-to-v2 changes) and `$S/drops_v2.md` (the scientist's sheet; copy it
  into `evals/litreview/hand_audit/` or `docs/litreview/`). The check stays
  off by default until a scientist marks that sheet.
- Task D: code, tests and docs done in `/Users/li/bedrock-recall` (see the
  entries). Left: its before-and-after needs a person to approve aliases and a
  re-collection plus a real-model benchmark run, both after C, as the plan
  says; write that into this plan's status.

Entries:

- 2026-09-30 ~01:50 PDT. Wrote `evals/litreview/entailment_audit.py`: reads the
  seven hand-audit sheets and `grades*.jsonl` (105 sentences, 12 not faithful),
  runs guard check 10, the file judge and the entailment check on each, prints
  confusion matrices, `--rows`, `--before`, `--sheet`. Dry run with no model:
  check 10 drops 4 (migraine_after S5, obesity_after S6, S18, S29) and keeps all
  93 faithful ones; the file judge answers 90 of 105. Edited the docstring of
  `report/entailment.py` to describe the narrowed check (code not yet changed).
- 2026-09-30 02:14 PDT. The first v1 run stopped after 7.5 minutes with
  `ModelServerUnavailable ... peer closed connection`. Cause: the Ollama app was
  shut down at 02:14:25 (SIGTERM in `~/.ollama/logs/app-2.log`), restarted at
  02:16 and updated itself from 0.34.4 to 0.35.0. Not memory.
- 2026-09-30 02:25 PDT. The user said: retry, keep this log, finish the other
  parts tonight. Remaining parts: C, then D. E waits on an embedding model
  and the parked items wait on decisions, so neither can be done tonight.
- 2026-09-30 02:29 PDT. Started step 1 from `../bedrock-v1` with a retrying
  script (`scratchpad/run_v1.sh`, log `v1.log`, table `table_v1.md`, rows
  `rows_v1.jsonl`).
- 2026-09-30 02:29 PDT. Step 2 done in the main tree:
  `report/entailment.evidence_text` now prints `Quote: "..."` and then, under
  "Shown beside the quote, for names, who was studied, a sample size or a
  limit only:", one labelled line per context, next or naming sentence
  (`BESIDE`); the `guard.source_text` import is gone. `llm/prompts/entail.md`
  is version 2: the main claim must be in a quote, and a result found only
  beside one is `not_supported`.
- 2026-09-30 02:29 PDT. Step 4, tests: `tests/litreview/test_entailment.py`
  gains the obesity S18 / E190 case (`TMEM135`), tests of the labelled layout
  and the v2 rule, and three tests of `evals/litreview/entailment_audit.py` on
  the committed sheets (105 cases, 12 not faithful; check 10 drops exactly the
  four #3 cases; measure, table, sheet and changes with a fake model). 25
  pass.
- 2026-09-30 02:34 PDT. Step 4, the rest: the methods line in
  `stages/s7_write.py` and its test now say "optional, not yet checked by a
  scientist"; the `entailment_check` note in `config.py` says the same; the
  docstring of `evals/litreview/entailment_agreement.py` says what the check
  is shown now and why the judge cannot see #3; `entailment_audit.py` takes
  its role labels from `entailment.BESIDE` and silences the judge's
  per-sentence warnings. Whole litreview suite: 977 pass.
- 2026-09-30 02:51 PDT. Task D started in the worktree `../bedrock-recall`,
  branch `litreview/recall` (from `main`, not committed). Citation snowballing
  written: `sources/snowball.py` (new: `SnowballPolicy`, `choose_seeds`,
  `follow`), `EuropePMC.linked`/`records`, `PubMed.linked` (elink), wired into
  `LiteratureSearcher.search` after the cap, `SearchReport.snowball_*`,
  `LiteratureSearcher.signature()` feeding
  `s1_collect.collection_signature`/`terms_signature(spec, extra)`, a
  `snowball` block in `coverage.acquisition_summary` only when it ran, and
  `search.snowball` knobs in `defaults/pipeline.yaml` (seeds 0 = off). Tests:
  `tests/litreview/test_snowball.py`, 10 pass. Next for D: alias import
  (`knowledge/aliases.py` + `bedrock-litreview aliases propose|approve`).
- 2026-09-30 02:54 PDT. Task D alias import written in `../bedrock-recall`:
  `knowledge/aliases.py` (new: `read_hgnc`, `read_mesh` (preferred concept
  only, no tree), `propose_targets` (table targets keyed by an HGNC symbol,
  plus `--symbol` for new ones), `propose_diseases` (one MeSH heading per
  disease), `refusal` (another entity's name, a near-miss, a name HGNC gives
  another gene, 2 characters or fewer, inverted MeSH forms), `write_proposals`
  (everything `approve: false`), `approve` (text edit that keeps comments,
  re-checks refusals, reloads and proves only the marked names changed)), and
  `bedrock-litreview aliases propose|approve` in `cli.py`. Tests:
  `tests/litreview/test_aliases.py`, 15 pass. Full suite running.
- 2026-09-30 02:57 PDT. v1 is slow: 20 answers in its first 30 minutes (about
  1.5 minutes each; reasoning is off and answers are about 60 tokens, so it is
  qwen3:8b running partly on the CPU). The harness stops a background job
  after 2 hours, so the run was moved into a detached pipeline,
  `$S/measure.sh`, which resumes v1 from the call cache and then runs v2. Task
  D: docs written (README searcher section and state table, the package
  CLAUDE.md state and two rules, `sources/base.py`, OPEN_ISSUES #10).
  `entities.yaml` is deliberately untouched: a comment change alone would
  change the table hash that `verify-run` compares. The whole litreview suite
  in the D worktree: 1002 pass.
- 2026-09-30 02:59 PDT. Task D review fixes: seed matching now uses each
  required entity's near-misses too, so a "psoriatic arthritis" paper is not a
  psoriasis seed (test added); a docstring that claimed a merge was corrected.
  Live check, identifiers only, of the four link endpoints on PMID 36115523
  (the deucravacitinib phase 3 trial from the frozen psoriasis corpus): Europe
  PMC references 6 and citations 10+, PubMed citedin 10+, PubMed refs 0 (that
  paper's references are not deposited in PMC), and `EuropePMC.records` and
  `PubMed.fetch` return full records by id. The response shapes match the
  mocks. 25 D tests pass.
- 2026-09-30 03:00 PDT. Step 5, the parts that need no numbers:
  `docs/litreview/README.md` entailment section rewritten (what v2 shows the
  model, the two measuring tools, stays off until a scientist marks the
  sheet); OPEN_ISSUES #5 rewritten and a bullet added to #3. Both carry the
  placeholder "MEASUREMENT PENDING", to replace with the v1 and v2 numbers;
  BENCHMARK_RESULTS gets a new top entry then, and the header of
  `report/entailment.py` loses "AND UNMEASURED".
- 2026-09-30 03:01 PDT. Plan status written: A and B merged (#32, #31); D code
  written and tested, not measured, with the steps left. Scheduled resumes in
  this Claude session: 05:23 (the user's) and 07:53 (after v2 is expected to
  finish, to write the numbers). Both are session-only: if the session is
  gone, start a new one and follow "Resume here". Full bedrock `pytest`
  running on the C tree.
- 2026-09-30 03:02 PDT. Wrote `$S/summarise.py`: once `rows_v1.jsonl` and
  `rows_v2.jsonl` exist, `python $S/summarise.py 2>/dev/null` prints the
  markdown for BENCHMARK_RESULTS (lost / caught / missed per version, over the
  101 sentences check 10 keeps and over all 105; the #3 table; every v1-to-v2
  change; every faithful sentence v2 drops). Tested on fake rows. The baseline
  it shows: of the 101 sentences check 10 keeps, 8 are unfaithful by the hand
  audit; those are what the entailment check can add.
- 2026-09-30 03:06 PDT. Full bedrock `pytest -x` on the C tree: 1318 passed,
  then
  `tests/test_agentpack_app.py::test_approval_prompt_approve_calls_backend_and_clears`
  failed. It passes alone 3 of 3 on C and on `main`: timing under load from
  the measurement, not this change. Rerunning the whole suite without `-x` on
  C and then on D.
- 2026-09-30 03:28 PDT. Full suites. D (`../bedrock-recall`): 2573 passed. C:
  2552 passed, 3 failed, all terminal-app timing tests C does not touch
  (`test_agentpack_app` approval prompt, `test_history_regressions` forget
  during stream, `test_queue_execution` switching chats); the three pass
  together when run alone on C. Load from the measurement is the likely cause;
  rerun the full suite once the measurement is finished, before committing. v1
  is at 48 of 105 at 03:27 (about one a minute now), so expect it near 04:30
  and v2 near 06:15.
- 2026-09-30 05:32 PDT. Measurement finished on its own: v1 at 03:50, v2 at
  04:22 (`ALL DONE` in `$S/measure.log`). Copied the record into the repo:
  `evals/litreview/entailment/rows_v1_qwen3-8b.jsonl`,
  `rows_v2_qwen3-8b.jsonl` (one line per sentence, usable as `--before`) and
  `drops_v2_qwen3-8b.md` (the scientist's sheet, 2 sentences). Results: no
  faithful sentence dropped by either prompt (0 of 93); unfaithful dropped 1
  of 12 (v1) and 2 of 12 (v2); over the 101 check 10 keeps, 1 and 2 of 8; all
  five #3 cases supported or partial under both prompts; 35 of 105 changed
  class between v1 and v2, nearly all supported and partial swapping.
- 2026-09-30 05:45 PDT. BENCHMARK_RESULTS.md: new top entry, "The entailment
  check against the hand audit (30 September 2026)": setup, commands, the
  v1/v2 table, the #3 table with the judge column, what it says (safe, catches
  little, the narrowing did not catch #3 on qwen3:8b, labels unstable, v2
  kept), the decision (stays off) with the three steps before it can be
  switched on, and caveats.
- 2026-09-30 05:46 PDT. Step 5 done. OPEN_ISSUES #5 and #3: the placeholders
  replaced with the numbers and what is left. `report/entailment.py`: header
  now "OFF BY DEFAULT.", the "partial" note cites the 35-of-105 instability,
  and the off-by-default paragraph states the result. README entailment
  section: one paragraph of result. This plan: a status paragraph under task
  C. Removed the `../bedrock-v1` worktree (its only change was a copy of
  `entailment_audit.py`). Next: the full suite on C, now that nothing else is
  running.
- 2026-09-30 05:46 PDT. Cancelled the 06:27 and 07:53 scheduled resumes: they
  existed only to write the v2 numbers, which are written. Full `pytest` on C
  running with the machine idle.
- 2026-09-30 06:08 PDT. Full `pytest` on C with the machine idle: 2555 passed,
  7 skipped, none failed; the three earlier failures were load. C and D are
  done as far as they can go without a scientist, a person's alias approval
  and the office model.

**D. Recall (#1 narrowed, #2).** Measured after C; its code can be written
while C is being measured.

- Alias import: synonym files a person downloads (HGNC for genes, MeSH entry
  terms for diseases) become proposed entries that a person approves into
  `knowledge/entities.yaml`.
- Snowballing inside Stage 1, only when a search was asked for, through Europe
  PMC's citations and references endpoints and PubMed's `elink`. Every added
  paper passes the same gate as any other.
- One task, because both change what a question's frozen paper set contains:
  together they need one search-signature change, one re-collection of the
  benchmark corpora and one before-and-after on target recall (OPEN_ISSUES
  #10: 0.11 for obesity, 0.06 for asthma).

*Status, 30 September 2026: code written and tested, not yet measured.*
Branch `litreview/recall` (worktree `/Users/li/bedrock-recall`), not yet
committed:

- Alias import: `bedrock-litreview aliases propose --hgnc FILE [--mesh FILE]
  [--symbol SYM ...] --out proposals.yaml`, then `aliases approve
  proposals.yaml` (`knowledge/aliases.py`). HGNC names of each gene the table
  keys by its symbol, and of symbols asked for; MeSH terms of a heading's
  preferred concept only. Code refuses, with a reason, names another entity
  holds, near-misses, names HGNC gives another gene, names of two characters
  or fewer and inverted MeSH forms. Nothing is used until a person marks it,
  and `approve` edits `entities.yaml` as text and proves it changed only the
  marked names.
- Snowballing: `sources/snowball.py`, inside `LiteratureSearcher.search`.
  Seeds are the first searched papers naming every required entity (code, in
  the searcher's fixed order), followed through Europe PMC's references and
  citations and PubMed's `elink`, identifiers only. Added papers come after
  the searched ones, say how they were found, and pass the same gate. The
  policy is in the search signature only when on. **Off**
  (`search.snowball.seeds: 0`).
- Left, in this order: a person downloads HGNC and MeSH and approves names;
  snowballing is turned on for the measurement; the benchmark corpora are
  collected again (`build-corpus --searcher`); the benchmark is run on the
  same model and settings as the last one, and target recall compared.

**E. Semantic retrieval (#8).** When IT has deployed an embedding model. The
plan is `NEXT_STEPS.md` §6, and it also serves `/digest`. It chooses Stage 5's
chunks and orders papers before Stage 3 grades them; the gate stays exact.

**Parked, each until its decision:**

| # | Waits for |
|---|---|
| 4 ClinicalTrials.gov | The owner's approval of a new host. Then as a disclosure ("registered trials with no published results"), never as evidence |
| 5 Tables | A way to verify a table cell as a quote is verified. JATS tables first; PDFs wait for GROBID |
| 11 Subgroups | A provenance design stored as a run artifact, worked out together with OPEN_ISSUES #1 |
| 15, its kernel: concurrent extraction on the office server | Determinism on vLLM proven (OPEN_ISSUES #14) |
| 20 DSPy | A gold set of hundreds of adjudicated findings, with a holdout |

## How to read the rest of this document

Below is the original proposal, kept for its description of each technique.
Each item now opens with its verdict; where the verdict and the item's
"Litreview Integration" disagree, the verdict wins, and the file paths the
proposal names are not a plan. The proposal's figures and attributions (for
example "over 70% of numerical evidence" or "93.9% factuality on SimpleQA")
were not checked against their sources, and no decision here depends on them.
Every item was judged against the rules in
[`src/bedrock/litreview/CLAUDE.md`](../../src/bedrock/litreview/CLAUDE.md) and
the failures recorded in [OPEN_ISSUES.md](OPEN_ISSUES.md).

---

## Background: what frontier systems do in production

Before implementing enhancements, we must separate **empirically proven production architectures** from speculative academic theory:

| Architectural Area | Popular Academic Speculation | Disclosed Frontier Production Reality | Sources cited (not re-checked) |
| :--- | :--- | :--- | :--- |
| **Search Planning & Backtracking** | Monte Carlo Tree Search (MCTS) / Tree-of-Thoughts over search queries | **No frontier production system uses MCTS over search queries.** Planning, query reformulation, and backtracking occur **autoregressively within the model's test-time compute chain of thought (CoT)**, paired with **Orchestrator-Worker parallel subagent swarms**. | **OpenAI** *Deep Research System Card* (Feb 2025); **Anthropic** *Building Effective Agents* (2024–2025) & *Multi-Agent Research System* (June 2025). |
| **Why MCTS Was Rejected in Production** | Theoretical optimality in game trees | Anthropic explicitly cautions against MCTS/graph topologies for agents: they introduce severe latency jitter, brittleness, and fail to outperform simple composable workflows (orchestrator-worker + evaluator-optimizer). | Anthropic Engineering: *Building Effective Agents*. |
| **Citation & Claim Grounding** | Unchecked prompt instructions | **Formal NLI Claim Entailment ($Passage \models Sentence$)** + **Sub-document span-level extraction**. | **Stanford STORM** (arXiv:2402.14207, NAACL 2024); **Perplexity** *AI-First Search API* (Sept 2025). |
| **Collaborative / Compounding Memory** | Monolithic chat context stuffing | **Tree-structured Dynamic Concept Mind Maps $\mathcal{M} = (\mathcal{C}, \mathcal{E})$** with algorithmic `insert` and `reorganize` operations. | **Stanford Co-STORM** (arXiv:2408.15232, EMNLP 2024). |
| **Context Management** | Passing full HTML/PDFs between agents | **External Filesystem / Artifact Stores**: subagents compress sources into artifacts on disk and pass only lightweight reference handles. | **Anthropic** *How we built our multi-agent research system* (June 2025). |

---

## The catalogue, with a verdict on each item

### Pillar I: Intelligent Ingestion & Retrieval (Stages 0 & 1)

#### 1. Ontological Term Expansion Without Drift

> **Verdict: build, narrowed (task D).** Import synonyms only, from files a person downloads (HGNC gene aliases matter most for Propeller's targets; MeSH entry terms for diseases), as proposed entries that a person approves into `knowledge/entities.yaml`. Never walk an ontology tree: MeSH files bovine RSV under the RSV heading, which is exactly the near-miss `entities.yaml` exists to keep out. Database ids such as ChEMBL numbers do not appear in paper text, so they add no recall. It fills a measured gap: 160 obesity and 65 asthma reference targets have no alias entry (OPEN_ISSUES #10).

* **The Frontier Technique:** Naive LLMs hallucinate synonyms. Frontier biomedical searchers traverse deterministic directed acyclic graphs: **MeSH (Medical Subject Headings)**, **RxNorm** (active ingredients vs brand formulations), and **UMLS**.
* **Litreview Integration:** In `knowledge/entities.yaml`, supplement static lists with automated local MeSH tree traversal. Expanding *"Tirzepatide"* automatically fetches `CHEMBL4297831`, `LY3298176`, `Mounjaro`, and `Zepbound` deterministically, guaranteeing complete literature recall without semantic drift.

#### 2. Forward & Backward Citation Snowballing

> **Verdict: build (task D).** Europe PMC's citations and references endpoints (on `www.ebi.ac.uk`) and PubMed's `elink` (on `eutils.ncbi.nlm.nih.gov`) are on the two hosts the searcher may already reach, so no new host is needed; OpenAlex, Semantic Scholar or Crossref would need the owner's approval. It runs only when a search was asked for, and every paper it adds passes the same gate as any other and is recorded in the frozen set's search signature. Code chooses the seed papers inside Stage 1, rather than "after Stage 3", so that what gets collected never depends on a model's grades.

* **The Frontier Technique:** High-performing research agents don't stop at keyword search. They traverse the academic citation graph:
  * **Backward Snowballing:** Automatically parse the references section of high-scoring papers to find seminal foundational literature.
  * **Forward Snowballing (The Game Changer):** Query citation indexes (OpenAlex / Semantic Scholar / Crossref APIs) to locate papers published *after* the landmark trial that cite it (catching 2025/2026 follow-up studies and letters to the editor).
* **Litreview Integration:** Extend `sources/literature.py` with a snowballing pass that runs after Stage 3 relevance ranking on the top 5 papers.

#### 3. Sub-Document / Span-Level Indexing (Perplexity Architecture)

> **Verdict: already built.** Stage 5 splits each section into chunks and sends the ones that best match the question (`plan_chunks` in `stages/s5_extract.py`, ranked by BM25). The model never sees a whole paper.

* **The Verified Source:** Perplexity AI Engineering (*Architecting and evaluating an AI-first Search API*, Sept 2025).
* **The Frontier Technique:** Instead of returning full documents or massive text chunks, the retrieval engine indexes and scores candidate evidence at the **sub-document span level** (100–300 token semantic passages). This prevents context window contamination and drastically cuts downstream hallucination rates (yielding 93.9% factuality on SimpleQA).
* **Litreview Integration:** In `parsing/` and `ingest.py`, segment full paper texts into semantically cohesive, offset-anchored spans prior to relevance scoring.

#### 4. ClinicalTrials.gov Protocol Registry Ingestion

> **Verdict: parked.** It needs the owner's approval for a new host. A registry entry is not a sentence in a paper, so it cannot be evidence under the quote rule. The useful form is a disclosure: "registered trials on this target with no published results".

* **The Frontier Technique:** Peer-reviewed publications suffer from publication bias (positive results get published; negative results are filed away). Frontier medical engines search ClinicalTrials.gov directly.
* **Litreview Integration:** Add a dedicated source adapter `sources/clinicaltrials.py`. Ingest protocol registrations (NCT numbers, primary completion status, terminated/withdrawn flags, and outcome measures).

#### 5. Multimodal PDF Complex Table Extraction

> **Verdict: parked.** Open-access JATS full text already carries its tables as XML (stripped today in `parsing/sections.py`), so no vision model is needed for those papers. The blocker is verification: quotes are anchored as sentences, and a table cell needs an anchor of its own before a number from a table can be cited. The PDF route waits on GROBID (`parsing/grobid.py`, not built).

* **The Frontier Technique:** Over 70% of numerical evidence (endpoints, hazard ratios, adverse event counts, patient demographics) is locked inside PDF tables. Standard text extractors turn tables into garbled string soups. Frontier models use vision-based table extractors (e.g., Table Transformer, Nougat, or Marker).
* **Litreview Integration:** In `parsing/`, replace or supplement raw PDF text extraction with a dedicated table parser that outputs normalized Markdown and JSON schemas for tables labeled *"Table 1"*, *"Table 2"*, etc.

---

### Pillar II: Gating, Ranking & Quality Stratification (Stages 2 & 3)

#### 6. Oxford CEBM Evidence Hierarchy Stratification

> **Verdict: already built.** `study_design.py` labels each paper's design from its publication type and its own wording; Stage 6 orders targets by human clinical, then human genetic, then animal evidence; "How the studies line up" states an evidence basis by a written rule. A rule that animal studies can never outrank human trials would be wrong for target identification, where knockout studies are primary evidence.

* **The Frontier Technique:** Not all literature is created equal. Frontier engines classify papers according to the **Oxford Centre for Evidence-Based Medicine (CEBM)** hierarchy:
  * Level 1: Systematic Reviews & Meta-Analyses of double-blind RCTs.
  * Level 2: Randomized Controlled Trials (RCTs).
  * Level 3: Controlled Cohort / Non-randomized trials.
  * Level 4: Case-series / Case-control studies.
  * Level 5: Animal models (*in vivo* C57BL/6 mice) and mechanistic cell cultures (*in vitro*).
* **Litreview Integration:** In `study_design.py`, assign explicit CEBM Level tags to each paper based on code heuristics (MeSH publication types, sample size, presence of placebo/blinding). Ensure animal studies can never outrank human RCTs in the final evidence summary.

#### 7. RetractionWatch & Expression of Concern Real-Time Quarantine

> **Verdict: already built.** `sources/integrity.py` and `publication_integrity.py`, following the owner's decisions of 26 September 2026: Europe PMC, PubMed and Crossref are checked, retracted papers are excluded, and papers with an expression of concern are ranked last.

* **The Frontier Technique:** Frontier research agents query Crossref’s retraction database to ensure discredited studies are never cited as fact.
* **Litreview Integration:** In `stages/s2_gate.py`, check paper DOIs against a local SQLite snapshot of RetractionWatch and Crossref retractions. Flag retracted papers with a quarantine warning and exclude them from evidence synthesis.

#### 8. ColBERTv2 / Late-Interaction Deterministic Re-Ranking

> **Verdict: already planned (task E), blocked.** This is the semantic-retrieval item (`NEXT_STEPS.md` §6, OPEN_ISSUES #29), waiting on a deployed embedding model. It adds to the ranking rather than replacing it: Stage 3's grade is the model's answer to narrow relevance questions, and BM25 only decides which papers get graded and breaks ties.

* **The Frontier Technique:** BM25 lexical matching misses semantic synonyms; naive dense bi-encoders (cosine similarity) lose token-level precision. Frontier searchers use **ColBERTv2 late interaction** (token-level MaxSim), which computes fine-grained alignment between question tokens and document tokens.
* **Litreview Integration:** In `stages/s3_rank.py`, replace simple keyword scoring with a fast, local, quantized ColBERTv2 engine (or BGE-Reranker-large), running locally on GPU/CPU with $100\%$ determinism.

---

### Pillar III: Grounding, Extraction & Semantic Verification (Stages 4 & 5)

#### 9. Natural Language Inference (NLI) Claim Entailment Guard

> **Verdict: build the next step (task C).** The check exists: `report/entailment.py`, off by default and unmeasured (OPEN_ISSUES #5). It asks for a label (supported, partial, not supported, contradicted) rather than a probability, which suits a temperature-0 structured call better than thresholds on P(entailment). The description of the current code below is out of date: the citation guard (`report/guard.py`) already drops sentences whose numbers, names or subjects are not in the cited text. Task C adds the measurement, and narrows what the check reads (see the plan).

* **The Verified Source:** Stanford STORM (*Assisting in Writing Wikipedia-like Articles From Scratch with LLMs*, arXiv:2402.14207, Section 4.2 & Appendix C.3) & ALCE Benchmark.
* **The Limitation in Existing Litreview:** Litreview's current `extract/verify.py` string-matches quotes against the paper. This proves the *quote exists*, but it does **not** prove that the model's *claim* accurately reflects the quote! (e.g., Quote: *"The drug showed non-significant trend towards tumor regression (p=0.12)"*; Model writes: *"The drug caused tumor regression"*).
* **The Frontier Technique:** Stanford formulated instruction-tuned models as **NLI Entailment Classifiers** to verify that the retrieved passage strictly entails the generated statement:
  $$P(\text{Entailment} \mid \text{Premise}=\text{VerbatimQuote}, \; \text{Hypothesis}=\text{ModelClaim})$$
* **Litreview Integration:** Add `extract/nli_guard.py`. If $P(\text{Entailment}) < 0.85$, or if $P(\text{Contradiction}) > 0.10$, automatically reject the finding or blank the field.

#### 10. Automated Quantitative Unit Normalizer

> **Verdict: rejected.** Standardised effect sizes need per-arm standard deviations that abstracts rarely give, and a computed number is in no paper, so the citation guard removes it by design. Pooling effects across studies is a meta-analysis, which the client has not asked for. Unit conversion factors differ by analyte, and one wrong factor would be shown as the paper's own number.

* **The Frontier Technique:** Different papers report endpoints in divergent units ($\text{mg/dL}$ vs $\text{mmol/L}$, $\mu\text{M}$ vs $\text{nM}$, $\text{pg/mL}$ vs $\text{ng/L}$).
* **Litreview Integration:** In `extract/provenance.py`, add a deterministic unit normalization pass. Automatically calculate standardized effect sizes (Cohen's $d$, Hedge's $g$, or normalized percentage change) so studies can be compared apple-to-apple.

#### 11. Subgroup & Cohort Disaggregation

> **Verdict: parked.** New fields on `EvidenceUnit` change every stored body's bytes and break `check_pinned` for old runs; new run state goes in versioned artifacts. The real problem, which experiment or subgroup backs which half of a claim, is OPEN_ISSUES #1 and needs a provenance design first.

* **The Frontier Technique:** Frontier models extract not just aggregate averages, but demographic and biomarker subgroups:
  * Baseline HbA1c $\ge 8.5\%$ vs $< 8.5\%$
  * Prior SGLT2i treatment vs drug-naive
  * Renal impairment stage (CKD Stage 3a/3b/4)
* **Litreview Integration:** Update `EvidenceUnit` in `models.py` with an optional `subgroup_cohort` string and `baseline_biomarkers` dictionary.

---

### Pillar IV: Contradiction Triangulation & Bias Auditing (Stage 6)

#### 12. Stance Triangulation & The Discrepancy Matrix

> **Verdict: build, narrowed (task B).** Stage 6 already finds conflicts, and "How the studies line up" already says when studies disagree. Task B adds the side-by-side view: each disagreeing study's design, who and how many were studied, what was read, the direction, and the paper's own words. Code shows how the studies differ; the scientist decides which difference explains the disagreement (dose, duration, formulation).

* **The Frontier Technique:** When clinical trials disagree (e.g., SURPASS-2 shows $p < 0.001$ for weight loss; another trial shows no significant difference), naive pipelines either hallucinate an average or pick one. Frontier research models build a **Discrepancy Matrix**.
* **Litreview Integration:** Add `stages/s6_triangulate.py`. When two `EvidenceUnit` items share the same target and endpoint but diverge in effect direction, code extracts:
  1. Dose and titration variance
  2. Baseline patient severity
  3. Trial duration (e.g., 24 weeks vs 72 weeks)
  4. Formulation differences
  The report displays a highlighted *"Discordant Findings & Methodological Differences"* section.

#### 13. Automated Risk of Bias (RoB 2 / ROBINS-I) Scoring

> **Verdict: rejected.** A green, yellow or red rating is a quality score, which the client refused: quality signals are looked up and shown, and a person judges them. RoB 2 is a judgement tool even for trained reviewers, and keyword heuristics under Cochrane's name would be trusted far more than they deserve.

* **The Frontier Technique:** Code-level extraction of Cochrane Risk of Bias indicators:
  * Random sequence generation (Low/High/Unclear risk)
  * Allocation concealment
  * Blinding of participants and personnel
  * Incomplete outcome data (attrition rate $> 20\%$)
  * Selective reporting
* **Litreview Integration:** In `stages/s6_score.py`, extract trial design flags deterministically and display an RoB traffic-light summary (Green/Yellow/Red) beside each clinical trial.

---

### Pillar V: Advanced Synthesis & Reporting (Stages 7 & 8)

#### 14. Dynamic Mind Map Delta-Search (Stanford Co-STORM)

> **Verdict: rejected.** Skipping search because an earlier review "saturated" a facet breaks run pinning and misses papers published since. Retractions found later would not reach the stored claims. `BEDROCK_USER` is not authentication, so one scientist's reviews would flow into another's. Co-STORM's insert needs embeddings (blocked), and its reorganise lets a model shape the structure. A safe version, if wanted later: list earlier reviews on the same entities as links, with no evidence reused.

* **The Verified Source:** Stanford Co-STORM (*Into the Unknown Unknowns*, arXiv:2408.15232, EMNLP 2024).
* **The Frontier Technique:** Co-STORM maintains a dynamic tree-structured concept mind map $\mathcal{M} = (\mathcal{C}, \mathcal{E})$.
  * **`Insert` Operation:** Computes cosine similarity between the question that yielded evidence and existing concept nodes, placing the finding under the closest semantic concept.
  * **`Reorganize` Operation:** Triggered when any concept exceeds $K=10$ findings. Automatically spawns child subtopics, re-distributes evidence, and collapses single-child redundant nodes.
* **Litreview Integration:**
  * When a review runs, save all verified findings into `atomic_claims` structured by the concept tree.
  * When Scientist B asks a new question, compute the Knowledge Frontier:
    $$\mathcal{F} = \text{TargetFacets} \setminus \text{SaturatedFacets}(\text{AVC})$$
  * Skip search for saturated baseline facts. Search *only* for missing delta facets, compounding the team's shared memory.

#### 15. Orchestrator-Worker Swarms with Artifact Stores (Anthropic Pattern)

> **Verdict: rejected as written.** It is the agent-loop design this package rejects: a pipeline, not a free-form agent. The model never passes whole papers around, so the context problem it solves does not arise here, and `BEDROCK_HISTORY_DIR` is the compliance journal, not scratch space. Its kernel is parked: concurrent extraction calls on the office server, once determinism on vLLM is proven (OPEN_ISSUES #14), because batching can change temperature-0 output.

* **The Verified Source:** Anthropic Engineering (*How we built our multi-agent research system*, June 2025).
* **The Frontier Technique:** Instead of passing massive web/paper text up and down nested agent chains (which causes severe degradation and context saturation), the Lead Orchestrator delegates sub-facets to parallel workers. Each worker saves its findings to an **external filesystem / artifact store** on disk and passes back only lightweight pointers/metadata.
* **Litreview Integration:** In `stages/s4_plan.py` and `runner.py`, parallelize facet extraction tasks, storing intermediate extraction matrices in the local workspace directory (`BEDROCK_HISTORY_DIR`), passing only `EvidenceUnit` references to `s7_write.py`.

#### 16. Structured Evidence Matrix (Comparative Markdown Tables)

> **Verdict: already built.** "Evidence at a glance" and "Key results in context" (`report/synthesis.py`) are built by code from verified fields and the papers' own sentences. The proposal has the model draft the table, which would put model-written numbers in the report.

* **The Frontier Technique:** Instead of dense walls of prose, synthesize evidence into structured comparative tables:
  | Target / Drug | Study / DOI | Cohort ($N$) | Primary Endpoint | Effect Size (95% CI) | $p$-value | CEBM Level | Risk of Bias |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
* **Litreview Integration:** In `stages/s7_write.py`, require the model to draft an `EvidenceMatrix` structure via guided JSON decoding, rendered in markdown by `stages/s8_render.py`.

#### 17. Asynchronous Stream-Aligned Verification Badges

> **Verdict: rejected.** It shows text before it is verified, which the quote rule forbids. The "~150ms" does not hold here: each check is a model call through the scheduler, which takes seconds on a laptop.

* **The Frontier Technique:** Text streams into the Textual UI immediately with subtle pending anchors:
  `Tirzepatide achieved 20.9% weight reduction [Ref 1: verifying...]`
  A background worker executes string-match and NLI entailment in parallel (~150ms latency), dynamically updating the UI to:
  `Tirzepatide achieved 20.9% weight reduction [Ref 1: verified ✓]`
* **Litreview Integration:** Add streaming events in `llm/scheduled.py` and `ui/review_command.py` for sentence-level verification state transitions.

#### 18. Explicit "Known Unknowns & Evidence Gaps" Chapter

> **Verdict: already built; task B adds one bullet.** "Limitations, gaps and open questions" is in every report layout (`defaults/report.yaml`). Task B adds the measures whose evidence contains no tested human study.

* **The Frontier Technique:** Frontier models explicitly catalogue what is *missing* from current literature to prevent dangerous extrapolation.
* **Litreview Integration:** In `report/invariants.py`, enforce an invariant that every report contains an *"Unanswered Questions & Literature Gaps"* section, listing subtopics where no peer-reviewed human trials were located.

---

### Pillar VI: Continuous Improvement & Self-Evolution (Post-Training)

#### 19. The 50-Question Golden Evaluation Harness ("The Nightly Exam")

> **Verdict: build, first (task A).** The benchmark exists (`bench/`: judge, perturbation, distractors, determinism). What it lacks is what makes its numbers trustworthy: a context-window check before model calls, gates that report "not measured", scores that name their judged fraction and raters, a comparison that refuses unequal runs, and a gold set labelled by a scientist (`evals/litreview/goldset.jsonl` holds only a placeholder). HLE and GAIA test open web research, and "redundant query rate" measures a model's search queries, while here code builds the queries; neither applies.

* **The Verified Sources:** OpenAI Deep Research evaluation methodology on Humanity's Last Exam (HLE, 26.6%) and GAIA (67.4%–72.6%); Anthropic BrowseComp evaluation.
* **The Frontier Technique:** Frontier labs never push a prompt or algorithm tweak without running benchmark regression suites measuring accuracy, pass rate vs tool calls, and citation precision.
* **Litreview Integration:** Add `bench/golden_eval.py`. A suite of 50 complex biomedical literature review questions with known ground-truth endpoints. Runs headlessly via `pytest bench/` or nightly cron, measuring:
  * Citation Precision ($100\%$ target)
  * Verification Survival Rate (how many extracted quotes pass code verification)
  * Redundant Query Rate ($< 15\%$ target)

#### 20. Programmatic Prompt Distillation (DSPy / MIPRO)

> **Verdict: parked.** Tuning prompts against a small evaluation set produces prompts that fit those papers and generalise poorly, which `src/bedrock/litreview/CLAUDE.md` warns against. Revisit when the gold set holds hundreds of adjudicated findings with a holdout.

* **The Verified Source:** Stanford STORM & Co-STORM (both natively implemented via DSPy).
* **The Frontier Technique:** Hand-crafting prompts is obsolete. Frontier labs use automated prompt optimizers (DSPy) that evaluate hundreds of prompt phrasing variations and few-shot examples against the evaluation harness.
* **Litreview Integration:** In `bench/optimize_prompts.py`, optimize the extraction and ranking prompts automatically using DSPy with zero-temperature local models.

#### 21. Local Direct Preference Optimization (DPO) for Verifiers

> **Verdict: rejected.** No feedback data exists, no machine here can train it, it would replace the models pinned in `defaults/models.yaml`, and it trains the verifier toward what scientists liked rather than what the paper says. The useful part is scientists' grades, which task A's blind re-judging sheet collects.

* **The Frontier Technique:** Fine-tune small 8B verifiers (or LoRA adapters) on human scientist interaction traces:
  * Accepted/Starred reports $\to y_{\text{win}}$
  * Edited/Deleted sentences $\to y_{\text{lose}}$
* **Litreview Integration:** Log scientist edits in `BEDROCK_HISTORY_DIR`. Provide a local CLI command `bedrock-litreview train-verifier` that runs LoRA DPO on a local GPU.
