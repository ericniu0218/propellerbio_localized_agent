# Working in `src/bedrock/litreview/`

Read this before changing anything in this package. The repository root's
`CLAUDE.md` also applies — this is the terminal client the review lives inside,
and its model-call rule, its privacy rule and its "not a website" rule are not
suspended here. `docs/litreview/README.md` has the full project context; this
file is the short version plus the rules that are easy to break by accident.

This package moved in from its own repository
(Bedrock-Technical-Consulting/Localized-Airgap-LLM-for-literature-review). What
changed in the move, and what deliberately did not, is
`docs/prompts/LITREVIEW_INTEGRATION.md`.

## What this is

An air-gapped literature review pipeline for Propeller Bio, a Bay Area biotech
developing biologics (antibodies, peptides, siRNAs) that hit multiple disease
pathways at once. Focus areas: immunology & inflammation, obesity, CNS.

A scientist types a question ("if I want to prevent psoriasis, what should I
target?"), as `/review` in the app, `bedrock-litreview ask` in a shell, or
`POST /ask` over the optional HTTP API. The paper searcher
(`sources/literature.py`) collects
candidates from Europe PMC, preprints and PubMed. This system reads the question, excludes papers about near-miss entities, ranks
every paper by relevance, plans which information to take from which papers,
extracts verified findings, and writes a cited report. It runs entirely on the
client's own in-office GPU. Nothing leaves their network.

## The architecture in one line

**A pipeline, not a free-form agent.** Python owns control flow. The model is
called only for narrow jobs — classify one abstract, extract from one section,
write one report section. The model never decides what happens next.

This is a deliberate rejection of the agent-loop design. Do not "improve" a
stage by letting the model plan or choose tools. If a change makes the model
responsible for sequencing, it is wrong for this project, regardless of how well
it works elsewhere.

## Read these first, in order

1. `models.py` — the data model. Everything is a view of `EvidenceUnit`
2. `extract/verify.py` — the fabrication guard
3. `extract/provenance.py` — the same guard, for every other field a finding carries
4. `report/guard.py` — the same guard, applied to report sentences
5. `report/invariants.py` — what code proves about a finished report
6. `knowledge/` — exact terms and near-misses, meaning words, outcome measures, a paper's own abbreviations, and the matcher
7. `docs/litreview/README.md` § The pipeline — what each stage owes the next
8. `service.py` and `llm/scheduled.py` — how a run reaches the model server

## Rules that are not style preferences

**Every EvidenceUnit carries a verbatim quote, verified in code.** The model
copies the sentence it took a finding from; `extract/verify.py` string-matches
it against the source. No match, the finding is destroyed. A fuzzy match is
typography, never meaning: if a differing word is a number or is listed in
`knowledge/meaning_words.yaml`, the quote fails. What is stored and shown is
the paper's own sentence at recorded offsets, never the model's copy. Never
relax this, never add a path that surfaces an unverified quote, never "trust" a
model that seems reliable. A citation that cannot be checked is worse than no
citation, because it reads as though somebody checked it.

**Every other field the model writes is found in the paper, or blanked.** A
correct quote can carry an invented sample size or "in mice".
`extract/provenance.py` looks for the target, agent, subjects, sample size and
result summary in the text the model read; a field found nowhere is blanked
(subjects show "not stated") and a target found nowhere drops the finding.
Never display or cite a model-written field whose provenance is `none`.

**No model ever scores paper quality.** The client said explicitly that they do
not trust AI judgement here and will check sources themselves. Quality signals
(journal, citations, preprint status, sample size, retraction) are looked up in
code and displayed. A human decides what they mean. The model answers narrow
questions about *relevance to the question* (Stage 3) and code turns them into
a grade by a written rule — never present that grade as quality,
and it must come with a verified justification quote.

**Exact means exact.** A question about RSV is answered from RSV papers, not
hMPV or bovine RSV papers. Exact terms and near-misses live in
`knowledge/entities.yaml`, are applied by code, and are shown in the
report. Never let a model add a synonym or search term: a wrong synonym
silently lets the wrong papers in. The model may only copy names that are
verifiably in the question.

**Every report sentence cites evidence, and code checks it.** `report/guard.py`
drops any sentence whose numbers, entity names or who-was-studied words are not
in the evidence it cites — and that evidence is the paper's text (quote and
context sentence), never `effect_detail` or other model-written fields. Never
bypass it, and never fill a planning gap with text that cites nothing — a gap
is stated as a gap.

**Every report is integrity-checked, and a run is pinned to its papers.**
`report/invariants.py` re-proves the guarantees on every finished report;
a violation is a bug, written on the report (warning banner) and makes
`bedrock-litreview ask` exit non-zero. `bedrock-litreview verify-run` and
`/reviews verify` re-check a stored run.
Runs record the exact paper versions they read (`paper_versions`,
`papers_manifest`); never read a past run's papers from the question's
current set. Merging duplicate papers never loses a review or retraction flag
(`ingest.merge_records`).

**The model copies; code normalises.** The model returns "C57BL/6 mice"; code
returns "mouse". Do not ask the model to normalise — a lookup table returns the
same answer every time and a model does not, and determinism is a feature the
client asked for by name.

**Ask narrow questions; let code decide.** Do not ask a small model for a
score or a pick-several list: on the first real run MedGemma 4B graded 39 of 40
papers 3/3 and ticked every information need on 36. Ask one-answer questions
whose answer code can check ("which molecule did the paper change?" — then
look for that name in the abstract), and write the rule that turns answers
into a decision in code. Where the model labels something (a direction, an
outcome), have it copy the words it based the label on, so code can check.

**`temperature=0.0` everywhere.** Do not raise it, including "just for the
writing stage". The client's loudest complaint about existing chatbots is
getting different answers to the same question.

**Structured calls pass a JSON schema.** vLLM's guided decoding makes malformed
output impossible rather than unlikely. Every structured call uses it.

**Never integrate Sci-Hub or any pirated source.** The client mentioned it; the
answer is the drop folder (`dropfolder.py`, to be built on bedrock's `share`
connector) — Propeller already pays
for journal access, and we ingest what their scientists legally download.

**Reviews are not evidence.** A review article citing a trial is not a trial.
Counting one would inflate the apparent weight of evidence.

**Negative results are first-class.** `no_effect` and `worsened` matter as much
as `improved`. Models skip these unless explicitly told not to.

## Conventions

- Stages are `stages/sN_name.py`, run in numeric order: s0_question,
  s1_collect, s2_gate, s3_rank, s4_plan, s5_extract, s6_score, s7_write,
  s8_render (s9_chat not built). `runner.py` calls them; nothing else sequences
- What each question type needs is `facets.py` (code). The report layout is
  `defaults/report.yaml`. Pipeline knobs are `defaults/pipeline.yaml`
- The paper searcher plugs in only through `sources/base.py`
- The searcher goes to the internet only when asked: `bedrock-litreview search`,
  `ask --search`, `/review --search`, or `"search": true` over HTTP. Never make
  it a default — a question's papers are frozen once collected, and the
  air-gapped machine must work from a JSON-lines file. Its queries come from
  `exact_terms` and `search.focus` in `defaults/pipeline.yaml`; never let a
  model add a term. A request may reach only `www.ebi.ac.uk` and
  `eutils.ncbi.nlm.nih.gov` (`SOURCE_HOSTS` in `sources/http.py`), redirects
  included: a search URL carries the scientist's terms in its query string
- A question's frozen paper set records the search signature it was
  collected under (`stages/s1_collect.py`, `question_collections`): the
  required entities and their exact terms, or the corpus hash. A run whose
  reading differs collects again when a search was asked for, and otherwise
  says in the report that its papers were never searched for its terms.
  Failed search queries are recorded with the set and shown in every report
  built on it. Never reuse a set silently across readings
- A disease named only in the question's population ("... in adults with
  obesity") is not required when another disease is (`s0_question`
  `population_span`): requiring it made the search drop the trials that
  measured the outcome in a related population. Who each study was about is
  shown instead
- Among papers of equal relevance grade, efficacy, safety and mechanism
  questions read randomised trials before observational studies before animal
  and cell work (`study_design.design_rank`, from the paper's own publication
  type and wording - a reading order, never a quality score). Target
  identification keeps lexical order: measured on the frozen corpora it gained
  nothing there. `rank.selection: evidence_aware` remains the separate, opt-in
  policy
- Retraction checks (`sources/integrity.py`) are the other way out, decided by
  the owner on 26 September 2026: automatic at the start of each *new* review
  unless the scientist chose manual (`/reviews settings`, `preferences.py`),
  and by hand with `bedrock-litreview integrity refresh --yes`. They send
  publication identifiers only, to `INTEGRITY_HOSTS` (the two source hosts
  plus `api.crossref.org`, which the search itself cannot reach). Never check
  from `verify-run`, a resume or report viewing; never add another host
  without the owner's approval; never let a check failure fail a review; and
  never add a retracted paper back automatically - a reinstated one waits
  for a person (`/reviews reinstate`)
- Tests never reach the network: `tests/conftest.py` turns the automatic check
  off for the whole suite; test it with `httpx.MockTransport`
- New run state goes in versioned auxiliary artifacts (`put_artifact`), not
  in `Report`, `Paper` or `EvidenceUnit`: a new defaulted field there changes
  every stored body's bytes and breaks `check_pinned` for old runs
- Sources are listed in code (`SOURCES` in `sources/literature.py`), not read
  from a config file, so nothing gets wired in by editing one
- Model names live only in `defaults/models.yaml`, or in a whole-file
  replacement at `~/.config/bedrock/litreview/models.yaml`. Nothing else
  hardcodes one. The *server* is bedrock's `api_base`, or in agent-pack mode
  the agent server's inference API (`agent_pack_api_base` + `/v1`, the
  `agentpack` model set, roles as model names), or `[litreview] api_base`
  (`config.resolve_endpoint`). The context window is what someone configured,
  else the machine profile's (local Ollama only), else what the server
  reports per model - and the reported one wins when smaller
  (`service.preflight`). Never let a local machine profile describe a remote
  model, and never assume a remote window
- **The only live model client is `SchedulerClient`** (`llm/scheduled.py`),
  which goes through bedrock's `Scheduler` at `Priority.USER_JOB`, so a review
  never gets in front of the employee's own chat. `tests/test_model_call_guard.py`
  fails if anything reaches the backend directly, and nothing here may talk
  HTTP to a model server
- Nothing else in this package imports `bedrock.inference`: only
  `llm/scheduled.py` and `service.py` do. Everything else depends on the
  `LLMClient` protocol, so tests use `FakeLLMClient`
- A model-using review starts through `service.run_review` and nothing else.
  It is what runs `preflight`, builds the client and puts the blocking pipeline
  on a worker thread
- A server that goes away raises `ReviewInterrupted`, which no stage catches, so
  the run stops and can be resumed. `LLMError` still means "skip this one item"
- Prompts live in `llm/prompts/*.md` with a `version` in front matter.
  Bump it on every change — it is recorded on each EvidenceUnit
- Everything that can be tested without a GPU is. The whole pure-code layer runs
  offline, because in week one the office GPU does not exist yet. Run
  `pytest tests/litreview`, and `pytest` before you commit

## Current state

| Part | State |
|---|---|
| Stages 0-8 (question → report) | Built and tested; runs end to end with `FakeLLMClient` |
| Paper searcher | Built and tested offline; run live. Europe PMC + preprints + PubMed, open-access JATS full text, on a two-host allowlist. GROBID/PDF rung, alias lookup and drop folder not built |
| Benchmark (`bench/`) | Built and tested with a fake model; sample corpora in `evals/litreview/corpora/` |
| Real-model benchmark results | Run on MedGemma 1.5 4B, both sample questions — see [BENCHMARK_RESULTS.md](../../../docs/litreview/BENCHMARK_RESULTS.md) |
| Front ends | `/review` and `/reviews` in the terminal app - both modes, `ui/review_mixin.py` - `bedrock-litreview` in a shell, the server's Research tab, the FastAPI app as the optional `api` extra. All call `service.run_review`. `bedrock-litreview doctor` (`readiness.py`) checks a machine can run one |
| Report | Reading view, then References, then an Appendix. `report/synthesis.py` computes study profiles, independent-study counts, evidence basis, disagreements with whole quotes and qualifiers, key results and abbreviations from verified fields and paper records. Synthesis, disagreements and the gaps bullet naming measures or targets without tested human effects share comparison rows. `check_disagreements` binds each quote to evidence from its cited paper; no new stored model fields |
| Coverage ledger, study design, drift disclosure | Built and tested offline: `coverage.v1`, `study_design.v1`, `verification_context.v1` run artifacts, never in the report; `verify-run` compares environments (`--require-same-rules`). `rank.selection: evidence_aware` is opt-in and unmeasured on a real model |
| Publication integrity | Built and tested against captured public responses: automatic check per new review (or manual), append-only observations, run snapshot, retraction exclusion, concerns ranked last and kept out of the Answer, reinstated papers wait for a person, `bedrock-litreview integrity`, `/reviews settings`, `/reviews reinstate` |
| Background from reviews | A code-written report section quoting reviews with links; never counted as evidence |
| Fresh-inference determinism | Harness built (`bench determinism`, runbook in `docs/litreview/DETERMINISM_RUNBOOK.md`); not run on vLLM |
| Stage 9 (chat), PPTX, patents, drop folder | Not started |

Working and tested today, with no GPU and no network: everything above except
real-model quality. Run `pytest tests/litreview` for this package, `pytest` for
all of bedrock. `tests/litreview/test_sources.py` plays Europe PMC and PubMed
with a mock transport; nothing there needs a network.

A model-using review needs a context window of at least 12,288 tokens
(`min_context_window` in `defaults/models.yaml`): extraction sends about 4,300
prompt tokens and allows a 4,096-token reply. `service.preflight` refuses to
start otherwise and says what to change, and `SchedulerClient` stops a run
whose server reports reading fewer prompt tokens than the prompt can hold
(`PromptTruncated`): the Ollama app's default window is 4,096 on a 16 GB Mac,
and it truncates silently.

Benchmark: `python -m bedrock.litreview.bench run evals/litreview/questions/*.yaml --models laptop`.
Read `scorecard.md`'s "where the pipeline lost points" before changing a prompt,
and bump the prompt's `version` when you do. `--from-run <id>` scores a report
an earlier version of the code wrote, which is how before/after is measured.

Every fix so far came from that list, and the failures it found are worth
knowing before writing a prompt: see
[BENCHMARK_RESULTS.md](../../../docs/litreview/BENCHMARK_RESULTS.md) § What broke.

## Who owns what

1. **Collector** — sources, dedupe, acquisition, GROBID, drop folder
2. **Reader** — screening, extraction, the gold set, accuracy
3. **Presenter** — setup form, report, review screen
4. **Foreman** — orchestration, cache, determinism, storage, chat tools

## If you are about to write extraction code

Build `evals/litreview/goldset.jsonl` first. See `evals/litreview/README.md`.
Everything in Stage 5 is guesswork until it exists, and a prompt tuned without it will be excellent on
the papers you happened to look at and mediocre everywhere else.
