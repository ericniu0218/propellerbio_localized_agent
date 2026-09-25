# Working in this repository

Read this before changing anything. `README.md` has the full project context;
this file is the short version plus the rules that are easy to break by accident.

## What this is

An air-gapped literature review pipeline for Propeller Bio, a Bay Area biotech
developing biologics (antibodies, peptides, siRNAs) that hit multiple disease
pathways at once. Focus areas: immunology & inflammation, obesity, CNS.

A scientist types a question ("if I want to prevent psoriasis, what should I
target?"). The paper searcher (`litreview/sources/literature.py`) collects
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

1. `litreview/models.py` — the data model. Everything is a view of `EvidenceUnit`
2. `litreview/extract/verify.py` — the fabrication guard
3. `litreview/extract/provenance.py` — the same guard, for every other field a finding carries
4. `litreview/report/guard.py` — the same guard, applied to report sentences
5. `litreview/report/invariants.py` — what code proves about a finished report
6. `litreview/knowledge/` — exact terms and near-misses, meaning words, and the matcher
7. `README.md` § The pipeline — what each stage owes the next

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
`litreview/knowledge/entities.yaml`, are applied by code, and are shown in the
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
`litreview ask` exit non-zero. `litreview verify-run` re-checks a stored run.
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
answer is the drop folder (`litreview/dropfolder.py`) — Propeller already pays
for journal access, and we ingest what their scientists legally download.

**Reviews are not evidence.** A review article citing a trial is not a trial.
Counting one would inflate the apparent weight of evidence.

**Negative results are first-class.** `no_effect` and `worsened` matter as much
as `improved`. Models skip these unless explicitly told not to.

## Conventions

- Stages are `litreview/stages/sN_name.py`, run in numeric order: s0_question,
  s1_collect, s2_gate, s3_rank, s4_plan, s5_extract, s6_score, s7_write,
  s8_render (s9_chat not built). `runner.py` calls them; nothing else sequences
- What each question type needs is `litreview/facets.py` (code). The report
  layout is `config/report.yaml`. Pipeline knobs are `config/pipeline.yaml`
- The paper searcher plugs in only through `litreview/sources/base.py`
- The searcher goes to the internet only when asked: `litreview search`,
  `ask --search`, `"search": true` over HTTP. Never make it a default — a
  question's papers are frozen once collected, and the air-gapped machine must
  work from a JSON-lines file. Its queries come from `exact_terms` and
  `search.focus` in `config/pipeline.yaml`; never let a model add a term
- Sources are listed in code (`SOURCES` in `sources/literature.py`), not read
  from `config/sources.yaml`, so nothing gets wired in by editing a config file
- Model names live only in `config/models.yaml`. Nothing else hardcodes one
- Prompts live in `litreview/llm/prompts/*.md` with a `version` in front matter.
  Bump it on every change — it is recorded on each EvidenceUnit
- Nothing outside `litreview/llm/` imports a concrete model client. Stages
  depend on the `LLMClient` protocol, so tests use `FakeLLMClient`
- Everything that can be tested without a GPU is. The whole pure-code layer runs
  offline, because in week one the office GPU does not exist yet

## Current state

| Part | State |
|---|---|
| Stages 0-8 (question → report) | Built and tested; runs end to end with `FakeLLMClient` |
| Paper searcher | Built and tested offline; run live. Europe PMC + preprints + PubMed, open-access JATS full text. GROBID/PDF rung, alias lookup and drop folder not built |
| Benchmark (`litreview/bench/`) | Built and tested with a fake model; sample corpora in `evals/corpora/` |
| Real-model benchmark results | Run on MedGemma 1.5 4B, both sample questions — see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) |
| Stage 9 (chat), PPTX, patents, drop folder | Not started |

Working and tested today, with no GPU and no network: everything above except
real-model quality. Run `pytest`. 217 tests, all passing (install the `api` extra, or the API test is skipped). `tests/test_sources.py`
plays Europe PMC and PubMed with a mock transport; nothing there needs a network.

Benchmark: `python -m litreview.bench run evals/questions/*.yaml --profile laptop`.
Read `scorecard.md`'s "where the pipeline lost points" before changing a prompt,
and bump the prompt's `version` when you do. `--from-run <id>` scores a report
an earlier version of the code wrote, which is how before/after is measured.

Every fix so far came from that list, and the failures it found are worth
knowing before writing a prompt: see BENCHMARK_RESULTS.md § What broke.

## Who owns what

1. **Collector** — sources, dedupe, acquisition, GROBID, drop folder
2. **Reader** — screening, extraction, the gold set, accuracy
3. **Presenter** — setup form, report, review screen
4. **Foreman** — orchestration, cache, determinism, storage, chat tools

## If you are about to write extraction code

Build `evals/goldset.jsonl` first. See `evals/README.md`. Everything in Stage 5
is guesswork until it exists, and a prompt tuned without it will be excellent on
the papers you happened to look at and mediocre everywhere else.
