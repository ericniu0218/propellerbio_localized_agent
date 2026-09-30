# Literature review: quality and agent-pack audit (28–29 September 2026)

Branch `litreview/agentpack-and-report-quality`. What was wrong with the
reports a scientist got, why, what changed, and how it was measured. Read with
[OPEN_ISSUES.md](OPEN_ISSUES.md) and [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

## The report that started it

"Does tirzepatide reduce liver fat in adults with obesity?", qwen3:8b through
Ollama on a 16 GB Mac, 28 September. Two papers of 200 were used. The Answer
was four disconnected sentences of p-values ("FLI significantly decreased from
baseline to 3 months (p < 0.001)"), one of them a case report's background
sentence; abbreviations were never expanded; no sample size, design or
comparator was stated; five sections read "No evidence for this section"; and
a 118-row table of excluded papers sat in the middle of the report.

## Root causes, in pipeline order

| # | Where | What went wrong | Evidence |
|---|---|---|---|
| 1 | Model server | The Ollama app's default window on a 16 GB Mac is **4,096 tokens**. Extraction sends ~4,300 prompt tokens with a 4,096-token reply allowance; Ollama keeps the end of the prompt and drops the start - the instructions. `BEDROCK_CONTEXT_WINDOW=12288` in the server's environment made preflight pass. | `/api/ps` reported `context_length: 4096`; a 14,000-token probe was read as 2,050 tokens and answered wrongly |
| 2 | Stage 1 | A question's paper set was frozen by question *text*. A first run without a model read the question as "tirzepatide, obesity" and froze 200 papers; the model run read "liver fat" too and reused them. Liver fat was never searched for. | both runs have the same `papers_sha256` |
| 3 | Knowledge table | "liver fat" was not in `entities.yaml`, so it matched verbatim only: papers about MASLD, NAFLD, hepatic steatosis or MRI-PDFF were "not about the subject". | 198 of 200 gated out |
| 4 | Stage 0 / query | "in adults with obesity" made obesity a required entity, ANDed into every query. That dropped the randomised trial measuring liver fat by MRI (SURPASS-3 MRI, in type 2 diabetes) and the SYNERGY-NASH papers from the search itself. | Europe PMC: 94 hits with obesity required, 149 without; SURPASS-3 MRI only in the latter |
| 5 | Stage 5 | A finding filed under "obesity with MASLD" was dropped as "another disease" - obesity, which the question names. These were the Results sentences with the effect sizes; only the abstract's p-values survived. | 8 of 16 proposed findings, all quantitative |
| 6 | Stage 5 | "Tirzepatide has been also reported to be beneficial..." (a case report's background) passed the other-people's-work filter, which knew only "it has been reported". | E8 |
| 7 | Stages 4/7 | Among equally relevant papers, reading order ignored study design, and each section's representative finding per paper was the alphabetically first quote. The randomised trial ranked 21st of a 16-paper plan; "A total of 276 participants were included" was offered to the Answer ahead of the same paper's result, the guard rightly refused it, and the Answer was written from two mouse studies. | first "after" run |
| 8 | Stage 7 | The writer expanded "BMS" as "best medical standard"; the paper's BMS is bariatric metabolic surgery. The guard checks numbers, names and subjects, not expansions. | "after" run 2 |
| 9 | Report | Structure: no synthesis, no study profiles, no independence of studies, no glossary, audit trail in the reading view. | |

## What changed

Retrieval and extraction (`stages/`, `knowledge/`, `db.py`, `runner.py`):

- The frozen set records the search it was collected under (entities and exact
  terms, or a corpus hash). A different reading collects again when a search is
  asked for, and otherwise says so at the top of Limitations. Sets frozen
  before this are traced to the run that collected them. Failed search queries
  are recorded with the set and disclosed in every report built on it.
- `hepatic steatosis` in the knowledge table: liver fat, hepatic fat,
  steatosis, MASLD/MAFLD/NAFLD, MRI-PDFF, intrahepatic lipid, fatty liver index.
  MASH is neither a term nor a near-miss.
- A disease named only in the population is not required when another disease
  is; the population is shown per study instead.
- The question's own diseases are never "another disease"; "X has been
  (also) reported/described/suggested" is someone else's work.
- Within a relevance grade, reading order follows study design from the paper's
  own publication type and wording (`study_design.design_rank`), and each
  section's evidence leads with results about the question's own outcome
  (`report/synthesis.section_preference`).

Model calls (`llm/scheduled.py`, `service.py`, `inference/`):

- Preflight takes each model's window from the server where it reports one
  (vLLM, the agent server, a loaded Ollama model) and refuses a remote model
  whose window is unknown and unconfigured.
- Every call is checked after it returns: a reported prompt count below
  `len(prompt)/7` means the server truncated it, and the run stops with the fix
  (`PromptTruncated`), before anything is cached.

Report (`report/synthesis.py`, `stages/s7_write.py`, `s8_render.py`, `defaults/report.yaml`):

- Reading view, then References, then an Appendix (how the question was read,
  plan, ranking, evidence table, method, excluded papers, quotes, removed
  sentences). Empty sections are not headings.
- Code-written, from verified fields and the papers' own sentences only:
  **Evidence at a glance** (design, who, n, full text or abstract; publications
  of one trial grouped), **How the studies line up** (per outcome: independent
  studies, direction, and an evidence level by a written rule printed beside
  it), **Key results in context** (the papers' quantitative sentences with
  their study), **Abbreviations** (the paper's own definition first;
  statistical abbreviations keep their statistical meaning; conflicting
  definitions are shown as such). Every written section says which studies it
  rests on.
- Baseline descriptions, demographics, aims and cut-off quotes are kept out of
  the synthesis and the front of every section (they stay in the Appendix).
- The writer is told to keep each effect with its comparison and time point
  (prompt v5), and a sentence whose abbreviation expansion is not the paper's
  is removed with the reason.

## Before and after, same papers, same model

Corpus: 173 papers from 28 September's searches (Europe PMC was returning 503
for part of the night; the complete search's result list, with open-access
full text from an earlier attempt where the later one failed - 32 papers with
full text). qwen3:8b, temperature 0, a 12,288-token Ollama. Old code is
`de7e043`.

| | before | after |
|---|---|---|
| Papers about the exact subject | 7 | 75 |
| Papers read | 5 | 16 |
| Verified findings | 3 | 65 |
| Independent studies behind the findings | – | 11 (2 randomised trials; SURPASS-3's two papers counted once) |
| The Answer | "Tirzepatide has been reported to be beneficial in reducing liver fat content in patients with obesity" - a case report's background sentence | "... tirzepatide 10 mg and 15 mg significantly reduced liver fat content compared to insulin degludec, with an absolute reduction of -8.09% at week 52 versus -3.38%" (SURPASS-3 MRI), then the post hoc genotype result |
| Integrity check / `verify-run` | pass | pass |

Hand audit of the final report against the quotes (by the auditor, not by the
generating model): the Answer's two sentences are faithful to their quotes; the
liver-fat line of the synthesis ("Supported, but limited: one randomised trial,
two observational studies agreeing") matches the evidence table; the glossary's
BMS, LFC, CAP and HOMA-IR are the papers' own definitions. Known remaining
faults in it are listed below.

## Agent-pack mode (P0.1–P0.5)

`/review` works in agent-pack mode: the pipeline runs on the laptop and every
model call goes to the agent server's inference API (`/v1`, roles as model
names, JSON schema, seed, temperature 0; unsupported settings refused by name;
token required in real mode). `bedrock-litreview ask` with
`BEDROCK_BACKEND=agentpack` and no config file uses it; the local machine
profile no longer describes the remote model. `bedrock-litreview doctor` and
the first-run banner separate "server reachable" from "a review can run".
Setup: [docs/server/DEPLOYMENT.md § The inference API](../server/DEPLOYMENT.md).

Verified live on 29 September against a real `bedrock-server` (real mode,
token set, roles served by qwen3:8b through Ollama at a 12,288 window), with
`BEDROCK_BACKEND=agentpack` and no bedrock config file on the client:

- `/health` lists `inference_api`; `/v1/models` without the token is 401 and
  with it lists the roles, their weights and 12,288-token windows.
- `bedrock-litreview doctor`: a wrong key gives "Authentication: FAIL", the
  right key "Literature review: operational".
- `bedrock-litreview ask` on five papers (SURPASS-3 MRI and its post hoc
  analysis, SYNERGY-NASH, a real-world cohort, a mouse study) ran end to end
  through the server in 11.5 minutes: 5 graded, 5 read, 16 verified findings,
  and an Answer stating the trial's result - liver fat -8.09% vs -3.38% with
  insulin degludec at week 52, treatment difference -4.71% (95% CI -6.72 to
  -2.70) - as the paper's abstract gives it. The run records
  `specialist=qwen3:8b` per job; the server's audit table holds one
  metadata-only row per call.

The live run found two server faults the in-process tests had not, both fixed
and now tested: the API dropped "thinking off" for a role configured as having
no thinking switch, though qwen3:8b thinks, so two relevance replies were cut
off; and the per-call audit went to a Python logger nothing prints under
uvicorn - it is now written to the workspace audit table.
Everything else is covered by in-process tests that drive the real app, the
real HTTP client and the real server router
(`tests/server/test_agentpack_review.py`).

## Second pass: a scientist's review of report a67d8b766a9a2632

A reviewer read the report the Research tab produced for the same question and
listed where it misled. Each case became a general rule with a test
(`tests/litreview/test_claim_reliability.py`, cases named by evidence id), in
[report/claims.py](../../src/bedrock/litreview/report/claims.py) and
[knowledge/endpoints.yaml](../../src/bedrock/litreview/knowledge/endpoints.yaml):

- **Attribution.** "HFHFr feeding induced ... hepatic steatosis" had counted as
  tirzepatide *worsening* liver fat. A finding whose agent is a disease model
  or another drug, or that names no agent and does not name the drug, is not
  counted for the drug; it stays in the evidence table, and each synthesis line
  says what it did not count and why.
- **What was compared.** Genotype-subgroup nulls, arms that all got
  tirzepatide (diet A vs diet B) and change from baseline are no longer read
  as treatment versus control.
- **What was measured.** Findings are grouped by endpoint class: mortality and
  MALO are clinical events, not "liver fat"; an index estimated from blood
  tests is not an MRI measurement; a covariate list ("adjusted for ... hepatic
  steatosis") does not name the endpoint.
- **Numbers.** Middle-dot decimals (15·71) and signs are read as printed; a
  sentence that states a decrease as an unsigned change must say it fell. A
  reversed confidence interval or a p-value above 1 is flagged in Limitations
  and kept out of Key results, never repaired; one paper giving two estimates
  for one outcome is noted.
- **Basis, not verdicts.** "Supported, but limited" is replaced by what stands
  behind each line ("Randomised comparison: 1 trial", "Animal or cell studies
  only", "Associations only"), with disagreement judged only within human or
  within preclinical studies.
- **Comparators.** A near-miss drug may be named when the cited evidence names
  it, the question's drug is named, and the sentence makes a comparison
  ("more common with liraglutide than tirzepatide"). A sentence about the
  near-miss drug alone is still removed.
- **Parsing.** A literal "&lt;" in Europe PMC text no longer swallows the
  p-value and the next heading.

Re-run on the same 177 papers after these changes (qwen3:8b, 12,288 window,
35 minutes on the laptop): 16 read, 85 verified findings, 15 sentences kept,
integrity check clean. The diet-induced steatosis is no longer counted as
tirzepatide worsening liver fat (its quote still appears in Mechanism of
action, where it says what the diet did and that tirzepatide reversed it); the
subgroup null and the diet-arm comparison are listed under "Not counted" with
their reasons; MALO and mortality are under clinical events; the reversed
interval is flagged. The re-run exposed further faults, fixed and tested the
same way: the "Animal or cell studies only" line said "no human evidence" while
three human studies reported associations; a weight-loss result was filed
under liver fat through its covariate list; a baseline comparison of people
with and without diabetes, and a diet-versus-control rat result, reached Key
results; every finding set aside was given the same reason ("a disease model or
another agent"), now one of three (what the disease model did / another agent's
result / does not say the treatment caused it). A second re-run from the call
cache (the same model replies; only code-made parts change) shows each of these
fixed in the stored report, and `verify-run --all` passes on all six stored
runs, the original included.

The same review found the chat unusable after a report: both safety questions
were waiting at the high-stakes approval gate, shown only in the Activity
panel while the chat showed dots. The chat now shows the waiting draft inline
with Approve / Send back / Stop, reports a stalled, failed or interrupted run
as such (with Retry that does not repeat the question), and gives the model the
report's Answer, glance, synthesis and limitations plus the sections the
question is about (`tests/server/test_chat_lifecycle.py`).

## Still open

- **Writer faithfulness beyond names, numbers, populations and comparison
  words.** The guard checks what a sentence names; it does not parse its full
  logical structure. The contrast rules above keep the synthesis honest, but a
  written sentence can still over-generalise a quote in ways no rule names.
- **Wrong quote chosen by the extraction model.** qwen3:8b often quotes the
  sentence next to the result ("A total of 276 participants were included").
  Nothing false reaches the report, but the result is lost. A larger model or
  a prompt change measured on the gold set is the fix.
- **Tables and figures are not read**; 9 of 16 papers were abstract only here.
- The call cache stores a model's raw reply before it is parsed, so a reply
  cut off by a misconfigured server is replayed on every re-run of that
  question until the cache entry is removed (pre-existing).
- **Arm labels a paper never defines.** A rat study names its tirzepatide
  arms "TRZ1-3" without defining the abbreviation, so its liver-lipid results
  are set aside as another agent's. Guessing arm labels would break the
  exactness rule; a scientist-reviewed alias (or the paper's methods table,
  when tables are read) is the fix.
- **Tables are not parsed as tables.** JATS tables would give the numbers a
  figure-free structure; not done.
- **The claim rules are engineering expectations, not a scientist's
  adjudication.** They were written from one report; a gold set scored by a
  scientist is what measures them.
- Only one question was run live; the benchmark corpora were not re-scored.
