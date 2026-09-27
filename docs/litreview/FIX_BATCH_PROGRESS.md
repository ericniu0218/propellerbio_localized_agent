# Literature review: mechanical fix batch

Eight small, well-understood fixes that need no scientist's judgment. Baseline
before the batch: `pytest tests/litreview` = 550 passed, 2 skipped.

Each item is implemented and tested on its own, then reviewed and committed
separately. Benchmark rerun happens only after items 1-6 land, because this
project's history is that the benchmark finds bugs the unit tests do not.

| # | Item | Touches | Status |
|---|------|---------|--------|
| 1 | Judge/reader display parity test | tests only | [x] |
| 2 | `stale_reasons()` note on verify-run | `runner.py`, `cli.py`, `ui/review_command.py` | [x] |
| 3 | Strip structured-abstract labels from displayed text | `extract/verify.py`, `s5_extract.py`, `extract/provenance.py` | [x] |
| 4 | Drop garbled duplicated-stem sentences | `stages/s7_write.py` | [x] |
| 5 | Strip leftover "However," ("Moreover," ruled out by measurement) | `stages/s7_write.py` | [x] |
| 6 | Guard: a sentence may not drop a hedge | `negation.py`, `report/guard.py` | [x] |
| 7 | A paper with no headings is not all "results" | `parsing/sections.py`, `extract/verify.py` | [x] |
| 8 | More evidenced other-work lead-in phrases | `extract/verify.py` | [x] |
| — | Benchmark rerun on all three corpora | `docs/litreview/BENCHMARK_RESULTS.md` | [ ] |
| — | `OPEN_ISSUES.md` updated | `docs/litreview/OPEN_ISSUES.md` | [ ] |

## Left for later, found while doing this batch

- `extract/provenance.py` `methods_subjects()` still stores its context
  sentence with the structured-abstract label attached (`"METHODS: ..."`), and
  `tests/litreview/test_methods_subjects.py`
  `test_a_structured_abstracts_methods_part_is_read` currently asserts that
  text. It is the same defect as item 3 on a separate code path, left alone
  deliberately rather than changed together with an assertion that encodes it.
- Guard check 10 (`rests_on_quote`) may have a separate live gap around
  obesity S18-shaped citations, where the quote reports one result and the
  claim actually rests only on the qualifier sentence. Found while building
  item 6's tests; `grades_after.jsonl` independently grades S18
  `weak_citation`. Not part of this batch and not investigated.
- **Numbered headings are never matched.** `parsing/sections.py`'s
  `_HEADING_MAP` patterns all anchor at `^\s*`, so "1. Introduction",
  "2. Materials and Methods" and "3. Results" match nothing. This is the real
  cause of item 7, found by fetching PMID41596081 and PMID41858165 rather than
  reasoning from the test fixture. Item 7 stops those papers being
  mis-filed as results; it does not recover their structure, and they are now
  abstract-only. Tolerating a numeral prefix would restore the coverage
  properly and is the natural next change.
- `"unknown"` is deliberately not added to `models.py`'s `Section` literal:
  under `defaults/pipeline.yaml`'s `sections: [abstract, results]` it can
  never reach an `EvidenceUnit`. If extraction is ever asked for more
  sections, that literal and `bench/judge.py`'s `CorpusIndex` both need it.
