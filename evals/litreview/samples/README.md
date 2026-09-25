# Sample reports

One report per sample question, exactly as the pipeline wrote it, with the
scorecard that graded it. These are committed so anyone can see what the
system produces without a GPU, a corpus or an hour to spare.

| Question | Corpus | Report | Scorecard |
|---|---|---|---|
| If I want to prevent psoriasis, what areas should I target? | frozen (hand-written queries) | [report.md](psoriasis-prevention-targets/report.md) · [report.html](psoriasis-prevention-targets/report.html) | [scorecard.md](psoriasis-prevention-targets/scorecard.md) |
| What should I target to prevent RSV infection in infants? | frozen (hand-written queries) | [report.md](rsv-infant-prevention-targets/report.md) · [report.html](rsv-infant-prevention-targets/report.html) | [scorecard.md](rsv-infant-prevention-targets/scorecard.md) |
| If I want to prevent psoriasis, what areas should I target? | searcher (`litreview/sources/literature.py`) | [report.md](psoriasis-prevention-targets-searcher/report.md) · [report.html](psoriasis-prevention-targets-searcher/report.html) | [scorecard.md](psoriasis-prevention-targets-searcher/scorecard.md) |
| What should I target to prevent RSV infection in infants? | searcher (`litreview/sources/literature.py`) | [report.md](rsv-infant-prevention-targets-searcher/report.md) · [report.html](rsv-infant-prevention-targets-searcher/report.html) | [scorecard.md](rsv-infant-prevention-targets-searcher/scorecard.md) |

Written by MedGemma 1.5 4B through Ollama on a laptop. The frozen-corpus
reports read from `evals/corpora/`; the searcher-corpus reports read from
`evals/corpora-searcher/` (Europe PMC, preprints and PubMed, built by the
paper searcher merged in commit c5d7765). Both sets were scored by the same
strong (non-MedGemma) file-based judge. The numbers, and what they mean, are
in [BENCHMARK_RESULTS.md](../../BENCHMARK_RESULTS.md).

`report.html` is self-contained: open it in a browser and hover a citation to
see the quote it rests on.

To regenerate (about an hour per question on a laptop, minutes when the call
cache is warm):

```
python -m litreview.bench run evals/questions/*.yaml --profile medgemma
```
