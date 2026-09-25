# Benchmark — What should I target to prevent RSV infection in infants?

Question id `rsv-infant-prevention-targets` · report `8dbd0d3c6296d04d` · 20260923T203932Z

| Metric | This pipeline | Chatbot, no papers | Chatbot + pasted abstracts |
|---|---|---|---|
| **Overall (0–100)** | 90.60 | 70.30 | 84.40 |
| Claims | 53 | 15 | 13 |
| Accuracy: claims supported by their citations | 0.98 | – | 0.77 |
| Groundedness: claims supported by the collected papers | 1.00 | 0.73 | 1.00 |
| Numbers present in the evidence | 1.00 | 0.33 | 1.00 |
| Claims mentioning a near-miss entity | 0 | 0 | 0 |
| Completeness: review facts covered (reachable) | 0.71 | 0.50 | 0.75 |
| Completeness: vital review facts covered | 0.90 | 0.40 | 0.90 |
| Completeness: reference targets named (reachable) | – | – | – |
| Detail: supported claims with numbers | 25 | 1 | 2 |
| Distinct sources cited | 34 | 0 | 9 |

## Pipeline checks

| Check | Result |
|---|---|
| quotes all verified (1.00) | PASS |
| every sentence cited | PASS |
| no near-miss claims | PASS |
| numbers all in cited evidence | PASS |
| no distractor leaked | PASS |
| same report twice | PASS |
| claims supported by citations >= 0.85 | PASS |
| judge catches flipped effects >= 0.7 (benchmark validity) | PASS |

| Measure | Value |
|---|---|
| quote_validity | 1.00 |
| citation_coverage | 1.00 |
| sentences_kept | 53 |
| sentences_dropped | 1 |
| evidence_units | 74 |
| papers_read | 11 |
| cited_papers_passing_gate | 1.00 |
| distractor_leak_rate | 0.00 |
| judge_catches_number | 1.00 |
| guard_or_judge_catches_number | 1.00 |
| judge_catches_flip | 1.00 |
| guard_or_judge_catches_flip | 1.00 |
| judge_catches_entity | 0.00 |
| guard_or_judge_catches_entity | 1.00 |
| perturbations | 6 |
| deterministic | yes |

## Notes

- 12 reference facts from reviews; 12 supported by the collected papers.
- The judge is medgemma1.5:4b, the same model family that wrote the report; its sensitivity is measured by the perturbation rows.
- 35 support verdict(s), across all systems, were settled in code because the claim is a verbatim passage of its evidence; the judge model made 107 call(s). In 1 the model's sentence was not in the evidence and code found the closest one; 0 detail(s) it called missing were in the evidence and struck off.

## Where the pipeline lost points

Claims the judge did not find fully supported by their citations:

- [unsupported] (targets:YLL) The estimated years of life lost (YLL) exceeded 268.8 per 100,000 older adults and 271.8 per 100,000 infants in the upper range of 95% uncertainty. — *most of what it says is not in the evidence (45% of its words). Judge: The evidence states YLL exceeded 280.2 per 100,000 older adults and 311.5 per 100,000 infants, which is different from the claim's values of 268.8 and 271.8.*

Reference facts the collected papers support but the report does not state:

- (okay) Monthly palivizumab injection provides passive immunization for high-risk infants.

Sentences removed by the citation guard, by reason: names ×1

## Chatbot, no papers: examples of failures

- not in any collected paper: Nirsevimab (Beyfortus): A newer humanized mAb targeting G protein (G2).
- not in any collected paper: Therefore, targeting the G protein with mAbs like Palivizumab or Nirsevimab is the most established and effective approach for preventing RSV infection in infants.
