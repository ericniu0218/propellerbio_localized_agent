# Benchmark — If I want to prevent psoriasis, what areas should I target?

Question id `psoriasis-prevention-targets` · report `8be152cae817d2e5` · 20260923T203929Z

| Metric | This pipeline | Chatbot, no papers | Chatbot + pasted abstracts |
|---|---|---|---|
| **Overall (0–100)** | 90.00 | 69.10 | 54.50 |
| Claims | 58 | 19 | 22 |
| Accuracy: claims supported by their citations | 1.00 | – | 0.45 |
| Groundedness: claims supported by the collected papers | 1.00 | 0.79 | 0.93 |
| Numbers present in the evidence | 1.00 | – | – |
| Claims mentioning a near-miss entity | 0 | 0 | 0 |
| Completeness: review facts covered (reachable) | 0.83 | 0.50 | 0.33 |
| Completeness: vital review facts covered | 0.83 | 0.50 | 0.33 |
| Completeness: reference targets named (reachable) | 0.50 | 0.21 | 0.00 |
| Detail: supported claims with numbers | 22 | 0 | 0 |
| Distinct sources cited | 40 | 0 | 8 |

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
| sentences_kept | 58 |
| sentences_dropped | 3 |
| evidence_units | 130 |
| papers_read | 11 |
| cited_papers_passing_gate | 1.00 |
| distractor_leak_rate | 0.00 |
| judge_catches_number | 1.00 |
| guard_or_judge_catches_number | 1.00 |
| judge_catches_flip | 1.00 |
| guard_or_judge_catches_flip | 1.00 |
| judge_catches_entity | 1.00 |
| guard_or_judge_catches_entity | 1.00 |
| perturbations | 5 |
| deterministic | yes |

## Notes

- 7 reference facts from reviews; 6 supported by the collected papers.
- 37 reference targets (Open Targets); 14 named in the collected papers. Pipeline raw recall against all of them: 0.19.
- The judge is medgemma1.5:4b, the same model family that wrote the report; its sensitivity is measured by the perturbation rows.
- 47 support verdict(s), across all systems, were settled in code because the claim is a verbatim passage of its evidence; the judge model made 87 call(s). In 0 the model's sentence was not in the evidence and code found the closest one; 0 detail(s) it called missing were in the evidence and struck off.

## Where the pipeline lost points

Every pipeline claim was judged supported by its citations.

Reference facts the collected papers support but the report does not state:

- (vital) Stress increases proinflammatory cytokines.

Reference targets named in the collected papers but not in the report: IL17RA, IL1RL2, PDE4B, PDE4D, TNF, GLP1R, JAK2

Sentences removed by the citation guard, by reason: number 100 is not in the cited evidence ×2; names ×1

## Chatbot + pasted abstracts: examples of failures

- not in any collected paper: Targeting iron regulation, potentially through hepcidin inhibitors or iron chelators, could be a strategy.

## Chatbot, no papers: examples of failures

- not in any collected paper: Receptors: Blocking the IL-23 receptor (IL-23R) (e.g., ustekinumab) is another major strategy, as IL-23 drives Th17 cell differentiation and IL-17 production.
- not in any collected paper: Lipids: Modulating the composition of the stratum corneum lipids is another potential strategy.
- not in any collected paper: VEGF: Inhibiting Vascular Endothelial Growth Factor (VEGF) (e.g., bapineuzumab, although development was halted) can reduce angiogenesis.
- not in any collected paper: While animal models (mice) are essential for understanding disease biology and testing drug candidates, the ultimate validation relies on human evidence.
