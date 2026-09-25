# Benchmark — What should I target to prevent RSV infection in infants?

Question id `rsv-infant-prevention-targets` · report `f27c5b46768c915b` · 20260924T035923Z

| Metric | This pipeline | Chatbot, no papers | Chatbot + pasted abstracts |
|---|---|---|---|
| **Overall (0–100)** | 67.90 | 36.70 | 32.80 |
| Claims | 53 | 15 | 9 |
| Accuracy: claims supported by their citations | 0.96 | – | 0.00 |
| Groundedness: claims supported by the collected papers | 0.96 | 0.20 | 0.89 |
| Numbers present in the evidence | 1.00 | 0.00 | 1.00 |
| Claims mentioning a near-miss entity | 0 | 0 | 0 |
| Completeness: review facts covered (reachable) | 0.00 | 0.36 | 0.00 |
| Completeness: vital review facts covered | 0.00 | 0.25 | 0.00 |
| Completeness: reference targets named (reachable) | – | – | – |
| Detail: supported claims with numbers | 15 | 0 | 1 |
| Distinct sources cited | 32 | 0 | 0 |

## Pipeline checks

| Check | Result |
|---|---|
| quotes all verified (1.00) | **FAIL** |
| every sentence cited | PASS |
| no near-miss claims | PASS |
| numbers all in cited evidence | PASS |
| no distractor leaked | PASS |
| same report twice | PASS |
| claims supported by citations >= 0.85 | PASS |
| judge catches flipped effects >= 0.7 (benchmark validity) | PASS |

| Measure | Value |
|---|---|
| quote_validity | 0.96 |
| citation_coverage | 1.00 |
| sentences_kept | 53 |
| sentences_dropped | 8 |
| evidence_units | 54 |
| papers_read | 16 |
| cited_papers_passing_gate | 1.00 |
| distractor_leak_rate | 0.00 |
| judge_catches_number | 1.00 |
| guard_or_judge_catches_number | 1.00 |
| judge_catches_flip | 1.00 |
| guard_or_judge_catches_flip | 1.00 |
| judge_catches_entity | 1.00 |
| guard_or_judge_catches_entity | 1.00 |
| perturbations | 6 |
| deterministic | yes |

## Notes

- 16 reference facts from reviews; 7 supported by the collected papers.
- The judge is file:answers.jsonl, not the model that wrote the report. Its answers are checked by the same code as any model's: the sentence it cites must be in the evidence, and the claim's names and numbers must be too.
- 32 support verdict(s), across all systems, were settled in code because the claim is a verbatim passage of its evidence; the judge model made 95 call(s). In 0 the model's sentence was not in the evidence and code found the closest one; 0 detail(s) it called missing were in the evidence and struck off.

## Where the pipeline lost points

Claims the judge did not find fully supported by their citations:

- [unsupported] (targets:nirsevimab) These mutations were found in the F genes of RSV A and B subtypes. — *no sentence of the evidence states it. Judge: this sentence does not name RSV A and B subtypes*
- [contradicted] (clinical) Maternal smoking exposure was reported in 27.0% of mothers. — *the evidence sentence says the effect went the other way. Judge: *

Reference facts the collected papers support but the report does not state:

- (vital) The long-acting monoclonal antibody nirsevimab and the maternal RSV prefusion F vaccine have recently been approved for RSV immunisation.
- (vital) The bivalent RSVpreF vaccine given in pregnancy transfers protective RSV IgG antibodies across the placenta.
- (vital) Long-acting monoclonal antibodies for infants are a recent advance against severe RSV lower respiratory tract infection.
- (vital) Extended half-life monoclonal antibodies nirsevimab and clesrovimab allow large-scale protection of all infants born in the winter season.
- (okay) Palivizumab given monthly was until recently the only RSV immunisation, limited to high-risk infants.
- (okay) Monthly palivizumab injection provides passive immunisation for high-risk infants and reduces RSV-related hospitalisations.
- (okay) Palivizumab is a monoclonal antibody indicated for the highest risk infants.

Sentences removed by the citation guard, by reason: names ×8

## Chatbot + pasted abstracts: examples of failures

- not in any collected paper: To prevent RSV infection in infants, several strategies are being explored, focusing on targeting key viral components or pathways.

## Chatbot, no papers: examples of failures

- not in any collected paper: Okay, focusing on preventing RSV infection in infants, particularly high-risk ones, the primary strategy involves targeting viral entry.
- not in any collected paper: The main targets are the RSV G proteins (G1 and G2), which are essential for viral entry into host cells.
- not in any collected paper: Monoclonal antibodies (mAbs) directed against these G proteins are the cornerstone of prevention.
- not in any collected paper: Target: RSV G protein (specifically G1 and G2) binding site.
- not in any collected paper: Palivizumab (Synagis): An older humanized mAb targeting G protein.
