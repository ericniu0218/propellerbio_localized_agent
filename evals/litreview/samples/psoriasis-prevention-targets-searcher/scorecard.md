# Benchmark — If I want to prevent psoriasis, what areas should I target?

Question id `psoriasis-prevention-targets` · report `dd8aa94928477944` · 20260924T035919Z

| Metric | This pipeline | Chatbot, no papers | Chatbot + pasted abstracts |
|---|---|---|---|
| **Overall (0–100)** | 72.90 | 22.50 | 56.30 |
| Claims | 78 | 19 | 13 |
| Accuracy: claims supported by their citations | 0.94 | – | 0.69 |
| Groundedness: claims supported by the collected papers | 0.96 | 0.00 | 0.77 |
| Numbers present in the evidence | 1.00 | – | – |
| Claims mentioning a near-miss entity | 0 | 0 | 0 |
| Completeness: review facts covered (reachable) | 0.00 | 0.33 | 0.00 |
| Completeness: vital review facts covered | 0.00 | 0.40 | 0.00 |
| Completeness: reference targets named (reachable) | 0.39 | 0.17 | 0.11 |
| Detail: supported claims with numbers | 6 | 0 | 0 |
| Distinct sources cited | 54 | 0 | 9 |

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
| judge catches flipped effects >= 0.7 (benchmark validity) | **FAIL** |

| Measure | Value |
|---|---|
| quote_validity | 1.00 |
| citation_coverage | 1.00 |
| sentences_kept | 78 |
| sentences_dropped | 1 |
| evidence_units | 155 |
| papers_read | 16 |
| cited_papers_passing_gate | 1.00 |
| distractor_leak_rate | 0.00 |
| judge_catches_flip | 0.67 |
| guard_or_judge_catches_flip | 0.67 |
| judge_catches_entity | 1.00 |
| guard_or_judge_catches_entity | 1.00 |
| perturbations | 7 |
| deterministic | yes |

## Notes

- 11 reference facts from reviews; 6 supported by the collected papers.
- 37 reference targets (Open Targets); 18 named in the collected papers. Pipeline raw recall against all of them: 0.19.
- The judge is file:answers.jsonl, not the model that wrote the report. Its answers are checked by the same code as any model's: the sentence it cites must be in the evidence, and the claim's names and numbers must be too.
- 52 support verdict(s), across all systems, were settled in code because the claim is a verbatim passage of its evidence; the judge model made 104 call(s). In 0 the model's sentence was not in the evidence and code found the closest one; 1 detail(s) it called missing were in the evidence and struck off.

## Where the pipeline lost points

Claims the judge did not find fully supported by their citations:

- [unsupported] (answer) IL-17 effectively suppressed core inflammatory pathways: IL-17, neutrophil extracellular trap formation, cytokine-cytokine receptor, and chemokine signaling in mice. — *no sentence of the evidence states it. Judge: evidence says three (unnamed) treatments suppressed a pathway list that includes IL-17 as an item being suppressed; it does not say IL-17 itself did the suppressing*
- [unsupported] (answer) CYSLTR1 was elevated expression of CYSLTR1 in skin lesions from patients with psoriasis and healthy controls with pigmented nevi. — *no sentence of the evidence states it. Judge: evidence sentence is only about tissue collection methodology; it does not state elevated expression, and does not mention pigmented nevi controls having elevated CYSLTR1*
- [partially_supported] (targets:IL17A) IL-17 biologics suppress active psoriasis. — *not in the evidence: psoriasis. Judge: *
- [unsupported] (targets:IL17A) IL-17 suppresses core inflammatory pathways in mice. — *no sentence of the evidence states it. Judge: evidence's subject is 'all three treatments', not IL-17; IL-17 is one of the pathways being suppressed, not the suppressor*
- [unsupported] (targets:CYSLTR1) Following treatment with montelukast cream and an intraperitoneal injection of montelukast reduced the PASI scores and delayed the disease progression, as evidenced by improvements in skin erythema, scaling, and thickness in the IMQ-induced psoriasis murine model. — *no sentence of the evidence states it. Judge: the cited evidence unit's own quote is about reduced immune-cell frequencies after montelukast, not PASI scores or disease progression; the correct sentence exists in the same paper but is not in this citation's evidence*

Reference facts the collected papers support but the report does not state:

- (vital) Psoriasis involves dysregulated cytokine signalling and altered keratinocyte function.
- (vital) Psoriasis pathogenesis involves keratinocytes, inflammatory cells and both adaptive and innate immunity.
- (vital) Keratinocytes act as inflammatory effectors and disease drivers, interacting with immune cells to amplify psoriatic inflammation.
- (vital) IL-23/IL-17 axis activation is a key pathogenic mechanism in psoriasis.
- (vital) IL-23 inhibitors such as risankizumab offer sustained efficacy in psoriasis.
- (okay) Psychological stress increases proinflammatory cytokines and ICAM-1 expression in psoriasis.

Reference targets named in the collected papers but not in the report: IL12B, IL1RL2, KEAP1, PDE4B, PDE4D, RXRA, VDR, HMGCR, JAK1, JAK2, JAK3

Sentences removed by the citation guard, by reason: names ×1

## Chatbot + pasted abstracts: examples of failures

- not in any collected paper: Preventing psoriasis requires targeting multiple pathways, as the disease is complex and relapse is common.
- not in any collected paper: The evidence presented is primarily from animal models, though some targets like hepcidin and epigenetic scars have human relevance.
- not in any collected paper: Targeting these specific molecular pathways or components could be a strategy to prevent psoriasis or its relapse.

## Chatbot, no papers: examples of failures

- not in any collected paper: Okay, let's focus on preventing psoriasis.
- not in any collected paper: While true *prevention* is challenging due to its complex interplay of genetics and environmental triggers, targeting key pathological pathways can significantly delay onset or manage early disease.
- not in any collected paper: Key areas to target involve modulating the immune response and restoring skin barrier function.
- not in any collected paper: Immune Dysregulation: This is central.
- not in any collected paper: Cytokines: Targeting pro-inflammatory cytokines like TNF-alpha (e.g., adalimumab, etanercept) and IL-17 (e.g., secukinumab, ixekizumab) is crucial.
