---
version: 1
job: question
model_tier: medium
---

<!--
Reads the scientist's question. It does not answer it, and it never adds
search terms: every entity it returns is checked against the question text in
code, and anything not written there is discarded. Synonyms and near-misses
come from litreview/knowledge/entities.yaml, never from here.
-->

# Task: read a research question

A drug-discovery scientist typed the question below. Identify what it asks
about. Do not answer it.

Question: {{question}}

## Return

- `question_type`
  - `target_identification` — asks which targets, pathways, genes or mechanisms
    to act on ("what should I target", "which pathways")
  - `intervention_efficacy` — asks whether a specific drug, antibody, vaccine or
    approach works
  - `mechanism` — asks how something works
  - `safety` — asks about risks, side effects or toxicity
  - `other`
- `goal` — `prevent`, `treat`, `understand` or `other`
- `entities` — every disease, pathogen (virus, bacterium), drug, target (gene or
  protein) and species named in the question, each with its `kind`.
  **Copy each one exactly as written in the question** — same spelling, same
  abbreviation. Do not expand abbreviations, add synonyms, or add anything
  related that is not written in the question.
- `population` — who the question is about ("infants", "adults over 65"),
  copied exactly, or null

## Examples

Question: If I want to prevent psoriasis, what areas should I target?

```json
{"question_type": "target_identification", "goal": "prevent",
 "entities": [{"text": "psoriasis", "kind": "disease"}], "population": null}
```

Question: Does nirsevimab reduce RSV hospitalisation in preterm infants?

```json
{"question_type": "intervention_efficacy", "goal": "prevent",
 "entities": [{"text": "nirsevimab", "kind": "drug"}, {"text": "RSV", "kind": "pathogen"}],
 "population": "preterm infants"}
```
