---
version: 4
job: extract
model_tier: medium
---

# Extraction prompt

<!--
Person 2 owns this file. Bump `version` in the front matter on every change —
it is recorded on each EvidenceUnit, so we can always tell which prompt
produced a given finding, and the eval harness reports scores per version.

v2: question-driven. The model now sees the scientist's question, the exact
entity with its near-misses, and the facets the answer needs; findings carry
`intervention`, `setting` and `facet`. `quote` moved first in the schema so the
model copies its sentence before it fills the fields from it.

v3: the model copies `change` (what was done to the target) and `effect` (what
happened to the disease) out of its quote, and code checks `direction` and
`outcome` against them (s5_extract.reconcile). Fixes MedGemma 4B reading
"PGGT1B deficiency aggravated the psoriasis-like lesions" as `improved`, and
listing a cell line ("HaCaT cells") as a target. Rule 3 and the third example
are new.

v4: the paper's text is now the last thing in the prompt, after the examples.
With the examples last, MedGemma 4B returned the guselkumab example as its
finding for a HuR-peptide abstract (quote verification discarded it, leaving
the paper with no findings); with the text last, the same abstract gave six
verified findings.

Keep the rules section short. Long rule lists get skimmed by small models the
same way they get skimmed by people. Every rule below earned its place by
fixing a failure — if you add one, note which failure it fixes.
-->

You are extracting research findings from one part of one scientific paper,
to help answer a scientist's question.

Question: {{question}}

The question is about **{{subject}}** exactly. Names that count: {{exact_terms}}.
{{near_miss_line}}

This paper was chosen because it may supply: {{job_facets}}.

## Information the answer needs (use these ids for `facet`)

{{facets}}

## Rules

1. **Copy the exact sentence** you took each finding from into `quote`. One
   sentence, word for word. Never paraphrase, never join two sentences, never
   tidy the wording. A sentence that is not in the text is discarded automatically.
2. **Only this paper's own results about {{subject}}.** Skip findings about any
   other disease, virus, drug or species, even closely related ones. Skip
   sentences that report other people's studies ("previous studies showed…",
   "has been reported [12]").
3. **`direction` is what was done to the target; `outcome` is what then
   happened to {{subject}}.** Deleting, knocking out, knocking down or
   inhibiting a target is `block`, even when the disease got worse. If the
   disease got worse ("aggravated", "exacerbated", "more severe") the outcome is
   `worsened` — that means the target protects against {{subject}}. Report
   these, and results where nothing changed (`no_effect`): they matter as much
   as `improved`. Copy the words for each into `change` and `effect`.
4. **Observations without an intervention** ("IL-17A was elevated in lesions")
   are findings too: `outcome` = `association`, `direction` = `unclear`.
5. `setting`: `prevention` if the intervention was given before onset, exposure
   or challenge, or to prevent relapse; `treatment` if given to established
   disease; otherwise `unclear`.
6. `evidence_type`: `human_clinical` for patients or volunteers,
   `human_genetic` for genetic association in people, `in_vivo` for animals,
   `in_vitro` for cells or tissue in a dish.
7. Take `species` and `n` from the Methods context when the sentence does not
   state them.
8. If no sentence states a finding relevant to the question, return an empty
   list. That is a correct and common answer. At most 8 findings.
9. If this text summarises other people's work rather than reporting its own,
   set `is_review` to true and return no findings.

## Output fields

- `quote` — the verbatim sentence
- `target` — the gene, molecule, pathway or cell type acted on or measured, as
  written in the paper. Never a cell line (HaCaT, HEK293), never {{subject}} itself
- `change` — the words from the quote saying what was done to the target
  ("deficiency", "knockout", "blocking"), or "" if nothing was
- `direction` — `block`, `activate`, or `unclear`
- `modality` — `antibody`, `peptide`, `siRNA`, `small_molecule`, `vaccine`, `genetic`, `other`
- `intervention` — the drug, antibody, vaccine or agent used, as written, or null
- `disease` — the disease or model, as written
- `evidence_type`, `species` (as written)
- `effect` — the words from the quote saying what happened to the disease
  ("reduced", "aggravated", "did not significantly reduce"), or "" if none
- `outcome`, `setting`
- `effect_detail` — the concrete result, with numbers exactly as the paper gives them
- `n` — subjects or animals, or null
- `combo_with` — the second target if the finding is about a combination, else null
- `target_is_primary` — false if the paper is about something else and mentions this in passing
- `facet` — one id from the list above

## Examples

<!--
Person 2: replace these with real examples drawn from evals/goldset.jsonl once
it exists. Pick them to cover the failures you expect, not the easy cases.
-->

**A positive human result**

Sentence: "At week 12, PASI-75 was achieved by 78% of patients receiving
guselkumab versus 8% receiving placebo (n=507)."

```json
{"findings": [{"quote": "At week 12, PASI-75 was achieved by 78% of patients receiving guselkumab versus 8% receiving placebo (n=507).",
"target": "IL-23", "change": "receiving guselkumab", "direction": "block", "modality": "antibody",
"intervention": "guselkumab", "disease": "plaque psoriasis", "evidence_type": "human_clinical",
"species": "patients", "effect": "PASI-75 was achieved by 78%", "outcome": "improved", "setting": "treatment",
"effect_detail": "PASI-75 in 78% vs 8% placebo at week 12", "n": 507,
"combo_with": null, "target_is_primary": true, "facet": "clinical"}], "is_review": false}
```

**A negative result, which must still be reported**

Sentence: "Anti-IL-23 treatment did not significantly reduce ear thickening
compared with vehicle control (p=0.41)."

```json
{"findings": [{"quote": "Anti-IL-23 treatment did not significantly reduce ear thickening compared with vehicle control (p=0.41).",
"target": "IL-23", "change": "Anti-IL-23 treatment", "direction": "block", "modality": "antibody",
"intervention": "anti-IL-23", "disease": "psoriasis-like inflammation", "evidence_type": "in_vivo",
"species": "mice", "effect": "did not significantly reduce", "outcome": "no_effect", "setting": "unclear",
"effect_detail": "no significant reduction in ear thickening, p=0.41", "n": null,
"combo_with": null, "target_is_primary": true, "facet": "negative"}], "is_review": false}
```

**Blocking a target made the disease worse: `block` and `worsened`**

Sentence: "Keratinocyte-specific SOCS1 deficiency aggravated imiquimod-induced
psoriasis-like skin inflammation in mice."

```json
{"findings": [{"quote": "Keratinocyte-specific SOCS1 deficiency aggravated imiquimod-induced psoriasis-like skin inflammation in mice.",
"target": "SOCS1", "change": "deficiency", "direction": "block", "modality": "genetic",
"intervention": null, "disease": "psoriasis-like skin inflammation", "evidence_type": "in_vivo",
"species": "mice", "effect": "aggravated", "outcome": "worsened", "setting": "unclear",
"effect_detail": "keratinocyte SOCS1 loss aggravated psoriasis-like inflammation", "n": null,
"combo_with": null, "target_is_primary": true, "facet": "causal"}], "is_review": false}
```

**Nothing to report**

Sentence: "Cytokine signalling has been reviewed extensively elsewhere."

```json
{"findings": [], "is_review": false}
```

## Methods context

<!-- Species and sample size live in Methods, outcomes live in Results.
     Pasting Methods in as background is what stops us producing findings
     that have an outcome but no species, or vice versa. -->

{{methods_text}}

## Text to extract from

Section: {{section_name}}

{{section_text}}

Now extract the findings from the text above — this paper's own results only, never the examples.
