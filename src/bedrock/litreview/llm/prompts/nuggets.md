---
version: 2
job: judge
model_tier: medium
---

<!--
Benchmark only. Turns an external reference — a published review's abstract —
into "nuggets": short, atomic facts that a good answer to the question should
contain. Each nugget carries the verbatim sentence it came from, verified in
code, so the reference itself cannot be hallucinated.

v2: v1 returned definitions and epidemiology ("Psoriasis affects 2-3% of the
global population", "Stress can worsen psoriasis") as `vital` facts for a
question about what to target. A fact must now name something one could act
on; code drops facts that name no molecule, pathway, cell type or agent.
-->

# Task: list the key facts this reference gives in answer to a question

Question: {{question}}

Reference ({{reference_label}}):
{{reference_text}}

List up to {{max_nuggets}} key facts from the reference that help answer the
question about {{subject}}. Each fact:

- `fact` — one short, self-contained statement (under 25 words), e.g.
  "Blocking IL-23 improves plaque psoriasis in patients."
- `quote` — the sentence of the reference it comes from, copied word for word
- `importance` — `vital` if an answer that left it out would be incomplete,
  otherwise `okay`

Every fact must name a specific molecule, pathway, cell type, drug or vaccine,
and say what it does in {{subject}} or what acting on it does — the kind of
fact a scientist choosing what to target needs. Definitions, prevalence,
symptoms, quality of life and risk factors such as stress are not key facts.

Only facts stated in the reference and about {{subject}}. No facts about other
diseases. Return an empty list if the reference does not help answer the
question.
