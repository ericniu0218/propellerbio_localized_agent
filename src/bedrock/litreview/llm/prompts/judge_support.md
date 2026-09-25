---
version: 3
job: judge
model_tier: medium
---

<!--
Benchmark only. Finds where the evidence states one claim. Code decides the
verdict (bench/judge.verdict_from): the sentence must be in the evidence,
every name and number in the claim must be in the evidence, and the claim
must not say the effect went the other way from the sentence.

v3: v1 asked for a verdict — MedGemma 4B rejected claims its own reasoning
said were stated. v2 asked "is the main point stated?" — it said yes to
nearly everything, including claims with the effect reversed. A small model
can find a sentence; code can check it.
-->

# Task: where does the evidence state this claim?

Claim:
"{{claim}}"

Evidence (quoted from research papers):
{{evidence}}

## Answer each field, in order

1. `reasoning`: one short sentence comparing the claim with the evidence.
2. `evidence_sentence`: copy, word for word, the one sentence from the evidence
   that states what the claim says — the same molecules or drugs, and the same
   direction of effect (better, worse, no change). If no sentence states it,
   write an empty string. Never write a sentence of your own.
3. `missing`: the details in the claim that the evidence does not state — a
   number, a drug, a species, a population. Copy each one word for word from
   the claim. An empty list if every detail is in the evidence.

Use only the evidence above, not what you know.
