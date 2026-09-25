---
version: 2
job: judge
model_tier: medium
---

<!--
Benchmark only. Finds the sentence of an answer that states a given key fact.
Code decides coverage (bench/judge.coverage_from): the sentence must be in the
answer, name what the fact names, and not say the effect went the other way.

v2: v1 asked for a verdict, and MedGemma 4B called nearly every fact covered
by every answer.
-->

# Task: which sentence of this answer states this fact?

Fact: "{{fact}}"

Answer:
{{answer}}

## Answer each field, in order

1. `reasoning`: one short sentence.
2. `answer_sentence`: copy, word for word, the one sentence from the answer that
   states this fact — the same molecule, drug or pathway, and the same effect.
   If no sentence states it, write an empty string. Never write a sentence of
   your own.
