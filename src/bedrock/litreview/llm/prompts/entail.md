---
version: 1
job: entail
model_tier: medium
---

<!--
The optional entailment check (report/entailment.py), off by default and
unmeasured: `entailment_check = true` in the [litreview] table turns it on.
It runs after the citation guard, on sentences the guard kept. The guard's
checks are word-level - numbers, names, who was studied, direction words -
and cannot tell whether a reworded sentence means what its quote means. This
asks that one question and nothing else.

Code decides what the answer does: "not_supported" and "contradicted" drop
the sentence, "partial" keeps it and is counted, "supported" keeps it.
The reason comes first so the model compares before it labels.
-->

# Task: does the paper's text support this sentence?

Sentence from a report:
"{{sentence}}"

The paper's text it cites:
{{evidence}}

## Rules

- Judge only from the paper's text above, not from what you know.
- "about:" names what that finding concerns; the paper names it nearby.
- A sentence that adds anything the text does not state - a subject or
  population, a number, a direction of effect, a target or drug, or more
  certainty ("proves", "significantly", "in humans") - is not fully supported.
- A sentence that leaves out a limit the text states ("not significant", "in
  mice", "a trend") is not fully supported.

## Answer each field, in order

1. `reason`: one short sentence comparing the sentence with the text.
2. `verdict`, one of:
   - `supported`: the text states everything the sentence says.
   - `partial`: the text states the main point, but the sentence adds or
     drops a detail.
   - `not_supported`: the text does not state the sentence's main point.
   - `contradicted`: the text says the opposite of the sentence.
