---
version: 2
job: entail
model_tier: medium
---

<!--
The optional entailment check (report/entailment.py), off by default: a
scientist has not yet checked what it drops. `entailment_check = true` in
the [litreview] table turns it on. It runs after the citation guard, on
sentences the guard kept. The guard's checks are word-level - numbers,
names, who was studied, direction words - and cannot tell whether a
reworded sentence means what its quote means. This asks that one question
and nothing else.

Version 2 shows each quote apart from the sentences beside it and says the
sentence's claim must be in a quote. Version 1 showed all of them as one
passage, so a sentence copied from a finding's naming or next sentence while
citing a quote that says something else was "supported" (OPEN_ISSUES #3).

Code decides what the answer does: "not_supported" and "contradicted" drop
the sentence, "partial" keeps it and is counted, "supported" keeps it.
The reason comes first so the model compares before it labels.
-->

# Task: does the quote support this sentence?

Sentence from a report:
"{{sentence}}"

The paper's text it cites:
{{evidence}}

## Rules

- Judge only from the paper's text above, not from what you know.
- The sentence's main claim - its result or finding - must be stated in a
  "Quote". The sentences shown beside a quote (context, next sentence,
  naming sentence) may only supply a name, who was studied, a sample size or
  a limit. A result that appears only beside a quote, and not in any quote,
  is not supported.
- "about:" names what that finding concerns; the paper names it nearby.
- A sentence that adds anything the text does not state - a subject or
  population, a number, a direction of effect, a target or drug, or more
  certainty ("proves", "significantly", "in humans") - is not fully supported.
- A sentence that leaves out a limit the text states ("not significant", "in
  mice", "a trend") is not fully supported.

## Answer each field, in order

1. `reason`: one short sentence comparing the sentence with the quote.
2. `verdict`, one of:
   - `supported`: a quote states the sentence's claim, and the quote or the
     sentences beside it state everything else it says.
   - `partial`: a quote states the main claim, but the sentence adds or drops
     a detail.
   - `not_supported`: no quote states the sentence's main claim, even if a
     sentence beside it does.
   - `contradicted`: the text says the opposite of the sentence.
