---
version: 6
job: write
model_tier: medium
---

<!--
Writes ONE report section from ONE slice of evidence. Never the whole report:
that is the long, vague task where a 30B model falls over.

Every sentence comes back with the evidence ids it relies on (citations are
generated first, and guided decoding restricts them to the ids in this
slice). report/guard.py then checks each sentence: numbers and names must
appear in what it cites, or the sentence is dropped. So the rules below are
not the safety mechanism — the guard is. They are here so fewer sentences
get dropped.

v2, from the MedGemma 4B benchmark run: the model wrote the same sentence up to
four times in a row, turned labels into claims ("activating SLC15A3 improved
patient outcomes", from a mouse study), and copied "we observed…" from the
papers. Rules 1, 4 and 7-9 are new or rewritten.

v3: v2 gave each item a code-written `Meaning` line and the direction/outcome
labels. MedGemma copied them into the report word for word ("an observation:
nothing was blocked or activated") and turned bad labels into claims
("Blocking participants improved psoriasis"). Items now carry only what was
studied, the quote and the result; the labels stay with code.

v4: items no longer carry "Result", the extraction model's own summary of the
finding. Sentences written from it carried numbers and subjects ("in human
tissues") no quote contained, and the guard now checks against the paper's
text only, so they were dropped anyway. Items carry the quote and, where the
paper names the subjects in another sentence, that sentence as "Context".

v5, from the first report a scientist ran (tirzepatide and liver fat, qwen3:8b):
the Answer listed p-values ("FLI significantly decreased (p < 0.001)") while
the quotes beside them gave the changes those p-values tested (baseline to
3 months, with units). Rule 3 now asks for the change, its comparison and its
time point together, and never a bare p-value when the quote gives the change.
Everything it asks for is copied from a quote, so the guard can check it.

v6, from a scientist's review of the same report (1 October): the Answer gave
"an absolute decrease of 8.09%" for a fall in liver fat measured in percent,
without the comparator's change, the between-group difference or the pooled
doses, and said a small observational study "led to" improvements. Rules 11-13
ask for the comparison the quote gives, percentage points for a change in a
percentage, and association wording for non-randomised studies (each item now
says its design). Code removes "led to" from such sentences anyway.
-->

# Task: write one section of a research report

A scientist asked: {{question}}

The question is about **{{subject}}** exactly.

You are writing the section **{{section_title}}**{{subsection}}.
{{instructions}}

## Evidence you may use

Each item is one finding. Its Quote, and its Context when there is one, are
the paper's own sentences.

{{evidence}}

## Rules

1. For each sentence, first list in `citations` the ids of the items it relies
   on, then write the sentence from those items' **Quote** (and **Context**):
   every sentence restates what a cited quote reports.
2. Use no outside knowledge. Do not name any drug, target, disease, number or
   study that does not appear in the items you cite.
3. Copy numbers exactly as they appear in the items. When a quote gives the
   size of an effect, keep it together with what it was compared with
   (placebo, another drug, baseline) and when it was measured, in the
   quote's own words - for example "fell from 97.7 kg at baseline to 85.7 kg
   after 3 months". Never give a p-value on its own when the quote also
   gives the change it refers to.
4. Say who or what was studied — patients, mice, cells — only in the words
   the item's Quote or Context uses, so animal and cell results are never read
   as human ones. If an item says "who was studied: not stated", do not name
   any species or population for it.
5. Report results where something did not work, or made {{subject}} worse, as
   plainly as results where it helped. Say "did not" or "no effect" only when a
   cited item says so.
6. Write {{sentences}} sentences, in plain scientific English, in the
   third person. Do not mention the evidence ids in the text; the citations
   carry them.
7. Every sentence must add a new fact. Never repeat or rephrase a sentence you
   have already written.
8. Do not write vague sentences such as "X improved outcomes". State what the
   paper measured and what happened.
9. Do not copy "we" or "our" from a paper: write "the authors" or name the
   finding directly.
10. If the items do not support a clear conclusion, say what they do show and
    stop. Fewer, correct sentences are better than more.
11. When a quote gives the comparator's result, the difference between groups
    or its confidence interval, include them beside the treatment's result. If
    the quote says doses were pooled ("pooled 10 mg and 15 mg groups"), say so.
12. A change in something already measured in percent (liver fat fraction,
    HbA1c) is a change in **percentage points**: write "fell by 8.09
    percentage points", not "fell by 8.09%", unless the quote says it is a
    relative change.
13. When an item's design says it is not randomised, write "was associated
    with" or "changed from baseline" - never "led to", "resulted in" or
    "caused".
