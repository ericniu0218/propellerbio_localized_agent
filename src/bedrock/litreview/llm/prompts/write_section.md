---
version: 4
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
3. Copy numbers exactly as they appear in the items.
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
