---
version: 1
job: write
model_tier: medium
---

# Task: summarise what competitors' patent filings say they cover

<!--
Bump `version` on every change. v1.

Writes ONE target's part of the competitor landscape from the filings found for
that target. Like write_section.md, the rules are not the safety mechanism: the
patent guard (report/patent_guard.py) checks every sentence in code and drops
what it cannot support. The rules are here so fewer sentences get dropped.

The one rule that is new, and the whole reason this is a separate prompt: a
patent says what its filer CLAIMS or DESCRIBES. It does not say the thing works.
Every sentence must therefore say whose claim it is, in verbs like "claims",
"describes", "recites", "is directed to", and none may state an outcome as fact.
-->

A scientist asked: {{question}}

You are summarising the patent filings found for **{{target}}**.

## Filings you may use

Each item is one filing. Its Quote is the filing's own words.

{{evidence}}

## Rules

1. For each sentence, first list in `citations` the ids of the filings it
   relies on, then write the sentence from those items' Quote, Covers and Kind.
2. **Say whose claim it is.** Use "claims", "describes", "recites", "discloses",
   "is directed to" or "covers": "Janssen's application WO... claims a method of
   treating plaque psoriasis by...". Never write that something works, is
   effective, is safe or "shows" an outcome: a filing is not evidence.
3. Use no outside knowledge. Name no company, drug, target, disease or number
   that is not in the items you cite. The company is the Filer shown on the item.
4. A filing marked "application" is pending; only one marked "granted" may be
   called granted.
5. Copy numbers exactly as they appear in the items.
6. Say what kind of thing is claimed (an antibody, a peptide, a method of
   treatment) only as the item says it.
7. Write {{sentences}} sentences, in plain English, in the third person. Do not
   mention the item ids in the text; the citations carry them.
8. Every sentence must add a new fact. Do not repeat a sentence. Group filings
   that claim the same thing into one sentence naming the filers.
9. If the items do not support a clear statement, say what they do show and
   stop. Fewer, correct sentences are better than more.
