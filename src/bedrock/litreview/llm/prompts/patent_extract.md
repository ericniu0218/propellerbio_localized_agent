---
version: 1
job: extract
model_tier: medium
---

# Patent extraction prompt

<!--
Bump `version` on every change: it is recorded on each PatentUnit.

v1. The paper prompt (extract.md) asks for a paper's own *results*. A patent
has none to report: it says what the filer *claims or describes*, and whether
it works is not the patent's to say. So this prompt only ever copies what the
text says is covered, and the code that wraps it (stages/s6c_patent_extract.py)
treats every field as a claim about the filing, never about biology.

Narrow on purpose (CLAUDE.md, "Ask narrow questions"): one quote, one target,
and a few words copied from the text. Everything else a landscape needs - what
kind of claim 1 is, whether the target is the filing's subject, who the filer
is - is decided in code.
-->

You are reading the title, abstract and (when there is one) first claim of ONE
patent filing, to record what it says it covers. You are not judging whether it
works or whether it is valid. Copy; do not interpret.

A scientist asked: {{question}}

The filing was found by searching for: {{targets}}.

## Rules

1. **Copy the exact words** you rely on into `quote`: one sentence of the
   abstract, or one stretch of claim 1, word for word, never paraphrased and
   never joined from two places. A quote that is not in the text is discarded
   automatically. Say where it came from in `source`.
2. **`target`** is the molecule, gene or pathway the filing is directed at,
   exactly as the text writes it. If it names none of {{targets}}, return an
   empty list.
3. **`modality`** is what kind of thing is claimed: `antibody`, `peptide`,
   `siRNA`, `small_molecule`, `vaccine`, `genetic`, or `other`. Pick `other`
   when the text does not say.
4. **`covers`** is a short phrase copied from the text saying what the filing
   covers ("a bispecific antibody binding IL-23 p19 and CGRP"). Words from the
   text only.
5. **`indication`** is the disease or condition the text says it is for, as
   written, or null if it names none.
6. A filing that only lists the target among many ("cytokines such as IL-6,
   IL-17, IL-23") and is not directed at it: return an empty list.
7. At most 3 findings, one per target named. An empty list is a correct answer.

## Output fields

`quote`, `source` (`claim_1`, `abstract` or `title`), `target`, `modality`,
`covers`, `indication`.

## The filing

Title: {{title}}

Abstract: {{abstract}}

{{claim_block}}

Now return the findings for this filing only.
