---
version: 3
job: relevance
model_tier: small
---

<!--
Reads ONE paper's title and abstract and answers narrow questions about it.
Code turns the answers into the 0-3 relevance grade (s3_rank.grade_from) and
decides which information needs the paper can supply (facets.paper_facets).

v3: the model no longer picks a grade or a list of needs. On the psoriasis
benchmark MedGemma 4B graded 39 of 40 papers 3 — including a study of breast
cancer risk in people with psoriasis — and ticked all six needs, in the order
listed, on 36 of 40. Narrow questions, one answer each, with the named molecule
checked against the abstract, are what a small model can do reliably.

This is relevance, not quality: nothing here asks the model to judge a paper.
The quote is verified against the title+abstract in code.
Field order matters under guided decoding: quote and reason come first.
-->

# Task: what does this paper's own work tell us about the question?

Question: {{question}}

The question is about **{{subject}}** exactly. These names count as it:
{{exact_terms}}.
{{near_miss_line}}

## The paper

Title: {{title}}
Publication type: {{publication_types}}

Abstract:
{{abstract}}

## Answer each field, in order

1. `quote`: copy, word for word, one complete sentence from the abstract (or the
   title) that best shows what the paper itself found. Never paraphrase, never
   shorten it.
2. `reason`: one short sentence — what the paper's own work did, and what it
   found.
3. `about_exact_entity`: true only if the paper's own work concerns {{subject}}
   itself. A paper about a related but different disease, virus, drug or species
   is false, even if it mentions {{subject}}.
4. `subject_role`: how {{subject}} figures in the paper's own work.
   - `studied` — {{subject}} is what the paper studies: its causes, biology,
     prevention or treatment
   - `exposure` — the paper studies {{subject}} only as a risk factor, a
     comorbidity or a cause of something else (for example, the risk of another
     disease in people who have {{subject}})
   - `passing` — {{subject}} is only mentioned, not studied
5. `study_kind`: `primary_research` if the paper reports its own experiments,
   trials or data; `review` if it summarises or pools other people's published
   studies (narrative reviews, systematic reviews, meta-analyses).
6. `tested`: {{tested_rule}}
