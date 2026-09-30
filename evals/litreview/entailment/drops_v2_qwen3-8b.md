# Entailment check: sentences it would remove

2 report sentences. For each, read the sentence and the paper's text it cites.
Each finding's quote is the sentence the report cites for its result; the context, next and
naming sentences are shown beside it in the report and may add a name, who was studied, a
sample size or a limit.

Mark one: **right** (the cited quote does not support the sentence, so removing it is correct) or
**wrong** (the quote supports it and removing it loses real evidence), and a note if useful.

## 1. qwen_asthma S34

> The Penh value of the OVA-challenged mice was higher than the control group, indicating increased airway hyperresponsiveness.

- E59, about AHR:
  - quote: "The effects of GRI977143 and H2L5186303 on AHR were measured. Results showed that the Penh value of the OVA-challenged mice was higher than the control group."

The check's reason: The sentence claims that the higher Penh value indicates increased airway hyperresponsiveness, but this interpretation is not stated in the quote.

Mark: right / wrong    Note:

## 2. qwen_asthma S41

> Montelukast reduced asthma exacerbations in children with high-moderate pre-treatment severity, as shown in a study of 45 participants randomized to montelukast and 44 to placebo.

- E233, about montelukast:
  - quote: "Key Results: Among 45 participants randomized to montelukast and 44 to placebo, median [IQR] ages were 6.6 [5.4, 10] and 6.7 [5.3, 9.9] years, with high-moderate pre-treatment exacerbation severity."
  - context: "Background and Purpose: Children with acute asthma exacerbations are frequently hospitalized due to insufficient response to inhaled β-agonist and systemic corticosteroid potentially due to leukotriene mediated airway inflammation."

The check's reason: The sentence claims that montelukast reduced asthma exacerbations, but the quote only states that participants had high-moderate pre-treatment severity, without mentioning any reduction in exacerbati…

Mark: right / wrong    Note:
