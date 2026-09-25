"""The benchmark: is a report good enough, with no scientist in the loop yet?

A general chatbot fails a scientist in three ways the client named: it
leaves things out, it states things the papers do not say (often from its
own memory), and it drifts to things that are merely close to what was
asked. Each failure gets its own measurement here, each measurement is
reference-based or mechanical, and the same measurements are applied to two
chatbot-style baselines run on the same local model — so "better than a
chatbot" is a number, not a claim.

    accuracy       every claim, judged against the exact evidence it cites
                   (a local model doing entailment, a far narrower job than
                   writing), plus mechanical checks: quotes re-verified
                   against the papers, numbers present in the cited evidence
    groundedness   every claim, judged against the best-matching passages of
                   the whole collected corpus: a claim no paper supports came
                   from somewhere else — the model's memory
    exactness      near-miss entities mentioned in the answer; and distractor
                   papers — real abstracts with the entity swapped for a
                   near-miss plus one passing mention — injected to see whether
                   any reach the evidence
    completeness   key facts ("nuggets") from published review articles, and
                   clinically validated targets from Open Targets, counted
                   only where the collected papers actually contain them, so
                   the pipeline is not blamed for the searcher's recall
    consistency    the same question over the same papers gives the same bytes

And the benchmark checks itself: supported claims are deliberately corrupted
(a number changed, an effect flipped, the entity swapped for a near-miss) and
the judge and guard must catch them. A judge that passes corrupted claims is
not measuring anything, and the scorecard says so.

User ratings, when they exist, are stored beside these scores (`ratings`
table) so the two can be compared; until then, this is the check.
"""
