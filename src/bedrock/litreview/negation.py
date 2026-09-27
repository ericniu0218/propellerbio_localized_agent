"""Negation scope, shared by the pipeline and the benchmark judge.

A null result restated without its negation is the opposite finding, and
every word it uses is still in the paper: "did not significantly reduce" ->
"did significantly reduce". Word-presence checks cannot see that, so both the
result-summary check (`extract/provenance.py`) and the judge
(`bench/judge.py`) read negation here, by the same rule.

This is a bounded lexical check over the cues in `_NULL`, not entailment.
It misses negation it has no cue for ("absence of", "lacking" - left out
because "mice lacking IL-23" is a faithful "IL-23-deficient mice" - and double
negation) and reads only the three words after a cue. A false "dropped"
blanks a faithful summary; a missed one lets the opposite claim through, so
cues are added only with tests.
"""

from __future__ import annotations

import re

from bedrock.litreview.extract.verify import normalize as normalize_text

_BETTER = re.compile(
    r"\b(reduc|decreas|attenuat|ameliorat|alleviat|suppress|prevent|protect|inhibit|mitigat|reliev|resolv|"
    r"lower|milder|diminish|abolish|abrogat|improv|block)\w*", re.I)
_WORSE = re.compile(
    r"\b(aggravat|exacerbat|worsen|enhanc|increas|promot|accelerat|intensif|augment|exaggerat|elevat|higher|"
    r"amplif|sensiti[sz])\w*", re.I)
# "didn't" as well as "did not": `normalize` keeps contractions, and papers
# use both apostrophes.
# "but not in IL-23R-deficient mice" and "without changes in food intake"
# are cues too: a summary that drops them claims the result where the paper
# says it was absent. Not "with or without", which negates nothing. The words
# after a cue are read with their hyphens, so "IL-23R-deficient" is one word.
_NULL = re.compile(
    r"\b(?:did not|does not|do not|failed to|no significant|not significantly|no effect|no difference|"
    r"not associated|had no|was not|were not|but not|not in|(?<!or )without|neither|nor|"
    r"(?:did|does|do|was|were|had)n['’]t)\b((?:\s+[\w-]+){0,3})", re.I)


def _stems(pattern: re.Pattern[str], text: str) -> set[str]:
    return {m.lower() for m in pattern.findall(text)}


_STOP = frozenset(
    "that this with from have were which their these those into also been than more such both other "
    "study showed show shows results result indicate indicates suggest suggests paper papers evidence "
    "findings finding data using used there where when while however further".split()
)
SAME_TOPIC = 0.5
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _stem_set(text: str, drop: frozenset[str] = _STOP) -> set[str]:
    """Content words, cut to five letters so 'patient' meets 'patients'."""
    return {w[:5] for w in _WORD.findall(normalize_text(text)) if len(w) > 3 and w not in drop}


def _negated(match: re.Match[str]) -> set[str]:
    """What one negation cue governs, in the three words after it: its effect
    words if it has any ("not significantly reduce" -> reduc), else the first
    word ("no difference" -> diffe)."""
    after = match.group(1)
    effects = _stems(_BETTER, after) | _stems(_WORSE, after)
    governed = [w[:5] for w in _WORD.findall(normalize_text(after)) if len(w) > 3]
    return effects or set(governed[:1])


def drops_negation(claim: str, sentence: str) -> bool:
    """Does the claim state, without the negation, something this sentence negates?

    Every negation in the sentence is read, not only the first ("did not
    affect body weight and did not reduce ear thickening"). One is dropped
    when the claim does not negate what it governs, and:

    - the claim uses the negated effect word: "did not reduce" -> "reduced"
    - or the claim reuses the first word the negation governs, and is
      otherwise about the same thing (`SAME_TOPIC`): "no significant
      difference in AE types" -> "a significant difference in AE types"

    A negation that governs something the claim is not about does not count:
    "did not affect body weight, but reduced lesion area" -> "reduced lesion
    area" is a faithful restatement.
    """
    kept = set().union(*(_negated(m) for m in _NULL.finditer(claim)))
    claim_effects = _stems(_BETTER, claim) | _stems(_WORSE, claim)
    claim_stems = _stem_set(claim)
    for match in _NULL.finditer(sentence):
        if _negated(match) & kept:
            continue
        after = match.group(1)
        if (_stems(_BETTER, after) | _stems(_WORSE, after)) & claim_effects:
            return True
        # The effect-word lists cover no such verb ("difference", "receive"),
        # so read the negated word itself.
        governed = [w[:5] for w in _WORD.findall(normalize_text(after)) if len(w) > 3]
        if governed and governed[0] in claim_stems \
                and len(claim_stems & _stem_set(sentence)) / len(claim_stems) >= SAME_TOPIC:
            return True
    return False


# A paper that says a result is "potentially due to" some mechanism is
# speculating, not reporting it. Restating that mechanism as settled fact
# ("X is associated with Y") is the same kind of error as dropping a
# negation: real information the paper carries - that it is a guess, not a
# finding - goes missing. Asthma S25: the paper's background sentence says
# insufficient response to inhaled beta-agonist and corticosteroid was
# "potentially due to leukotriene mediated airway inflammation"; the report
# sentence stated the same thing as "is associated with", unhedged.
#
# This is the same bounded lexical check as `_NULL` above, over a small,
# closed set of epistemic-hedge cues - not entailment, and not a general
# hedge detector. It reads only the cues in `_HEDGE` and the few words after
# one, so a hedge with no cue here ("indicates a possible link", "suggests
# that") is invisible to it, same as `_NULL` misses negation it has no cue
# for. Cues are added only alongside a real example that needs them.
_HEDGE = re.compile(
    r"\b(?:potentially|possibly|presumably|may reflect|might reflect|could be due to|"
    r"is thought to|appears to|seems to)\b((?:\s+[\w-]+){0,6})", re.I,
)


def _hedged(match: re.Match[str]) -> set[str]:
    """What one hedge cue governs: the content words in the few words after
    it ("potentially due to leukotriene mediated airway inflammation" ->
    leuko, media, airway, inflam)."""
    after = match.group(1)
    return {w[:5] for w in _WORD.findall(normalize_text(after)) if len(w) > 3}


def drops_hedge(claim: str, sentence: str) -> bool:
    """Does the claim state, as settled fact, something this sentence only
    speculates about?

    Mirrors `drops_negation`. Every hedge cue in `sentence` is read. One is
    dropped when its governed words are also what the claim is about
    (`SAME_TOPIC`) and the claim does not cover those same words with a hedge
    of its own: "potentially due to leukotriene mediated airway inflammation"
    -> "leukotriene mediated airway inflammation is associated with ..." has
    dropped "potentially". A claim that keeps its own hedge over the same
    words is a faithful restatement and is left alone - "presumably due to
    the already rapid ... weight gain" restated with "presumably" intact is
    not penalized, even in different words, as long as it names the same
    governed content the source hedge names.

    Lexical, not entailment: a hedge the claim rephrases into different
    governing words it does not otherwise state is invisible to this check,
    same as `drops_negation`.
    """
    kept = set().union(*(_hedged(m) for m in _HEDGE.finditer(claim)))
    claim_stems = _stem_set(claim)
    if not claim_stems:
        return False
    for match in _HEDGE.finditer(sentence):
        governed = _hedged(match)
        if not governed or governed & kept:
            continue
        if governed & claim_stems \
                and len(claim_stems & _stem_set(sentence)) / len(claim_stems) >= SAME_TOPIC:
            return True
    return False
