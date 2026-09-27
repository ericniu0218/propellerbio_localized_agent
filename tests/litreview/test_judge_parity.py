"""Issue #24 — judge/reader display parity.

`report/guard.py`'s `_shown` decides which of the paper's own sentences the
*reader* sees beside a citation: the quote, the context sentence, the
qualifying next sentence and the naming sentence. `bench/run.py`'s
`_unit_evidence` is supposed to show the *judge* exactly those same four
sentences and no more — its docstring explains the incident this guards
against: the judge used to also see the extraction model's own written
fields ("Result", "subjects"), and so scored a claim "supported" for
something the reader could never have checked against the paper themselves.

Both functions read exactly `quote, context_quote, qualifier_quote,
naming_quote` today, but nothing in the code keeps them in sync — a change to
one function's field list is not required to touch the other's. So this test
does not hand-write that field list a second time (`unit.quote`,
`unit.context_quote`, ...): that copy could itself drift from whichever list
`_shown` actually reads. Instead it takes the sentences straight out of
`_shown`'s own return value and checks each one turns up in
`_unit_evidence`'s text. Add a fifth field to `_shown` (and populate it on a
unit) without also adding it to `_unit_evidence`, or the reverse, and this
test fails the moment that field carries a marker `_shown` reports but
`_unit_evidence` does not. The `len(shown) == 4` assertion is the other half
of the tripwire: adding a fifth displayed field trips it, so whoever adds one
has to come here and decide deliberately whether the judge sees it too.
"""

from bedrock.litreview.bench.run import _unit_evidence
from bedrock.litreview.models import EvidenceUnit
from bedrock.litreview.report.guard import _shown


def _unit(**kw) -> EvidenceUnit:
    """One evidence unit with a distinct, unmistakable marker sentence in
    each of the four fields the reader is shown beside a citation. Matches
    the minimal-unit style of `tests/litreview/test_rests_on_quote.py`;
    offsets are not read by anything this test calls, so they are only made
    valid."""
    base = dict(
        id="u1", paper_id="P1", question_id="q", target="MC4R", target_raw="MC4R", direction="unclear",
        modality="other", intervention=None, disease="obesity", disease_raw="obesity", evidence_type="in_vivo",
        species="mouse", species_raw="mice", outcome="association", effect_detail="", n=None, combo_with=None,
        target_is_primary=True, quote_status="exact", section="results",
        extracted_by_model="fake", prompt_version="1", provenance={"target": "quote"},
        quote="Marker sentence MARK-QUOTE-7f2a reports the treatment's own result.",
        context_quote="Marker sentence MARK-CONTEXT-9c1d supplies who was studied.",
        context_start=0, context_end=10,
        qualifier_quote="Marker sentence MARK-QUALIFIER-3b6e qualifies that result.",
        qualifier_start=0, qualifier_end=10,
        naming_quote="Marker sentence MARK-NAMING-5e8f names the target or agent.",
        naming_start=0, naming_end=10,
    )
    base.update(kw)
    return EvidenceUnit(**base)


def test_every_sentence_shown_to_the_reader_reaches_the_judge():
    unit = _unit()
    shown = _shown(unit)

    # Confirms the fixture actually exercises all four fields `_shown` reads
    # today (and that they are distinct markers, not accidental substrings of
    # one another) - not the drift check itself, just its precondition.
    assert len(shown) == 4
    assert len(set(shown)) == 4

    judge_text = _unit_evidence(unit)
    for sentence in shown:
        assert sentence in judge_text, (
            f"the reader is shown {sentence!r} beside this citation, "
            "but the judge in bench/run.py's _unit_evidence is not"
        )
