"""Every `/reviews` verb is discoverable, in both app modes.

`settings` and `reinstate` shipped working in agent-pack mode and listed in
neither of that mode's help tables, so the retraction configuration was
undiscoverable there: the feature existed and nothing said so. The cause was
three hand-maintained copies of the same table - `REVIEWS_HELP`,
`ui/commands.py`'s `/help`, and `agentpack_app.py`'s - and adding a verb meant
remembering all three.

They are now generated from `REVIEWS_VERBS`, so this file's job is to keep that
list honest against the code that actually dispatches: a verb someone adds to
`run_reviews_command` and forgets to describe fails here, which is the drift
that happened.

The claims *about* the commands are tested too. "reads local files only" was
true when it was written and became false when `settings` started saving a
preference and `reinstate` started recording a decision. A scientist reads that
line to decide whether a command is safe to try, so it has to keep matching
what the commands do.
"""

from __future__ import annotations

import re

import pytest

from bedrock.agentpack_app import AGENTPACK_HELP_TEXT, AGENTPACK_REVIEW_NOTICE
from bedrock.ui.commands import HELP_TEXT
from bedrock.ui.review_command import REVIEWS_HELP, REVIEWS_VERBS, reviews_rows

ALL_TABLES = {
    "/reviews help": REVIEWS_HELP,
    "default-mode /help": HELP_TEXT,
    "agent-pack /help": AGENTPACK_HELP_TEXT,
}


def dispatched_verbs() -> set[str]:
    """The verbs `run_reviews_command` really dispatches, read from its source.

    From the source rather than a hand-written list, so a verb added there and
    described nowhere is caught. `help` is excluded: it is how you reach the
    table, not a row in it.
    """
    import inspect

    from bedrock.ui import review_command

    body = inspect.getsource(review_command.run_reviews_command)
    found = set(re.findall(r'text\.casefold\(\)\.startswith\("(\w+)"\)', body))
    found |= set(re.findall(r'text\.casefold\(\) in \{"(\w+)"', body))
    return found - {"help"}


def test_every_dispatched_verb_is_described_once():
    described = {v.verb for v in REVIEWS_VERBS if v.verb}
    missing = dispatched_verbs() - described
    assert missing == set(), (
        f"/reviews dispatches {sorted(missing)} but REVIEWS_VERBS does not describe "
        "them, so they appear in no help table and nobody can find them."
    )


def test_no_described_verb_has_been_removed_from_the_code():
    """The other direction: a row for a verb that no longer exists sends
    someone to a command that answers "I do not know"."""
    described = {v.verb for v in REVIEWS_VERBS if v.verb}
    stale = described - dispatched_verbs()
    assert stale == set(), f"REVIEWS_VERBS lists {sorted(stale)}, which /reviews no longer dispatches"


@pytest.mark.parametrize("verb", [v for v in REVIEWS_VERBS if v.compact])
def test_the_compact_verbs_reach_both_app_modes(verb):
    """The actual regression: working in agent-pack mode, listed nowhere."""
    for name, table in ALL_TABLES.items():
        assert f"`{verb.command}`" in table, f"{verb.command} is missing from {name}"


def test_the_full_table_lists_everything():
    for verb in REVIEWS_VERBS:
        assert f"`{verb.command}`" in REVIEWS_HELP


def test_the_tables_are_generated_not_retyped():
    """If someone hand-edits a row back in, the generated block stops matching
    and this fails - which is the point of generating them."""
    assert reviews_rows(compact=True) in HELP_TEXT
    assert reviews_rows(compact=True) in AGENTPACK_HELP_TEXT
    assert reviews_rows() in REVIEWS_HELP


def test_no_table_still_claims_reviews_only_reads():
    """`settings` saves a preference; `reinstate` records a decision."""
    assert any(v.writes for v in REVIEWS_VERBS), "the claim below depends on this"
    for name, table in ALL_TABLES.items():
        for claim in ("`/reviews` read local files only", "and `/reviews` read local files only"):
            assert claim not in table, f"{name} still says /reviews only reads"


def test_the_agent_pack_refusal_does_not_call_reviews_read_only():
    assert "only reads" not in AGENTPACK_REVIEW_NOTICE


def test_the_agent_pack_help_does_not_dangle_a_missing_review_row():
    """It used to end "see `/review` below" with no `/review` row anywhere."""
    assert "`/review` below" not in AGENTPACK_HELP_TEXT
    assert "| `/review <question>` |" in AGENTPACK_HELP_TEXT, (
        "agent-pack help should say what /review is and that this mode cannot run it"
    )


def test_the_agent_pack_review_row_says_where_it_does_work():
    row = next(line for line in AGENTPACK_HELP_TEXT.split("\n") if line.startswith("| `/review <question>`"))
    assert "Not available" in row
    assert "bedrock-litreview ask" in row
