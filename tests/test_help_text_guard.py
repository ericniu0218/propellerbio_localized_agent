"""Guards against the class of bug review finding 9 found: nothing in this
suite ever compared what a mode's help text advertises against what the app
actually dispatches, which is how findings 6, 7 and 8 were able to happen at
all without a single red test.

- Finding 7: ``AGENTPACK_HELP_TEXT`` could name a command ``AgentPackApp``
  never runs, or the app could quietly grow a command nobody documented.
- Finding 6: the narrow case of the same thing - a notice pointing at a
  command the surrounding mode refuses.
- Finding 8: two near-identical constants in two files, describing the same
  fact, that drifted apart because nothing ever diffed them.

These tests read the real source with ``ast`` and ``inspect`` (the same
approach ``tests/test_model_call_guard.py`` and
``tests/test_import_boundaries.py`` use) rather than hand-maintaining a
second list of command names here - a hand-maintained list is exactly the
kind of duplicate copy that caused finding 8.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap

import bedrock.app as app_module
import bedrock.agentpack_app as agentpack_module
import bedrock.ui.local_commands as local_commands_module
import bedrock.ui.review_command as review_command
from bedrock.agentpack_app import AGENTPACK_HELP_TEXT
from bedrock.ui.commands import HELP_TEXT

# Any backtick-quoted token starting with "/", anywhere in a help text - not
# just at the start of a table row, so a command only mentioned in prose
# (e.g. "(also `/clear`)") still counts as documented.
_COMMAND_REF = re.compile(r"`/(\w[\w-]*)")

# Aliases every app dispatches on purpose but never advertises: `litreview`
# is the old name for `review`, `team` is the old name for `workspace`, and
# `quit` is `exit` by another name (`Ctrl+C` is the documented way to quit).
# A name added here needs the same justification - a real, working alias,
# not a command someone forgot to document.
DELIBERATELY_UNDOCUMENTED = {"litreview", "team", "quit"}


def documented_commands(help_text: str) -> set[str]:
    """Every command name a help text names in a backtick span, e.g.
    "`/fold <path>`" -> "fold"."""
    return set(_COMMAND_REF.findall(help_text))


def _pattern_names(pattern: ast.pattern) -> set[str]:
    """String literals one `case` pattern matches.

    `case "a" | "b":` is an `ast.MatchOr` of two `ast.MatchValue`s; a plain
    `case "a":` is just one. The wildcard `case _:` is an `ast.MatchAs` with
    no value to read, so it falls through to the empty set - which is
    exactly right, since it dispatches nothing by name.
    """
    if isinstance(pattern, ast.MatchOr):
        names: set[str] = set()
        for sub in pattern.patterns:
            names |= _pattern_names(sub)
        return names
    if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Constant):
        return {pattern.value.value}
    return set()


def dispatched_names_in_source(source: str) -> set[str]:
    """Every command name any `match ...:` block in `source` handles."""
    tree = ast.parse(textwrap.dedent(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Match):
            for case in node.cases:
                names |= _pattern_names(case.pattern)
    return names


def dispatched_commands(*funcs) -> set[str]:
    """The union of command names dispatched across one or more functions/methods."""
    names: set[str] = set()
    for func in funcs:
        names |= dispatched_names_in_source(inspect.getsource(func))
    return names


# ---------------------------------------------------------------------------
# Self-check: the extractors themselves behave as claimed above, on made-up
# source rather than the real app - so a change to how these functions work
# is caught here even if it happens to leave the real comparisons green.
# ---------------------------------------------------------------------------


def test_the_dispatch_extractor_reads_or_patterns_and_ignores_the_wildcard():
    source = '''
def run(self, command):
    match command.name:
        case "a" | "b":
            pass
        case "c":
            pass
        case _:
            pass
'''
    assert dispatched_names_in_source(source) == {"a", "b", "c"}


def test_the_documented_extractor_finds_backtick_commands_anywhere_in_the_text():
    text = "| `/foo <x>` | does foo | (also `/bar`) |\nSee `/foo --flag` for more."
    assert documented_commands(text) == {"foo", "bar"}


# ---------------------------------------------------------------------------
# Finding 7: AGENTPACK_HELP_TEXT <-> what AgentPackApp actually dispatches
# ---------------------------------------------------------------------------


def _agentpack_dispatch() -> set[str]:
    return dispatched_commands(
        local_commands_module.LocalCommandsMixin.run_local_command,
        agentpack_module.AgentPackApp.run_command,
    )


def test_every_agentpack_help_command_is_something_the_app_dispatches():
    documented = documented_commands(AGENTPACK_HELP_TEXT)
    dispatched = _agentpack_dispatch()
    stale = documented - dispatched
    assert stale == set(), (
        f"AGENTPACK_HELP_TEXT advertises {stale}, but AgentPackApp.run_command "
        "(by way of run_local_command or its own match) never dispatches it. "
        "Either the command was removed and the help row was not, or it was "
        "renamed on one side only."
    )


def test_every_agentpack_dispatched_command_is_documented_or_allowlisted():
    documented = documented_commands(AGENTPACK_HELP_TEXT)
    dispatched = _agentpack_dispatch()
    unexplained = dispatched - documented - DELIBERATELY_UNDOCUMENTED
    assert unexplained == set(), (
        f"AgentPackApp dispatches {unexplained}, but it is named neither in "
        "AGENTPACK_HELP_TEXT nor in DELIBERATELY_UNDOCUMENTED above. Add a row "
        "to AGENTPACK_HELP_TEXT, or add the name to the allowlist with a reason "
        "if it is a deliberate, unadvertised alias."
    )


# ---------------------------------------------------------------------------
# The same guard for the plain chat app's HELP_TEXT (bedrock.ui.commands)
# ---------------------------------------------------------------------------


def _bedrock_app_dispatch() -> set[str]:
    return dispatched_commands(
        local_commands_module.LocalCommandsMixin.run_local_command,
        app_module.BedrockApp.run_command,
    )


def test_every_chat_help_command_is_something_the_app_dispatches():
    documented = documented_commands(HELP_TEXT)
    dispatched = _bedrock_app_dispatch()
    stale = documented - dispatched
    assert stale == set(), (
        f"HELP_TEXT advertises {stale}, but BedrockApp.run_command never "
        "dispatches it."
    )


def test_every_chat_dispatched_command_is_documented_or_allowlisted():
    documented = documented_commands(HELP_TEXT)
    dispatched = _bedrock_app_dispatch()
    unexplained = dispatched - documented - DELIBERATELY_UNDOCUMENTED
    assert unexplained == set(), (
        f"BedrockApp dispatches {unexplained}, but it is named neither in "
        "HELP_TEXT nor in DELIBERATELY_UNDOCUMENTED above."
    )


# ---------------------------------------------------------------------------
# Finding 6, the narrow version: `/review` works in both modes, and the one
# notice that used to point at a refused command (the empty `/reviews` list)
# never again names anything but `/review` itself.
# ---------------------------------------------------------------------------


def test_review_is_dispatchable_in_both_app_modes():
    assert "review" in _agentpack_dispatch()
    assert "review" in _bedrock_app_dispatch()


def test_the_empty_reviews_notice_names_only_review_itself():
    """Read straight from the source that builds the notice, so this holds
    even without driving a full app with an empty database (both app-level
    tests in tests/test_review_command.py and tests/test_agentpack_app.py do
    that too, behaviourally; this pins the same fact structurally)."""
    source = inspect.getsource(review_command.run_reviews_command)
    assert "/review <question> --corpus <file.jsonl>" in source
    assert "/review <question> --search" in source
    assert "bedrock-litreview" not in source


# ---------------------------------------------------------------------------
# Finding 8: two notices describing "reviews don't work here" in two files,
# which drifted apart because nothing ever compared them. Both were deleted
# by the commit that let /review run in agent-pack mode; this keeps them
# deleted, under their old names or under the old wording.
# ---------------------------------------------------------------------------


def test_the_deleted_notice_constants_have_not_come_back():
    for name in ("AGENTPACK_NOTICE", "AGENTPACK_REVIEW_NOTICE"):
        assert not hasattr(agentpack_module, name), (
            f"{name} was deleted because it was a second, drifting copy of the "
            "same 'reviews don't work in agent-pack mode' fact AGENTPACK_NOTICE "
            "in review_command.py also stated - and that fact is no longer even "
            "true, since /review runs in both modes now. If this exists again, "
            "delete it and read the review it names its notice from instead."
        )
        assert not hasattr(review_command, name), (
            f"{name} was deleted from bedrock.ui.review_command for the same "
            "reason - see the message on the agentpack_module assertion above."
        )


def test_the_old_refusal_wording_has_not_come_back_anywhere():
    """The two deleted constants shared one distinctive clause. A literal
    reappearance of it - even under a new constant name - means the same
    false claim ("reviews need the default mode") has come back, which is
    exactly the drift finding 8 was about."""
    import bedrock

    package_dir = __import__("pathlib").Path(bedrock.__file__).parent
    fingerprint = "in the default mode (start `bedrock` without `BEDROCK_BACKEND=agentpack`)"
    offenders = [
        str(path)
        for path in sorted(package_dir.rglob("*.py"))
        if fingerprint in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        f"{offenders} still say reviews need the default mode - false since "
        "/review now runs in agent-pack mode too (see src/bedrock/ui/review_mixin.py)."
    )
