"""How text is made safe to show in the terminal.

`elide_path` is here because a real message lost the thing it existed to say.
"Saved at /very/long/directory/…" tells the reader nothing they can act on:
the file name is the part they need, and it was the part being cut off.
"""

from bedrock.ui.rendering import as_code_path, elide_path, safe_line

# The exact shape that found the bug: a pytest temporary directory under a long
# user name. 164 characters, against a 160-character limit. The same path under
# a short user name is 154 and never truncates, which is why this passed in CI
# and failed on one developer's machine.
LONG_PATH = (
    "/private/var/folders/2w/237wzl7x5vxb6jxtw3tzdchr0000gn/T/"
    "pytest-of-alejandrojarquin/pytest-51/test_pymol_fetches_a_structure0/"
    "work/sources/563bb95eccdb-STRUCT-9.pdb"
)


def test_a_short_path_is_left_alone():
    assert elide_path("/home/maya/work/STRUCT-9.pdb") == "/home/maya/work/STRUCT-9.pdb"


def test_a_long_path_keeps_its_file_name():
    """The whole point: the reader can still see which file was written."""
    rendered = elide_path(LONG_PATH)
    assert rendered.endswith("563bb95eccdb-STRUCT-9.pdb")
    assert len(rendered) <= 160


def test_a_long_path_keeps_its_beginning_too():
    """So the reader can tell which tree it landed in, not just the name."""
    assert elide_path(LONG_PATH).startswith("/private/var/folders/")


def test_the_elision_is_visible():
    assert "…" in elide_path(LONG_PATH)


def test_end_truncation_would_have_lost_the_name():
    """Pins why this helper exists rather than reusing safe_line."""
    assert not safe_line(LONG_PATH).endswith("563bb95eccdb-STRUCT-9.pdb")


def test_a_file_name_longer_than_the_limit_falls_back_to_end_truncation():
    """Nothing left to protect, so do the ordinary thing rather than return a
    string that is all ellipsis."""
    rendered = elide_path("/x/" + "a" * 300, limit=40)
    assert len(rendered) <= 40


def test_control_characters_never_survive_a_path():
    """A path is still untrusted text: it reaches a terminal."""
    assert "\x1b" not in elide_path("/home/maya/\x1b[31mred.pdb")


def test_as_code_path_wraps_in_a_span_that_cannot_close_early():
    assert "`" not in as_code_path("/home/maya/we`ird.pdb")[1:-1]
