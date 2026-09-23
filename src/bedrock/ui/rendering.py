"""Turning values from disk into text that is safe to show in a terminal.

Two hazards this file exists to remove, both of them caused by the same
thing: run manifests, file names and project labels are *data the employee
chose*, and they end up inside Markdown widgets and a terminal.

1. Terminal control characters. A file called ``report\x1b[2J.fasta`` would
   otherwise clear the screen when its name is printed. Everything that comes
   off disk goes through :func:`safe_line` first.
2. Markdown. A name containing a backtick or a pipe would otherwise break out
   of its code span or its table cell. :func:`as_code` and :func:`as_cell`
   handle those two placements, which are the only two we use.

These are deliberately plain functions with no Textual import, for the same
reason ``bedrock.ui.commands`` is: they are the easiest thing in the codebase
to test, and the hardest to reason about by eye.
"""

from __future__ import annotations

import re

# C0 and C1 control characters, plus DEL. Tab and newline are included on
# purpose: every value we render this way belongs on exactly one line.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")

# Unicode direction overrides can visually reorder a file name so that what
# the employee reads is not what is on disk.
_BIDI = re.compile(r"[‪-‮⁦-⁩]")


def safe_line(value: object, limit: int = 160) -> str:
    """One line of plain text, with nothing in it that a terminal will obey."""
    text = _BIDI.sub("", _CONTROL.sub(" ", str(value)))
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def as_code(value: object, limit: int = 160) -> str:
    """A Markdown code span. Backticks are dropped rather than escaped -
    a file name with a backtick in it is vanishingly rare, and closing the
    span early is the failure we actually have to prevent."""
    return f"`{safe_line(value, limit).replace('`', '')}`"


def elide_path(value: object, limit: int = 160) -> str:
    """A filesystem path shortened from the middle, so the file name survives.

    ``safe_line`` cuts the end off, which is right for a sentence and wrong for
    a path: "saved at /very/long/directory/…" has told the reader nothing they
    can act on, because the name of the file is the part they need. Keeping the
    front as well means they can still see which tree it landed in.

    Found by a test that passes or fails depending on how long the developer's
    user name is - a temporary directory under "pytest-of-alejandrojarquin"
    pushes the path past the limit where "pytest-of-runner" does not. Ordinary
    home directories are short enough that this rarely triggers in real use,
    which is exactly why it went unnoticed.
    """
    text = _BIDI.sub("", _CONTROL.sub(" ", str(value)))
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    name = text.rsplit("/", 1)[-1]
    # A file name that is itself longer than the limit leaves nothing to
    # protect, so fall back to the ordinary end-truncation.
    if len(name) + 2 >= limit:
        return safe_line(text, limit)
    return f"{text[: limit - len(name) - 2].rstrip()}…/{name}"


def as_code_path(value: object, limit: int = 160) -> str:
    """A Markdown code span holding a path, keeping its file name visible."""
    return f"`{elide_path(value, limit).replace('`', '')}`"


def as_cell(value: object, limit: int = 60) -> str:
    """Text inside a Markdown table cell, where ``|`` ends the cell."""
    return safe_line(value, limit).replace("\\", "\\\\").replace("|", "\\|")


def as_code_cell(value: object, limit: int = 60) -> str:
    return f"`{as_cell(value, limit).replace('`', '')}`"


def bullet_lines(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def safe_block(value: object, limit: int = 2000) -> str:
    """Multi-line text - an approval preview, a tool message - with control
    characters removed but its line breaks kept."""
    text = _BIDI.sub("", str(value))
    lines = [_CONTROL.sub(" ", line).rstrip() for line in text.splitlines()]
    block = "\n".join(lines).strip()
    if len(block) > limit:
        block = block[: limit - 1].rstrip() + "…"
    return block
