"""Argument parsing for slash commands, in a program that must not exit.

``argparse`` is the right tool for ``--project X --since 2026-07-01``: it
gives predictable options, useful messages and no invented syntax. Its one
unacceptable habit is calling ``sys.exit`` on a typo, which inside a Textual
application means the employee's terminal closes mid-sentence.

:class:`CommandParser` turns both of argparse's exits into
:class:`CommandUsageError`, which the command handlers catch and show as an
ordinary error notice. ``shlex`` splits the words, so a quoted path with
spaces in it works the same way it would in a shell - without ever invoking
one.
"""

from __future__ import annotations

import argparse
import shlex


class CommandUsageError(ValueError):
    """The employee typed something this command cannot read."""


class CommandParser(argparse.ArgumentParser):
    def __init__(self, prog: str, **kwargs) -> None:
        # ``add_help`` off: ``-h`` would print to stdout and exit, and these
        # commands document themselves with their own help verb instead.
        super().__init__(prog=prog, add_help=False, **kwargs)

    def error(self, message: str) -> None:  # type: ignore[override]
        raise CommandUsageError(f"{self.prog}: {message}")

    def exit(self, status: int = 0, message: str | None = None) -> None:  # type: ignore[override]
        raise CommandUsageError(message or f"{self.prog}: could not read those arguments")


def split_arguments(text: str) -> list[str]:
    """Shell-style word splitting, so quoted file names survive intact."""
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise CommandUsageError(f"Could not read those arguments: {exc}") from exc
