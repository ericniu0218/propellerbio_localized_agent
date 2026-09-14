"""Slash-command parsing.

Kept as plain functions with no Textual imports so they are trivial to test.
The app decides what to *do* with a command; this module only recognises it.
"""

from __future__ import annotations

from dataclasses import dataclass

HELP_TEXT = """\
**Commands**

| Command | What it does |
|---|---|
| `/model` | List available models |
| `/model <name>` | Switch to a different model |
| `/digest <path>` | Compress a file so a model can read it cheaply |
| `/digest` | List and search what you have compressed |
| `/clear` | Start a new conversation |
| `/help` | Show this help |
| `/exit` | Quit (also `Ctrl+C`) |

**Keys**

- `Enter` sends your message
- `Esc` stops a response that is still streaming
- `Ctrl+L` clears the screen (same as `/clear`)
"""


@dataclass(frozen=True)
class Command:
    name: str
    args: str


KNOWN_COMMANDS = {"model", "digest", "clear", "help", "exit", "quit"}


def parse_command(text: str) -> Command | None:
    """Return a ``Command`` if ``text`` is a slash command, otherwise ``None``.

    Anything starting with ``/`` is treated as a command attempt, even if the
    name is unknown - the app reports unknown commands rather than sending
    them to the model by mistake.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    name, _, args = stripped[1:].partition(" ")
    return Command(name=name.lower(), args=args.strip())
