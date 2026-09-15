"""Slash-command parsing.

Kept as plain functions with no Textual imports so they are trivial to test.
The app decides what to *do* with a command; this module only recognises it.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

HELP_TEXT = """\
**Commands**

| Command | What it does |
|---|---|
| `/model` | List available models |
| `/model <name>` | Switch to a different model |
| `/digest <path>` | Compress a file so a model can read it cheaply |
| `/digest` | List and search what you have compressed |
| `/workspace` | List private team workspaces |
| `/workspace help` | Show workspace actions |
| `/history` | List saved conversations |
| `/resume <number or name>` | Reopen a saved conversation, e.g. `/resume 2` or `/resume platypus` |
| `/new` | Start a new conversation (also `/clear`) |
| `/forget` | Delete this conversation from disk |
| `/help` | Show this help |
| `/exit` | Quit (also `Ctrl+C`) |

**Keys**

- `Enter` sends your message
- `Esc` stops a response that is still streaming
- `Ctrl+L` starts a new conversation (same as `/new`)
- `Ctrl+B` shows or hides the conversation list; click a conversation to open it

Conversations are saved on this machine as you go, and the most recent one
reopens the next time you start. Nothing is sent anywhere but your model server.
"""


@dataclass(frozen=True)
class Command:
    name: str
    args: str


KNOWN_COMMANDS = {
    "model",
    "digest",
    "workspace",
    "team",
    "clear",
    "new",
    "history",
    "resume",
    "forget",
    "help",
    "exit",
    "quit",
}


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


def parse_scientific_request(text: str) -> Command | None:
    """Recognize explicit requests to run locally integrated scientific tools."""
    stripped = text.strip()
    if not stripped or not _looks_actionable(stripped):
        return None
    lowered = stripped.casefold()
    path = _extract_supported_path(stripped)
    if (re.search(r"\balpha\s*fold\s*2?\b", lowered) and re.search(r"\b(fold|predict|prediction|model|run|use)\b", lowered)) or (path and _has_suffix(path, (".fasta", ".fa", ".faa")) and re.search(r"\b(fold|predict(?:ion)?|structure prediction)\b", lowered)):
        return Command("fold", path)
    if "pymol" in lowered and re.search(r"\b(open|view|show|display|visuali[sz]e|launch|use)\b", lowered) and (path or re.search(r"\b(latest|last|recent|structure|prediction|result|it|this|that)\b", lowered)):
        return Command("pymol", path)
    return None


def _looks_actionable(text: str) -> bool:
    lowered = text.casefold().strip()
    if re.match(r"^(how|what|why|where|when|does|is|are|explain|tell me)\b", lowered) or re.search(r"\b(explain|tell me|show me how)\b", lowered):
        return False
    if re.match(r"^(can|could|would|will)\s+(?!you\b)", lowered):
        return False
    return bool(re.match(r"^(?:please\s+)?(?:open|view|show|display|visuali[sz]e|launch|use|run|fold|predict|model)\b", lowered) or re.match(r"^(?:can|could|would|will)\s+you\b", lowered) or re.match(r"^i(?:'d| would) like (?:you )?to\b", lowered) or re.match(r"^i (?:want|need) (?:you )?to\b", lowered) or re.match(r"^let(?:'s| us)\b", lowered) or re.match(r"^(?:alpha\s*fold\s*2?|pymol)\s*[:,]?\s*(?:please\s+)?(?:open|view|show|display|visuali[sz]e|launch|use|run|fold|predict|model)\b", lowered))


def _extract_supported_path(text: str) -> str:
    try:
        words = shlex.split(text)
    except ValueError:
        return ""
    for word in reversed(words):
        candidate = word.rstrip(".,!?;:")
        if _has_suffix(candidate, (".fasta", ".fa", ".faa", ".pdb", ".cif", ".mmcif")):
            return candidate
    return ""


def _has_suffix(value: str, suffixes: tuple[str, ...]) -> bool:
    return value.casefold().endswith(suffixes)
