"""Slash-command parsing.

Kept as plain functions with no Textual imports so they are trivial to test.
The app decides what to *do* with a command; this module only recognises it.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from bedrock.connectors.base import parse_source_ref

HELP_TEXT = """\
**Commands**

| Command | What it does |
|---|---|
| `/model` | List available models |
| `/model <name>` | Switch to a different model |
| `@<path>` in a message | Attach a text file (or drag the file onto the terminal) |
| `/digest <path>` | Compress a file so a model can read it cheaply |
| `/digest` | List and search what you have compressed |
| `/sources` | List the company data sources you can pull from |
| `/digest <source>://<id>` | Pull a record from a company source and compress it |
| `/runs` | Prediction runs recorded on this machine |
| `/runs show <run-id>` | Settings, lineage and integrity of one run |
| `/confidence <run-id>` | pLDDT bands, low-confidence regions and PAE |
| `/audit` | What this client has done on this machine |
| `/workspace` | List private team workspaces |
| `/workspace help` | Show workspace actions |
| `/history` | List saved conversations |
| `/resume <number or name>` | Reopen a saved conversation, e.g. `/resume 2` or `/resume platypus` |
| `/new` | Start a new conversation (also `/clear`) |
| `/forget` | Delete this conversation from disk |
| `/recall <what to look for>` | Pull matching parts of past chats into this one (also `/context`) |
| `/recall` | Show what has been pulled in; `/recall clear` removes it |
| `/help` | Show this help |
| `/exit` | Quit (also `Ctrl+C`) |

**Examples**

- Runs for project X this quarter:
  `/runs --project X --since 2026-07-01 --before 2026-10-01`
- Export the action history: `/audit export "/approved/review/actions.jsonl"`

`/runs`, `/confidence` and `/audit` read local files only. They work with the
model server switched off, and nothing they show is sent to a model.

**Attaching files**

Drag a file onto the terminal window, or write `@` before its path anywhere in
a message: `summarise @notes.txt`. Quote a path with spaces: `@"My Data.csv"`.
A small text file is sent whole; a large one is compressed and its digest is
sent instead, and a notice says so. The model is told to treat attached files
as data, not instructions. Attached files are saved with the conversation, so
follow-up questions still see them, and `/forget` deletes them with it.

**Keys**

- `Enter` sends your message
- `Esc` stops a response that is still streaming
- `Ctrl+L` starts a new conversation (same as `/new`)
- `Ctrl+B` shows or hides the conversation list; click a conversation to open it

Conversations are saved on this machine as you go, and the most recent one
reopens the next time you start. Past chats stay on disk and are only searched
when you use `/recall` or ask something like "do you remember when we set up the
deploy workflow?". Nothing is sent anywhere but your model server.
"""


@dataclass(frozen=True)
class Command:
    name: str
    args: str


KNOWN_COMMANDS = {
    "model",
    "digest",
    "sources",
    "runs",
    "confidence",
    "audit",
    "workspace",
    "team",
    "clear",
    "new",
    "history",
    "resume",
    "forget",
    "recall",
    "context",
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
    source = _extract_source_ref(stripped)
    mentions_alphafold = bool(re.search(r"\balpha\s*fold\s*2?\b", lowered))
    requests_prediction = bool(
        re.search(r"\b(fold|predict|prediction|model|run|use)\b", lowered)
    )
    requests_folding = bool(re.search(r"\b(fold|predict(?:ion)?|structure prediction)\b", lowered))
    if (
        (mentions_alphafold and requests_prediction)
        or (path and _has_suffix(path, (".fasta", ".fa", ".faa")) and requests_folding)
        # A company record such as registry://SEQ-00123 has no file suffix to
        # go on, so an explicit fold/predict verb is what marks it as a job.
        or (source and requests_folding)
    ):
        return Command("fold", path or source)

    mentions_pymol = "pymol" in lowered
    requests_view = bool(
        re.search(r"\b(open|view|show|display|visuali[sz]e|launch|use)\b", lowered)
    )
    references_known_structure = bool(
        re.search(r"\b(latest|last|recent|structure|prediction|result|it|this|that)\b", lowered)
    )
    if mentions_pymol and requests_view and (path or references_known_structure):
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
        # "fold @candidate.fasta" names the file the same way attaching does.
        candidate = word.rstrip(".,!?;:").removeprefix("@")
        if _has_suffix(candidate, (".fasta", ".fa", ".faa", ".pdb", ".cif", ".mmcif")):
            return candidate
    return ""


def _extract_source_ref(text: str) -> str:
    """The first ``<source>://<id>`` reference in ``text``, or ``""``."""
    try:
        words = shlex.split(text)
    except ValueError:
        return ""
    for word in words:
        candidate = word.rstrip(".,!?;:")
        if parse_source_ref(candidate) is not None:
            return candidate
    return ""


def _has_suffix(value: str, suffixes: tuple[str, ...]) -> bool:
    return value.casefold().endswith(suffixes)
