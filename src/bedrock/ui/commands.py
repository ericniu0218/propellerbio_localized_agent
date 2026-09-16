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
| `/digest <path>` | Compress a file so a model can read it cheaply |
| `/digest` | List and search what you have compressed |
| `/sources` | List the company data sources you can pull from |
| `/digest <source>://<id>` | Pull a record from a company source and compress it |
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


KNOWN_COMMANDS = {"model", "digest", "sources", "clear", "help", "exit", "quit"}


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
    """Recognize clear requests to run the two locally integrated tools.

    This intentionally handles action language, not every mention of a tool.
    Questions and explanations still go to the agent. The returned value uses
    the same small command interface as slash commands, so execution, errors,
    and authorization behavior cannot drift between the two input styles.
    """
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
    if re.match(r"^(how|what|why|where|when|does|is|are|explain|tell me)\b", lowered):
        return False
    if re.search(r"\b(explain|tell me|show me how)\b", lowered):
        return False
    if re.match(r"^(can|could|would|will)\s+(?!you\b)", lowered):
        return False
    return bool(
        re.match(
            r"^(?:please\s+)?(?:open|view|show|display|visuali[sz]e|launch|use|run|fold|predict|model)\b",
            lowered,
        )
        or re.match(r"^(?:can|could|would|will)\s+you\b", lowered)
        or re.match(r"^i(?:'d| would) like (?:you )?to\b", lowered)
        or re.match(r"^i (?:want|need) (?:you )?to\b", lowered)
        or re.match(r"^let(?:'s| us)\b", lowered)
        or re.match(
            r"^(?:alpha\s*fold\s*2?|pymol)\s*[:,]?\s*"
            r"(?:please\s+)?(?:open|view|show|display|visuali[sz]e|launch|use|run|fold|predict|model)\b",
            lowered,
        )
    )


def _extract_supported_path(text: str) -> str:
    try:
        words = shlex.split(text)
    except ValueError:
        return ""
    supported = (".fasta", ".fa", ".faa", ".pdb", ".cif", ".mmcif")
    for word in reversed(words):
        candidate = word.rstrip(".,!?;:")
        if _has_suffix(candidate, supported):
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
