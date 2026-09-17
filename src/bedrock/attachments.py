"""Attaching files to a chat message by writing ``@path`` in it.

Before this module, getting a file in front of the model took two steps:
``/digest <path>`` first, then a question that happened to share words with
it. Now the employee writes the file into the message itself::

    what changed between @results/march.csv and @results/april.csv?

Dragging a file onto most terminals pastes its path, so typing ``@`` and then
dropping the file in works too. Paths with spaces can be quoted
(``@"My Data.csv"``) or escaped the way terminals paste them (``@My\\ Data.csv``).

**What gets sent.** A small text file goes to the model whole. A file too large
to send whole goes through the same compression as ``/digest`` and the model
receives its digest instead - a few hundred tokens of structure and statistics.
The full content is then in the ``/digest`` store, so ``/digest search`` finds
exact passages in it later. The notice after sending says which of the two
happened, so nobody believes the model read every row of a file it only saw
summarised.

**Only real files count.** ``@`` followed by something that is not a file on
this machine (``@alice``, an email address) is left alone and sent as typed.
A word that looks like a file name (``@notes.txt``) but cannot be read is an
error instead, because the employee clearly meant to attach something and
should not get an answer that silently ignored it.

**Where it is kept.** The typed text stays the message's ``content``; each
attachment is stored beside it in ``Message.extra`` and added back when the
conversation is sent (``Message.to_api``). So the transcript on screen shows
what was typed, follow-up questions still see the file, and ``/forget`` removes
the copy along with the chat. Nothing here touches the network.

Binary formats (PDF, Excel, images) are refused with a plain message for now,
for the same reason ``bedrock.context.adapters.base.Source`` gives: each needs
its own parser and dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bedrock.context import ContextStore, ingest
from bedrock.context.adapters.base import MAX_DECODE_BYTES, Source

MAX_INLINE_CHARS = 24_000
"""Largest file sent to the model whole, roughly 6,000 tokens. Beyond this a
typical local model's context fills up after a couple of turns, so the digest
is sent instead."""

MAX_ATTACHMENTS = 5
"""Per message. Five whole files is already most of a local model's context."""

_BINARY_SNIFF_BYTES = 8192

_REFERENCE = re.compile(
    r"""(?:^|(?<=\s))@(?:"(?P<double>[^"]+)"|'(?P<single>[^']+)'|(?P<bare>(?:\\.|[^\s"'])+))"""
)
"""``@`` at the start of a word, then a quoted path or a bare one in which a
backslash escapes the next character (how terminals paste ``My\\ Data.csv``)."""

_TRAILING_PUNCTUATION = ".,;:!?)"

_LOOKS_LIKE_FILE = re.compile(r"[/\\~]|\.[A-Za-z0-9]{1,8}$")
"""A path separator, a home directory, or a file extension at the end."""


class AttachmentError(ValueError):
    """An ``@`` reference the employee meant as a file, but which cannot be sent."""


@dataclass(frozen=True)
class Attachment:
    """One file as it goes to the model.

    ``mode`` is ``"whole"`` when ``text`` is the file's full content and
    ``"digest"`` when it is the compressed digest of a larger file.
    """

    name: str
    path: str
    byte_size: int
    mode: str
    text: str

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "bytes": self.byte_size, "mode": self.mode, "text": self.text}

    @classmethod
    def from_dict(cls, data: object) -> Attachment | None:
        """Rebuild from a saved file, or None for an entry this app cannot use."""
        if not isinstance(data, dict):
            return None
        name, text = data.get("name"), data.get("text")
        if not isinstance(name, str) or not isinstance(text, str):
            return None
        size = data.get("bytes")
        return cls(
            name=name,
            path=data.get("path") if isinstance(data.get("path"), str) else name,
            byte_size=size if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else 0,
            mode="digest" if data.get("mode") == "digest" else "whole",
            text=text,
        )

    @property
    def label(self) -> str:
        """``march.csv (12 KB)``, or ``(12 MB, sent as a digest)``."""
        size = format_size(self.byte_size)
        return f"{self.name} ({size}, sent as a digest)" if self.mode == "digest" else f"{self.name} ({size})"


def find_references(text: str) -> list[str]:
    """Every ``@`` path written in ``text``, in order, without duplicates.

    Includes words that may turn out not to be files; ``resolve`` decides.
    """
    found: list[str] = []
    for match in _REFERENCE.finditer(text):
        if match.group("bare") is not None:
            raw = re.sub(r"\\(.)", r"\1", match.group("bare")).rstrip(_TRAILING_PUNCTUATION)
        else:
            raw = match.group("double") or match.group("single")
        if raw and raw not in found:
            found.append(raw)
    return found


def resolve(text: str, cwd: Path | None = None) -> list[Path]:
    """The files ``text`` asks to attach. Cheap: checks the paths, reads nothing.

    Raises :class:`AttachmentError` for a reference that looks like a file but
    is not one, or for too many attachments. Called when the message is typed,
    so a mistake is reported before the prompt waits in the queue.
    """
    base = cwd or Path.cwd()
    paths: list[Path] = []
    for reference in find_references(text):
        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = base / path
        try:
            is_file, is_dir = path.is_file(), path.is_dir()
        except OSError:
            is_file = is_dir = False
        if is_file:
            if path not in paths:
                paths.append(path)
        elif is_dir:
            raise AttachmentError(f"`{reference}` is a folder. Attach the files inside it one at a time.")
        elif _LOOKS_LIKE_FILE.search(reference):
            raise AttachmentError(f"Could not find the file `{reference}`. Check the path; nothing was sent.")
    if len(paths) > MAX_ATTACHMENTS:
        raise AttachmentError(f"A message can attach at most {MAX_ATTACHMENTS} files. Nothing was sent.")
    return paths


def load(paths: list[Path], store: ContextStore) -> list[Attachment]:
    """Read each file and decide whether it goes whole or as a digest.

    Blocking (file reads, and compression for a large file), so the apps call
    it with ``asyncio.to_thread``. Raises :class:`AttachmentError` with a
    message fit to show the employee.
    """
    return [_load_one(path, store) for path in paths]


def _load_one(path: Path, store: ContextStore) -> Attachment:
    try:
        size = path.stat().st_size
        if size > MAX_DECODE_BYTES:
            raise AttachmentError(
                f"`{path.name}` is {format_size(size)}, over the {format_size(MAX_DECODE_BYTES)} limit for an attached file."
            )
        with path.open("rb") as handle:
            head = handle.read(_BINARY_SNIFF_BYTES)
        if b"\x00" in head:
            raise AttachmentError(
                f"`{path.name}` is not a text file. Only text files (notes, CSV, JSON, code, logs, FASTA) "
                "can be attached for now."
            )
        source = Source.from_path(path)
    except OSError as exc:
        raise AttachmentError(f"Could not read `{path.name}`: {exc.strerror or exc}.") from exc
    except ValueError as exc:
        if isinstance(exc, AttachmentError):
            raise
        raise AttachmentError(str(exc)) from exc

    if len(source.text) <= MAX_INLINE_CHARS:
        return Attachment(name=source.name, path=str(path), byte_size=source.byte_size, mode="whole", text=source.text)
    try:
        artifact = ingest(source, store=store)
    except Exception as exc:
        raise AttachmentError(f"`{path.name}` is too large to send whole, and compressing it failed: {exc}") from exc
    return Attachment(
        name=source.name, path=str(path), byte_size=source.byte_size, mode="digest", text=artifact.digest_text()
    )


def prompt_block(attachments: list[Attachment]) -> str:
    """The attachments as text for the model, clearly fenced off from the question.

    Placed *after* the typed text: the question is what the employee asked,
    and the files are material to answer it from, not further instructions.
    """
    blocks = []
    for attachment in attachments:
        if attachment.mode == "digest":
            heading = (
                f"[Attached file: {attachment.name} - too large to include whole, "
                "so this is a compressed digest of it]"
            )
        else:
            heading = f"[Attached file: {attachment.name}]"
        blocks.append(f"{heading}\n{attachment.text.rstrip()}\n[End of {attachment.name}]")
    return "\n\n".join(blocks)


def with_attachments(text: str, attachments: list[Attachment]) -> str:
    """The typed text followed by its attachments. Unchanged when there are none."""
    if not attachments:
        return text
    return f"{text}\n\n{prompt_block(attachments)}"


def display_text(text: str, attachments: list[Attachment]) -> str:
    """What the chat shows for a message: what was typed, then one line per file."""
    if not attachments:
        return text
    return text + "".join(f"\n📎 {attachment.label}" for attachment in attachments)


def summary(attachments: list[Attachment]) -> str | None:
    """A notice for any file that was *not* sent whole, or None if all were."""
    digested = [attachment for attachment in attachments if attachment.mode == "digest"]
    if not digested:
        return None
    names = ", ".join(f"`{attachment.name}`" for attachment in digested)
    return (
        f"{names} {'is' if len(digested) == 1 else 'are'} too large to send whole, so the model received "
        "a compressed digest (structure and statistics, not every line). `/digest search <words>` "
        "finds exact passages."
    )


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} bytes"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.0f} MB"
