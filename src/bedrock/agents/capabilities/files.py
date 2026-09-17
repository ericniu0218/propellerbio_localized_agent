"""Reading and writing files, inside directories somebody chose on purpose.

Three capabilities live here because they share one rule, and a rule enforced
in one place is a rule::

    [agents.capabilities.files]
    roots = ["~/work/projects", "/mnt/share/protocols"]  # what may be read
    output_root = "~/work/agent-output"                  # what may be written
    extensions = [".md", ".txt", ".csv"]                 # optional filter
    max_bytes = 1000000
    max_results = 50
    max_scan_files = 20000

``files.search`` and ``files.read`` are ``READ_ONLY`` and may run before anyone
looks at the plan. ``files.write`` is ``CHANGES_THINGS`` and never runs without
a grant - a check the supervisor makes before calling it, not one repeated
here; see ``capability_support`` for why that is the right place for it.

Three decisions shape this file:

*A path outside the roots is an error, not a warning.* Every path an agent
supplies is resolved with symlinks followed and then checked to still be inside
a configured root. ``../`` cannot climb out and a symlink cannot point out,
because the check happens after resolution rather than before it.

*Reading and writing get different roots.* ``roots`` is what may be read;
``output_root`` is the single directory that may be written. They are separate
settings so that "let the agent read the protocol folder" never silently means
"let the agent rewrite the protocol folder".

*Nothing is ever overwritten.* ``files.write`` creates a new file or it creates
``name (2).md`` and says which - there is no argument that makes it replace
something. An agent that can overwrite can destroy work that took a week, and
no approval prompt read at 6pm reliably catches that.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bedrock.agents.capability_support import (
    CapabilityConfigError,
    as_int,
    as_list,
    as_path,
    as_paths,
    capability_config,
    create_without_replacing,
    done,
    failed,
)
from bedrock.agents.contract import Outcome, Risk
from bedrock.agents.protocol import Capability

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


# ---- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class FilesConfig:
    """Which directories these capabilities may touch, and how much of them."""

    roots: tuple[Path, ...] = ()
    output_root: Path | None = None
    extensions: tuple[str, ...] = ()
    max_bytes: int = 1_000_000
    max_results: int = 50
    max_scan_files: int = 20_000

    @classmethod
    def from_settings(cls, settings: Any = None) -> "FilesConfig":
        table = capability_config("files", settings)
        extensions = tuple(
            (item if item.startswith(".") else f".{item}").lower()
            for item in as_list(table, "extensions")
        )
        return cls(
            roots=as_paths(table, "roots"),
            output_root=as_path(table, "output_root"),
            extensions=extensions,
            max_bytes=as_int(table, "max_bytes", 1_000_000),
            max_results=as_int(table, "max_results", 50),
            max_scan_files=as_int(table, "max_scan_files", 20_000),
        )

    def readable_roots(self) -> tuple[Path, ...]:
        if not self.roots:
            raise CapabilityConfigError(
                "No folders are configured for the agent to read. Add `roots` to "
                "[agents.capabilities.files] in ~/.config/bedrock/config.toml."
            )
        return self.roots

    def writable_root(self) -> Path:
        if self.output_root is None:
            raise CapabilityConfigError(
                "No folder is configured for the agent to write into. Add "
                "`output_root` to [agents.capabilities.files] in "
                "~/.config/bedrock/config.toml."
            )
        return self.output_root

    def serves(self, path: Path) -> bool:
        return not self.extensions or path.suffix.lower() in self.extensions


class _FilesCapability:
    """Shared configuration handling for the three capabilities below.

    ``config`` is read the first time it is needed rather than at import, so
    listing what capabilities exist never opens the employee's config file.
    Tests pass a ``FilesConfig`` in and never reach the real settings.
    """

    def __init__(self, config: FilesConfig | None = None) -> None:
        self._config = config

    @property
    def config(self) -> FilesConfig:
        if self._config is None:
            self._config = FilesConfig.from_settings()
        return self._config


# ---- The one path rule -------------------------------------------------------


class PathRefused(Exception):
    """A path an agent asked for is not one it may have."""


def _check_shape(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise PathRefused("No path was given.")
    if _CONTROL.search(text):
        raise PathRefused("A path cannot contain control characters.")
    return text


def resolve_for_reading(roots: tuple[Path, ...], raw: str) -> Path:
    """The path ``raw`` names, proven to be inside one of ``roots``.

    An absolute path is taken as given. A relative path is tried against each
    root in order and the first one that exists wins, so an agent can say
    ``protocols/amendment-3.md`` without knowing which share it is on.

    Resolution happens *before* the containment check, so a symlink inside a
    root that points at ``/etc`` is refused rather than followed.
    """
    text = _check_shape(raw)
    resolved_roots = [root.resolve() for root in roots]
    candidates = (
        [Path(text)]
        if Path(text).is_absolute()
        else [root / text for root in roots]
    )

    inside: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if any(_is_within(resolved, root) for root in resolved_roots):
            inside.append(resolved)

    if not inside:
        where = ", ".join(str(root) for root in roots)
        raise PathRefused(
            f"`{text}` is outside the folders this agent may read ({where})."
        )
    for path in inside:
        if path.exists():
            return path
    return inside[0]


def resolve_for_writing(output_root: Path, raw: str) -> Path:
    """Where ``raw`` would be written, proven to be inside ``output_root``.

    Absolute paths are refused outright rather than checked. An agent writing
    to ``/etc/passwd`` is not a path mistake to be corrected, and refusing the
    whole shape means there is no clever spelling to get wrong.
    """
    text = _check_shape(raw)
    if Path(text).is_absolute() or text.startswith("~"):
        raise PathRefused(
            f"`{text}` is an absolute path. Files are written under "
            f"{output_root} using a relative name such as `notes/summary.md`."
        )
    root = output_root.resolve()
    target = (root / text).resolve()
    if not _is_within(target, root):
        raise PathRefused(f"`{text}` points outside {output_root}.")
    if target == root:
        raise PathRefused("A file name is needed, not the output folder itself.")
    return target


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


# ---- files.search ------------------------------------------------------------


class FilesSearch(_FilesCapability):
    """Find files whose name or contents mention every word in the query."""

    name = "files.search"
    risk = Risk.READ_ONLY

    def describe(self, arguments: dict[str, str]) -> str:
        query = str(arguments.get("query", "")).strip()
        return f"Search the configured folders for {query!r}."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        terms = str(arguments.get("query", "")).casefold().split()
        if not terms:
            return failed("files.search needs a `query` to look for.")
        try:
            config = self.config
            roots = config.readable_roots()
            limit = min(int(arguments.get("limit", config.max_results)), config.max_results)
        except (CapabilityConfigError, ValueError) as exc:
            return failed(str(exc))

        hits = await asyncio.to_thread(_walk_for, roots, config, terms, max(limit, 1))
        if not hits:
            return done((f"No file mentions all of: {' '.join(terms)}.",))
        return done(
            (f"{len(hits)} match(es) for: {' '.join(terms)}", *hits),
        )


def _walk_for(
    roots: tuple[Path, ...],
    config: FilesConfig,
    terms: list[str],
    limit: int,
) -> list[str]:
    """Walk the roots once, matching names first and contents second.

    Symlinked directories are not followed: a link inside a root that points
    outside it would otherwise walk straight out of the sandbox, and the
    containment check only runs on paths an agent names.
    """
    found: list[str] = []
    scanned = 0
    for root in roots:
        base = root.resolve()
        if not base.is_dir():
            continue
        for directory, subdirectories, filenames in os.walk(base, followlinks=False):
            subdirectories[:] = sorted(d for d in subdirectories if not d.startswith("."))
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                path = Path(directory) / filename
                if path.is_symlink() or not config.serves(path):
                    continue
                scanned += 1
                if scanned > config.max_scan_files:
                    found.append(
                        f"[stopped after {config.max_scan_files} files; narrow the query]"
                    )
                    return found
                shown = str(path)
                if all(term in shown.casefold() for term in terms):
                    found.append(shown)
                else:
                    line = _first_matching_line(path, terms, config.max_bytes)
                    if line is not None:
                        found.append(f"{shown}:{line[0]}: {line[1]}")
                if len(found) >= limit:
                    return found
    return found


def _first_matching_line(
    path: Path, terms: list[str], max_bytes: int
) -> tuple[int, str] | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                folded = line.casefold()
                if all(term in folded for term in terms):
                    return number, line.strip()[:300]
    except OSError:
        # A file that cannot be read is not a search failure. The employee
        # asked what mentions "enrolment", not for an audit of the share's
        # permissions, and one unreadable file must not end the walk.
        return None
    return None


# ---- files.read --------------------------------------------------------------


class FilesRead(_FilesCapability):
    """Read one file, as text, from inside the configured folders."""

    name = "files.read"
    risk = Risk.READ_ONLY

    def describe(self, arguments: dict[str, str]) -> str:
        return f"Read {str(arguments.get('path', '')).strip() or '(no path given)'}."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        try:
            config = self.config
            path = resolve_for_reading(
                config.readable_roots(), str(arguments.get("path", ""))
            )
        except (CapabilityConfigError, PathRefused) as exc:
            return failed(str(exc))

        try:
            limit = min(int(arguments.get("max_bytes", config.max_bytes)), config.max_bytes)
        except (TypeError, ValueError):
            limit = config.max_bytes

        try:
            text, size = await asyncio.to_thread(_read_text, path, max(limit, 1))
        except FileNotFoundError:
            return failed(f"There is no file at {path}.")
        except IsADirectoryError:
            return failed(f"{path} is a folder. Use files.search to look inside it.")
        except PermissionError:
            return failed(f"No permission to read {path}.")
        except OSError as exc:
            return failed(f"Could not read {path}: {exc.strerror or exc}")

        header = f"{path} ({size} bytes)"
        if size > limit:
            header += f", first {limit} bytes shown"
        return done((header, text))


def _read_text(path: Path, limit: int) -> tuple[str, int]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(limit)
    return raw.decode("utf-8", errors="replace"), size


# ---- files.write -------------------------------------------------------------


class FilesWrite(_FilesCapability):
    """Create a new file under the configured output folder. Never replace one."""

    name = "files.write"
    risk = Risk.CHANGES_THINGS

    def describe(self, arguments: dict[str, str]) -> str:
        path = str(arguments.get("path", "")).strip() or "(no path given)"
        size = len(str(arguments.get("content", "")))
        return f"Write a new file `{path}` ({size} characters). Nothing is overwritten."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        try:
            target = resolve_for_writing(
                self.config.writable_root(), str(arguments.get("path", ""))
            )
        except (CapabilityConfigError, PathRefused) as exc:
            return failed(str(exc))

        content = str(arguments.get("content", ""))
        try:
            written = await asyncio.to_thread(
                create_without_replacing, target, content.encode("utf-8")
            )
        except PermissionError:
            return failed(f"No permission to write into {target.parent}.")
        except OSError as exc:
            return failed(f"Could not write {target}: {exc.strerror or exc}")

        lines = [f"Wrote {written} ({len(content)} characters)."]
        if written != target:
            lines.insert(0, f"{target.name} already existed, so nothing was replaced.")
        return done(lines, artifacts=(written,))


CAPABILITY: tuple[Capability, ...] = (FilesSearch(), FilesRead(), FilesWrite())
"""What the roster finds in this file.

A tuple rather than one capability because this adapter offers several verbs
over one set of rules: searching, reading and writing all check the same roots
with the same containment check, and splitting them across three files would
split that check across three files too. ``roster.discover_capability`` takes
either shape and picks the entry whose ``name`` the roster asked for.
"""

CAPABILITIES = CAPABILITY
"""Plural alias, for reading a module's contents rather than the roster's."""
