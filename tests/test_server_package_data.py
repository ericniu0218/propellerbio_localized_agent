"""Every file the server reads at runtime ships in the wheel.

The workspace's HTML/JS/CSS, the default ``models.yaml``/``budgets.yaml`` and
the sample corpora are not Python, so setuptools only packages them if a
``package-data`` glob in ``pyproject.toml`` names them. An editable install
(what CI and developers use) finds them either way, so a missing glob would only
show up after a real install - on the server, as a blank page or a crash at
startup. This compares the files on disk with the globs instead.
"""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

import bedrock

PACKAGE = Path(bedrock.__file__).parent
PYPROJECT = PACKAGE.parent.parent / "pyproject.toml"
RUNTIME_DIRS = ("server/static", "server/defaults", "server/sample_data")


def _globs() -> list[str]:
    return tomllib.loads(PYPROJECT.read_text())["tool"]["setuptools"]["package-data"]["bedrock"]


def _covered(relative: str, glob: str) -> bool:
    # setuptools globs match one path segment per `*`, like shell globs.
    parts, pattern = relative.split("/"), glob.split("/")
    return len(parts) == len(pattern) and all(fnmatch.fnmatchcase(p, g) for p, g in zip(parts, pattern))


def test_every_server_runtime_file_is_package_data():
    files = [
        path.relative_to(PACKAGE).as_posix()
        for directory in RUNTIME_DIRS
        for path in (PACKAGE / directory).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".py"
    ]
    assert files, "no runtime files found - has the server moved?"
    missing = [f for f in files if not any(_covered(f, g) for g in _globs())]
    assert missing == [], "Add a package-data glob in pyproject.toml for: " + ", ".join(missing)


def test_the_matcher_is_one_segment_per_star():
    assert _covered("server/static/index.html", "server/static/*.html")
    assert not _covered("server/static/workspace/app.js", "server/static/*.html")
    assert _covered("server/static/workspace/app.js", "server/static/workspace/*")
