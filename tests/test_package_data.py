"""Every runtime file that is not Python is listed in ``package-data``.

The prompts, the knowledge tables and the three YAML defaults are read from
inside the installed package at run time, exactly like a ``.py`` file. Unlike a
``.py`` file, setuptools does not ship them unless ``[tool.setuptools.package-data]``
in ``pyproject.toml`` names them.

Nothing catches a miss before a user does. CI installs with ``pip install -e``,
which leaves the source tree in place, so a file left out of the globs is still
found by every test - and then missing from the wheel a scientist installs with
``uv tool install``, where the failure is a ``FileNotFoundError`` deep in a
stage. This test compares the tree against the globs so the gap closes at
commit time instead.

Add a data file and this fails: add its glob to ``pyproject.toml``. Add a
document meant for developers and not for the wheel, and add it to
``DEVELOPER_DOCS`` below.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import bedrock

PACKAGE = Path(bedrock.__file__).parent
PYPROJECT = PACKAGE.parents[1] / "pyproject.toml"

DEVELOPER_DOCS = {"AGENTS.md", "CLAUDE.md"}
"""Instructions for whoever edits the code, deliberately not shipped. They sit
beside the code they describe so they are found when that code is changed, and
nothing reads them at run time."""


def package_data_globs() -> list[str]:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return config["tool"]["setuptools"]["package-data"]["bedrock"]


def runtime_data_files() -> list[Path]:
    """Every non-Python file under the package, as a path relative to it."""
    return sorted(
        path.relative_to(PACKAGE)
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and path.suffix != ".py"
        and "__pycache__" not in path.parts
        and path.name not in DEVELOPER_DOCS
        # A stale editable install can leave compiled or cached files behind;
        # they are build output, not source.
        and path.suffix not in {".pyc", ".pyo", ".so", ".pyd"}
    )


def test_package_data_covers_every_runtime_file():
    globs = package_data_globs()
    missing = [
        str(path) for path in runtime_data_files()
        if not any(path.match(glob) for glob in globs)
    ]
    assert missing == [], (
        "These files are read at run time but would not be in the wheel. Add a "
        f"glob for them to [tool.setuptools.package-data] in pyproject.toml: {missing}. "
        f"Current globs: {globs}"
    )


def test_every_glob_still_matches_something():
    """A glob left behind after its files were moved or deleted is dead weight
    that reads as coverage. This is the half that fails when a directory is
    renamed and the glob is not."""
    files = runtime_data_files()
    unused = [glob for glob in package_data_globs() if not any(path.match(glob) for path in files)]
    assert unused == [], f"These package-data globs match no file any more: {unused}"


def test_the_matching_is_anchored_not_just_by_suffix():
    """The guard itself. ``Path.match`` matches a relative pattern from the
    right, so a bug here would be a glob that appears to cover a file in a
    directory it says nothing about - and this file would then pass while the
    wheel shipped incomplete."""
    covered = Path("litreview/defaults/models.yaml")
    assert covered.match("litreview/defaults/*.yaml")
    # The same file name one directory across is *not* covered by that glob.
    assert not Path("litreview/knowledge/entities.yaml").match("litreview/defaults/*.yaml")
    assert not Path("litreview/defaults/deep/models.yaml").match("litreview/defaults/*.yaml")
