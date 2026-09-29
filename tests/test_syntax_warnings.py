r"""No source file compiles with a `SyntaxWarning`.

(This docstring carries the `r` prefix, making it raw, because it quotes the
very escape it is about. Written as an ordinary string it trips the test
itself - which is exactly what happened on the first run.)

The warning this exists for is an invalid escape sequence: `"\\|"` written as
`"\|"` inside an ordinary string. Python still accepts it today and leaves the
backslash in place, so the text is right and nothing misbehaves - but it prints
a `SyntaxWarning` on import, and **a future Python turns it into a
`SyntaxError`**, at which point the module stops importing at all. That is a
working feature breaking on a Python upgrade, for a Markdown-escaped pipe in a
help table.

It went unnoticed because nothing looks for it. CI runs plain `pytest -q`, and
pytest does not fail on a warning raised at *compile* time in a module that
imports successfully. So this test compiles every file itself, with
`SyntaxWarning` promoted to an error.

Compiling is not importing: no module body runs, nothing is executed, and a file
with a missing dependency still compiles. The whole tree takes about a second.

If this fails, the fix is almost always to double the backslash (`\\|`) or make
the string raw - not to delete the backslash, which would change what the user
sees.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import bedrock

ROOT = Path(bedrock.__file__).parents[2]
TREES = ("src", "tests", "scripts")


def python_files() -> list[Path]:
    return sorted(
        path
        for tree in TREES
        for path in (ROOT / tree).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def compile_warnings(path: Path) -> list[str]:
    """Warnings raised while compiling one file, as readable strings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    # Repo-relative when it is in the repo; the guard tests below compile a
    # file in pytest's tmp_path, which is not.
    name = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return [
        f"{name}:{w.lineno}: {w.category.__name__}: {w.message}"
        for w in caught
        if issubclass(w.category, SyntaxWarning)
    ]


def test_no_source_file_raises_a_syntax_warning():
    files = python_files()
    assert len(files) > 100, f"only found {len(files)} files; the tree walk is wrong"

    offenders = [line for path in files for line in compile_warnings(path)]
    assert offenders == [], (
        "These raise a SyntaxWarning at compile time, and a future Python will "
        "refuse to import them. Double the backslash (`\\\\|`) or use a raw "
        "string; do not delete the backslash, which changes what is displayed:\n"
        + "\n".join(offenders)
    )


def test_the_guard_would_catch_an_invalid_escape(tmp_path):
    """The guard itself. Without this, a mistake in the walk above would make
    this file pass while the tree went unchecked."""
    sneaky = tmp_path / "help.py"
    sneaky.write_text('TEXT = "a \\| b"\n')

    found = compile_warnings(sneaky)

    assert len(found) == 1
    assert "invalid escape sequence" in found[0]


def test_a_correctly_escaped_pipe_is_not_flagged(tmp_path):
    ok = tmp_path / "help.py"
    ok.write_text('TEXT = "a \\\\| b"\n')
    assert compile_warnings(ok) == []


@pytest.mark.parametrize("module", ["bedrock.agentpack_app", "bedrock.app"])
def test_the_app_modules_import_with_syntax_warnings_as_errors(module):
    """The failure a user would actually hit: launching the app. Import, not
    compile, so a warning from anything these pull in is caught too."""
    import importlib

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        importlib.reload(importlib.import_module(module))
