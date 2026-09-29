"""The terminal app and the server stay separable.

``bedrock.server`` (the agent-pack server, the ``[server]`` extra) may use
bedrock's engines - inference, meetings, context, connectors - but the terminal
app must never need the server: an employee's laptop installs bedrock without
FastAPI or SQLAlchemy. And the server must never import the terminal UI, which
would drag Textual into a web process for no reason. CLAUDE.md's scope note on
the terminal-only rule points here.

Read with ``ast`` so a docstring that merely mentions a module does not count.
"""

from __future__ import annotations

import ast
from pathlib import Path

import bedrock

SOURCE = Path(bedrock.__file__).parent
SERVER = SOURCE / "server"
TERMINAL_UI = ("bedrock.app", "bedrock.agentpack_app", "bedrock.ui")


def imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return found


def _within(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def test_nothing_outside_the_server_imports_it():
    offenders = [
        f"{path.relative_to(SOURCE.parent)}:{line} imports {module}"
        for path in sorted(SOURCE.rglob("*.py"))
        if SERVER not in path.parents
        for line, module in imported_modules(path)
        if _within(module, "bedrock.server")
    ]
    assert offenders == [], "The terminal app must not need the server: " + "; ".join(offenders)


def test_the_server_never_imports_the_terminal_ui():
    offenders = [
        f"{path.relative_to(SOURCE.parent)}:{line} imports {module}"
        for path in sorted(SERVER.rglob("*.py"))
        for line, module in imported_modules(path)
        if any(_within(module, ui) for ui in TERMINAL_UI)
    ]
    assert offenders == [], "The server must not import the terminal UI: " + "; ".join(offenders)


def test_the_check_would_catch_a_violation(tmp_path):
    sneaky = tmp_path / "feature.py"
    sneaky.write_text('"""Mentions bedrock.server in prose, which is fine."""\nfrom bedrock.server.config import get_settings\n')
    assert imported_modules(sneaky) == [(2, "bedrock.server.config")]
