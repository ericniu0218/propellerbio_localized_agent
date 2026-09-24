"""Nothing outside ``bedrock.inference`` calls ``stream_chat``.

Every model call has to go through the ``Scheduler``, which decides when it
may run: one call at a time on a laptop, the chat first, background work only
when idle. A feature that called the backend directly would skip all of that
and bring back the problem the scheduler fixes - a title or a summary
competing with the employee's own reply. This test reads the source with
``ast`` (so comments and docstrings that merely mention the name don't count)
and fails on any call it finds.

If this fails for new code: call ``scheduler.stream(...)`` or
``scheduler.complete(...)`` with a ``Priority`` and a ``RequestOptions``
carrying a reply limit instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import bedrock

SOURCE = Path(bedrock.__file__).parent
ALLOWED = SOURCE / "inference"


def stream_chat_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "stream_chat"
    ]


def test_only_the_inference_package_calls_stream_chat():
    offenders = [
        f"{path.relative_to(SOURCE.parent)}:{line}"
        for path in sorted(SOURCE.rglob("*.py"))
        if ALLOWED not in path.parents
        for line in stream_chat_calls(path)
    ]
    assert offenders == [], (
        "These call stream_chat directly; go through Scheduler instead: " + ", ".join(offenders)
    )


def test_the_guard_would_catch_a_direct_call(tmp_path):
    sneaky = tmp_path / "feature.py"
    sneaky.write_text(
        '"""Mentions stream_chat in a docstring, which is fine."""\n'
        "async def go(backend):\n"
        "    async for part in backend.stream_chat('m', []):\n"
        "        yield part\n"
    )
    assert stream_chat_calls(sneaky) == [3]
