"""Only three modules in the review package import ``bedrock.inference``.

The rule, from ``litreview/CLAUDE.md``: ``llm/scheduled.py``, ``service.py`` and
the doctor (``readiness.py``) import it; everything else depends on the
``LLMClient`` protocol. That is what
lets the whole pipeline - every stage, the guard, the report, the benchmark -
be tested offline with ``FakeLLMClient``, on a machine with no GPU and no
server, which is how this package was built before the client's GPU existed.

It is also the seam that keeps the model-call discipline enforceable. A stage
that imported ``Scheduler`` could reach a server on its own schedule, and the
guarantees this package makes about reproducibility depend on every call going
through one place at one priority.

``tests/test_model_call_guard.py`` is the neighbouring rule, and a different
one: that nothing outside ``bedrock.inference`` *calls* ``stream_chat``. It
would not notice a stage that imported ``Scheduler`` and used it correctly -
which would still be wrong here, because the point is that a stage does not
know a scheduler exists.

Read with ``ast``, so a docstring or a comment naming the module - this file's
own, or CLAUDE.md's - does not count as an import.
"""

from __future__ import annotations

import ast
from pathlib import Path

import bedrock.litreview

PACKAGE = Path(bedrock.litreview.__file__).parent

ALLOWED = {"llm/scheduled.py", "service.py", "readiness.py"}
"""The modules that may. ``llm/scheduled.py`` is the only live model client
(`SchedulerClient`); ``service.py`` is the one door a model-using review comes
through, and it needs `Scheduler` for its type and `InferenceError` in
preflight. ``readiness.py`` (`bedrock-litreview doctor`) asks the server which
models it has and catches `InferenceError` when it cannot: it imports the
exception type and builds a backend through `service.review_backend`, and it
makes no model call and is no stage, so what this rule protects - a stage that
could reach a server on its own schedule - is not in play."""


def inference_imports(path: Path) -> list[int]:
    """Line numbers of statements importing ``bedrock.inference``, or anything under it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_inference(alias.name) for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            # `level` > 0 is a relative import, which cannot reach outside
            # this package, so it can never be `bedrock.inference`.
            if node.level == 0 and node.module and _is_inference(node.module):
                lines.append(node.lineno)
    return sorted(lines)


def _is_inference(name: str) -> bool:
    return name == "bedrock.inference" or name.startswith("bedrock.inference.")


def test_only_scheduled_and_service_import_the_inference_layer():
    offenders = [
        f"{path.relative_to(PACKAGE)}:{line}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if str(path.relative_to(PACKAGE)) not in ALLOWED
        for line in inference_imports(path)
    ]
    assert offenders == [], (
        "These import bedrock.inference. Depend on the LLMClient protocol instead, so the "
        "pipeline stays testable with FakeLLMClient and every model call keeps going through "
        f"SchedulerClient: {', '.join(offenders)}"
    )


def test_both_allowed_modules_still_exist_and_still_import_it():
    """An allow-list that names a deleted file, or one that no longer needs the
    exemption, silently widens the rule."""
    for name in sorted(ALLOWED):
        path = PACKAGE / name
        assert path.is_file(), f"{name} is on the allow-list but does not exist"
        assert inference_imports(path), (
            f"{name} no longer imports bedrock.inference; remove it from ALLOWED so the "
            "rule stays as tight as it reads"
        )


def test_the_guard_reads_imports_not_mentions(tmp_path):
    sneaky = tmp_path / "stage.py"
    sneaky.write_text(
        '"""Talks about bedrock.inference and the Scheduler, which is fine."""\n'
        "# import bedrock.inference would be wrong here\n"
        "from bedrock.litreview.llm import LLMClient\n"
        "from bedrock.inference import Scheduler\n"
        "import bedrock.inference.scheduler\n"
    )
    assert inference_imports(sneaky) == [4, 5]
