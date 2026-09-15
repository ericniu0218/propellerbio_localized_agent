"""Source code.

For code the summary writes itself: the signatures *are* the overview and the
bodies are the detail. Python gets this exactly right because ``ast`` is in
the standard library, so the outline is parsed rather than guessed. Other
languages fall back to a regular expression over common declaration keywords,
which is rougher but still better than treating a source file as prose.

Each function and class becomes its own segment, so retrieval returns whole
definitions instead of arbitrary slices of code.
"""

from __future__ import annotations

import ast
import re

from bedrock.context.adapters.base import Source
from bedrock.context.types import Extraction, Segment

CODE_SUFFIXES = frozenset(
    {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".c", ".h", ".cpp", ".sh"}
)
_DECL = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?"
    r"(?:def|class|function|func|fn|type|interface|struct|impl)\s+([A-Za-z_][\w]*)",
)
_LINES_PER_BLOCK = 60


class CodeAdapter:
    name = "code"

    def can_handle(self, source: Source) -> bool:
        return source.suffix in CODE_SUFFIXES

    def extract(self, source: Source) -> Extraction:
        if source.suffix == ".py":
            outline, segments = self._python(source)
        else:
            outline, segments = self._generic(source)

        lines = source.text.splitlines()
        header = [f"Source file ({source.suffix or 'unknown'}): {len(lines):,} lines, {len(outline)} definitions."]
        if outline:
            header.append("")
            header.append("Definitions:")
            header.extend(f"- {entry}" for entry in outline)
        overview = "\n".join(header)

        return Extraction(
            source_kind="code",
            overview=overview,
            segments=segments,
            metadata={"lines": len(lines), "definitions": len(outline), "language": source.suffix},
        )

    def _python(self, source: Source) -> tuple[list[str], list[Segment]]:
        try:
            tree = ast.parse(source.text)
        except SyntaxError:
            # A file that does not parse is still worth storing; just treat it
            # like any other language rather than failing the ingest.
            return self._generic(source)

        lines = source.text.splitlines()
        outline: list[str] = []
        segments: list[Segment] = []

        docstring = ast.get_docstring(tree)
        if docstring:
            segments.append(
                Segment(text=f'"""{docstring}"""', kind="docstring", label="module docstring", weight=4.0)
            )

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            signature = _signature(node)
            outline.append(signature)
            own_docstring = ast.get_docstring(node)
            if own_docstring:
                outline[-1] += f" - {own_docstring.splitlines()[0]}"
            body = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
            segments.append(Segment(text=body, kind="definition", label=signature, weight=2.0))
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        outline.append(f"    {_signature(child)}")

        if not segments:
            return outline, _line_blocks(lines)
        return outline, segments

    def _generic(self, source: Source) -> tuple[list[str], list[Segment]]:
        lines = source.text.splitlines()
        outline = []
        for line in lines:
            match = _DECL.match(line)
            if match:
                outline.append(line.strip().rstrip("{").strip())
        return outline, _line_blocks(lines)


def _signature(node) -> str:
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    args = [arg.arg for arg in node.args.args]
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)})"


def _line_blocks(lines: list[str]) -> list[Segment]:
    segments = []
    for start in range(0, len(lines), _LINES_PER_BLOCK):
        block = lines[start : start + _LINES_PER_BLOCK]
        segments.append(
            Segment(
                text="\n".join(block),
                kind="code",
                label=f"lines {start + 1}-{start + len(block)}",
            )
        )
    return segments
