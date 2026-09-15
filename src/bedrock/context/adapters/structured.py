"""JSON and JSON Lines.

The useful summary of a structured document is its *shape*: which key paths
exist, what type each holds, and how often each appears. A thousand records
that all share the same nine fields collapse to nine lines, and that is almost
always what someone means when they ask what is in the file.

Records themselves are kept as segments so an exact value is still one
retrieval away.
"""

from __future__ import annotations

import json

from bedrock.context.adapters.base import Source
from bedrock.context.types import Extraction, Segment

RECORDS_PER_SEGMENT = 10
_MAX_PATHS = 40
_MAX_DEPTH = 6


class StructuredAdapter:
    name = "structured"

    def can_handle(self, source: Source) -> bool:
        if source.suffix in {".json", ".jsonl", ".ndjson"}:
            return True
        stripped = source.text.lstrip()
        if not stripped.startswith(("{", "[")):
            return False
        try:
            json.loads(source.text)
        except (ValueError, RecursionError):
            return False
        return True

    def extract(self, source: Source) -> Extraction:
        records, shape_note = self._load(source)
        paths: dict[str, set[str]] = {}
        for record in records:
            _walk(record, "", paths)

        lines = [shape_note, "", "Fields:"]
        for path, types in list(paths.items())[:_MAX_PATHS]:
            lines.append(f"- {path}: {'|'.join(sorted(types))}")
        if len(paths) > _MAX_PATHS:
            lines.append(f"- ... and {len(paths) - _MAX_PATHS} more fields")
        overview = "\n".join(lines)

        segments = [Segment(text=overview, kind="schema", label="shape", weight=5.0)]
        for start in range(0, len(records), RECORDS_PER_SEGMENT):
            block = records[start : start + RECORDS_PER_SEGMENT]
            body = "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in block)
            segments.append(
                Segment(
                    text=body,
                    kind="records",
                    label=f"records {start + 1}-{start + len(block)}",
                )
            )

        return Extraction(
            source_kind="structured",
            overview=overview,
            segments=segments,
            metadata={"records": len(records), "fields": len(paths)},
        )

    def _load(self, source: Source) -> tuple[list, str]:
        """Return a flat list of records plus a one-line description of the shape."""
        try:
            data = json.loads(source.text)
        except ValueError:
            # Fall through to JSON Lines: one object per line.
            records = []
            for line in source.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
            return records, f"JSON Lines: {len(records):,} records."

        if isinstance(data, list):
            return data, f"JSON array: {len(data):,} items."
        if isinstance(data, dict):
            return [data], f"JSON object: {len(data)} top-level keys."
        return [data], f"JSON {type(data).__name__} value."


def _walk(node, prefix: str, paths: dict[str, set[str]], depth: int = 0) -> None:
    """Collect ``key.path -> {types seen}`` across every record.

    Arrays are collapsed to ``[]`` rather than indexed, so a list of a thousand
    items contributes one path, not a thousand. That collapse is the entire
    reason this summary stays small on real data.
    """
    if depth > _MAX_DEPTH:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                _walk(value, path, paths, depth + 1)
            else:
                paths.setdefault(path, set()).add(_type_name(value))
    elif isinstance(node, list):
        path = f"{prefix}[]" if prefix else "[]"
        if not node:
            paths.setdefault(path, set()).add("empty")
        for item in node[:50]:
            if isinstance(item, (dict, list)):
                _walk(item, path, paths, depth + 1)
            else:
                paths.setdefault(path, set()).add(_type_name(item))


def _type_name(value) -> str:
    if value is None:
        return "null"
    return {bool: "bool", int: "int", float: "float", str: "string"}.get(type(value), type(value).__name__)
