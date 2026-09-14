"""Plain text and Markdown - the fallback adapter.

There is no structure to exploit here, so the deterministic overview is
modest: size, shape, and the heading outline if the document has one. This is
the one data type where the optional model pass in ``digest.py`` earns its
keep, because prose is exactly what rules cannot summarise.
"""

from __future__ import annotations

import re

from bedrock.context.adapters.base import Source
from bedrock.context.types import Extraction, Segment

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_MAX_OUTLINE = 15


class TextAdapter:
    name = "text"

    def can_handle(self, source: Source) -> bool:
        # Always true: this is the last-resort adapter, so anything that no
        # other adapter claimed still gets sensibly handled.
        return True

    def extract(self, source: Source) -> Extraction:
        paragraphs = [block.strip() for block in re.split(r"\n\s*\n", source.text) if block.strip()]
        headings = self._headings(source.text)

        segments: list[Segment] = []
        current_heading: str | None = None
        for block in paragraphs:
            match = _HEADING.match(block.splitlines()[0].strip())
            if match:
                current_heading = match.group(2)
            segments.append(Segment(text=block, kind="prose", label=current_heading))

        words = len(source.text.split())
        lines = [f"Text document: {words:,} words, {len(paragraphs):,} paragraphs."]
        if headings:
            lines.append("")
            lines.append("Outline:")
            for depth, title in headings[:_MAX_OUTLINE]:
                lines.append(f"{'  ' * (depth - 1)}- {title}")
            if len(headings) > _MAX_OUTLINE:
                lines.append(f"  ... and {len(headings) - _MAX_OUTLINE} more headings")

        return Extraction(
            source_kind="text",
            overview="\n".join(lines),
            segments=segments,
            metadata={"words": words, "paragraphs": len(paragraphs), "headings": len(headings)},
        )

    def _headings(self, text: str) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        for line in text.splitlines():
            match = _HEADING.match(line.strip())
            if match:
                found.append((len(match.group(1)), match.group(2)))
        return found
