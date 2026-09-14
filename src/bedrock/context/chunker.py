"""Packs segments into bounded chunks.

Why chunk at all: retrieval needs pieces small enough that pulling one into a
prompt is cheap, but large enough that the piece still makes sense on its own.
A few hundred tokens is the usual sweet spot.

Two rules, both about cutting in sensible places:

* Segments are packed whole. A paragraph, a function, or a block of table rows
  is never split across two chunks just to fill one up, because half a
  function retrieved on its own is close to useless.
* A segment that is bigger than the whole budget is the one exception. It gets
  split, and we look for the most natural boundary available in order:
  blank line, then line break, then a hard character cut as a last resort.
  Consecutive pieces of a split segment overlap slightly, so a sentence that
  straddles the cut is still findable from both sides.
"""

from __future__ import annotations

from bedrock.context.types import CHARS_PER_TOKEN, Chunk, Segment

DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 40


def chunk_segments(
    segments: list[Segment],
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Turn segments into chunks of at most roughly ``max_tokens`` each."""
    budget = max(1, max_tokens) * CHARS_PER_TOKEN
    overlap = max(0, overlap_tokens) * CHARS_PER_TOKEN

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0
    buffer_label: str | None = None

    def flush() -> None:
        nonlocal buffer, buffer_len, buffer_label
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        if text:
            chunks.append(Chunk(index=len(chunks), text=text, label=buffer_label))
        buffer, buffer_len, buffer_label = [], 0, None

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        if len(text) > budget:
            flush()
            for piece in _split_oversized(text, budget, overlap):
                chunks.append(Chunk(index=len(chunks), text=piece, label=segment.label))
            continue

        if buffer and buffer_len + len(text) > budget:
            flush()
        if not buffer:
            buffer_label = segment.label
        buffer.append(text)
        buffer_len += len(text) + 2

    flush()
    return chunks


def _split_oversized(text: str, budget: int, overlap: int) -> list[str]:
    """Break one too-large segment at the most natural boundary available."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + budget, len(text))
        if end < len(text):
            end = _best_boundary(text, start, end)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _best_boundary(text: str, start: int, end: int) -> int:
    """Find where to cut, preferring a blank line, then a newline, then here.

    Only looks back over the last third of the window - cutting a chunk far
    short of the budget to land on a boundary wastes more than the tidy cut
    is worth.
    """
    window = max(start + 1, end - (end - start) // 3)
    for separator in ("\n\n", "\n"):
        found = text.rfind(separator, window, end)
        if found > start:
            return found + len(separator)
    return end
