"""WEBVTT parsing for meeting transcripts.

Zoom delivers transcripts as WEBVTT: a cue number, a timestamp range, then
a line of the form ``Speaker Name: what they said``. Handing that to a model
verbatim spends a large fraction of the context window on cue numbers and
millisecond timestamps that carry no meaning for the question being asked.

This module turns that into structured cues so callers can search the words,
keep the speaker, and cite a real timestamp — without paying for the syntax.
Malformed or unrecognized blocks are skipped rather than raising: a partial
transcript is still worth answering from, and a meeting is not worth losing
because one cue was truncated mid-delivery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "00:00:01.000 --> 00:00:04.000", with the hours field optional because some
# writers omit it for meetings under an hour.
_TIMING = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)

# "Alice Chen: we're two weeks behind" — but not "12:30: let's break", and not
# a bare URL like "https://example.com/x". Requires the name to start with a
# letter and stay short enough to plausibly be a name.
_SPEAKER = re.compile(r"^([A-Za-z][^:\n]{0,60}?):\s+(.*)$", re.S)


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    speaker: str | None
    text: str

    @property
    def timestamp(self) -> str:
        """``hh:mm:ss`` offset from the start of the meeting, for citations."""
        total = self.start_ms // 1000
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    def render(self) -> str:
        who = f"{self.speaker}: " if self.speaker else ""
        return f"[{self.timestamp}] {who}{self.text}"


def _to_ms(hours: str | None, minutes: str, seconds: str, fraction: str) -> int:
    return (
        int(hours or 0) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1000
        + int(fraction.ljust(3, "0"))
    )


def parse_vtt(raw: str) -> list[Cue]:
    """Parse WEBVTT into cues, skipping anything that doesn't parse."""
    cues: list[Cue] = []
    # Blocks are separated by blank lines; \r\n is normalized first because
    # Zoom's files are CRLF and a stray \r would end up inside the spoken text.
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if lines[0].strip().upper().startswith("WEBVTT"):
            continue

        timing_index = next(
            (i for i, line in enumerate(lines) if _TIMING.match(line.strip())), None
        )
        if timing_index is None:
            continue  # a NOTE block, a stray cue id, or a truncated cue

        match = _TIMING.match(lines[timing_index].strip())
        assert match is not None  # guarded by the search above
        start_ms = _to_ms(match.group(1), match.group(2), match.group(3), match.group(4))
        end_ms = _to_ms(match.group(5), match.group(6), match.group(7), match.group(8))

        body = " ".join(line.strip() for line in lines[timing_index + 1 :]).strip()
        if not body:
            continue

        speaker: str | None = None
        speaker_match = _SPEAKER.match(body)
        if speaker_match:
            candidate, remainder = speaker_match.group(1).strip(), speaker_match.group(2).strip()
            # A "speaker" containing sentence punctuation is almost certainly
            # just a sentence with a colon in it, not an attribution.
            if remainder and not re.search(r"[.!?]", candidate):
                speaker, body = candidate, remainder

        cues.append(Cue(start_ms=start_ms, end_ms=end_ms, speaker=speaker, text=body))

    return cues


def speakers(cues: list[Cue]) -> list[str]:
    """Distinct speakers, in first-appearance order."""
    seen: list[str] = []
    for cue in cues:
        if cue.speaker and cue.speaker not in seen:
            seen.append(cue.speaker)
    return seen
