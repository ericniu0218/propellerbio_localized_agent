"""WEBVTT parsing for meeting transcripts.

Zoom and whisper.cpp both emit WEBVTT: a cue number, a timestamp range, then a
line of the form ``Speaker Name: what they said``. Handing that to a model
verbatim spends a large share of the context window on cue numbers and
millisecond timestamps that carry no meaning for the question being asked.

Malformed blocks are skipped rather than raising. A partial transcript is still
worth summarising, and a meeting is not worth losing because one cue arrived
truncated.

The agent-pack server's meeting_transcripts tool (bedrock.server.tools) imports
this module rather than keeping its own copy. It uses Cue, parse_vtt and
speakers, plus Cue.timestamp and Cue.render(). Changing any of those needs a
matching change there, in the same commit; tests/server/test_meeting_transcripts.py
fails if they drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TIMING = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)

# "Alice Chen: we're two weeks behind" — but not a sentence that happens to
# contain a colon, and not a bare URL.
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
    """Parse WEBVTT into cues, skipping anything that does not parse."""
    cues: list[Cue] = []
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n", normalized):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines or lines[0].strip().upper().startswith("WEBVTT"):
            continue

        timing_index = next(
            (i for i, line in enumerate(lines) if _TIMING.match(line.strip())), None
        )
        if timing_index is None:
            continue

        match = _TIMING.match(lines[timing_index].strip())
        assert match is not None
        start_ms = _to_ms(*match.group(1, 2, 3, 4))
        end_ms = _to_ms(*match.group(5, 6, 7, 8))

        body = " ".join(line.strip() for line in lines[timing_index + 1 :]).strip()
        if not body:
            continue

        speaker: str | None = None
        speaker_match = _SPEAKER.match(body)
        if speaker_match:
            candidate = speaker_match.group(1).strip()
            remainder = speaker_match.group(2).strip()
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


def word_count(cues: list[Cue]) -> int:
    return sum(count_words(cue.text) for cue in cues)


# Han, Hiragana, Katakana and Hangul: scripts written without spaces between
# words, where splitting on whitespace counts a whole sentence as one word.
_CJK = re.compile(
    "[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\uf900-\ufaff]"
)


def count_words(text: str) -> int:
    """Words in a line, for budgets and thresholds rather than for linguistics.

    Whitespace-separated tokens, plus one for each Chinese, Japanese or Korean
    character. A Mandarin sentence of twenty characters is about that many
    words of English; counted by whitespace it was one, and a twenty-minute
    Mandarin meeting came out under the forty-word floor below which no
    overview is written.
    """
    cjk = len(_CJK.findall(text))
    # What is left of a token once its CJK characters are gone counts only if
    # it still has a letter or digit in it: "KRAS" does, "，。" does not.
    latin = sum(1 for token in text.split() if _WORDISH.search(_CJK.sub("", token)))
    return cjk + latin


_WORDISH = re.compile(r"[^\W_]")
