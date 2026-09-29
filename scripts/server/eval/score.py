#!/usr/bin/env python3
"""Score the meeting recorder against audio whose truth we know.

    ./.venv/bin/python scripts/server/eval/score.py

Builds a short meeting with text-to-speech — who says what, and when, is known
exactly — runs the real pipeline over it, and reports word error, how many
speakers were found, and what share of cues were attributed to the right
person. Run it before and after touching the pipeline; the numbers move.

Two things learned building this, both worth keeping in mind before trusting a
result:

* **Voices from one TTS engine are far more alike than real people.** The first
  version used two macOS female voices and they merged into a single speaker,
  which looked like a bug in the product and was a property of the test audio.
  They are now picked across gender and accent. For anything about speaker
  identification, trust a real recording over this one.
* **Word error here is flattering.** Synthesised speech has no crosstalk, no
  room, no accent the model has not heard. Treat these numbers as a regression
  check, not as what the product will do in a meeting room.

Set `EVAL_AUDIO` and `EVAL_TRUTH` to score a real recording instead, where
`EVAL_TRUTH` is JSON: `[{"who":…, "start":…, "end":…, "text":…}, …]`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bedrock.server.meetings.config import RecordingConfig  # noqa: E402
from bedrock.server.meetings.pipeline import transcribe  # noqa: E402

HERE = Path(__file__).parent
BUILD = HERE / "build"

# Across gender and accent on purpose — see the module docstring.
SCRIPT = [
    ("Samantha", "Alice", "Good morning everyone. Let's start with enrollment for the KRAS trial."),
    ("Daniel", "Bob", "We are at one hundred forty eight of two hundred patients."),
    ("Samantha", "Alice", "That's behind the October target. What is blocking us?"),
    ("Daniel", "Bob", "The site coordinator in Boston left last week, so we are not opening a new site this month."),
    ("Moira", "Cara", "I can cover Boston remotely until we backfill the coordinator role."),
    ("Samantha", "Alice", "Good. Decision: we push the interim analysis to November tenth."),
    ("Moira", "Cara", "I will update the protocol document by Friday."),
    ("Daniel", "Bob", "And I will send the enrollment tracker tonight."),
    ("Samantha", "Alice", "One more thing. The G12C cohort showed a partial response in three of twelve patients."),
    ("Moira", "Cara", "Is that above the futility boundary for the interim look?"),
    ("Daniel", "Bob", "Yes, the boundary is two responses, so we continue."),
    ("Samantha", "Alice", "Agreed. Let's reconvene next Tuesday."),
]
GAP_S = 0.45
VOCABULARY = "KRAS, G12C, Boston, cohort, enrollment, protocol, interim"


def build() -> tuple[Path, list[dict]]:
    """Synthesise the meeting and write down exactly who spoke when."""
    BUILD.mkdir(parents=True, exist_ok=True)
    audio = BUILD / "meeting.wav"
    truth_path = BUILD / "truth.json"
    if audio.exists() and truth_path.exists():
        return audio, json.loads(truth_path.read_text())

    if not shutil_which("say"):
        raise SystemExit("`say` is macOS-only. Point EVAL_AUDIO/EVAL_TRUTH at a real recording instead.")

    parts = []
    for index, (voice, _who, text) in enumerate(SCRIPT):
        aiff, wav = BUILD / f"t{index:02d}.aiff", BUILD / f"t{index:02d}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)],
            check=True,
        )
        parts.append(wav)

    gap = BUILD / "gap.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-t", str(GAP_S),
         "-i", "anullsrc=r=16000:cl=mono", str(gap)],
        check=True,
    )
    listing = BUILD / "list.txt"
    with listing.open("w") as handle:
        for index, wav in enumerate(parts):
            if index:
                handle.write(f"file '{gap.resolve()}'\n")
            handle.write(f"file '{wav.resolve()}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-ar", "16000", "-ac", "1", str(audio)],
        check=True,
    )

    at, truth = 0.0, []
    for index, (_voice, who, text) in enumerate(SCRIPT):
        if index:
            at += GAP_S
        with wave.open(str(parts[index])) as handle:
            seconds = handle.getnframes() / handle.getframerate()
        truth.append({"who": who, "start": round(at, 3), "end": round(at + seconds, 3), "text": text})
        at += seconds
    truth_path.write_text(json.dumps(truth, indent=1))
    return audio, truth


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def word_error(reference: list[str], hypothesis: list[str]) -> float:
    matcher = SequenceMatcher(None, reference, hypothesis)
    correct = sum(block.size for block in matcher.get_matching_blocks())
    return (len(reference) - correct) / max(1, len(reference))


async def main() -> None:
    audio_env, truth_env = os.environ.get("EVAL_AUDIO"), os.environ.get("EVAL_TRUTH")
    if audio_env and truth_env:
        audio, truth = Path(audio_env), json.loads(Path(truth_env).read_text())
    else:
        audio, truth = build()

    os.environ.setdefault("MEETING_VOCABULARY", VOCABULARY)
    expected = len({turn["who"] for turn in truth})

    def speaker_at(ms: int) -> str | None:
        for turn in truth:
            if turn["start"] * 1000 <= ms <= turn["end"] * 1000:
                return turn["who"]
        return None

    started = time.time()
    outcome = await transcribe(audio, RecordingConfig.from_env(), BUILD / "work", expected_speakers=expected)
    elapsed = time.time() - started

    reference = words(" ".join(turn["text"] for turn in truth))
    hypothesis = words(" ".join(cue.text for cue in outcome.cues))

    # A label is only ever a placeholder, so score the mapping that fits best
    # rather than the arbitrary number the clustering happened to assign.
    pairs = [(cue, speaker_at((cue.start_ms + cue.end_ms) // 2)) for cue in outcome.cues]
    ghosts = sum(1 for _cue, who in pairs if who is None)
    votes: dict[str | None, dict[str, int]] = {}
    for cue, who in pairs:
        if who:
            votes.setdefault(cue.speaker, {}).setdefault(who, 0)
            votes[cue.speaker][who] += 1
    mapping = {label: max(tally, key=tally.get) for label, tally in votes.items()}
    real = [(cue, who) for cue, who in pairs if who]
    attribution = sum(1 for cue, who in real if mapping.get(cue.speaker) == who) / max(1, len(real))

    seconds = outcome.duration_ms / 1000
    print(f"audio              {seconds:.1f}s ({len(truth)} turns, {expected} speakers)")
    print(f"wall clock         {elapsed:.1f}s   ({elapsed / max(seconds, 1):.2f}x real time)")
    print(f"word error         {word_error(reference, hypothesis) * 100:.1f}%")
    print(f"speakers found     {outcome.speaker_count}/{expected}")
    print(f"attribution        {attribution * 100:.1f}%  ({len(real)} attributable cues)")
    print(f"cues               {len(outcome.cues)}  (ghost cues in silence: {ghosts})")
    if outcome.vocabulary.total:
        print(f"vocabulary         {outcome.vocabulary.describe()}")


if __name__ == "__main__":
    asyncio.run(main())
