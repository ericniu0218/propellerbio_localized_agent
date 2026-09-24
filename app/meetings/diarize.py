"""Working out who was speaking, from a single mixed microphone.

Until now a recording made here was honestly unattributed: one microphone in a
room produces one track, and the transcript could say what was said but not who
said it. That is the single thing people asked for first, and for a meeting of
four people it is most of the value — "Priya owns the protocol update" is a
different note from "somebody owns the protocol update".

How it works. A segmentation model marks where speech is and where the voice
changes; an embedding model turns each stretch into a vector; the vectors are
clustered, and each cluster becomes a speaker. Then every whisper cue is given
the speaker whose segments it overlaps most. All three models are ONNX running
locally through sherpa-onnx — no PyTorch, no service call, nothing leaves the
machine, which is the only kind of diarisation this product can use.

**Say how many people were there.** Left to decide the number itself, the
clustering badly over-splits on real meeting audio: on a genuine three-person
call recorded through a laptop it found 46 speakers at its default setting and
still 24 at the most conservative one, because people talked over each other
and laughed, and a half-second of two voices at once embeds like neither of
them. Given the number, it is reliable — the same recording resolves cleanly.
So `expected_speakers` is asked for in the UI and passed straight through. With
no number the estimate is still made, and it is still worth having, but it is
the weaker mode and the caller is told which one it used.

Diarisation is optional everywhere. Without the models installed a recording
transcribes exactly as before and is reported as unattributed, the same way the
whole feature degrades without whisper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import sherpa_onnx
    import soundfile

    DIARIZATION_AVAILABLE = True
except ImportError:  # pragma: no cover - deployments without the extra
    DIARIZATION_AVAILABLE = False

DIARIZATION_MISSING = (
    "Speaker identification is unavailable: sherpa-onnx is not installed. "
    "Install requirements.txt to enable it."
)

# Segments shorter than this are not embedded. A very short stretch of speech
# produces an unreliable vector — it is mostly the start of a word — and those
# are what the clustering splinters on. Raising it from 0.3s to 1.0s roughly
# halved the spurious speaker count on a real recording.
MIN_SPEECH_S = 1.0
# How long a pause inside one person's turn may be before it is cut in two.
MIN_SILENCE_S = 0.5
# Only consulted when no speaker count is given. Higher merges more.
DEFAULT_THRESHOLD = 0.8


@dataclass(frozen=True)
class SpeakerTurn:
    start_ms: int
    end_ms: int
    speaker: int


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    speaker_count: int
    estimated: bool
    """True when nobody said how many people were there, so the number of
    speakers was guessed — much less reliable on overlapping speech."""
    note: str | None = None
    """Why there is no result, when there is none."""

    @property
    def ok(self) -> bool:
        return bool(self.turns)


def models_present(segmentation: Path, embedding: Path) -> list[str]:
    missing = []
    if not segmentation.exists():
        missing.append(
            f"No speaker-segmentation model at {segmentation}. Get "
            "sherpa-onnx-pyannote-segmentation-3-0 from the sherpa-onnx releases."
        )
    if not embedding.exists():
        missing.append(
            f"No speaker-embedding model at {embedding}. Get "
            "nemo_en_titanet_large.onnx from the sherpa-onnx releases."
        )
    return missing


def diarize(
    wav: Path,
    segmentation: Path,
    embedding: Path,
    expected_speakers: int | None = None,
    threads: int = 2,
) -> DiarizationResult:
    """Who spoke when, in one 16 kHz mono WAV.

    Returns an empty result rather than raising: a meeting whose speakers could
    not be told apart is still a meeting worth keeping, and the transcript is
    the part that cannot be recreated.
    """
    if not DIARIZATION_AVAILABLE:
        return DiarizationResult([], 0, False, DIARIZATION_MISSING)

    missing = models_present(segmentation, embedding)
    if missing:
        return DiarizationResult([], 0, False, "; ".join(missing))

    try:
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation)
                ),
                num_threads=threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding), num_threads=threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                # -1 means "work it out", which is the weaker mode. A real
                # number is used exactly as given.
                num_clusters=expected_speakers if expected_speakers and expected_speakers > 0 else -1,
                threshold=DEFAULT_THRESHOLD,
            ),
            min_duration_on=MIN_SPEECH_S,
            min_duration_off=MIN_SILENCE_S,
        )
        engine = sherpa_onnx.OfflineSpeakerDiarization(config)
        audio, _ = soundfile.read(str(wav), dtype="float32", always_2d=True)
        segments = engine.process(audio[:, 0]).sort_by_start_time()
    except Exception as exc:
        # Deliberately broad. Nothing here is worth losing a meeting over.
        logger.warning("Diarisation failed for %s: %s", wav.name, exc)
        return DiarizationResult([], 0, False, f"Speaker identification failed: {exc}")

    turns = [
        SpeakerTurn(int(s.start * 1000), int(s.end * 1000), int(s.speaker)) for s in segments
    ]
    return DiarizationResult(
        turns=turns,
        speaker_count=len({t.speaker for t in turns}),
        estimated=not (expected_speakers and expected_speakers > 0),
    )


def label_for(index: int) -> str:
    """The placeholder name a speaker carries until somebody renames them.

    One-based, because "Speaker 0" reads like a bug to everyone who is not a
    programmer, and these labels are shown to scientists.
    """
    return f"Speaker {index + 1}"


def flatten(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Resolve overlapping turns into one speaker at a time.

    The diariser reports simultaneous speech as overlapping turns, which is
    honest — two people really were talking — but it makes "who owns this
    moment" ambiguous, and naively taking the largest overlap hands every
    moment to whoever spoke longest. On a real two-person recording that
    collapsed the whole meeting onto one speaker: one long turn covered the
    monologue, the other person's interjections sat inside it, and the long
    turn won every comparison.

    So the shortest turn claims its span first and longer ones fill only what
    is left. A short turn sitting inside a long one is an interjection, and it
    wins the seconds it covers, which is exactly the case that was being lost.
    """
    if not turns:
        return []
    timeline: list[SpeakerTurn] = []
    for turn in sorted(turns, key=lambda t: t.end_ms - t.start_ms):
        remaining = [(turn.start_ms, turn.end_ms)]
        for placed in timeline:
            carved: list[tuple[int, int]] = []
            for start, end in remaining:
                if placed.end_ms <= start or placed.start_ms >= end:
                    carved.append((start, end))
                    continue
                if start < placed.start_ms:
                    carved.append((start, placed.start_ms))
                if placed.end_ms < end:
                    carved.append((placed.end_ms, end))
            remaining = carved
        timeline += [SpeakerTurn(s, e, turn.speaker) for s, e in remaining if e > s]
    return sorted(timeline, key=lambda t: t.start_ms)


def speaker_change_points(result: DiarizationResult) -> list[int]:
    """Only the moments where the speaker actually changes.

    This is what the transcriber is told to cut on, and the distinction is not
    cosmetic. Handing it every turn start and end cuts the audio at every pause
    the diariser noticed, including the ones where the same person carried on
    talking — on an 18-minute recording that produced 440 slices.

    Whisper pads every slice it is given to its 30-second window, so a
    one-second slice costs the same as a thirty-second one. 440 slices is
    therefore about 3.7 hours of encoder work for 18 minutes of audio, which is
    exactly the 1011 seconds it took. Cutting only where the voice changes
    keeps the property that matters — one speaker per cue — at a fraction of
    the slices.
    """
    timeline = flatten(result.turns)
    changes: list[int] = []
    for earlier, later in zip(timeline, timeline[1:]):
        if earlier.speaker != later.speaker:
            # Cut midway through any gap between them, so neither side is
            # clipped and the cut lands in silence where there is any.
            changes.append((earlier.end_ms + later.start_ms) // 2)
    return changes


def assign(cues: list, result: DiarizationResult) -> tuple[list, dict[int, int]]:
    """Give every cue the speaker who holds most of it.

    Overlap rather than containment, because the two are segmented separately:
    whisper cuts on silence and the diariser cuts on voice change, so their
    boundaries never line up exactly. A cue that straddles a handover belongs to
    whoever holds more of it.

    Returns the relabelled cues and a count of cues per speaker, which is what
    tells the caller whether a speaker is real or is two seconds of somebody
    coughing.
    """
    from dataclasses import replace

    if not result.ok or not cues:
        return cues, {}

    timeline = flatten(result.turns)
    counts: dict[int, int] = {}
    relabelled = []
    for cue in cues:
        totals: dict[int, int] = {}
        for turn in timeline:
            overlap = min(cue.end_ms, turn.end_ms) - max(cue.start_ms, turn.start_ms)
            if overlap > 0:
                totals[turn.speaker] = totals.get(turn.speaker, 0) + overlap
        if not totals:
            relabelled.append(cue)
            continue
        speaker = max(totals, key=totals.get)
        counts[speaker] = counts.get(speaker, 0) + 1
        relabelled.append(replace(cue, speaker=label_for(speaker)))
    return relabelled, counts
