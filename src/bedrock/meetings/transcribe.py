"""Transcribing a Zoom *local* recording, on this machine.

Zoom only generates transcripts for **cloud** recordings, and cloud recording
is a paid feature. Local recording is available on every plan including the
free one, writes to the host's own disk, and uploads nothing. So the transcript
is ours to produce - which is better for an air-gapped customer anyway, because
the audio never reaches a third party at all.

The part worth understanding is where the speaker labels come from. Zoom has a
setting, "Record a separate audio file for each participant who speaks", and
when it is on a local recording produces one full-length track per person with
silence wherever they are not talking. Transcribing each track separately gives
perfect speaker attribution for free: every word in Bob's file was said by Bob,
by construction. No diarisation model, no clustering, no voice fingerprinting -
and better labels than any of those would produce.

Timestamps stay comparable across tracks because every file spans the whole
meeting, so merging is a sort by start time.

Requires `whisper-cli` (brew install whisper.cpp), `ffmpeg`, and a ggml model.
`readiness_issues()` reports whatever is missing rather than failing mid-run,
matching the AlphaFold adapter next door.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from bedrock.biotech.errors import ToolIntegrationError
from bedrock.biotech.process import CommandRunner, SubprocessCommandRunner
from bedrock.meetings.transcript import Cue, parse_vtt

# Zoom writes per-participant tracks as "audio<digits>_<Display Name>.m4a".
_PARTICIPANT_FILE = re.compile(r"^audio\d+_(?P<name>.+)$", re.IGNORECASE)

# Whisper wants 16 kHz mono; anything else is resampled internally and slower.
TARGET_SAMPLE_RATE = 16_000
AUDIO_SUFFIXES = (".m4a", ".mp4", ".wav", ".mp3", ".aac", ".aiff")

# "Speech segment 3: start = 3322.00, end = 3558.00" — centiseconds.
_VAD_LINE = re.compile(r"start\s*=\s*(?P<start>[\d.]+),\s*end\s*=\s*(?P<end>[\d.]+)")

# VAD cuts tight enough to clip a leading consonant.
_SLICE_PAD_MS = 150

# Whisper writes these in place of words when a slice turns out to hold no
# speech. They are markers, not something anybody said.
_NON_SPEECH = re.compile(r"^\[?(BLANK_AUDIO|SILENCE|INAUDIBLE|NO_SPEECH|MUSIC|SOUND)\]?$", re.IGNORECASE)


def is_speech(text: str) -> bool:
    return bool(text.strip()) and not _NON_SPEECH.match(text.strip())


def _group_regions(regions: list[tuple[int, int]], max_gap_ms: int) -> list[tuple[int, int]]:
    """Join speech regions separated by less than a breath into one turn.

    VAD splits on every pause, including mid-sentence ones, which would
    otherwise send a single sentence through whisper in three pieces and lose
    the context that makes it transcribe well.
    """
    if not regions:
        return []
    grouped = [regions[0]]
    for start, end in regions[1:]:
        last_start, last_end = grouped[-1]
        if start - last_end <= max_gap_ms:
            grouped[-1] = (last_start, end)
        else:
            grouped.append((start, end))
    return grouped


@dataclass(frozen=True)
class WhisperConfig:
    model: Path
    vad_model: Path
    """Silero VAD weights, and not optional.

    Without VAD, whisper's segment boundaries span silence: a cue begins where
    the previous one ended rather than where speech starts. On a
    per-participant track - which is mostly silence by design - that puts a
    line minutes away from when it was said. Measured here: Bob's opening line
    was reported at 00:00:00 without VAD and 00:00:08.830 with it, against a
    true 8.9s. Every citation and the whole cross-track merge rest on these
    numbers, so a missing VAD model is a readiness failure rather than a
    downgrade. Get it from huggingface.co/ggml-org/whisper-vad.
    """
    binary: str = "whisper-cli"
    ffmpeg: str = "ffmpeg"
    threads: int = 4
    language: str = "en"
    turn_gap_ms: int = 600
    """Silence longer than this starts a new turn. Below it, the pause is
    treated as mid-sentence and the speech is transcribed as one piece."""
    initial_prompt: str | None = None
    """Vocabulary hint. Whisper renders unfamiliar proper nouns phonetically -
    it writes "PREA" for "Priya" and will do worse with drug and assay names.
    Seeding the decoder with the study's real terms is the cheapest fix there
    is, and it is per-customer, so it belongs in configuration rather than code.
    """


@dataclass
class TranscriptionResult:
    cues: list[Cue]
    speakers: list[str]
    per_speaker_counts: dict[str, int] = field(default_factory=dict)
    unattributed: bool = False
    """True when the recording had no per-participant tracks, so nothing can be
    attributed to a named person."""


class WhisperTranscriber:
    """Hide whisper.cpp behind one operation: audio in, cues out."""

    def __init__(self, config: WhisperConfig, runner: CommandRunner | None = None) -> None:
        self.config = config
        self.runner = runner or SubprocessCommandRunner()

    def readiness_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if shutil.which(self.config.binary) is None:
            issues.append(
                f"{self.config.binary} is not on PATH. Install it with "
                "`brew install whisper.cpp`."
            )
        if shutil.which(self.config.ffmpeg) is None:
            issues.append(
                f"{self.config.ffmpeg} is not on PATH. Install it with `brew install ffmpeg`."
            )
        if not self.config.model.exists():
            issues.append(
                f"No speech model at {self.config.model}. Download a ggml model, e.g. "
                "ggml-small.en.bin from huggingface.co/ggerganov/whisper.cpp."
            )
        if not self.config.vad_model.exists():
            issues.append(
                f"No VAD model at {self.config.vad_model}. Without it, timestamps on "
                "tracks containing silence are wrong by minutes — see WhisperConfig. "
                "Download ggml-silero-v5.1.2.bin from huggingface.co/ggml-org/whisper-vad."
            )
        return tuple(issues)

    async def to_wav(self, source: Path, work_dir: Path) -> Path:
        """Normalise any audio Zoom produced into 16 kHz mono PCM."""
        work_dir.mkdir(parents=True, exist_ok=True)
        target = work_dir / f"{source.stem}.16k.wav"
        await self.runner.run(
            [
                self.config.ffmpeg, "-y", "-v", "error",
                "-i", str(source),
                "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
                str(target),
            ],
            log_path=work_dir / f"{source.stem}.ffmpeg.log",
        )
        if not target.exists():
            raise ToolIntegrationError(f"ffmpeg produced no output for {source.name}.")
        return target

    async def detect_speech_regions(self, wav: Path, work_dir: Path) -> list[tuple[int, int]]:
        """Where speech actually is, in milliseconds from the start of the file."""
        log = work_dir / f"{wav.stem}.vad.log"
        await self.runner.run(
            [
                "whisper-vad-speech-segments",
                "-vm", str(self.config.vad_model),
                "-f", str(wav),
                "-t", str(self.config.threads),
            ],
            log_path=log,
        )
        regions: list[tuple[int, int]] = []
        for line in log.read_text(errors="replace").splitlines():
            match = _VAD_LINE.search(line)
            if match:
                # The tool reports centiseconds.
                regions.append((int(float(match.group("start")) * 10), int(float(match.group("end")) * 10)))
        return regions

    async def transcribe_track(self, audio: Path, work_dir: Path) -> list[Cue]:
        """One audio file to cues, each timestamped where it was actually said.

        Whisper is run over one slice per turn rather than over the whole file.
        Running it over the file - even with VAD enabled - produces segments
        that span the silence between turns, so a single cue can cover words
        said twenty-five seconds apart and its timestamp is only right for the
        first few. Slicing first means every cue's offset is its slice's known
        start, so a citation lands on the sentence it quotes.

        All slices go to one whisper invocation, so the model is loaded once
        per track no matter how many turns there are.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        wav = audio if audio.suffix == ".wav" else await self.to_wav(audio, work_dir)

        regions = await self.detect_speech_regions(wav, work_dir)
        if not regions:
            return []

        turns = _group_regions(regions, self.config.turn_gap_ms)
        slice_dir = work_dir / f"{wav.stem}.slices"
        slice_dir.mkdir(parents=True, exist_ok=True)

        slices: list[tuple[Path, int]] = []
        for index, (start_ms, end_ms) in enumerate(turns):
            # A little padding either side: VAD trims tight and can clip the
            # first consonant or a trailing syllable.
            padded_start = max(0, start_ms - _SLICE_PAD_MS)
            padded_end = end_ms + _SLICE_PAD_MS
            target = slice_dir / f"turn{index:04d}.wav"
            await self.runner.run(
                [
                    self.config.ffmpeg, "-y", "-v", "error",
                    "-i", str(wav),
                    "-ss", f"{padded_start / 1000:.3f}",
                    "-to", f"{padded_end / 1000:.3f}",
                    "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
                    str(target),
                ],
                log_path=slice_dir / f"turn{index:04d}.ffmpeg.log",
            )
            if target.exists():
                slices.append((target, padded_start))

        if not slices:
            return []

        args = [
            self.config.binary,
            "-m", str(self.config.model),
            "-l", self.config.language,
            "-t", str(self.config.threads),
            "-ovtt", "--no-prints",
        ]
        if self.config.initial_prompt:
            args += ["--prompt", self.config.initial_prompt]
        for path, _ in slices:
            args += ["-f", str(path)]

        await self.runner.run(args, log_path=work_dir / f"{wav.stem}.whisper.log")

        cues: list[Cue] = []
        for path, offset_ms in slices:
            # With several -f inputs whisper writes "<input path>.vtt" rather
            # than using a single -of prefix.
            vtt = path.with_suffix(path.suffix + ".vtt")
            if not vtt.exists():
                continue
            for cue in parse_vtt(vtt.read_text(errors="replace")):
                if not is_speech(cue.text):
                    continue
                cues.append(
                    Cue(
                        start_ms=cue.start_ms + offset_ms,
                        end_ms=cue.end_ms + offset_ms,
                        speaker=None,
                        text=cue.text,
                    )
                )

        if not cues:
            raise ToolIntegrationError(
                f"whisper produced no transcript for {audio.name}. "
                f"See {work_dir / (wav.stem + '.whisper.log')}."
            )
        cues.sort(key=lambda cue: cue.start_ms)
        return cues


def discover_participant_tracks(folder: Path) -> dict[str, Path]:
    """Map speaker name to audio file for a Zoom local recording.

    Zoom puts per-participant tracks in an "Audio Record" subfolder; the
    display name is in the filename. Returns empty when the recording has no
    per-participant tracks, which is the signal to fall back to the mixed file.
    """
    tracks: dict[str, Path] = {}
    for candidate in sorted(folder.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        match = _PARTICIPANT_FILE.match(candidate.stem)
        if match:
            name = match.group("name").strip()
            if name:
                tracks[name] = candidate
    return tracks


def find_mixed_audio(folder: Path) -> Path | None:
    """The single combined recording, for when per-participant tracks are off."""
    candidates = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in AUDIO_SUFFIXES
        and not _PARTICIPANT_FILE.match(path.stem)
    ]
    if not candidates:
        return None
    # Prefer an audio-only file over the video; both carry the same audio, but
    # the m4a is far smaller and decodes faster.
    audio_only = [p for p in candidates if p.suffix.lower() in (".m4a", ".mp3", ".wav", ".aac")]
    return (audio_only or candidates)[0]


async def transcribe_recording(
    folder: Path,
    transcriber: WhisperTranscriber,
    work_dir: Path | None = None,
) -> TranscriptionResult:
    """Transcribe a Zoom local recording folder into speaker-attributed cues."""
    issues = transcriber.readiness_issues()
    if issues:
        raise ToolIntegrationError(" ".join(issues))
    if not folder.is_dir():
        raise ToolIntegrationError(f"{folder} is not a folder.")

    work_dir = work_dir or folder / ".bedrock-work"
    tracks = discover_participant_tracks(folder)

    if not tracks:
        mixed = find_mixed_audio(folder)
        if mixed is None:
            raise ToolIntegrationError(f"No audio files found under {folder}.")
        cues = await transcriber.transcribe_track(mixed, work_dir)
        return TranscriptionResult(
            cues=cues,
            speakers=[],
            per_speaker_counts={},
            unattributed=True,
        )

    # Sequential on purpose: whisper.cpp already saturates the GPU, so running
    # tracks concurrently makes the whole job slower, not faster.
    merged: list[Cue] = []
    counts: dict[str, int] = {}
    for speaker, audio in tracks.items():
        cues = await transcriber.transcribe_track(audio, work_dir)
        merged += [
            Cue(start_ms=c.start_ms, end_ms=c.end_ms, speaker=speaker, text=c.text)
            for c in cues
        ]
        counts[speaker] = len(cues)

    # Every track spans the whole meeting, so start time orders them correctly
    # even when two people talk over each other.
    merged.sort(key=lambda cue: (cue.start_ms, cue.speaker or ""))
    return TranscriptionResult(
        cues=merged,
        speakers=[s for s in tracks if counts.get(s)],
        per_speaker_counts=counts,
    )


def to_vtt(cues: list[Cue]) -> str:
    """Render cues back out as WEBVTT, so the result is interchangeable with a
    transcript Zoom would have produced."""
    lines = ["WEBVTT", ""]
    for index, cue in enumerate(cues, 1):
        who = f"{cue.speaker}: " if cue.speaker else ""
        lines += [str(index), f"{_stamp(cue.start_ms)} --> {_stamp(cue.end_ms)}", f"{who}{cue.text}", ""]
    return "\n".join(lines)


def _stamp(ms: int) -> str:
    seconds, millis = divmod(ms, 1000)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}.{millis:03d}"
