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

# Zoom's intermediate format. A local recording lands as these and is only
# readable once Zoom converts it, which does not always happen on its own -
# quitting Zoom before conversion finishes leaves them behind indefinitely.
UNCONVERTED_SUFFIX = ".zoom"

# "Speech segment 3: start = 3322.00, end = 3558.00" — centiseconds.
_VAD_LINE = re.compile(r"start\s*=\s*(?P<start>[\d.]+),\s*end\s*=\s*(?P<end>[\d.]+)")

# VAD cuts tight enough to clip a leading consonant.
_SLICE_PAD_MS = 150

# Whisper writes these in place of words when a slice turns out to hold no
# speech. They are markers, not something anybody said.
_NON_SPEECH = re.compile(r"^\[?(BLANK_AUDIO|SILENCE|INAUDIBLE|NO_SPEECH|MUSIC|SOUND)\]?$", re.IGNORECASE)


def is_speech(text: str) -> bool:
    """Whether a cue holds words, rather than whisper's way of saying nothing.

    Two shapes get through. The marker tokens above are the documented ones.
    The other is punctuation on its own: given a stretch of room noise,
    whisper emits a cue whose entire text is "," or ".". Seen twice in one
    real two-minute recording, where they appeared as cues spanning
    eighteen and four seconds and read, to somebody looking at the
    transcript, like the recorder had simply broken. A cue has to contain at
    least one letter or digit to be something a person said.
    """
    stripped = text.strip()
    if not stripped or _NON_SPEECH.match(stripped):
        return False
    return any(character.isalnum() for character in stripped)


def clean_cue_text(text: str) -> str:
    """Tidy the edges of one cue without touching what was said.

    Slicing the audio on a speech boundary regularly leaves whisper starting a
    cue with the punctuation that ended the previous one, so a transcript line
    reads ", and I'm also going to...". It is only ever leading punctuation and
    whitespace; the words are untouched.
    """
    return text.strip().lstrip(",.;:-–— ").strip()


# Whisper processes audio in 30-second windows, so a slice longer than that
# comes back as one cue covering the whole window. Somebody talking without a
# real pause for half a minute then gets a single cue holding six sentences,
# and every citation into it collapses onto that cue's start - observed on a
# real recording, where "enrollment is 148 of 200" and "interim analysis moves
# to March", said sixteen seconds apart, both cited 00:00:30. Capping turns
# below the window keeps citations near the sentence they quote.
MAX_TURN_MS = 20_000


def _group_regions(
    regions: list[tuple[int, int]],
    max_gap_ms: int,
    max_turn_ms: int = MAX_TURN_MS,
) -> list[tuple[int, int]]:
    """Join speech regions separated by less than a breath into one turn.

    VAD splits on every pause, including mid-sentence ones, which would
    otherwise send a single sentence through whisper in three pieces and lose
    the context that makes it transcribe well. Turns are capped at
    ``max_turn_ms`` for the reason above.
    """
    if not regions:
        return []
    grouped = [regions[0]]
    for start, end in regions[1:]:
        last_start, last_end = grouped[-1]
        joinable = start - last_end <= max_gap_ms
        within_cap = end - last_start <= max_turn_ms
        if joinable and within_cap:
            grouped[-1] = (last_start, end)
        else:
            grouped.append((start, end))

    # A single uninterrupted region can still exceed the cap on its own; split
    # it on even boundaries so no slice reaches whisper's window limit.
    capped: list[tuple[int, int]] = []
    for start, end in grouped:
        span = end - start
        if span <= max_turn_ms:
            capped.append((start, end))
            continue
        pieces = -(-span // max_turn_ms)  # ceiling division
        step = -(-span // pieces)
        capped += [(start + i * step, min(start + (i + 1) * step, end)) for i in range(pieces)]
    return capped


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
    vad_binary: str = "whisper-vad-speech-segments"
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
    multi_file_speakers: list[str] = field(default_factory=list)
    """Speakers whose audio arrived as several files, which are treated as
    consecutive parts of a split recording.

    Two participants sharing one Zoom display name look exactly the same on
    disk, and for them this assumption is wrong: their speech is concurrent,
    not consecutive, so the second is timestamped after the meeting ended.
    Nothing on disk distinguishes the two cases, so callers are told which
    speakers this applies to rather than left to wonder.
    """


class WhisperTranscriber:
    """Hide whisper.cpp behind one operation: audio in, cues out."""

    def __init__(self, config: WhisperConfig, runner: CommandRunner | None = None) -> None:
        self.config = config
        # An injected runner owns command resolution (and is the seam used by
        # hermetic tests). Only the real subprocess runner depends on this
        # process's PATH; model files are required in both cases.
        self._check_host_tools = runner is None
        self.runner = runner or SubprocessCommandRunner()

    def readiness_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self._check_host_tools and shutil.which(self.config.binary) is None:
            issues.append(
                f"{self.config.binary} is not on PATH. Install it with "
                "`brew install whisper.cpp`."
            )
        if self._check_host_tools and shutil.which(self.config.vad_binary) is None:
            issues.append(
                f"{self.config.vad_binary} is not on PATH. Install it with "
                "`brew install whisper.cpp`."
            )
        if self._check_host_tools and shutil.which(self.config.ffmpeg) is None:
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

    async def duration_ms(self, audio: Path) -> int:
        """How long the file actually is.

        Taken from the container rather than from the last cue, because a file
        that ends in silence would otherwise report short and make the next
        split part overlap it.
        """
        log = audio.parent / f".{audio.stem}.probe.log"
        try:
            await self.runner.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(audio),
                ],
                log_path=log,
            )
            return int(float(log.read_text().strip().splitlines()[0]) * 1000)
        except (ToolIntegrationError, ValueError, IndexError, OSError):
            return 0
        finally:
            log.unlink(missing_ok=True)

    async def detect_speech_regions(self, wav: Path, work_dir: Path) -> list[tuple[int, int]]:
        """Where speech actually is, in milliseconds from the start of the file."""
        log = work_dir / f"{wav.stem}.vad.log"
        await self.runner.run(
            [
                self.config.vad_binary,
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

    async def transcribe_track(
        self, audio: Path, work_dir: Path, split_at_ms: list[int] | None = None
    ) -> list[Cue]:
        """One audio file to cues, each timestamped where it was actually said.

        Whisper is run over one slice per turn rather than over the whole file.
        Running it over the file - even with VAD enabled - produces segments
        that span the silence between turns, so a single cue can cover words
        said twenty-five seconds apart and its timestamp is only right for the
        first few. Slicing first means every cue's offset is its slice's known
        start, so a citation lands on the sentence it quotes.

        All slices go to one whisper invocation, so the model is loaded once
        per track no matter how many turns there are.

        ``split_at_ms`` forces an extra cut at each of those offsets. VAD alone
        splits on silence, and two people talking without a gap between them is
        one region - on a real recording that produced a single nineteen-second
        cue holding both halves of "What's your name? My name's Luke", which no
        amount of speaker labelling afterwards can separate, because the cue is
        the unit a label attaches to. A caller that knows where the voice
        changes passes those offsets here so each cue holds one speaker.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        wav = audio if audio.suffix == ".wav" else await self.to_wav(audio, work_dir)

        regions = await self.detect_speech_regions(wav, work_dir)
        if not regions:
            return []

        turns = _group_regions(regions, self.config.turn_gap_ms)
        if split_at_ms:
            turns = _split_turns_at(turns, split_at_ms)
        slice_dir = work_dir / f"{wav.stem}.slices"
        slice_dir.mkdir(parents=True, exist_ok=True)

        slices: list[tuple[Path, int, int]] = []
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
                slices.append((target, padded_start, padded_end))

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
        for path, _, _ in slices:
            args += ["-f", str(path)]

        await self.runner.run(args, log_path=work_dir / f"{wav.stem}.whisper.log")

        cues: list[Cue] = []
        transcribed_any = False
        for path, offset_ms, slice_end_ms in slices:
            # With several -f inputs whisper writes "<input path>.vtt" rather
            # than using a single -of prefix.
            vtt = path.with_suffix(path.suffix + ".vtt")
            if not vtt.exists():
                continue
            transcribed_any = True
            for cue in parse_vtt(vtt.read_text(errors="replace")):
                if not is_speech(cue.text):
                    continue
                text = clean_cue_text(cue.text)
                if not is_speech(text):
                    continue
                # Whisper reports a cue's end at its 30-second window edge even
                # when the slice is shorter, which makes a 5-second remark look
                # half a minute long and inflates the meeting's duration. The
                # slice's own end is the real bound.
                start_ms = cue.start_ms + offset_ms
                cues.append(
                    Cue(
                        start_ms=start_ms,
                        end_ms=min(cue.end_ms + offset_ms, slice_end_ms),
                        speaker=None,
                        text=text,
                    )
                )

        # An empty result is only a failure if whisper never ran. Somebody who
        # joined and said nothing is ordinary: VAD picks up a cough or a door,
        # whisper reports [BLANK_AUDIO], and every cue is filtered out. Raising
        # there would fail the entire meeting because one person stayed quiet.
        if not transcribed_any:
            raise ToolIntegrationError(
                f"whisper produced no transcript for {audio.name}. "
                f"See {work_dir / (wav.stem + '.whisper.log')}."
            )
        cues.sort(key=lambda cue: cue.start_ms)
        return cues


# A cut is only worth making if both halves are long enough to transcribe.
# Below this, whisper is being handed a fragment of a word, and the result is
# worse than leaving the two speakers in one cue.
_MIN_SPLIT_PIECE_MS = 700


def _split_turns_at(turns: list[tuple[int, int]], points: list[int]) -> list[tuple[int, int]]:
    """Cut every turn at each offset that falls inside it.

    Used to make cues speaker-homogeneous: see ``transcribe_track``. A cut that
    would leave a fragment too short to transcribe is skipped, so a speaker
    boundary landing a tenth of a second inside a turn does not produce a cue
    holding half a syllable.
    """
    ordered = sorted(set(points))
    result: list[tuple[int, int]] = []
    for start, end in turns:
        cuts = [
            point
            for point in ordered
            if start + _MIN_SPLIT_PIECE_MS <= point <= end - _MIN_SPLIT_PIECE_MS
        ]
        previous = start
        for point in cuts:
            result.append((previous, point))
            previous = point
        result.append((previous, end))
    return result


def _audio_files(folder: Path) -> list[Path]:
    """Every audio file in the recording, ignoring our own working directory.

    Conversions and slices are written under a dot-directory inside the
    recording folder. Without this filter a second run rediscovers the 16 kHz
    copies it made on the first, transcribes everybody twice, and reports
    speakers named "Alice Chen.16k" — so the tool silently corrupts itself the
    moment it is run twice on the same meeting.
    """
    return [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in AUDIO_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(folder).parts)
    ]


def find_unconverted(folder: Path) -> list[Path]:
    """Raw Zoom files awaiting conversion.

    Their presence alongside no audio is the difference between "this recording
    is not ready yet" and "there is nothing here", and the two need different
    advice: one is waiting, the other is a job nobody started.
    """
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == UNCONVERTED_SUFFIX
    )


def discover_participant_tracks(folder: Path) -> dict[str, list[Path]]:
    """Map speaker name to their audio files for a Zoom local recording.

    Zoom puts per-participant tracks in an "Audio Record" subfolder with the
    display name in the filename. **A participant can have more than one file**:
    Zoom splits long meetings, so a two-hour call produces several per person.
    Keeping only one of them loses most of the meeting without any error, which
    is why this returns every file per speaker rather than the last one seen.

    Files are ordered by name, which is Zoom's own chronological order.
    Returns empty when there are no per-participant tracks, which is the signal
    to fall back to the mixed recording.
    """
    tracks: dict[str, list[Path]] = {}
    for candidate in _audio_files(folder):
        match = _PARTICIPANT_FILE.match(candidate.stem)
        if match:
            name = match.group("name").strip()
            if name:
                tracks.setdefault(name, []).append(candidate)
    return tracks


def find_mixed_audio(folder: Path) -> list[Path]:
    """The combined recording, for when per-participant tracks are off.

    A list, not one file, for the same reason as above: Zoom splits long
    recordings, and returning the first would drop everything after it.
    """
    candidates = [
        path for path in _audio_files(folder) if not _PARTICIPANT_FILE.match(path.stem)
    ]
    if not candidates:
        return []
    # Prefer audio-only files over the video: both carry the same audio, but
    # the m4a is far smaller and decodes faster. Only mix the two if there are
    # no audio-only files at all.
    audio_only = [p for p in candidates if p.suffix.lower() in (".m4a", ".mp3", ".wav", ".aac")]
    return audio_only or candidates


async def _transcribe_all(
    transcriber: "WhisperTranscriber", files: list[Path], work_dir: Path
) -> list[Cue]:
    """Transcribe a speaker's files in order, offsetting each by what came before.

    Zoom's split files are consecutive, so part two's timestamps continue where
    part one's audio ended rather than restarting at zero.
    """
    cues: list[Cue] = []
    elapsed_ms = 0
    for path in files:
        part = await transcriber.transcribe_track(path, work_dir)
        cues += [
            Cue(
                start_ms=c.start_ms + elapsed_ms,
                end_ms=c.end_ms + elapsed_ms,
                speaker=c.speaker,
                text=c.text,
            )
            for c in part
        ]
        elapsed_ms += await transcriber.duration_ms(path)
    return cues


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

    # Dot-prefixed so _audio_files skips it; see the note there about what
    # happens on a second run if it does not.
    work_dir = work_dir or folder / ".bedrock-work"
    tracks = discover_participant_tracks(folder)

    if not tracks:
        mixed = find_mixed_audio(folder)
        if not mixed:
            unconverted = find_unconverted(folder)
            if unconverted:
                raise ToolIntegrationError(
                    f"This recording has not been converted yet: {folder} holds "
                    f"{len(unconverted)} .zoom file(s). Double-click "
                    f"{unconverted[0].name} in Finder, or in Zoom open "
                    "Meetings > Recorded and press Convert. Then run this again."
                )
            raise ToolIntegrationError(f"No audio files found under {folder}.")
        cues = await _transcribe_all(transcriber, mixed, work_dir)
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
    for speaker, audio_files in tracks.items():
        cues = await _transcribe_all(transcriber, audio_files, work_dir)
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
        multi_file_speakers=[s for s, files in tracks.items() if len(files) > 1],
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
