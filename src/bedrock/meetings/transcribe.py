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

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from bedrock.biotech.errors import ToolIntegrationError
from bedrock.biotech.process import CommandRunner, SubprocessCommandRunner
from bedrock.meetings.transcript import Cue, parse_vtt

# Zoom has written per-participant tracks two different ways, and a recording
# folder tells you which only by what is in it. Older builds separate the
# display name with an underscore: "audio<digits>_<Display Name>.m4a". The
# builds seen in 2026 write "audio<Display Name><recording id>.m4a" with no
# separator at all, into the "Audio Record" subfolder. Matching only the first
# shape is not a cosmetic miss: the tracks go unrecognised, the meeting falls
# back to the mixed recording, and the tool advises turning on a Zoom setting
# that is already on.
_PARTICIPANT_FILE = re.compile(r"^audio\d+_(?P<name>.+)$", re.IGNORECASE)

# Zoom's own name for the folder holding per-participant tracks.
_PARTICIPANT_DIR = "audio record"

_AUDIO_PREFIX = "audio"

# Whisper wants 16 kHz mono; anything else is resampled internally and slower.
TARGET_SAMPLE_RATE = 16_000

# What `normalize_audio` applies. highpass=80 Hz is below every voice and above
# table rumble and HVAC. dynaudnorm: 300 ms frames (f) smoothed over 21 of
# them (g) so a word is levelled without pumping; at most 20x gain (m), so a
# whisper across the room comes up but the silence between sentences does not
# become a roar; 0.9 peak (p) leaves headroom for whisper's own resampling.
LEVELLING_FILTER = "highpass=f=80,dynaudnorm=f=300:g=21:m=20:p=0.9"


# `language` values with a meaning of their own, beside whisper's codes.
DETECT = "detect"
"""Listen first, then decide. A sample of the speech is run through whisper's
language detection; if one language carries most of it the whole recording is
transcribed in that language (a fixed language is a bias the decoder trusts,
and is more accurate on accented speech than guessing each turn), otherwise
every turn is detected on its own so a meeting that switches between Mandarin
and English gets each turn in its own script."""
AUTO = "auto"
"""whisper's own per-file detection: each slice is guessed separately."""

# Whisper decides between simplified and traditional characters from its
# prompt, and with no prompt a Mandarin meeting comes out in whichever it
# feels like line by line. This sentence ("The following are sentences in
# Mandarin.") is the community's standard nudge towards simplified, and in
# testing it also kept an English sentence in the middle of a Mandarin clip
# that was dropped without it. It goes before the vocabulary because whisper
# weights the start of its prompt most.
MANDARIN_PROMPT = "以下是普通话的句子。"

# How much of the sampled speech one language must carry for the whole
# recording to be transcribed in it. Below this the meeting is treated as
# mixed and every turn is detected separately. 0.8 rather than 0.5: a meeting
# in Mandarin with a fifth of it in English is a mixed meeting, and forcing
# those English turns into Chinese characters (which is what a fixed `zh`
# does to them) is the worse mistake.
DOMINANT_SHARE = 0.8
# Slices sampled for detection. Detection reads a 30-second window per slice,
# about a second each on a laptop, so a dozen keeps it under fifteen seconds
# for any length of meeting. The longest slices are taken: a long turn is a
# sentence or more, which detection is sure about; a one-word turn is not.
DETECT_SAMPLES = 12

_DETECTED_LINE = re.compile(r"auto-detected language: (?P<lang>[a-z]{2,3}) \(p = (?P<p>[0-9.]+)\)")
_PROCESSING_LINE = re.compile(r"processing '(?P<path>[^']+)'")


@dataclass(frozen=True)
class LanguageDecision:
    """What the recording was transcribed as, and why."""

    language: str
    """The value whisper ran with: a language code, or ``auto`` for per-turn."""
    heard: dict[str, float] = field(default_factory=dict)
    """Share of sampled speech time per detected language, most first."""
    sampled: int = 0
    """How many slices were listened to. 0 when nothing was detected because a
    language was chosen up front."""

    @property
    def mixed(self) -> bool:
        return self.language == AUTO and len(self.heard) > 1


def decide_language(heard: dict[str, float]) -> str:
    """The rule, on its own so it can be tested without whisper."""
    if not heard:
        return AUTO
    top, share = max(heard.items(), key=lambda item: item[1])
    return top if share >= DOMINANT_SHARE else AUTO


def is_english_only_model(model: Path) -> bool:
    """whisper.cpp's English-only checkpoints carry ``.en`` in the file name
    (``ggml-small.en.bin``). They are faster and slightly more accurate on
    clear English, and produce gibberish for anything else."""
    return _ENGLISH_ONLY.search(model.name) is not None


# ``.en`` as its own dotted component: ``ggml-small.en.bin``,
# ``ggml-medium.en-q5_0.bin``; not the ``en`` inside ``ggml-tiny-encoder.bin``.
_ENGLISH_ONLY = re.compile(r"\.en(?=[.\-_]|$)")

AUDIO_SUFFIXES = (".m4a", ".mp4", ".wav", ".mp3", ".aac", ".aiff")

# Zoom's intermediate format. A local recording lands as these and is only
# readable once Zoom converts it, which does not always happen on its own -
# quitting Zoom before conversion finishes leaves them behind indefinitely.
UNCONVERTED_SUFFIX = ".zoom"

# "Speech segment 3: start = 3322.00, end = 3558.00" — centiseconds.
#
# Anchored to that exact line, and it has to be. The VAD binary also logs its
# own workings on the way there:
#
#     whisper_vad_segments_from_probs: VAD segment 654: start = 2334.47, ...
#     Speech segment 0: start = 109.00, end = 966.00
#
# The first is a debug line in *seconds*; the second is the result in
# centiseconds. A pattern that matched any "start = …, end = …" read both, so
# every transcript was built from twice as many regions as existed, half of
# them at the wrong time scale and running backwards through the meeting.
#
# Short recordings survived it — the bogus regions land early and whisper still
# got roughly the right audio — which is why it went unnoticed through a whole
# eval suite. A forty-minute meeting did not: the two scales diverge far enough
# apart to produce a turn ending before it starts, and ffmpeg refuses that with
# "-to value smaller than -ss".
_VAD_LINE = re.compile(
    r"^\s*Speech segment\s+\d+\s*:\s*start\s*=\s*(?P<start>[\d.]+),\s*end\s*=\s*(?P<end>[\d.]+)"
)

# VAD cuts tight enough to clip a leading consonant.
_SLICE_PAD_MS = 150

# Whisper writes these in place of words when a slice turns out to hold no
# speech. They are markers, not something anybody said.
_NON_SPEECH = re.compile(r"^\[?(BLANK_AUDIO|SILENCE|INAUDIBLE|NO_SPEECH|MUSIC|SOUND)\]?$", re.IGNORECASE)


# What whisper says when handed a second of room noise. These are the model's
# training-data reflexes — the end of a YouTube video — and they arrive as a
# confident, well-punctuated cue at 00:00:00. Dropped only when one of them is
# the *entire* content of a short slice: a real "thank you" in the middle of a
# conversation sits among other words and is left alone.
HALLUCINATIONS = frozenset({
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "thanks for listening", "please subscribe",
    "like and subscribe", "subtitles by the amara org community", "you", "bye",
    "the end", "so", "okay", "hmm", "mm hmm", "uh",
})
HALLUCINATION_MAX_MS = 3000


def looks_hallucinated(texts: list[str], slice_ms: int) -> bool:
    """Whether a short slice's whole output is one of whisper's reflexes."""
    if slice_ms > HALLUCINATION_MAX_MS:
        return False
    joined = " ".join(texts)
    normalised = re.sub(r"[^a-z ]+", " ", joined.casefold()).split()
    return " ".join(normalised) in HALLUCINATIONS


def _join_into_windows(turns: list[tuple[int, int]], window_ms: int) -> list[tuple[int, int]]:
    """Consecutive turns become one window while it stays under ``window_ms``.

    Only across a real gap: two pieces that touch (end == next start) are a
    turn that was cut where the speaker changed mid-flow, or a long turn cut
    to fit whisper's window, and both cuts are there for a reason.
    """
    windows: list[tuple[int, int]] = []
    for start, end in turns:
        if windows:
            w_start, w_end = windows[-1]
            if start > w_end and end - w_start <= window_ms:
                windows[-1] = (w_start, end)
                continue
        windows.append((start, end))
    return windows


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
    normalize_audio: bool = False
    """Level the recording before anything listens to it.

    A laptop on the table hears the person beside it at full volume and the
    person at the far end of the room 20 dB quieter, and Opus at a meeting
    bitrate then spends its bits on the loud voice. ``dynaudnorm`` lifts the
    quiet passages towards the loud ones a few hundred milliseconds at a time,
    and a high-pass drops the rumble of the table and the air conditioning
    that would otherwise be lifted with them. Off by default because the Zoom
    per-participant tracks this module was written for are already levelled
    per speaker; the room recorder turns it on.
    """
    vad_threshold: float = 0.5
    """Silero's speech probability above which a frame counts as speech.
    0.5 is upstream's default and is tuned for one clear voice. Lower it for a
    room: a far-away or soft-spoken person sits at 0.3-0.5 for whole
    sentences, and everything below the threshold is silently thrown away
    before whisper ever hears it."""
    vad_speech_pad_ms: int = 30
    """Extra audio kept either side of each detected region. VAD cuts tight
    and quiet speech starts below its threshold, so a bigger pad keeps the
    first syllable of a soft opening."""
    window_ms: int = 24_000
    """How much consecutive speech whisper is handed at once.

    Whisper reads a whole window of context and decodes far better with a
    few sentences in front of it than with one. Cutting the audio at every
    turn gave it one-line slices: on a real library recording the same five
    minutes read "No. No, it's nowhere close to that" in slices and "No.
    Wait, homework five" in whole windows, with the same words recovered in
    both. Turns are joined into windows up to this long, across the silences
    between them; a cut made *inside* a turn because the speaker changed with
    no pause is never joined back, since whisper would not separate those two
    people itself. Just under whisper's 30-second window, less the padding.
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
        self.last_language: LanguageDecision | None = None
        """Set by every ``transcribe_track`` call: what that track was run as."""

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
                "ggml-large-v3-turbo.bin from huggingface.co/ggerganov/whisper.cpp."
            )
        elif is_english_only_model(self.config.model) and self.config.language not in ("en", DETECT):
            issues.append(
                f"{self.config.model.name} is an English-only model and cannot transcribe "
                f"language '{self.config.language}'. Download a multilingual model such as "
                "ggml-large-v3-turbo.bin from huggingface.co/ggerganov/whisper.cpp, or set "
                "the language to English."
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
        filters = ["-af", LEVELLING_FILTER] if self.config.normalize_audio else []
        await self.runner.run(
            [
                self.config.ffmpeg, "-y", "-v", "error",
                "-i", str(source),
                *filters,
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
                "-vt", f"{self.config.vad_threshold:g}",
                "-vp", str(self.config.vad_speech_pad_ms),
            ],
            log_path=log,
        )
        regions: list[tuple[int, int]] = []
        for line in log.read_text(errors="replace").splitlines():
            match = _VAD_LINE.search(line)
            if match:
                # The tool reports centiseconds.
                start = int(float(match.group("start")) * 10)
                end = int(float(match.group("end")) * 10)
                # A region that ends before it starts is not a region. Nothing
                # downstream checks, and everything downstream assumes these
                # are sorted and forward-going: a single bad pair produced a
                # turn with a negative span, which survived the turn grouping
                # untouched (its span is below every cap) and reached ffmpeg as
                # "-ss 20.005 -to 9.660", which aborts and takes the whole
                # meeting with it. Cheap to check here, and this is the only
                # place the numbers enter the pipeline.
                if end > start >= 0:
                    regions.append((start, end))
        regions.sort()
        return regions

    async def _language_for(self, slices: list[tuple[Path, int, int]], work_dir: Path) -> LanguageDecision:
        """The language this track runs as: chosen up front, or detected."""
        chosen = self.config.language
        if chosen != DETECT:
            return LanguageDecision(language=chosen)
        if is_english_only_model(self.config.model):
            # The model can only do English; detection would be theatre.
            return LanguageDecision(language="en")
        return await self.detect_language(slices, work_dir)

    async def detect_language(self, slices: list[tuple[Path, int, int]], work_dir: Path) -> LanguageDecision:
        """Listen to a sample of the slices and decide how to run the rest.

        The longest slices are sampled, each detection weighted by its
        slice's length, and ``decide_language`` turns the shares into either
        one fixed language or per-turn detection. A detection whisper is not
        sure of (below 0.5) is left out rather than counted for a language
        that is probably wrong.
        """
        sample = sorted(slices, key=lambda item: item[2] - item[1], reverse=True)[:DETECT_SAMPLES]
        args = [
            self.config.binary,
            "-m", str(self.config.model),
            "-dl",
            "-t", str(self.config.threads),
        ]
        for path, _, _ in sample:
            args += ["-f", str(path)]
        log = work_dir / "language.log"
        await self.runner.run(args, log_path=log)

        weights = {str(path): end - start for path, start, end in sample}
        heard: dict[str, float] = {}
        current: str | None = None
        for line in log.read_text(errors="replace").splitlines():
            if match := _PROCESSING_LINE.search(line):
                current = match.group("path")
                continue
            match = _DETECTED_LINE.search(line)
            if not match or current is None:
                continue
            if float(match.group("p")) < 0.5:
                continue
            heard[match.group("lang")] = heard.get(match.group("lang"), 0.0) + weights.get(current, 0)
            current = None
        total = sum(heard.values())
        shares = {lang: ms / total for lang, ms in sorted(heard.items(), key=lambda i: -i[1])} if total else {}
        return LanguageDecision(language=decide_language(shares), heard=shares, sampled=len(sample))

    async def _slice(self, wav: Path, turns: list[tuple[int, int]], slice_dir: Path) -> list[tuple[Path, int, int]]:
        """Cut the audio into one file per turn (or window), with a little padding."""
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

        return slices

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
            turns = _split_turns_at(turns, split_at_ms, regions)
        turn_slices = await self._slice(wav, turns, work_dir / f"{wav.stem}.slices")
        if not turn_slices:
            return []

        # The language is decided on the turns, which is also the unit
        # per-turn detection needs: a mixed meeting is transcribed turn by
        # turn so each gets its own language. Once one language is fixed,
        # whisper is handed whole windows instead — several turns and the
        # silences between them — because it reads far better with a few
        # sentences of context than with one line.
        decision = await self._language_for(turn_slices, work_dir)
        self.last_language = decision
        slices = turn_slices
        if decision.language != AUTO:
            windows = _join_into_windows(turns, self.config.window_ms)
            if len(windows) < len(turns):
                slices = await self._slice(wav, windows, work_dir / f"{wav.stem}.windows")
        prompt = self.config.initial_prompt
        if decision.language == "zh" and (not prompt or MANDARIN_PROMPT not in prompt):
            prompt = f"{MANDARIN_PROMPT} {prompt}" if prompt else MANDARIN_PROMPT

        args = [
            self.config.binary,
            "-m", str(self.config.model),
            "-l", decision.language,
            "-t", str(self.config.threads),
            "-ovtt", "--no-prints",
            "--suppress-nst",  # no "*coughs*" or "[MUSIC]" as if somebody said it
        ]
        if prompt:
            args += ["--prompt", prompt]
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
            slice_cues = parse_vtt(vtt.read_text(errors="replace"))
            if looks_hallucinated([c.text for c in slice_cues], slice_end_ms - offset_ms):
                continue
            for cue in slice_cues:
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

# And each half has to contain this much actual speech. This is the guard that
# matters, and it was learned the hard way: cutting a turn wherever a speaker
# boundary falls produced slices that were mostly silence, and whisper does not
# return nothing for near-silence - it invents. On a twelve-turn test recording,
# splitting without this check turned 12 cues into 21, of which 8 were
# hallucinated filler ("and", "and summer.", "and year-end.") sitting in the
# gaps between turns. Speaker boundaries and speech boundaries are found by two
# different models and do not agree, so a boundary regularly lands mid-silence.
_MIN_SPLIT_SPEECH_MS = 400


def _speech_within(regions: list[tuple[int, int]], start: int, end: int) -> int:
    """How many milliseconds of detected speech fall inside a span."""
    return sum(
        max(0, min(end, region_end) - max(start, region_start))
        for region_start, region_end in regions
    )


def _split_turns_at(
    turns: list[tuple[int, int]],
    points: list[int],
    regions: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Cut every turn at each offset that falls inside it.

    Used to make cues speaker-homogeneous: see ``transcribe_track``. A cut is
    refused when either side would be too short to transcribe, or - given
    ``regions``, the speech the VAD found - when either side would be mostly
    silence. Both guards exist to stop whisper being handed a slice with no
    words in it, because what comes back is not silence but invention.
    """
    ordered = sorted(set(points))
    result: list[tuple[int, int]] = []
    for start, end in turns:
        previous = start
        for point in ordered:
            if not (previous + _MIN_SPLIT_PIECE_MS <= point <= end - _MIN_SPLIT_PIECE_MS):
                continue
            if regions is not None and (
                _speech_within(regions, previous, point) < _MIN_SPLIT_SPEECH_MS
                or _speech_within(regions, point, end) < _MIN_SPLIT_SPEECH_MS
            ):
                continue
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
    unseparated: list[Path] = []

    for candidate in _audio_files(folder):
        match = _PARTICIPANT_FILE.match(candidate.stem)
        if match:
            name = match.group("name").strip()
            if name:
                tracks.setdefault(name, []).append(candidate)
        elif _in_participant_dir(candidate, folder):
            unseparated.append(candidate)

    if unseparated:
        recording_id = _recording_id(folder, [path.stem for path in unseparated])
        for path in unseparated:
            name = _name_without_recording_id(path.stem, recording_id)
            if name:
                tracks.setdefault(name, []).append(path)

    return tracks


def _in_participant_dir(path: Path, folder: Path) -> bool:
    """True when a file sits inside Zoom's per-participant subfolder.

    Location is the only reliable signal for the unseparated layout: the
    filename alone cannot be told apart from the mixed recording's, which is
    also "audio<digits>". The mixed track lives at the top level.
    """
    parents = path.relative_to(folder).parts[:-1]
    return any(part.lower() == _PARTICIPANT_DIR for part in parents)


def _recording_id(folder: Path, stems: list[str]) -> str:
    """The trailing digits Zoom appends to every file of one recording.

    With no separator, splitting a name by eye is genuinely ambiguous:
    "audioGoat11441831751" is "Goat1" followed by the id, or "Goat11" followed
    by a shorter one, and nothing in the filename decides it. The recording
    does. `recording.conf` names the mixed track — "audio1441831751.m4a" —
    whose digits are that id, so anything before it is the display name.

    Without that file, the longest digit run every per-participant track ends
    with is the same answer whenever two people spoke. It is not, for a
    one-participant recording: there the trailing digits of a name ("Goat1")
    are indistinguishable from the id, and such a name loses its last digits.
    A slightly clipped label beats dropping speaker attribution entirely, and
    Zoom writes recording.conf, so this is the unlikely path.
    """
    config = folder / "recording.conf"
    if config.is_file():
        try:
            items = json.loads(config.read_text(encoding="utf-8")).get("items", [])
        except (OSError, ValueError, AttributeError):
            items = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                mixed = Path(str(item.get("audio") or "")).stem
                if not mixed.lower().startswith(_AUDIO_PREFIX):
                    continue
                digits = mixed[len(_AUDIO_PREFIX) :]
                if digits.isdigit() and all(stem.endswith(digits) for stem in stems):
                    return digits
    return _shared_digit_suffix(stems)


def _shared_digit_suffix(stems: list[str]) -> str:
    """The longest run of digits every stem ends with."""
    if not stems:
        return ""
    shortest = min(stems, key=len)
    found = ""
    for size in range(1, len(shortest) + 1):
        candidate = shortest[-size:]
        if not candidate.isdigit() or not all(stem.endswith(candidate) for stem in stems):
            break
        found = candidate
    return found


def _name_without_recording_id(stem: str, recording_id: str) -> str:
    name = stem[len(_AUDIO_PREFIX) :] if stem.lower().startswith(_AUDIO_PREFIX) else stem
    if recording_id and name.endswith(recording_id):
        name = name[: -len(recording_id)]
    return name.strip()


def find_mixed_audio(folder: Path) -> list[Path]:
    """The combined recording, for when per-participant tracks are off.

    A list, not one file, for the same reason as above: Zoom splits long
    recordings, and returning the first would drop everything after it.
    """
    candidates = [
        path
        for path in _audio_files(folder)
        if not _PARTICIPANT_FILE.match(path.stem) and not _in_participant_dir(path, folder)
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
