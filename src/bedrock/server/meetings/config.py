"""Where the recorder finds whisper, and what it is allowed to spend.

Everything here comes from environment variables so a deployment can point at
its own model files without editing code, matching how `DATACENTER_BASE_URL`
and friends are configured next door in `src/bedrock/server/tools/datacenter_client.py`.

The one setting worth reading twice is `vocabulary`. Whisper renders words it
does not know phonetically: in testing it wrote "CROS" for "KRAS" and turned
"Priya can own" into "Priya Kanone". Seeding the decoder with the study's real
terms fixes both, costs nothing, and is different for every customer — so it
belongs in configuration rather than in code.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "whisper-models"
# Multilingual, and chosen for who is in the room rather than for the
# benchmark. The English-only `small.en` is a little faster on clear native
# English and is lost on everything else: heavily accented English drifts
# into phonetic spellings, and a sentence of Mandarin comes out as the
# previous English sentence repeated. large-v3-turbo transcribes the same
# 49-second clip in 2.1s against small.en's 1.5s on an M-series laptop,
# handles a Chinese sentence and an English one in the same recording, and
# is the model whisper.cpp itself recommends for anything not English-only.
DEFAULT_SPEECH_MODEL = "ggml-large-v3-turbo.bin"
# What an installation set up before the model change still has. Used only
# when the multilingual model is absent, so nothing that worked stops.
FALLBACK_SPEECH_MODEL = "ggml-small.en.bin"
# Used for recordings decided to be Mandarin when it is present: measured
# better on Chinese technical terms than turbo, worse on noisy English, and
# 2.2x slower — so only for Mandarin, and only when somebody downloaded it.
MANDARIN_SPEECH_MODEL = "ggml-large-v3.bin"

# Languages the Record dialog offers. Keys are whisper language codes, plus
# two of our own. "detect" is the default and listens first: a sample of the
# speech is identified, one dominant language becomes a fixed language for
# the whole recording (more accurate on accented speech than guessing each
# turn), and a genuinely mixed meeting falls back to "auto", whisper's
# per-turn detection, so Mandarin and English turns each come out in their
# own script. The named languages are for overriding that.
LANGUAGES = {
    "detect": "Detect automatically",
    "en": "English",
    "zh": "Chinese (Mandarin)",
    "auto": "Detect per speaker turn (mixed-language meetings)",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "pt": "Portuguese",
    "hi": "Hindi",
}
DEFAULT_LANGUAGE = "detect"

# The room recorder's sensitivity. These are the values the server applies
# unless overridden; see WhisperConfig for what each one does. 0.4 rather than
# Silero's 0.5 because a far-away voice sits around the threshold for whole
# sentences, and 60 ms of pad rather than 30 keeps the first syllable of a
# soft opening.
#
# Levelling is off, and that is measured rather than assumed. On a real
# 31-minute library recording, dynaudnorm lowered the speech-to-noise ratio
# from 6.8 dB to 5.1 dB — it lifts the room between words as readily as it
# lifts a quiet voice — and the transcript came out the same length with or
# without it. It stays available for a recording whose problem really is one
# quiet speaker in a quiet room.
DEFAULT_VAD_THRESHOLD = 0.4
DEFAULT_VAD_PAD_MS = 60
DEFAULT_NORMALIZE_AUDIO = False
DEFAULT_VAD_MODEL = "ggml-silero-v5.1.2.bin"
# Speaker identification. Optional: without these a recording still
# transcribes, and is reported as unattributed exactly as before.
DEFAULT_DIARIZATION_DIR = "diarization"
DEFAULT_SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
# TitaNet *small*, not large, and that is measured rather than assumed. On a
# three-speaker recording with known ground truth, small scored 84.6% speaker
# attribution against large's 76.9%, ran the 3-minute case in 14.7s against
# 25.8s, and is 38 MB against 97 MB. Both found the right number of speakers on
# every recording tested. A bigger embedding model is not automatically a
# better one for short conversational turns.
DEFAULT_EMBEDDING_MODEL = "nemo_en_titanet_small.onnx"

# Whisper is CPU-bound and scales nearly linearly until it runs out of
# performance cores. The upstream default of 4 left most of this machine idle
# (18 logical cores), so the default here is derived instead of guessed, and
# capped because the efficiency cores contribute little and the server still
# has to answer requests while a transcription runs.
def _default_threads() -> int:
    return max(2, min(10, (os.cpu_count() or 4) - 2))


# An hour of meeting audio is roughly 14 MB as Opus, so eight hours is about
# 115 MB. The cap sits well above that: a recording is streamed to disk in
# chunks rather than held in memory, so the number is about refusing an
# obviously wrong upload, not about what fits.
MAX_RECORDING_BYTES = 512 * 1024 * 1024

# Recording is the only workspace operation whose cost grows with wall-clock
# time rather than with input size, so it gets its own ceiling. Eight hours
# covers an all-day workshop; above it the upload is refused with an
# explanation rather than tying the box up overnight.
#
# The cost is real and worth knowing: at 0.30x real time, eight hours of audio
# is about two and a half hours of processing, and the whole recording is held
# as 32-bit samples while it runs — roughly 1.8 GB for eight hours.
MAX_RECORDING_SECONDS = 8 * 60 * 60


@dataclass(frozen=True)
class RecordingConfig:
    """Everything the transcription pipeline needs to run."""

    speech_model: Path
    vad_model: Path
    segmentation_model: Path
    embedding_model: Path
    threads: int
    vocabulary: str | None
    language: str
    storage_dir: Path
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    vad_speech_pad_ms: int = DEFAULT_VAD_PAD_MS
    normalize_audio: bool = DEFAULT_NORMALIZE_AUDIO
    speech_model_zh: Path | None = None
    """``MEETING_SPEECH_MODEL_ZH``, else ``ggml-large-v3.bin`` beside the
    speech model when that file exists; None means Mandarin uses the main
    model like everything else."""

    @property
    def multilingual(self) -> bool:
        from bedrock.meetings.transcribe import is_english_only_model

        return not is_english_only_model(self.speech_model)

    @classmethod
    def from_env(cls) -> "RecordingConfig":
        model_dir = Path(os.environ.get("MEETING_MODEL_DIR", str(DEFAULT_MODEL_DIR))).expanduser()
        return cls(
            speech_model=_speech_model(model_dir),
            vad_model=Path(
                os.environ.get("MEETING_VAD_MODEL", str(model_dir / DEFAULT_VAD_MODEL))
            ).expanduser(),
            segmentation_model=Path(
                os.environ.get(
                    "MEETING_SEGMENTATION_MODEL",
                    str(model_dir / DEFAULT_DIARIZATION_DIR / DEFAULT_SEGMENTATION_MODEL),
                )
            ).expanduser(),
            embedding_model=Path(
                os.environ.get(
                    "MEETING_EMBEDDING_MODEL",
                    str(model_dir / DEFAULT_DIARIZATION_DIR / DEFAULT_EMBEDDING_MODEL),
                )
            ).expanduser(),
            threads=int(os.environ.get("MEETING_WHISPER_THREADS", _default_threads())),
            vocabulary=os.environ.get("MEETING_VOCABULARY") or None,
            language=os.environ.get("MEETING_LANGUAGE", DEFAULT_LANGUAGE),
            storage_dir=_recording_dir(),
            vad_threshold=float(os.environ.get("MEETING_VAD_THRESHOLD", DEFAULT_VAD_THRESHOLD)),
            vad_speech_pad_ms=int(os.environ.get("MEETING_VAD_PAD_MS", DEFAULT_VAD_PAD_MS)),
            normalize_audio=os.environ.get("MEETING_NORMALIZE_AUDIO", "false").strip().lower()
            in ("1", "true", "yes", "on"),
            speech_model_zh=_mandarin_model(model_dir),
        )


def _mandarin_model(model_dir: Path) -> Path | None:
    explicit = os.environ.get("MEETING_SPEECH_MODEL_ZH")
    if explicit:
        return Path(explicit).expanduser()
    candidate = model_dir / MANDARIN_SPEECH_MODEL
    return candidate if candidate.exists() else None


def _speech_model(model_dir: Path) -> Path:
    """``MEETING_SPEECH_MODEL`` if set; else the multilingual default; else,
    so an installation from before the model change keeps working, the
    English-only model it already has. Readiness names whichever was chosen."""
    explicit = os.environ.get("MEETING_SPEECH_MODEL")
    if explicit:
        return Path(explicit).expanduser()
    preferred = model_dir / DEFAULT_SPEECH_MODEL
    fallback = model_dir / FALLBACK_SPEECH_MODEL
    if not preferred.exists() and fallback.exists():
        return fallback
    return preferred


def _recording_dir() -> Path:
    """``MEETING_RECORDING_DIR``, else ``recordings`` in the server's data directory."""
    explicit = os.environ.get("MEETING_RECORDING_DIR")
    if explicit:
        return Path(explicit).expanduser()
    from bedrock.server.config import get_settings

    return get_settings().data_dir / "recordings"


def readiness_issues(config: RecordingConfig) -> list[str]:
    """Everything missing that would stop a recording from being transcribed.

    Reported all at once, before anything is recorded, so somebody setting this
    up fixes every problem in one pass instead of discovering them one failed
    meeting at a time. Same approach as `WhisperTranscriber.readiness_issues`,
    which this calls into for the parts it owns.
    """
    issues: list[str] = []
    for binary, package in (
        ("whisper-cli", "whisper-cpp"),
        ("whisper-vad-speech-segments", "whisper-cpp"),
        ("ffmpeg", "ffmpeg"),
        ("ffprobe", "ffmpeg"),
    ):
        if shutil.which(binary) is None:
            issues.append(f"`{binary}` is not installed. Run `brew install {package}`.")
    if not config.speech_model.exists():
        issues.append(
            f"No speech model at {config.speech_model}. Download "
            f"{DEFAULT_SPEECH_MODEL} from huggingface.co/ggerganov/whisper.cpp."
        )
    elif not config.multilingual and config.language not in ("en", "detect"):
        issues.append(
            f"{config.speech_model.name} is an English-only model and cannot transcribe "
            f"'{LANGUAGES.get(config.language, config.language)}'. Download "
            f"{DEFAULT_SPEECH_MODEL} from huggingface.co/ggerganov/whisper.cpp into "
            f"{config.speech_model.parent}, or record in English."
        )
    if not config.vad_model.exists():
        issues.append(
            f"No voice-activity model at {config.vad_model}. Download "
            f"{DEFAULT_VAD_MODEL} from huggingface.co/ggml-org/whisper-vad. "
            "Without it, timestamps drift by minutes and every citation is wrong."
        )
    return issues
