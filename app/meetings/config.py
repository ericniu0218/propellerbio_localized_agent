"""Where the recorder finds whisper, and what it is allowed to spend.

Everything here comes from environment variables so a deployment can point at
its own model files without editing code, matching how `DATACENTER_BASE_URL`
and friends are configured next door in `app/tools/datacenter_client.py`.

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
DEFAULT_SPEECH_MODEL = "ggml-small.en.bin"
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


# An hour of meeting audio is roughly 14 MB as Opus. The cap is well above that
# so a long meeting is never silently refused, and well below anything that
# would exhaust memory when read into a single bytes object.
MAX_RECORDING_BYTES = 200 * 1024 * 1024

# Recording is the only workspace operation whose cost grows with wall-clock
# time rather than with input size, so it gets its own ceiling. Above this the
# upload is refused with an explanation rather than tying the box up for hours.
MAX_RECORDING_SECONDS = 4 * 60 * 60


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
    summary_model: str
    summary_endpoint: str

    @classmethod
    def from_env(cls) -> "RecordingConfig":
        model_dir = Path(os.environ.get("MEETING_MODEL_DIR", str(DEFAULT_MODEL_DIR))).expanduser()
        return cls(
            speech_model=Path(
                os.environ.get("MEETING_SPEECH_MODEL", str(model_dir / DEFAULT_SPEECH_MODEL))
            ).expanduser(),
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
            language=os.environ.get("MEETING_LANGUAGE", "en"),
            storage_dir=Path(
                os.environ.get("MEETING_RECORDING_DIR", "data/recordings")
            ).expanduser(),
            # The notes are written by whatever local model this deployment
            # already runs. Ollama is the development default, the same one
            # bedrock-private-ai's own meeting CLI assumes.
            summary_model=os.environ.get("MEETING_SUMMARY_MODEL", "llama3.1:8b"),
            summary_endpoint=os.environ.get(
                "MEETING_SUMMARY_ENDPOINT", "http://localhost:11434/v1"
            ),
        )


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
    if not config.vad_model.exists():
        issues.append(
            f"No voice-activity model at {config.vad_model}. Download "
            f"{DEFAULT_VAD_MODEL} from huggingface.co/ggml-org/whisper-vad. "
            "Without it, timestamps drift by minutes and every citation is wrong."
        )
    return issues
