"""Text appearing while the meeting is still happening.

Transcription runs at about 0.3x real time, so an hour-long meeting produces
nothing for eighteen minutes after it ends. That is the single biggest
complaint available about this feature, and it is a complaint about *waiting*
rather than about accuracy: people want to see it working, and to catch a
mis-set microphone in the first minute rather than the sixty-first.

**Why this is a second recorder and not a stream.** MediaRecorder emits chunks,
but only the first carries the WebM header — hand ffmpeg a later one on its own
and it fails with "EBML header parsing failed". Chunks are only decodable
concatenated from the beginning, so a naive "upload each chunk and transcribe
it" does not work at all, and transcribing the whole accumulated file every few
seconds is quadratic.

So the browser runs a second MediaRecorder over the same mixed audio, restarted
every few seconds. Each restart produces a small, complete, independently
decodable file. Those are the preview; the continuous recorder is untouched and
still produces the authoritative recording.

**The preview is not the transcript, and the UI has to say so.** It misses the
fraction of a second at each restart boundary, it has no speaker labels —
diarisation needs the whole meeting to cluster voices, which by definition does
not exist yet — and it has had no vocabulary repair or citation checking. When
the meeting ends the real pipeline runs over the real recording and replaces
it. Presenting a preview as a record of what was said would be the worst kind
of wrong: readable, plausible, and not what the product stands behind.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# A live session with nothing arriving for this long is assumed abandoned — a
# closed laptop, a crashed tab — and is swept so preview text does not
# accumulate for ever in a process that may run for weeks.
IDLE_TIMEOUT_S = 30 * 60

# Preview text is bounded. A three-hour meeting should not be able to grow this
# without limit; the authoritative transcript is not subject to this cap.
MAX_PREVIEW_CHARS = 200_000


@dataclass
class LiveSession:
    id: str
    owner: str
    started_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    clips: int = 0
    lines: list[str] = field(default_factory=list)
    """Each clip's text, in arrival order. Kept as lines rather than one string
    so a clip that transcribes to nothing can be dropped without leaving a gap
    in the middle of a sentence."""
    failures: int = 0

    @property
    def text(self) -> str:
        return " ".join(line for line in self.lines if line)[:MAX_PREVIEW_CHARS]

    def public(self) -> dict:
        return {
            "id": self.id,
            "clips": self.clips,
            "failures": self.failures,
            "seconds": round(time.monotonic() - self.started_at, 1),
            "text": self.text,
            "provisional": True,
            "note": (
                "Live preview — no speaker names, and not corrected. The full "
                "transcript is produced when the recording is saved."
            ),
        }


class LiveSessions:
    """In-process registry of meetings currently being previewed.

    In memory on purpose, unlike recording jobs. A preview has no value once
    the meeting is over — the real transcript replaces it — so there is nothing
    worth surviving a restart, and persisting it would mean writing partial,
    unverified transcript text to disk for no benefit.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, LiveSession] = {}

    def open(self, session_id: str, owner: str) -> LiveSession:
        self.sweep()
        session = LiveSession(id=session_id, owner=owner)
        self._sessions[session_id] = session
        return session

    def get(self, owner: str, session_id: str) -> LiveSession | None:
        session = self._sessions.get(session_id)
        return session if session and session.owner == owner else None

    def append(self, session: LiveSession, text: str) -> None:
        session.clips += 1
        session.updated_at = time.monotonic()
        if text and text.strip():
            session.lines.append(text.strip())

    def note_failure(self, session: LiveSession) -> None:
        session.failures += 1
        session.updated_at = time.monotonic()

    def close(self, owner: str, session_id: str) -> None:
        session = self.get(owner, session_id)
        if session:
            del self._sessions[session_id]

    def sweep(self) -> None:
        cutoff = time.monotonic() - IDLE_TIMEOUT_S
        for session_id, session in list(self._sessions.items()):
            if session.updated_at < cutoff:
                del self._sessions[session_id]


async def transcribe_clip(clip: Path, config, work_dir: Path) -> str:
    """Words from one short, self-contained clip. Never raises.

    Deliberately skips everything the real pipeline does after whisper — no
    diarisation, no vocabulary repair, no citation checking. Those need either
    the whole meeting or a second model pass, and the point of a preview is
    that it arrives within a second or two of the words being said.

    A clip that fails is worth nothing and must not interrupt a meeting, so the
    failure is counted and swallowed.
    """
    from app.meetings.pipeline import PIPELINE_AVAILABLE, _whisper_config

    if not PIPELINE_AVAILABLE:
        return ""
    try:
        from bedrock.meetings.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber(_whisper_config(config))
        work_dir.mkdir(parents=True, exist_ok=True)
        wav = await transcriber.to_wav(clip, work_dir)
        cues = await transcriber.transcribe_track(wav, work_dir)
        return " ".join(cue.text for cue in cues).strip()
    except Exception as exc:
        logger.debug("Live clip %s produced nothing: %s", clip.name, exc)
        return ""
