"""Audio file in, meeting notes out — all of it on this machine.

The heavy lifting already exists in bedrock-private-ai, the employee terminal
client, which transcribes Zoom's *local* recordings for its `bedrock-meeting`
command. This module reuses that code rather than keeping a second copy, the
same arrangement `app/tools/meeting_transcripts.py` has with the VTT parser:
one implementation, pinned by commit in requirements.txt, so a fix reaches both
products.

What this module adds is the part that was missing — a recording that did not
come from Zoom. Browser audio arrives as WebM/Opus, gets normalised to the
16 kHz mono PCM whisper wants, and comes back as WEBVTT plus notes.

**One speaker track, so no speaker names.** Zoom's local recordings can write a
separate file per participant, which is where bedrock-private-ai's perfect
speaker labels come from — every word in Bob's file was said by Bob, by
construction. A microphone in a meeting room is one mixed track and offers no
such thing, and guessing with a diarisation model would put confident names on
the wrong sentences. So these transcripts are honestly unattributed, and
`TranscriptionOutcome.attributed` says so to whoever displays them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from app.meetings.config import RecordingConfig, readiness_issues

logger = logging.getLogger(__name__)

# Imported the way chemistry_tools.py imports RDKit: without the package the
# server still starts and only this feature reports that it is unavailable.
try:
    from bedrock.inference.openai_compatible import OpenAICompatibleBackend
    from bedrock.meetings.summarize import MeetingSummary, summarize_meeting
    from bedrock.meetings.transcribe import (
        WhisperConfig,
        WhisperTranscriber,
        to_vtt,
    )
    from bedrock.meetings.transcript import Cue

    PIPELINE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by deployments without it
    PIPELINE_AVAILABLE = False

PIPELINE_MISSING = (
    "Meeting recording is unavailable: the transcription pipeline from "
    "bedrock-private-ai is not installed. Install requirements.txt, which pins it."
)


class RecordingError(RuntimeError):
    """Raised when a recording cannot be turned into a transcript."""


@dataclass
class TranscriptionOutcome:
    """Everything one recording produced."""

    vtt: str
    cues: list["Cue"]
    duration_ms: int
    summary: "MeetingSummary | None" = None
    attributed: bool = False
    """False whenever the audio was one mixed track, which is every recording
    this module makes. Kept as a field rather than hardcoded so a future
    per-participant capture path does not have to change every caller."""
    notes: list[str] = field(default_factory=list)
    """Plain-language caveats worth showing the person who pressed Record."""

    @property
    def is_empty(self) -> bool:
        return not self.cues


def _whisper_config(config: RecordingConfig) -> "WhisperConfig":
    return WhisperConfig(
        model=config.speech_model,
        vad_model=config.vad_model,
        threads=config.threads,
        language=config.language,
        initial_prompt=config.vocabulary,
    )


async def transcribe(
    audio: Path,
    config: RecordingConfig,
    work_dir: Path,
    on_progress=None,
) -> TranscriptionOutcome:
    """Turn one recorded audio file into timestamped cues and WEBVTT.

    `on_progress` is an optional callable taking a short status string. It is
    how the browser gets told what is happening during the minutes a long
    meeting spends in whisper, instead of watching a spinner and assuming the
    thing has hung.
    """
    if not PIPELINE_AVAILABLE:
        raise RecordingError(PIPELINE_MISSING)

    issues = readiness_issues(config)
    if issues:
        raise RecordingError(
            "This server is not set up to transcribe recordings yet:\n- "
            + "\n- ".join(issues)
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    transcriber = WhisperTranscriber(_whisper_config(config))

    if on_progress:
        on_progress("Converting audio")
    try:
        wav = await transcriber.to_wav(audio, work_dir)
    except Exception as exc:
        raise RecordingError(
            "The recording could not be decoded. It may have been cut off "
            f"before the browser finished writing it. ({exc})"
        ) from exc

    duration_ms = await transcriber.duration_ms(wav)

    if on_progress:
        on_progress("Transcribing speech")
    try:
        cues = await transcriber.transcribe_track(wav, work_dir)
    except Exception as exc:
        # An empty recording is a real, ordinary outcome — somebody pressed
        # Record on a muted microphone. It is reported as an empty transcript
        # with an explanation, not as a server error, because the person needs
        # to know their meeting was not captured.
        message = str(exc)
        if "no transcript" in message.lower():
            return TranscriptionOutcome(
                vtt="WEBVTT\n",
                cues=[],
                duration_ms=duration_ms,
                notes=[
                    "No speech was found in this recording. Check that the "
                    "right microphone was selected and that it was not muted."
                ],
            )
        raise RecordingError(f"Transcription failed: {exc}") from exc

    outcome = TranscriptionOutcome(
        vtt=to_vtt(cues),
        cues=cues,
        duration_ms=duration_ms,
        attributed=False,
        notes=[
            "Speakers are not identified. This was recorded as a single mixed "
            "audio track, so the transcript records what was said, not who said it."
        ],
    )
    if not cues:
        outcome.notes.append("No speech was found in this recording.")
    return outcome


async def summarise(
    outcome: TranscriptionOutcome,
    config: RecordingConfig,
    *,
    title: str,
    meeting_date: date | None = None,
    notes: str | None = None,
    on_progress=None,
) -> TranscriptionOutcome:
    """Write the notes, and keep going if the model is unavailable.

    `summarize_meeting` already degrades to computed facts — duration, word
    count, speaker count — when inference is down, and sets `degraded_reason`
    to say why. That behaviour is the whole reason a meeting is never lost to a
    model outage, so this function must not turn a degraded summary into an
    error. The transcript is the thing that cannot be recreated; the notes can
    always be rewritten later.
    """
    if outcome.is_empty:
        return outcome

    if on_progress:
        on_progress("Writing notes")

    backend = OpenAICompatibleBackend(config.summary_endpoint)
    try:
        outcome.summary = await summarize_meeting(
            outcome.cues,
            backend=backend,
            model=config.summary_model,
            title=title,
            meeting_date=meeting_date,
            notes=notes,
        )
    except Exception as exc:
        # Deliberately broad: a transcript that exists must be saved even if
        # the summariser fails in a way its own degradation path did not catch.
        logger.warning("Meeting summarisation failed for %r: %s", title, exc)
        outcome.notes.append(
            "The transcript was saved, but notes could not be written because "
            f"the model at {config.summary_endpoint} did not answer. "
            "You can ask about this meeting in chat regardless."
        )
        return outcome

    if outcome.summary and outcome.summary.degraded_reason:
        outcome.notes.append(
            f"Notes are limited to computed facts: {outcome.summary.degraded_reason}."
        )
    return outcome


# A heading in the rewritten notes, in either style the model uses: "# Topic"
# or a line that is entirely bold. Both name the section rather than say
# anything, so as a card subtitle they repeat the title and tell the reader
# nothing.
_MARKDOWN_HEADING = re.compile(r"^\s*(?:#{1,6}\s|\*\*[^*]+\*\*\s*$|__[^_]+__\s*$)")
_MARKDOWN_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")


def _first_paragraph(text: str, limit: int = 400) -> str:
    """The first real sentence of the rewritten notes, for the meeting card.

    The card shows plain text, so the markdown the model writes has to come
    off — a card reading "**Enrollment Numbers**" looks like a rendering bug.
    """
    for block in (text or "").split("\n\n"):
        lines = [line for line in block.splitlines() if not _MARKDOWN_HEADING.match(line)]
        cleaned = _MARKDOWN_EMPHASIS.sub("", " ".join(line.strip() for line in lines)).strip()
        cleaned = cleaned.lstrip("-•* ").strip()
        if cleaned:
            return cleaned[:limit]
    return ""


def to_record_fields(outcome: TranscriptionOutcome) -> dict:
    """Shape the summary the way the workspace Meetings view already renders it.

    The view expects lists of strings under `decisions`, `actions`, `deadlines`,
    `questions` and `findings` — the shape `app/workspace/files.py` produces for
    an uploaded transcript. Rendering each point through `Point.render()` keeps
    the owner and the verified timestamp attached to the sentence they belong to.
    """
    summary = outcome.summary
    if summary is None:
        return {
            "summary": "",
            "decisions": [],
            "actions": [],
            "deadlines": [],
            "questions": [],
            "findings": [],
            "people": [],
            "method": "local transcription; no notes written",
        }

    citations = getattr(summary, "citations", None)
    method = "whisper.cpp transcription on this machine; notes written by "
    method += summary.model_used or "no model"
    if citations is not None:
        # Present only when bedrock-private-ai is new enough to verify
        # timestamps. Saying so is the point: an unverified timestamp looks
        # exactly like a verified one to a reader.
        method += (
            f"; {citations.checked} timestamps checked, "
            f"{citations.moved} corrected, {citations.removed} removed"
        )
    else:
        method += "; timestamps NOT verified (update the bedrock-private-ai pin)"

    # `summarize_meeting` fills in *either* the overview or the rewritten
    # notes, never both: when somebody typed notes during the call, those
    # become the output and no separate overview is generated. The Meetings
    # card shows `summary`, so without this fallback every meeting where the
    # person actually took notes displays a blank card.
    overview = summary.overview or _first_paragraph(summary.enhanced_notes)

    return {
        "summary": overview,
        "decisions": [point.render() for point in summary.decisions],
        "actions": [point.render() for point in summary.actions],
        "findings": [point.render() for point in summary.details],
        "deadlines": [],
        "questions": [],
        "people": list(summary.facts.speakers),
        "enhancedNotes": summary.enhanced_notes,
        "durationMinutes": summary.facts.duration_minutes,
        "wordCount": summary.facts.word_count,
        "method": method,
    }
