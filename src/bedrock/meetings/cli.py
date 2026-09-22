"""Command line for turning a Zoom local recording into notes.

    bedrock-meeting --list            what recordings are on this machine
    bedrock-meeting                   the most recent one
    bedrock-meeting <folder>          a specific one
    bedrock-meeting --notes notes.md  finish the notes you typed during the call

Everything runs on this machine. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date, datetime
from pathlib import Path

from bedrock.biotech.errors import ToolIntegrationError
from bedrock.inference.base import InferenceError
from bedrock.inference.openai_compatible import OpenAICompatibleBackend
from bedrock.meetings.summarize import summarize_meeting
from bedrock.meetings.transcribe import (
    WhisperConfig,
    WhisperTranscriber,
    discover_participant_tracks,
    find_mixed_audio,
    find_unconverted,
    to_vtt,
    transcribe_recording,
)

DEFAULT_ZOOM_DIR = Path.home() / "Documents" / "Zoom"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "whisper-models"
SPEECH_MODEL = "ggml-small.en.bin"
VAD_MODEL = "ggml-silero-v5.1.2.bin"

SETUP_HINT = f"""
One-time setup:

  brew install whisper.cpp ffmpeg
  mkdir -p {DEFAULT_MODEL_DIR}
  cd {DEFAULT_MODEL_DIR}
  curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{SPEECH_MODEL}
  curl -LO https://huggingface.co/ggml-org/whisper-vad/resolve/main/{VAD_MODEL}

And in Zoom: Settings > Recording > turn on "Record a separate audio file for
each participant who speaks". Without it everyone is transcribed as one
unnamed speaker.
""".strip()


def recordings_in(folder: Path) -> list[Path]:
    """Zoom writes one folder per meeting. Ordered oldest first by modification
    time, so the caller's "most recent" is the last entry."""
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.iterdir() if p.is_dir() and any(p.rglob("*"))),
        key=lambda p: p.stat().st_mtime,
    )


def meeting_title(folder: Path) -> str:
    """Zoom names folders "2026-09-15 09.00.12 Phase II weekly sync"."""
    name = folder.name
    parts = name.split(" ", 2)
    return parts[2].strip() if len(parts) == 3 else name


def meeting_date(folder: Path) -> date | None:
    try:
        return datetime.strptime(folder.name.split(" ", 1)[0], "%Y-%m-%d").date()
    except (ValueError, IndexError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bedrock-meeting",
        description="Transcribe and summarise a Zoom local recording, on this machine.",
    )
    parser.add_argument("folder", nargs="?", type=Path, help="a Zoom recording folder")
    parser.add_argument("--list", action="store_true", help="list recordings and exit")
    parser.add_argument("--notes", type=Path, help="rough notes you typed during the call")
    parser.add_argument("--terms", help="comma-separated names and jargon to expect")
    parser.add_argument("--model", default=os.environ.get("BEDROCK_MODEL", "qwen2.5:7b"))
    parser.add_argument(
        "--speech-model",
        default=os.environ.get("BEDROCK_SPEECH_MODEL", SPEECH_MODEL),
        help=f"ggml filename in {DEFAULT_MODEL_DIR} (default: {SPEECH_MODEL})",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("BEDROCK_API_BASE", "http://localhost:11434/v1"),
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="skip the summary; just write the transcript",
    )
    parser.add_argument("--zoom-dir", type=Path, default=DEFAULT_ZOOM_DIR)
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.list:
        found = recordings_in(args.zoom_dir)
        if not found:
            print(f"No recordings under {args.zoom_dir}.")
            print("\nIn Zoom, record a meeting with 'Record on this Computer'.")
            return 1
        for folder in found:
            tracks = discover_participant_tracks(folder)
            files = sum(len(f) for f in tracks.values())
            if tracks:
                who = f"{len(tracks)} participants, {files} audio files"
            elif find_unconverted(folder):
                who = "NOT CONVERTED — double-click the .zoom file in Finder"
            elif find_mixed_audio(folder):
                who = "no per-participant audio (speakers will not be named)"
            else:
                who = "no audio"
            print(f"{folder.name}\n    {who}")
        return 0

    # Validate arguments before doing any work: being told a path is wrong
    # after a minute of transcription is the same mistake as not checking.
    if args.notes and not args.notes.exists():
        # Summarising without them would look like success while ignoring the
        # thing the user most wanted included.
        print(f"No notes file at {args.notes}.", file=sys.stderr)
        return 1

    folder = args.folder
    if folder is None:
        found = recordings_in(args.zoom_dir)
        if not found:
            print(f"No recordings under {args.zoom_dir}. Pass a folder, or use --list.")
            return 1
        folder = found[-1]
        print(f"Using the most recent recording: {folder.name}\n")

    config = WhisperConfig(
        model=DEFAULT_MODEL_DIR / args.speech_model,
        vad_model=DEFAULT_MODEL_DIR / VAD_MODEL,
        initial_prompt=args.terms,
    )
    transcriber = WhisperTranscriber(config)
    issues = transcriber.readiness_issues()
    if issues:
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        print(f"\n{SETUP_HINT}", file=sys.stderr)
        return 1

    tracks = discover_participant_tracks(folder)
    if tracks:
        total = sum(len(files) for files in tracks.values())
        extra = f" ({total} files)" if total > len(tracks) else ""
        print(f"Transcribing {len(tracks)} participant tracks{extra}: {', '.join(tracks)}")
    elif find_mixed_audio(folder):
        print("No per-participant audio in this recording — speakers will not be named.")
        print('Turn on "Record a separate audio file for each participant" in Zoom for next time.')
    elif unconverted := find_unconverted(folder):
        # Telling someone to wait is wrong here: nothing is running. Zoom leaves
        # these behind when it is quit before conversion finishes, and they sit
        # there until a human starts it.
        print(f"\nThis recording has not been converted yet.", file=sys.stderr)
        print(f"  {folder} holds {len(unconverted)} .zoom file(s).\n", file=sys.stderr)
        print("Convert it, then run this again:", file=sys.stderr)
        print(f'  open "{folder}"   then double-click {unconverted[0].name}', file=sys.stderr)
        print("  or in Zoom: Meetings > Recorded > Convert", file=sys.stderr)
        return 1
    else:
        # Don't advise changing a Zoom setting when the real problem is that
        # there is no audio here at all — that sends people to fix the wrong thing.
        print(f"\nNo audio files in {folder}.", file=sys.stderr)
        return 1

    try:
        result = await transcribe_recording(folder, transcriber)
    except ToolIntegrationError as exc:
        print(f"\nTranscription failed: {exc}", file=sys.stderr)
        return 1

    if result.multi_file_speakers:
        print(
            f"  Note: {', '.join(result.multi_file_speakers)} had several audio files, "
            "joined as consecutive parts of a split recording."
        )
        print(
            "  If two people shared that Zoom display name, their timings will be wrong — "
            "rename one in Zoom and record again."
        )
    if result.speakers and len(result.speakers) < len(result.per_speaker_counts):
        quiet = [s for s, n in result.per_speaker_counts.items() if not n]
        print(f"  {', '.join(quiet)} never spoke — no lines attributed to them.")

    transcript_path = folder / "transcript.vtt"
    transcript_path.write_text(to_vtt(result.cues))
    print(f"  {len(result.cues)} cues -> {transcript_path}")

    if args.transcript_only:
        return 0

    notes = args.notes.read_text() if args.notes else None
    backend = OpenAICompatibleBackend(args.api_base, timeout=600.0)
    try:
        if not await backend.check_connection():
            print(
                f"\nNo model server at {args.api_base}, so the transcript above is all you get.\n"
                "Start one with `ollama serve`, then re-run.",
                file=sys.stderr,
            )
            return 1
        print(f"Summarising with {args.model}…")
        summary = await summarize_meeting(
            result.cues,
            backend=backend,
            model=args.model,
            title=meeting_title(folder),
            meeting_date=meeting_date(folder),
            notes=notes,
        )
    except InferenceError as exc:
        print(f"\nThe model server failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await backend.close()

    notes_path = folder / "notes.md"
    notes_path.write_text(summary.render())
    print(f"  notes -> {notes_path}")
    print()
    print(summary.render())
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
