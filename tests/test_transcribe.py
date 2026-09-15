"""Tests for transcribing a Zoom local recording.

whisper.cpp is faked throughout — the runner is a stub that writes the VTT a
real run would have produced. These check the parts that are ours: finding the
participant tracks, grouping speech into turns, offsetting each slice back onto
the meeting timeline, and merging speakers. Whisper's own accuracy is measured
by hand against real audio; see docs/MEETINGS.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bedrock.biotech.errors import ToolIntegrationError
from bedrock.meetings.transcribe import (
    WhisperConfig,
    WhisperTranscriber,
    _group_regions,
    discover_participant_tracks,
    find_mixed_audio,
    is_speech,
    to_vtt,
    transcribe_recording,
)
from bedrock.meetings.transcript import Cue, parse_vtt


def make_config(tmp_path: Path) -> WhisperConfig:
    model = tmp_path / "model.bin"
    vad = tmp_path / "vad.bin"
    model.write_bytes(b"x")
    vad.write_bytes(b"x")
    return WhisperConfig(model=model, vad_model=vad)


class FakeRunner:
    """Stands in for whisper.cpp and ffmpeg.

    `regions` is what VAD "detects"; `texts` is what whisper "hears", one per
    slice in order.
    """

    def __init__(self, regions: list[tuple[int, int]], texts: list[str]):
        self.regions = regions
        self.texts = list(texts)
        self.commands: list[list[str]] = []

    async def run(self, args, log_path, env=None):
        args = list(args)
        self.commands.append(args)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if args[0] == "whisper-vad-speech-segments":
            lines = [
                f"Speech segment {i}: start = {s / 10:.2f}, end = {e / 10:.2f}"
                for i, (s, e) in enumerate(self.regions)
            ]
            log_path.write_text("\n".join(lines))
            return

        if args[0] == "ffmpeg":
            Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(args[-1]).write_bytes(b"RIFFfake")
            log_path.write_text("")
            return

        if args[0] == "whisper-cli":
            inputs = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
            for path, text in zip(inputs, self.texts):
                # Timestamps are relative to the slice, as whisper reports them.
                Path(path + ".vtt").write_text(
                    f"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n{text}\n"
                )
            log_path.write_text("")
            return

        raise AssertionError(f"unexpected command: {args[0]}")


# --- discovery -----------------------------------------------------------


def test_finds_participant_tracks_by_zoom_naming(tmp_path):
    rec = tmp_path / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1757900001_Alice Chen.m4a").write_bytes(b"")
    (rec / "audio1757900002_Bob Ruiz.m4a").write_bytes(b"")
    (tmp_path / "zoom_0.mp4").write_bytes(b"")

    tracks = discover_participant_tracks(tmp_path)
    assert set(tracks) == {"Alice Chen", "Bob Ruiz"}
    assert "zoom_0" not in tracks  # the mixed recording is not a participant


def test_mixed_audio_prefers_audio_only_over_video(tmp_path):
    (tmp_path / "zoom_0.mp4").write_bytes(b"")
    (tmp_path / "audio_only.m4a").write_bytes(b"")
    assert find_mixed_audio(tmp_path).name == "audio_only.m4a"


# --- turn grouping -------------------------------------------------------


def test_short_pauses_join_into_one_turn():
    # 990-2080 and 2310-8450: a 230ms gap, mid-sentence.
    assert _group_regions([(990, 2080), (2310, 8450)], 600) == [(990, 8450)]


def test_a_real_silence_starts_a_new_turn():
    assert _group_regions([(990, 2080), (14370, 15290)], 600) == [(990, 2080), (14370, 15290)]


def test_blank_audio_markers_are_not_speech():
    assert is_speech("We're at 148 of 200.")
    assert not is_speech("[BLANK_AUDIO]")
    assert not is_speech("   ")


# --- transcription -------------------------------------------------------


async def test_slice_timestamps_land_on_the_meeting_timeline(tmp_path):
    """The whole reason for slicing: a cue from the second turn must carry the
    second turn's offset, not whisper's slice-relative zero."""
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.wav").write_bytes(b"RIFFfake")

    runner = FakeRunner(
        regions=[(1000, 3000), (20000, 23000)],
        texts=["First thing.", "Much later thing."],
    )
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    result = await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")

    assert [c.text for c in result.cues] == ["First thing.", "Much later thing."]
    # 1000ms and 20000ms, each less 150ms of padding.
    assert [c.start_ms for c in result.cues] == [850, 19850]
    assert all(c.speaker == "Alice Chen" for c in result.cues)


async def test_speakers_merge_in_chronological_order(tmp_path):
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.wav").write_bytes(b"RIFFfake")
    (rec / "audio2_Bob Ruiz.wav").write_bytes(b"RIFFfake")

    # Alice speaks first and third; Bob second. Same fake regions per track,
    # so ordering has to come from the offsets.
    runner = FakeRunner(regions=[(0, 2000), (20000, 22000)], texts=["one", "three"])
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    result = await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")

    assert result.speakers == ["Alice Chen", "Bob Ruiz"]
    starts = [c.start_ms for c in result.cues]
    assert starts == sorted(starts)


async def test_one_whisper_process_per_track(tmp_path):
    """Slicing must not mean reloading the model once per turn."""
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.wav").write_bytes(b"RIFFfake")

    runner = FakeRunner(
        regions=[(0, 1000), (5000, 6000), (9000, 10000)],
        texts=["a", "b", "c"],
    )
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")

    whisper_calls = [c for c in runner.commands if c[0] == "whisper-cli"]
    assert len(whisper_calls) == 1
    assert whisper_calls[0].count("-f") == 3


async def test_vad_is_always_requested(tmp_path):
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.wav").write_bytes(b"RIFFfake")
    runner = FakeRunner(regions=[(0, 1000)], texts=["a"])
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")
    assert any(c[0] == "whisper-vad-speech-segments" for c in runner.commands)


async def test_a_recording_with_no_participant_tracks_is_marked_unattributed(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir(parents=True)
    (rec / "audio_only.wav").write_bytes(b"RIFFfake")
    runner = FakeRunner(regions=[(0, 2000)], texts=["Someone said something."])
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    result = await transcribe_recording(rec, transcriber, work_dir=tmp_path / "w")

    assert result.unattributed is True
    assert result.speakers == []
    assert result.cues[0].speaker is None


async def test_a_missing_vad_model_is_refused_up_front(tmp_path):
    """Rather than silently producing timestamps that are wrong by minutes."""
    config = WhisperConfig(model=tmp_path / "m.bin", vad_model=tmp_path / "missing.bin")
    (tmp_path / "m.bin").write_bytes(b"x")
    rec = tmp_path / "rec"
    rec.mkdir()
    transcriber = WhisperTranscriber(config, runner=FakeRunner([], []))

    with pytest.raises(ToolIntegrationError, match="VAD model"):
        await transcribe_recording(rec, transcriber, work_dir=tmp_path / "w")


# --- round trip ----------------------------------------------------------


def test_cues_render_back_to_parseable_vtt():
    cues = [
        Cue(start_ms=850, end_ms=3000, speaker="Alice Chen", text="First thing."),
        Cue(start_ms=19850, end_ms=23000, speaker="Bob Ruiz", text="Much later."),
    ]
    reparsed = parse_vtt(to_vtt(cues))
    assert [(c.speaker, c.text, c.start_ms) for c in reparsed] == [
        (c.speaker, c.text, c.start_ms) for c in cues
    ]


# --- CLI helpers ---------------------------------------------------------


def test_meeting_title_comes_from_zooms_folder_name():
    from bedrock.meetings.cli import meeting_date, meeting_title

    folder = Path("2026-09-15 09.00.12 Phase II weekly sync")
    assert meeting_title(folder) == "Phase II weekly sync"
    assert meeting_date(folder).isoformat() == "2026-09-15"


def test_an_unrecognised_folder_name_still_yields_a_title():
    from bedrock.meetings.cli import meeting_date, meeting_title

    assert meeting_title(Path("some-export")) == "some-export"
    assert meeting_date(Path("some-export")) is None


def test_recordings_are_listed_oldest_first_and_empty_ones_skipped(tmp_path):
    from bedrock.meetings.cli import recordings_in

    older = tmp_path / "2026-09-01 10.00.00 Older"
    newer = tmp_path / "2026-09-15 09.00.12 Newer"
    empty = tmp_path / "2026-09-20 11.00.00 Empty"
    for folder in (older, newer, empty):
        folder.mkdir()
    (older / "audio_only.m4a").write_bytes(b"")
    (newer / "audio_only.m4a").write_bytes(b"")
    os.utime(older, (1_600_000_000, 1_600_000_000))
    os.utime(newer, (1_700_000_000, 1_700_000_000))

    found = recordings_in(tmp_path)
    assert [f.name for f in found] == [older.name, newer.name]
    assert found[-1].name == newer.name  # the CLI's "most recent" default


def test_no_zoom_folder_is_not_an_error(tmp_path):
    from bedrock.meetings.cli import recordings_in

    assert recordings_in(tmp_path / "nope") == []
