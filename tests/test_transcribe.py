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

    def __init__(self, regions: list[tuple[int, int]], texts: list[str],
                 file_duration_s: float = 60.0):
        self.regions = regions
        self.texts = list(texts)
        self.file_duration_s = file_duration_s
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

        if args[0] == "ffprobe":
            log_path.write_text(f"{self.file_duration_s:.3f}\n")
            return

        if args[0] == "ffmpeg":
            Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(args[-1]).write_bytes(b"RIFFfake")
            log_path.write_text("")
            return

        if args[0] == "whisper-cli":
            inputs = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
            for path in inputs:
                text = self.texts.pop(0) if self.texts else ""
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
    assert all(len(files) == 1 for files in tracks.values())


def test_mixed_audio_prefers_audio_only_over_video(tmp_path):
    (tmp_path / "zoom_0.mp4").write_bytes(b"")
    (tmp_path / "audio_only.m4a").write_bytes(b"")
    assert [p.name for p in find_mixed_audio(tmp_path)] == ["audio_only.m4a"]


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
        # Far enough apart not to share one whisper window.
        regions=[(1000, 3000), (40000, 43000)],
        texts=["First thing.", "Much later thing."],
    )
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    result = await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")

    assert [c.text for c in result.cues] == ["First thing.", "Much later thing."]
    # 1000ms and 40000ms, each less 150ms of padding.
    assert [c.start_ms for c in result.cues] == [850, 39850]
    assert all(c.speaker == "Alice Chen" for c in result.cues)


async def test_speakers_merge_in_chronological_order(tmp_path):
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.wav").write_bytes(b"RIFFfake")
    (rec / "audio2_Bob Ruiz.wav").write_bytes(b"RIFFfake")

    # Alice speaks first and third; Bob second. Same fake regions per track,
    # so ordering has to come from the offsets.
    runner = FakeRunner(
        regions=[(0, 2000), (20000, 22000)],
        texts=["alice one", "alice two", "bob one", "bob two"],
    )
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
        # Too far apart to share a window, so three slices go to one process.
        regions=[(0, 1000), (30000, 31000), (60000, 61000)],
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


def test_real_runner_checks_every_executable_it_will_invoke(tmp_path):
    config = make_config(tmp_path)
    config = WhisperConfig(
        model=config.model,
        vad_model=config.vad_model,
        binary="bedrock-missing-whisper",
        vad_binary="bedrock-missing-vad",
        ffmpeg="bedrock-missing-ffmpeg",
    )

    issues = WhisperTranscriber(config).readiness_issues()

    assert any("bedrock-missing-whisper" in issue for issue in issues)
    assert any("bedrock-missing-vad" in issue for issue in issues)
    assert any("bedrock-missing-ffmpeg" in issue for issue in issues)


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


# --- regressions ---------------------------------------------------------


def test_every_file_of_a_split_recording_is_kept(tmp_path):
    """Zoom splits long meetings into several files per participant. Keeping
    one per speaker silently lost the rest of a long call."""
    rec = tmp_path / "Audio Record"
    rec.mkdir(parents=True)
    for name in (
        "audio1757900001_Alice Chen.m4a",
        "audio1757900002_Alice Chen.m4a",
        "audio1757900003_Alice Chen.m4a",
        "audio1757900010_Bob Ruiz.m4a",
    ):
        (rec / name).write_bytes(b"")

    tracks = discover_participant_tracks(tmp_path)
    assert len(tracks["Alice Chen"]) == 3
    assert len(tracks["Bob Ruiz"]) == 1
    # Zoom's filenames are chronological, so file order must be too.
    assert [p.name for p in tracks["Alice Chen"]] == sorted(p.name for p in tracks["Alice Chen"])


def test_a_split_mixed_recording_keeps_every_part(tmp_path):
    (tmp_path / "audio_only1.m4a").write_bytes(b"")
    (tmp_path / "audio_only2.m4a").write_bytes(b"")
    assert len(find_mixed_audio(tmp_path)) == 2


async def test_later_parts_of_a_split_recording_continue_the_timeline(tmp_path):
    """Part two restarts at zero in its own file; its cues have to land after
    part one, not on top of it."""
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.wav").write_bytes(b"RIFFfake")
    (rec / "audio2_Alice Chen.wav").write_bytes(b"RIFFfake")

    # Each file: one region at 1s. Each file is 600s long.
    runner = FakeRunner(regions=[(1000, 3000)], texts=["part one", "part two"], file_duration_s=600.0)
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    result = await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")

    assert [c.text for c in result.cues] == ["part one", "part two"]
    assert result.cues[0].start_ms == 850
    assert result.cues[1].start_ms == 850 + 600_000  # offset by part one's length


async def test_running_twice_does_not_rediscover_its_own_output(tmp_path):
    """The work directory sits inside the recording folder, so a second run
    used to find the 16 kHz copies from the first, transcribe everyone twice,
    and invent speakers called "Alice Chen.16k"."""
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.m4a").write_bytes(b"")

    def run_once():
        runner = FakeRunner(regions=[(1000, 3000)], texts=["hello"])
        return WhisperTranscriber(make_config(tmp_path), runner=runner)

    first = await transcribe_recording(tmp_path / "rec", run_once())
    second = await transcribe_recording(tmp_path / "rec", run_once())

    assert first.speakers == ["Alice Chen"]
    assert second.speakers == ["Alice Chen"], "a second run must see the same speakers"
    assert len(second.cues) == len(first.cues), "a second run must not double the transcript"


def test_our_working_directory_is_not_mistaken_for_audio(tmp_path):
    rec = tmp_path / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.m4a").write_bytes(b"")
    work = tmp_path / ".bedrock-work"
    work.mkdir()
    (work / "audio1_Alice Chen.16k.wav").write_bytes(b"")

    tracks = discover_participant_tracks(tmp_path)
    assert list(tracks) == ["Alice Chen"]
    assert len(tracks["Alice Chen"]) == 1


async def test_a_participant_who_never_speaks_does_not_fail_the_meeting(tmp_path):
    """Somebody joins and stays quiet. VAD still fires on a cough or a door,
    whisper reports [BLANK_AUDIO], and every cue is filtered out. That must not
    take the whole meeting down with it."""
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.wav").write_bytes(b"RIFFfake")
    (rec / "audio2_Silent Sam.wav").write_bytes(b"RIFFfake")

    runner = FakeRunner(regions=[(1000, 3000)], texts=["Real words.", "[BLANK_AUDIO]"])
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    result = await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")

    assert [c.text for c in result.cues] == ["Real words."]
    assert result.speakers == ["Alice Chen"]          # Sam contributed nothing
    assert result.per_speaker_counts["Silent Sam"] == 0  # but is still accounted for


async def test_a_genuine_whisper_failure_still_raises(tmp_path):
    """The silent-participant fix must not swallow a real tool failure."""

    class NoOutputRunner(FakeRunner):
        async def run(self, args, log_path, env=None):
            if list(args)[0] == "whisper-cli":
                # Writes no .vtt at all, as a crashed whisper would.
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("segmentation fault")
                return
            await super().run(args, log_path, env)

    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.wav").write_bytes(b"RIFFfake")
    transcriber = WhisperTranscriber(
        make_config(tmp_path), runner=NoOutputRunner(regions=[(1000, 3000)], texts=["x"])
    )
    with pytest.raises(ToolIntegrationError, match="no transcript"):
        await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")


async def test_a_speaker_with_several_files_is_reported(tmp_path):
    """Two people sharing a display name is indistinguishable from one person's
    split recording, so the caller has to be told it happened."""
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alex.wav").write_bytes(b"RIFFfake")
    (rec / "audio2_Alex.wav").write_bytes(b"RIFFfake")

    runner = FakeRunner(regions=[(1000, 3000)], texts=["one", "two"])
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    result = await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")

    assert result.multi_file_speakers == ["Alex"]


def test_unusual_display_names_survive_discovery(tmp_path):
    rec = tmp_path / "Audio Record"
    rec.mkdir(parents=True)
    for name in (
        "audio1_José García.m4a",
        "audio2_Mary_Jane Smith.m4a",
        "audio3_12345.m4a",
        "AUDIO4_Bob Ruiz.m4a",
    ):
        (rec / name).write_bytes(b"")
    assert set(discover_participant_tracks(tmp_path)) == {
        "José García",
        "Mary_Jane Smith",
        "12345",
        "Bob Ruiz",
    }


def test_a_file_with_no_participant_suffix_is_not_a_participant(tmp_path):
    rec = tmp_path / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1234567.m4a").write_bytes(b"")
    assert discover_participant_tracks(tmp_path) == {}


# --- the unseparated layout Zoom writes today ----------------------------


def _unseparated_recording(tmp_path, *names, recording_id="1441831751", conf=True):
    """A real 2026 Zoom folder: no underscore, id glued to the display name."""
    rec = tmp_path / "Audio Record"
    rec.mkdir(parents=True)
    for name in names:
        (rec / f"audio{name}{recording_id}.m4a").write_bytes(b"")
    (tmp_path / f"audio{recording_id}.m4a").write_bytes(b"")
    (tmp_path / f"video{recording_id}.mp4").write_bytes(b"")
    if conf:
        (tmp_path / "recording.conf").write_text(
            '{"magic_number":"441831751","items":[{"process":100,'
            f'"audio":"audio{recording_id}.m4a","prefix":"",'
            f'"video":"video{recording_id}.mp4"}}]}}',
            encoding="utf-8",
        )
    return tmp_path


def test_participant_tracks_without_a_separator_are_found(tmp_path):
    """The regression. These filenames matched nothing, so a recording that
    did have per-participant tracks was transcribed as one anonymous mix —
    while the tool told the user to switch on a setting already switched on."""
    folder = _unseparated_recording(tmp_path, "303+G12+Eric2", "Goat1")
    assert set(discover_participant_tracks(folder)) == {"303+G12+Eric2", "Goat1"}


def test_a_name_ending_in_digits_keeps_them(tmp_path):
    """"audioGoat11441831751" splits as "Goat1" or "Goat11" and the filename
    cannot say which. recording.conf names the mixed track, whose digits are
    the recording id, which settles it."""
    folder = _unseparated_recording(tmp_path, "Goat1")
    assert set(discover_participant_tracks(folder)) == {"Goat1"}


def test_the_mixed_recording_is_not_mistaken_for_a_participant(tmp_path):
    folder = _unseparated_recording(tmp_path, "Alice Chen")
    assert set(discover_participant_tracks(folder)) == {"Alice Chen"}
    assert [p.name for p in find_mixed_audio(folder)] == ["audio1441831751.m4a"]


def test_both_naming_layouts_can_coexist(tmp_path):
    folder = _unseparated_recording(tmp_path, "Goat1")
    (folder / "Audio Record" / "audio999_Bob Ruiz.m4a").write_bytes(b"")
    assert set(discover_participant_tracks(folder)) == {"Goat1", "Bob Ruiz"}


def test_speakers_are_still_found_without_recording_conf(tmp_path):
    """Two speakers share the recording id, so it is recoverable from the
    filenames alone."""
    folder = _unseparated_recording(tmp_path, "Alice Chen", "Bob Ruiz", conf=False)
    assert set(discover_participant_tracks(folder)) == {"Alice Chen", "Bob Ruiz"}


def test_an_unconverted_recording_is_recognised(tmp_path):
    """Zoom leaves .zoom files behind when it is quit before conversion
    finishes. They sit there forever, so 'wait for it to finish' is the wrong
    advice — somebody has to start it."""
    from bedrock.meetings.transcribe import find_unconverted

    (tmp_path / "double_click_to_convert_01.zoom").write_bytes(b"")
    (tmp_path / "double_click_to_convert_02.zoom").write_bytes(b"")
    (tmp_path / "zoomver.tag").write_bytes(b"")

    assert len(find_unconverted(tmp_path)) == 2
    assert discover_participant_tracks(tmp_path) == {}
    assert find_mixed_audio(tmp_path) == []  # .zoom is not playable audio


async def test_an_unconverted_recording_says_how_to_convert_it(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "double_click_to_convert_01.zoom").write_bytes(b"")
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=FakeRunner([], []))

    with pytest.raises(ToolIntegrationError, match="not been converted"):
        await transcribe_recording(rec, transcriber, work_dir=tmp_path / "w")


def test_a_converted_recording_is_not_flagged_as_unconverted(tmp_path):
    from bedrock.meetings.transcribe import find_unconverted

    rec = tmp_path / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.m4a").write_bytes(b"")
    assert find_unconverted(tmp_path) == []


def test_a_long_uninterrupted_turn_is_split_below_whispers_window():
    """Whisper caps a segment at 30s, so a longer slice returns one cue holding
    everything said in it and every citation collapses onto its start."""
    from bedrock.meetings.transcribe import MAX_TURN_MS

    # One continuous 50-second region, as a monologue produces.
    pieces = _group_regions([(0, 50_000)], max_gap_ms=600)
    assert len(pieces) > 1
    assert all(end - start <= MAX_TURN_MS for start, end in pieces)
    assert pieces[0][0] == 0 and pieces[-1][1] == 50_000  # nothing lost


def test_joining_stops_at_the_cap():
    regions = [(i * 3000, i * 3000 + 2500) for i in range(12)]  # 36s of near-continuous speech
    grouped = _group_regions(regions, max_gap_ms=600)
    assert len(grouped) > 1
    assert all(end - start <= 20_000 for start, end in grouped)


def test_short_turns_are_untouched():
    assert _group_regions([(990, 2080), (2310, 8450)], max_gap_ms=600) == [(990, 8450)]


async def test_a_cue_cannot_outlast_the_slice_it_came_from(tmp_path):
    """Whisper reports a cue ending at its 30s window edge even for a short
    slice, which made brief remarks look half a minute long."""
    rec = tmp_path / "rec" / "Audio Record"
    rec.mkdir(parents=True)
    (rec / "audio1_Alice Chen.wav").write_bytes(b"RIFFfake")

    class WindowRunner(FakeRunner):
        async def run(self, args, log_path, env=None):
            if list(args)[0] == "whisper-cli":
                inputs = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
                for path in inputs:
                    # A 2-second remark reported as filling a 30-second window.
                    Path(path + ".vtt").write_text(
                        "WEBVTT\n\n00:00:00.000 --> 00:00:30.000\nA short remark.\n"
                    )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("")
                return
            await super().run(args, log_path, env)

    runner = WindowRunner(regions=[(1000, 3000)], texts=["A short remark."])
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    result = await transcribe_recording(tmp_path / "rec", transcriber, work_dir=tmp_path / "w")

    cue = result.cues[0]
    # Region 1000-3000 padded by 150ms each side ends at 3150, not 30150.
    assert cue.end_ms == 3150
    assert cue.end_ms - cue.start_ms < 5000


def test_the_speech_model_can_be_chosen():
    """Comparing speech models on real audio needs to be a flag, not an edit."""
    from bedrock.meetings.cli import SPEECH_MODEL, build_parser

    assert build_parser().parse_args([]).speech_model == SPEECH_MODEL
    assert (
        build_parser().parse_args(["--speech-model", "ggml-large-v3-turbo.bin"]).speech_model
        == "ggml-large-v3-turbo.bin"
    )


# --- Hearing the far end of the room, and other languages --------------------


async def test_levelling_is_applied_only_when_asked(tmp_path):
    from dataclasses import replace

    from bedrock.meetings.transcribe import LEVELLING_FILTER

    src = tmp_path / "in.webm"
    src.write_bytes(b"x")

    runner = FakeRunner(regions=[], texts=[])
    await WhisperTranscriber(make_config(tmp_path), runner=runner).to_wav(src, tmp_path / "w")
    (ffmpeg,) = [c for c in runner.commands if c[0] == "ffmpeg"]
    assert "-af" not in ffmpeg

    runner = FakeRunner(regions=[], texts=[])
    levelled = replace(make_config(tmp_path), normalize_audio=True)
    await WhisperTranscriber(levelled, runner=runner).to_wav(src, tmp_path / "w")
    (ffmpeg,) = [c for c in runner.commands if c[0] == "ffmpeg"]
    assert ffmpeg[ffmpeg.index("-af") + 1] == LEVELLING_FILTER
    # The filter goes between the input and the output settings.
    assert ffmpeg.index("-i") < ffmpeg.index("-af") < ffmpeg.index("-ar")


async def test_vad_sensitivity_is_passed_through(tmp_path):
    from dataclasses import replace

    runner = FakeRunner(regions=[(0, 1000)], texts=[])
    config = replace(make_config(tmp_path), vad_threshold=0.4, vad_speech_pad_ms=60)
    await WhisperTranscriber(config, runner=runner).detect_speech_regions(tmp_path / "a.wav", tmp_path / "w")
    (vad,) = [c for c in runner.commands if c[0] == "whisper-vad-speech-segments"]
    assert vad[vad.index("-vt") + 1] == "0.4"
    assert vad[vad.index("-vp") + 1] == "60"


def test_an_english_only_model_cannot_be_asked_for_another_language(tmp_path):
    from dataclasses import replace

    from bedrock.meetings.transcribe import is_english_only_model

    assert is_english_only_model(Path("ggml-small.en.bin"))
    assert is_english_only_model(Path("ggml-medium.en-q5_0.bin"))
    assert not is_english_only_model(Path("ggml-large-v3-turbo.bin"))
    assert not is_english_only_model(Path("ggml-small.bin"))

    english_only = tmp_path / "ggml-small.en.bin"
    english_only.write_bytes(b"x")
    config = replace(make_config(tmp_path), model=english_only)
    runner = FakeRunner(regions=[], texts=[])

    assert WhisperTranscriber(config, runner=runner).readiness_issues() == ()
    issues = WhisperTranscriber(replace(config, language="zh"), runner=runner).readiness_issues()
    assert len(issues) == 1 and "English-only" in issues[0] and "'zh'" in issues[0]
    assert WhisperTranscriber(replace(config, model=tmp_path / "model.bin", language="zh"), runner=runner).readiness_issues() == ()


# --- Listening before deciding the language ---------------------------------


class DetectingRunner(FakeRunner):
    """A FakeRunner that also answers whisper's language detection.

    `languages` maps a slice's turn number (from its file name) to the code
    whisper "detects" for it; slices not listed are detected as English.
    """

    def __init__(self, regions, texts, languages=None, **kw):
        super().__init__(regions, texts, **kw)
        self.languages = languages or {}

    async def run(self, args, log_path, env=None):
        args = list(args)
        if args[0] == "whisper-cli" and "-dl" in args:
            self.commands.append(args)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for path in (args[i + 1] for i, a in enumerate(args) if a == "-f"):
                turn = int(Path(path).stem.replace("turn", ""))
                code = self.languages.get(turn, "en")
                lines += [f"main: processing '{path}' (x samples), 4 threads ...",
                          f"whisper_full_with_state: auto-detected language: {code} (p = 0.98)"]
            log_path.write_text("\n".join(lines))
            return
        await super().run(args, log_path, env)


def _whisper_command(runner):
    (cmd,) = [c for c in runner.commands if c[0] == "whisper-cli" and "-dl" not in c]
    return cmd


def _detection_command(runner):
    found = [c for c in runner.commands if c[0] == "whisper-cli" and "-dl" in c]
    return found[0] if found else None


# Six turns: four long ones of eight seconds and two short ones of a second
# and a half. Slices carry a little padding either side, so shares are
# compared loosely rather than to the round numbers the regions suggest.
REGIONS = [(0, 8000), (9000, 10500), (12000, 20000), (21000, 29000), (30000, 31500), (33000, 41000)]


async def test_one_dominant_language_is_used_for_the_whole_recording(tmp_path):
    from dataclasses import replace

    from bedrock.meetings.transcribe import MANDARIN_PROMPT

    # The four long turns are Mandarin: about 32 of 35 seconds.
    runner = DetectingRunner(REGIONS, ["x"] * 6, languages={0: "zh", 2: "zh", 3: "zh", 5: "zh"})
    config = replace(make_config(tmp_path), language="detect", initial_prompt="KRAS")
    transcriber = WhisperTranscriber(config, runner=runner)
    await transcriber.transcribe_track(tmp_path / "a.wav", tmp_path / "w")

    detect = _detection_command(runner)
    assert detect is not None and detect.count("-f") == 6
    cmd = _whisper_command(runner)
    assert cmd[cmd.index("-l") + 1] == "zh"
    assert cmd[cmd.index("--prompt") + 1] == f"{MANDARIN_PROMPT} KRAS"
    assert transcriber.last_language.language == "zh"
    assert transcriber.last_language.heard["zh"] == pytest.approx(0.91, abs=0.03)
    assert list(transcriber.last_language.heard) == ["zh", "en"]
    assert transcriber.last_language.sampled == 6


async def test_a_mixed_meeting_is_detected_turn_by_turn(tmp_path):
    from dataclasses import replace

    # Two of the long turns are Mandarin: about 16 of 35 seconds.
    runner = DetectingRunner(REGIONS, ["x"] * 6, languages={0: "zh", 2: "zh"})
    config = replace(make_config(tmp_path), language="detect")
    transcriber = WhisperTranscriber(config, runner=runner)
    await transcriber.transcribe_track(tmp_path / "a.wav", tmp_path / "w")

    cmd = _whisper_command(runner)
    assert cmd[cmd.index("-l") + 1] == "auto"
    assert "--prompt" not in cmd
    assert transcriber.last_language.mixed
    assert transcriber.last_language.heard["zh"] == pytest.approx(0.46, abs=0.03)
    assert transcriber.last_language.heard["en"] == pytest.approx(0.54, abs=0.03)


async def test_detection_samples_the_longest_slices_only(tmp_path):
    from dataclasses import replace

    from bedrock.meetings.transcribe import DETECT_SAMPLES

    regions = [(i * 3000, i * 3000 + 1000 + (i % 5) * 200) for i in range(30)]
    runner = DetectingRunner(regions, ["x"] * 30)
    config = replace(make_config(tmp_path), language="detect")
    await WhisperTranscriber(config, runner=runner).transcribe_track(tmp_path / "a.wav", tmp_path / "w")
    assert _detection_command(runner).count("-f") == DETECT_SAMPLES
    # Detection sampled the turns; transcription got them joined into windows.
    assert 1 <= _whisper_command(runner).count("-f") < 30


async def test_an_english_only_model_never_runs_detection(tmp_path):
    from dataclasses import replace

    english_only = tmp_path / "ggml-small.en.bin"
    english_only.write_bytes(b"x")
    runner = DetectingRunner(REGIONS, ["x"] * 6, languages={0: "zh"})
    config = replace(make_config(tmp_path), model=english_only, language="detect")
    transcriber = WhisperTranscriber(config, runner=runner)
    assert transcriber.readiness_issues() == ()
    await transcriber.transcribe_track(tmp_path / "a.wav", tmp_path / "w")
    assert _detection_command(runner) is None
    assert _whisper_command(runner)[_whisper_command(runner).index("-l") + 1] == "en"
    assert transcriber.last_language.sampled == 0


async def test_a_chosen_language_skips_detection_and_mandarin_gets_its_hint(tmp_path):
    from dataclasses import replace

    from bedrock.meetings.transcribe import MANDARIN_PROMPT

    runner = DetectingRunner(REGIONS, ["x"] * 6)
    config = replace(make_config(tmp_path), language="zh")
    await WhisperTranscriber(config, runner=runner).transcribe_track(tmp_path / "a.wav", tmp_path / "w")
    assert _detection_command(runner) is None
    cmd = _whisper_command(runner)
    assert cmd[cmd.index("-l") + 1] == "zh"
    assert cmd[cmd.index("--prompt") + 1] == MANDARIN_PROMPT

    runner = DetectingRunner(REGIONS, ["x"] * 6)
    config = replace(make_config(tmp_path), language="zh", initial_prompt=f"{MANDARIN_PROMPT} KRAS")
    await WhisperTranscriber(config, runner=runner).transcribe_track(tmp_path / "a.wav", tmp_path / "w")
    cmd = _whisper_command(runner)
    assert cmd.count(MANDARIN_PROMPT) == 0 and cmd[cmd.index("--prompt") + 1].count(MANDARIN_PROMPT) == 1


def test_the_dominance_rule():
    from bedrock.meetings.transcribe import decide_language

    assert decide_language({"zh": 1.0}) == "zh"
    assert decide_language({"zh": 0.8, "en": 0.2}) == "zh"
    assert decide_language({"zh": 0.79, "en": 0.21}) == "auto"
    assert decide_language({}) == "auto"


async def test_an_unsure_detection_is_not_counted(tmp_path):
    from dataclasses import replace

    class Unsure(DetectingRunner):
        async def run(self, args, log_path, env=None):
            if args[0] == "whisper-cli" and "-dl" in list(args):
                self.commands.append(list(args))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                paths = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
                lines = []
                for n, path in enumerate(paths):
                    p = "0.3" if n == 0 else "0.99"
                    lines += [f"main: processing '{path}'", f"auto-detected language: {'ja' if n == 0 else 'en'} (p = {p})"]
                log_path.write_text("\n".join(lines))
                return
            await FakeRunner.run(self, args, log_path, env)

    runner = Unsure(REGIONS, ["x"] * 6)
    transcriber = WhisperTranscriber(replace(make_config(tmp_path), language="detect"), runner=runner)
    await transcriber.transcribe_track(tmp_path / "a.wav", tmp_path / "w")
    assert transcriber.last_language.heard == {"en": 1.0}


# --- Whole windows, and whisper's reflexes -----------------------------------


def test_turns_are_joined_into_windows_but_not_across_a_cut_inside_a_turn():
    from bedrock.meetings.transcribe import _join_into_windows

    # Four turns with silences between them fit one 24-second window.
    assert _join_into_windows([(0, 4000), (5000, 9000), (10000, 15000), (16000, 22000)], 24_000) == [(0, 22000)]
    # A fifth that would push past the window starts a new one.
    assert _join_into_windows([(0, 4000), (5000, 9000), (10000, 20000), (21000, 26000)], 24_000) == [(0, 20000), (21000, 26000)]
    # Pieces that touch were cut on purpose — a speaker change with no pause —
    # and are never joined back together.
    assert _join_into_windows([(0, 4000), (4000, 8000), (9000, 12000)], 24_000) == [(0, 4000), (4000, 12000)]
    assert _join_into_windows([], 24_000) == []


async def test_whisper_gets_windows_not_turns(tmp_path):
    regions = [(0, 3000), (4000, 7000), (8000, 11000), (40000, 43000)]
    runner = FakeRunner(regions=regions, texts=["a b", "c"])
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    cues = await transcriber.transcribe_track(tmp_path / "a.wav", tmp_path / "w")
    (cmd,) = [c for c in runner.commands if c[0] == "whisper-cli" and "-dl" not in c]
    assert cmd.count("-f") == 2  # three close turns became one window; the far one its own
    assert "--suppress-nst" in cmd
    assert [c.text for c in cues] == ["a b", "c"]
    assert cues[1].start_ms >= 39000  # timestamps still land on the meeting timeline


def test_a_reflex_on_a_second_of_noise_is_recognised():
    from bedrock.meetings.transcribe import looks_hallucinated

    assert looks_hallucinated(["Thank you."], 1500)
    assert looks_hallucinated(["Thanks for watching!"], 2900)
    assert looks_hallucinated(["*coughs*", "you"], 1000) is False  # not a listed phrase as a whole
    assert not looks_hallucinated(["Thank you."], 3500)  # long enough to be real
    assert not looks_hallucinated(["Thank you for the numbers."], 1500)
    assert not looks_hallucinated(["Okay, so the controls came back clean."], 1500)


async def test_a_hallucinated_slice_is_dropped_and_a_real_short_one_kept(tmp_path):
    regions = [(0, 1500), (30000, 31500), (60000, 62000)]
    runner = FakeRunner(regions=regions, texts=["Thank you.", "Order the antibody.", "Thanks for watching"])
    transcriber = WhisperTranscriber(make_config(tmp_path), runner=runner)
    cues = await transcriber.transcribe_track(tmp_path / "a.wav", tmp_path / "w")
    assert [c.text for c in cues] == ["Order the antibody."]


async def test_the_decided_language_picks_the_checkpoint(tmp_path):
    from dataclasses import replace

    zh_model = tmp_path / "large.bin"
    zh_model.write_bytes(b"x")
    runner = DetectingRunner(REGIONS, ["x"] * 6, languages={0: "zh", 2: "zh", 3: "zh", 5: "zh"})
    config = replace(make_config(tmp_path), language="detect", models_by_language={"zh": zh_model})
    await WhisperTranscriber(config, runner=runner).transcribe_track(tmp_path / "a.wav", tmp_path / "w")
    detect, cmd = _detection_command(runner), _whisper_command(runner)
    assert detect[detect.index("-m") + 1] == str(config.model)   # detection on the main model
    assert cmd[cmd.index("-m") + 1] == str(zh_model)              # transcription on the Mandarin one

    # An English decision, or a Mandarin model file that is missing, uses the main model.
    runner = DetectingRunner(REGIONS, ["x"] * 6)
    await WhisperTranscriber(config, runner=runner).transcribe_track(tmp_path / "a.wav", tmp_path / "w2")
    assert _whisper_command(runner)[_whisper_command(runner).index("-m") + 1] == str(config.model)
    runner = DetectingRunner(REGIONS, ["x"] * 6, languages={0: "zh", 2: "zh", 3: "zh", 5: "zh"})
    gone = replace(config, models_by_language={"zh": tmp_path / "missing.bin"})
    await WhisperTranscriber(gone, runner=runner).transcribe_track(tmp_path / "a.wav", tmp_path / "w3")
    assert _whisper_command(runner)[_whisper_command(runner).index("-m") + 1] == str(config.model)
