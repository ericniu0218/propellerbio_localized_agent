"""Reading the video call's own interface to find out who is talking.

For an online meeting the answer is already on screen: every client draws
participant names and rings whoever is speaking. Inferring it from voices is
doing badly, from scratch, what the meeting software already did.
"""

from __future__ import annotations

from dataclasses import dataclass

from bedrock.server.meetings.screen import FrameReading, Label, ScreenTimeline, looks_like_a_name


@dataclass
class Cue:
    start_ms: int
    end_ms: int


def label(text: str) -> Label:
    return Label(text, 0, 0, 10, 10)


def timeline(*observations: tuple[int, str]) -> ScreenTimeline:
    line = ScreenTimeline()
    for at_ms, name in observations:
        line.observe(at_ms, FrameReading(names=[label(name)], active=name))
    return line


# --- Telling a name from a button --------------------------------------------


def test_a_persons_name_is_a_name():
    assert looks_like_a_name("Alejandro Jarquin")
    assert looks_like_a_name("Luke Li")


def test_interface_chrome_is_not_a_name():
    """A roster polluted with "Unmute" would attach button labels to spoken
    lines, which is worse than no roster: it looks like data and is not."""
    for text in ["Unmute", "Participants", "Gallery View", "Share Screen", "Leave"]:
        assert not looks_like_a_name(text), text


def test_clocks_and_counters_are_not_names():
    for text in ["12:04", "(2)", "1 / 4", "00:45:12"]:
        assert not looks_like_a_name(text), text


def test_a_whole_sentence_is_not_a_name():
    assert not looks_like_a_name("Recording in progress, everyone can see this")


# --- Who was speaking, and when ----------------------------------------------


def test_the_window_says_who_was_speaking():
    assert timeline((5000, "Luke Li")).speaker_at(4000, 6000) == "Luke Li"


def test_observations_inside_a_line_beat_ones_nearby():
    """Turns run three to five seconds, so a window wide enough to be "nearby"
    swallows the turns either side. Sampling every four seconds with a
    four-second window swapped two speakers' names — confidently, with nothing
    to show it was wrong."""
    line = timeline((1000, "Alice"), (5000, "Bob"), (9000, "Cara"))
    assert line.speaker_at(4000, 6000) == "Bob"


def test_a_stretch_nobody_was_highlighted_in_gets_no_name():
    assert timeline((60_000, "Luke Li")).speaker_at(1000, 2000) is None


def test_a_tie_is_not_guessed():
    """Two people highlighted across one stretch is exactly when not to pick."""
    line = timeline((4500, "Alice"), (5500, "Bob"))
    assert line.speaker_at(4000, 6000) is None


# --- Naming the lines --------------------------------------------------------


def test_each_line_is_named_from_its_own_moment():
    """The payoff, and the thing that was wrong first time. Naming diarised
    clusters made the names inherit every clustering error: when two people
    merged into one cluster, 3 of 10 lines came out right. Naming each line
    from what the window showed at that moment makes it independent of how
    well the voices separated."""
    line = timeline((1000, "Alice"), (5000, "Bob"), (9000, "Cara"))
    cues = [Cue(0, 2000), Cue(4000, 6000), Cue(8000, 10000)]
    assert line.name_cues(cues) == {0: "Alice", 1: "Bob", 2: "Cara"}


def test_lines_the_window_cannot_account_for_are_left_alone():
    """They keep whatever the audio worked out, rather than being guessed at."""
    line = timeline((1000, "Alice"))
    assert line.name_cues([Cue(0, 2000), Cue(60_000, 62_000)]) == {0: "Alice"}


def test_an_empty_timeline_names_nothing():
    assert ScreenTimeline().name_cues([Cue(0, 1000)]) == {}
    assert ScreenTimeline().ok is False


# --- The roster ---------------------------------------------------------------


def test_the_roster_collects_everyone_seen():
    line = ScreenTimeline()
    line.observe(0, FrameReading(names=[label("Alice"), label("Bob")], active="Alice"))
    line.observe(5000, FrameReading(names=[label("Bob"), label("Cara")], active="Cara"))
    assert line.roster == ["Alice", "Bob", "Cara"]


def test_a_frame_with_nobody_highlighted_still_adds_to_the_roster():
    """Nobody talking is an ordinary frame, and the names are still worth having."""
    line = ScreenTimeline()
    line.observe(0, FrameReading(names=[label("Alice")], active=None))
    assert line.roster == ["Alice"]
    assert line.ok is False
