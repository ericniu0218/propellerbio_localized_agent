"""Reading the video call's own interface to find out who is talking.

For an online meeting the answer to "who said that?" is already on the screen.
Zoom, Meet and Teams all draw every participant's name under their tile and put
a coloured ring around whoever is speaking. Working that out from the audio —
clustering voiceprints and hoping — is doing badly, from scratch, something the
meeting software has already done perfectly.

And the video is already here. `getDisplayMedia` only offers tab audio if video
is requested too, so the recorder has always asked for a video track and thrown
it away. This reads it.

What comes out is a timeline of real names: *Luke Li was the active speaker from
0:12 to 0:31*. That is better than any amount of voice matching, because it is
not an inference — it is what the meeting software displayed at the time. It
needs no enrolment, no renaming, and no guess about how many people are there.

**No frame is ever stored.** A meeting recording is audio; a meeting *video* is
people's faces in their homes, and keeping those would be a far bigger thing to
hold than a transcript. Frames are read and discarded in the same function
call; what survives is a list of names and the seconds they were speaking.

**Everything here is a hint, never a requirement.** Layouts differ, names can
be hidden, someone may be screen-sharing instead of showing tiles, and the
accent colour changes between versions and platforms. Every function returns
nothing rather than guessing, and the audio pipeline works exactly as it did
without any of it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    import numpy
    from rapidocr_onnxruntime import RapidOCR

    SCREEN_AVAILABLE = True
except ImportError:  # pragma: no cover - deployments without the extra
    SCREEN_AVAILABLE = False

SCREEN_MISSING = (
    "Reading names from the meeting window is unavailable: rapidocr-onnxruntime "
    "is not installed. Speakers fall back to being worked out from the audio."
)

# A video call is grey chrome and skin tones. The ring around the active
# speaker is the one strongly saturated thing on screen, in every client I
# have looked at — Zoom's green, Teams' purple, Meet's blue. Matching
# *saturation* rather than a specific hue is what makes this survive a
# re-theme, a platform change, and dark mode.
MIN_SATURATION = 60
MIN_BRIGHTNESS = 90

# Below this the "ring" is a notification badge or somebody's jumper.
MIN_HIGHLIGHT_PIXELS = 1500

# OCR picks up clocks, buttons and meeting titles as well as names.
_NOT_A_NAME = frozenset(
    """mute unmute start stop video participants chat share screen record
    reactions apps more leave end view speaker gallery you host co-host
    recording am pm zoom meet teams meeting join leave audio settings
    everyone waiting room security invite raise hand""".split()
)
MAX_NAME_WORDS = 4
MIN_NAME_LENGTH = 2


@dataclass(frozen=True)
class Label:
    text: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def centre(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)


@dataclass
class FrameReading:
    """What one frame of the meeting window said."""

    names: list[Label] = field(default_factory=list)
    active: str | None = None
    """Whoever the meeting software was highlighting, if it was highlighting
    anybody. None is an ordinary answer: nobody talking, a shared screen, a
    layout without rings."""

    @property
    def roster(self) -> list[str]:
        return [label.text for label in self.names]


@lru_cache(maxsize=1)
def _reader():
    return RapidOCR()


def enabled() -> bool:
    return SCREEN_AVAILABLE and os.environ.get("MEETING_READ_SCREEN", "true").lower() != "false"


def looks_like_a_name(text: str) -> bool:
    """Whether a piece of on-screen text is plausibly somebody's name.

    Deliberately strict. A roster polluted with "Unmute" and "Gallery View"
    would attach button labels to spoken lines, which is worse than having no
    roster — it looks like data and is not.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) < MIN_NAME_LENGTH or len(cleaned.split()) > MAX_NAME_WORDS:
        return False
    if any(word.strip(".,()").casefold() in _NOT_A_NAME for word in cleaned.split()):
        return False
    # Names are mostly letters. Timestamps, counters and "(2)" are not.
    letters = sum(character.isalpha() or character.isspace() for character in cleaned)
    return letters / len(cleaned) >= 0.7 and any(c.isalpha() for c in cleaned)


def find_highlight(image) -> tuple[int, int, int, int] | None:
    """The box the meeting software is drawing around the active speaker.

    Returns None when nothing on screen is saturated enough to be a ring, which
    is the common case for "nobody is speaking" and for a shared screen.
    """
    pixels = numpy.asarray(image, dtype=numpy.int16)
    if pixels.ndim != 3:
        return None
    brightest = pixels.max(axis=2)
    mask = ((brightest - pixels.min(axis=2)) > MIN_SATURATION) & (brightest > MIN_BRIGHTNESS)
    if int(mask.sum()) < MIN_HIGHLIGHT_PIXELS:
        return None
    rows, columns = numpy.nonzero(mask)
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def read(image) -> FrameReading:
    """Names and the active speaker from one frame. Never raises."""
    if not enabled():
        return FrameReading()
    try:
        raw, _ = _reader()(numpy.asarray(image))
    except Exception as exc:  # pragma: no cover - model-level failure
        logger.debug("Could not read the meeting window: %s", exc)
        return FrameReading()

    labels: list[Label] = []
    for item in raw or []:
        box, text = item[0], item[1]
        if not looks_like_a_name(str(text)):
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        labels.append(
            Label(" ".join(str(text).split()), int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        )

    reading = FrameReading(names=labels)
    highlight = find_highlight(image)
    if highlight and labels:
        left, top, right, bottom = highlight
        inside = [
            label
            for label in labels
            if left <= label.centre[0] <= right and top <= label.centre[1] <= bottom
        ]
        # Exactly one name inside the ring is the good case. Several means the
        # "ring" was something else — a shared slide, a bright background — and
        # guessing between them would invent an attribution.
        if len(inside) == 1:
            reading.active = inside[0].text
        elif len(labels) == 1:
            # Speaker view: one tile, one name, and it is whoever is talking.
            reading.active = labels[0].text
    return reading


@dataclass
class ScreenTimeline:
    """Who the meeting window said was speaking, and when.

    Built from frames sampled every few seconds during the call, so it is a
    sequence of observations rather than a continuous record: each one says
    "at 0:42, the ring was around Luke Li".
    """

    observations: list[tuple[int, str]] = field(default_factory=list)
    roster: list[str] = field(default_factory=list)
    frames: int = 0

    def observe(self, at_ms: int, reading: FrameReading) -> None:
        self.frames += 1
        for name in reading.roster:
            if name not in self.roster:
                self.roster.append(name)
        if reading.active:
            self.observations.append((at_ms, reading.active))

    @property
    def ok(self) -> bool:
        return bool(self.observations)

    def speaker_at(self, start_ms: int, end_ms: int, window_ms: int = 1500) -> str | None:
        """Whoever was highlighted during a stretch of the meeting.

        Observations inside the stretch are used on their own, and only if
        there are none does this widen by `window_ms` either side. That
        ordering matters more than it looks: real turns run three to five
        seconds, so a window wide enough to be "nearby" also swallows the
        turns on both sides completely. Sampling every four seconds with a
        four-second window put each turn's neighbours into its vote and
        produced a transcript where two speakers' names were swapped —
        confidently, and with nothing to show it was wrong.

        A tie means two people were highlighted across the stretch, which is
        exactly when not to guess.
        """
        inside = [name for at_ms, name in self.observations if start_ms <= at_ms <= end_ms]
        if not inside:
            inside = [
                name
                for at_ms, name in self.observations
                if start_ms - window_ms <= at_ms <= end_ms + window_ms
            ]
        if not inside:
            return None

        votes: dict[str, int] = {}
        for name in inside:
            votes[name] = votes.get(name, 0) + 1
        ranked = sorted(votes.items(), key=lambda pair: -pair[1])
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    def name_cues(self, cues) -> dict[int, str]:
        """Name each *line* directly, from what the window showed at that moment.

        This is the whole reason reading the screen beats inferring speakers
        from audio, and it took getting it wrong to see why. Naming diarised
        *clusters* and letting the labels flow down to their lines means the
        names inherit every clustering error: when two people merged into one
        cluster, every one of their lines got a single name, and only 3 of 10
        lines came out right.

        The window already knew who was speaking at 0:07. Using that for the
        line at 0:07 makes naming independent of how well the voices clustered
        — which is the point, because clustering is the part that is hard and
        the screen is the part that is certain.

        Returns only the lines it is sure about, keyed by index.
        """
        names: dict[int, str] = {}
        for index, cue in enumerate(cues):
            name = self.speaker_at(cue.start_ms, cue.end_ms)
            if name:
                names[index] = name
        return names

    def name_clusters(self, turns, minimum_share: float = 0.5) -> dict[int, str]:
        """Map each diarised speaker to the name highlighted during their turns.

        This is the payoff: the labels stop being "Speaker 2" and become what
        the meeting software already knew. A cluster only takes a name if that
        name won a clear majority of its speaking time — otherwise the cluster
        spans a handover, or the ring was on somebody else, and a placeholder
        is the honest answer.

        One name is never given to two clusters. If diarisation split a person
        in two, the cluster with more of their speech keeps the name.
        """
        by_speaker: dict[int, dict[str, int]] = {}
        for turn in turns:
            name = self.speaker_at(turn.start_ms, turn.end_ms)
            if name:
                weight = turn.end_ms - turn.start_ms
                by_speaker.setdefault(turn.speaker, {})
                by_speaker[turn.speaker][name] = by_speaker[turn.speaker].get(name, 0) + weight

        scored: list[tuple[float, int, str]] = []
        for speaker, votes in by_speaker.items():
            total = sum(votes.values())
            name, weight = max(votes.items(), key=lambda pair: pair[1])
            if total and weight / total >= minimum_share:
                scored.append((weight / total, speaker, name))

        scored.sort(reverse=True)
        names: dict[int, str] = {}
        taken: set[str] = set()
        for _share, speaker, name in scored:
            if speaker in names or name in taken:
                continue
            names[speaker] = name
            taken.add(name)
        return names
