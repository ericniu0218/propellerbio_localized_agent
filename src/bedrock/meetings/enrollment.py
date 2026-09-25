"""Voice prints, so a meeting says "Priya" instead of "Speaker 2".

Diarisation can tell voices apart but has no idea whose they are, so every
recorded meeting arrived with placeholder labels and somebody renamed them by
hand, every time. That is the part people actually disliked: an action item
owned by "Speaker 2" is no more use than one owned by nobody, and the renaming
never happens for a meeting nobody gets round to opening.

Somebody records ten seconds of themselves once. That becomes a 192-dimension
vector, and afterwards any cluster of speech in any meeting can be scored
against it. The audio sample is not kept — only the vector, which cannot be
played back.

**A voice print is not authentication.** It is a similarity score against a
recording, it can be fooled by a good impersonation and confused by a cold, and
the store holds no proof that the person who enrolled is who they said. It is
here to save people typing names into a form, and must never be load-bearing
for who may read what: that is `governance.py`, which works off group
membership from the identity provider.

Matching is deliberately conservative. Below the threshold a speaker keeps its
"Speaker 2" label rather than being given the closest name, because a meeting
that attributes a decision to the wrong colleague is worse than one that
attributes it to nobody — the second is obviously incomplete and the first
looks finished and is wrong.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bedrock.meetings.vectors import centroid, cosine

logger = logging.getLogger(__name__)

# Recording and embedding a voice print needs the models; comparing two of them
# afterwards does not. identify() below is deliberately outside this guard.
try:
    import numpy
    import sherpa_onnx
    import soundfile

    ENROLLMENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ENROLLMENT_AVAILABLE = False

ENROLLMENT_MISSING = (
    "Voice prints are unavailable: sherpa-onnx is not installed. "
    "Install requirements.txt to enable them."
)

# Cosine similarity a cluster has to reach before it is given a name. Tuned to
# refuse rather than guess: see the module docstring.
DEFAULT_THRESHOLD = 0.55

# Ten seconds is enough for a stable vector and short enough that people will
# actually do it. Below about three, the vector is mostly the start of a word.
MIN_SAMPLE_S = 3.0

# The TitaNet ONNX export has a hard ceiling. Its encoder is built for 12,288
# frames at a 10ms hop - a little over 123 seconds - and audio longer than that
# does not degrade, it raises, from deep inside onnxruntime:
#
#   Attempting to broadcast an axis by a dimension other than 1. 12288 by 17998
#
# cluster_embeddings used to concatenate a speaker's three longest turns and
# embed the lot in one call, with no cap. In a four-minute two-person recording
# that is already 130 seconds, so the call raised, the exception was logged and
# swallowed, and the speaker came back with no vector at all. With no vector
# they cannot be matched against an enrolled voice, so they keep "Speaker 1".
#
# The person it hits first is whoever talked most, which is why this looked
# like "enrolment does not work" rather than like a crash: it was silent, and
# it took out the main speaker of every meeting long enough to matter. It only
# ever worked on short test clips.
#
# 60 seconds is the cap now, less than half the ceiling, with no accuracy cost:
# a voice print stops improving long before a minute of speech.
MAX_EMBED_S = 60.0
MAX_SAMPLE_S = MAX_EMBED_S


def threshold() -> float:
    return float(os.environ.get("MEETING_VOICE_THRESHOLD", DEFAULT_THRESHOLD))


@dataclass(frozen=True)
class VoicePrint:
    person_id: str
    name: str
    created_at: str
    seconds: float

    def public(self) -> dict:
        """What the browser may see. Never the vector itself."""
        return {
            "id": self.person_id,
            "name": self.name,
            "createdAt": self.created_at,
            "seconds": round(self.seconds, 1),
        }


class VoicePrintStore:
    """One SQLite table of vectors, scoped by owner.

    Scoped rather than global because a voice print is personal data. A
    deployment that wants one company-wide directory of voices passes the same
    owner for everybody, which is a decision worth making on purpose.
    """

    def __init__(self, root: Path) -> None:
        self.path = Path(root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS voice_prints (
                       person_id TEXT NOT NULL,
                       owner     TEXT NOT NULL,
                       name      TEXT NOT NULL,
                       vector    TEXT NOT NULL,
                       seconds   REAL NOT NULL,
                       created   TEXT NOT NULL,
                       PRIMARY KEY (owner, person_id))"""
            )

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, owner: str, person_id: str, name: str, vector, seconds: float) -> VoicePrint:
        created = datetime.now(timezone.utc).isoformat()
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO voice_prints VALUES (?,?,?,?,?,?)",
                (person_id, owner, name, json.dumps([float(x) for x in vector]), seconds, created),
            )
        return VoicePrint(person_id=person_id, name=name, created_at=created, seconds=seconds)

    def list(self, owner: str) -> list[VoicePrint]:
        with self._db() as db:
            rows = db.execute(
                "SELECT person_id,name,created,seconds FROM voice_prints WHERE owner=? ORDER BY name",
                (owner,),
            ).fetchall()
        return [VoicePrint(person_id=r[0], name=r[1], created_at=r[2], seconds=r[3]) for r in rows]

    def vectors(self, owner: str) -> list[tuple[str, list[float]]]:
        with self._db() as db:
            rows = db.execute(
                "SELECT name,vector FROM voice_prints WHERE owner=?", (owner,)
            ).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    def delete(self, owner: str, person_id: str) -> bool:
        with self._db() as db:
            return db.execute(
                "DELETE FROM voice_prints WHERE owner=? AND person_id=?", (owner, person_id)
            ).rowcount > 0


def _extractor(model: Path, threads: int = 2):
    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=threads)
    )


def embed_samples(extractor, samples, sample_rate: int, max_seconds: float = MAX_EMBED_S):
    """One vector for one stretch of audio, truncated to what the model accepts.

    The cap is applied here, at the one place that talks to the extractor, so
    no caller can reach the ceiling described at MAX_EMBED_S by accident. Every
    caller already had a reason to believe its audio was short enough, and one
    of them was wrong for months.
    """
    limit = int(max_seconds * sample_rate)
    if len(samples) > limit:
        samples = samples[:limit]
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()
    return numpy.array(extractor.compute(stream), dtype=numpy.float32)


def enroll_from_wav(wav: Path, model: Path, threads: int = 2):
    """A vector for one person, from one 16 kHz mono sample.

    Raises ValueError when the sample is too short to produce a stable vector,
    which is the common mistake and worth saying plainly rather than storing
    something that will mislabel meetings later.
    """
    if not ENROLLMENT_AVAILABLE:
        raise RuntimeError(ENROLLMENT_MISSING)
    audio, sample_rate = soundfile.read(str(wav), dtype="float32", always_2d=True)
    seconds = len(audio) / sample_rate
    if seconds < MIN_SAMPLE_S:
        raise ValueError(
            f"That sample is {seconds:.1f} seconds. Record at least "
            f"{MIN_SAMPLE_S:.0f} seconds of ordinary speech."
        )
    if seconds > MAX_SAMPLE_S:
        audio = audio[: int(MAX_SAMPLE_S * sample_rate)]
        seconds = MAX_SAMPLE_S
    return embed_samples(_extractor(model, threads), audio[:, 0], sample_rate), seconds


def cluster_embeddings(
    wav: Path, turns, model: Path, threads: int = 2, samples=None, sample_rate: int = 16_000
) -> dict[int, list[float]]:
    """One vector per diarised speaker, from the longest speech they have.

    The longest turns are used rather than all of them because a vector built
    from somebody's two-second interjection is mostly noise, and averaging that
    into their profile makes the match worse rather than better.
    """
    if not ENROLLMENT_AVAILABLE or not turns:
        return {}
    if samples is None:
        audio, sample_rate = soundfile.read(str(wav), dtype="float32", always_2d=True)
        samples = audio[:, 0]
    channel = samples
    extractor = _extractor(model, threads)

    by_speaker: dict[int, list] = {}
    for turn in turns:
        by_speaker.setdefault(turn.speaker, []).append(turn)

    vectors: dict[int, list[float]] = {}
    for speaker, speaker_turns in by_speaker.items():
        longest = sorted(speaker_turns, key=lambda t: t.end_ms - t.start_ms, reverse=True)[:3]
        # Each turn is embedded on its own and the results averaged, rather
        # than the turns being glued together and embedded once. Two reasons.
        # It cannot reach the model's length ceiling, because no single call
        # carries more than one turn. And one turn that happens to be recorded
        # badly - somebody turning away from the microphone - pulls the average
        # rather than contaminating the whole vector.
        embedded = []
        for turn in longest:
            piece = channel[
                int(turn.start_ms / 1000 * sample_rate) : int(turn.end_ms / 1000 * sample_rate)
            ]
            if len(piece) <= sample_rate * 0.5:
                continue
            try:
                embedded.append(embed_samples(extractor, piece, sample_rate))
            except Exception as exc:  # pragma: no cover - model-level failure
                logger.warning("Could not embed one turn of speaker %s: %s", speaker, exc)
        if embedded:
            vectors[speaker] = [float(x) for x in centroid(embedded)]
        else:
            logger.warning("No usable audio to embed for speaker %s", speaker)
    return vectors


def rescore_cues(
    wav: Path,
    cues: list,
    enrolled: list[tuple[str, list[float]]],
    model: Path,
    threads: int = 2,
    margin: float = 0.08,
    samples=None,
    sample_rate: int = 16_000,
) -> dict[int, str]:
    """Second opinion on individual cues, from their own audio.

    Clustering assigns a whole stretch of speech to one speaker, so a cue that
    sits near a handover inherits whichever cluster won the overlap — which is
    where most misattribution comes from. A cue long enough to embed on its own
    can be checked directly against the enrolled voices.

    Returns only the cues to *change*, keyed by index, and only where the new
    name beats the current one by `margin`. Requiring a margin rather than a
    bare win matters: the two scores are computed from different amounts of
    audio, so near-ties are noise, and flipping on noise would trade a
    systematic error for a random one — which is harder to reason about and no
    more accurate.
    """
    if not ENROLLMENT_AVAILABLE or not enrolled or not cues:
        return {}
    if samples is None:
        try:
            audio, sample_rate = soundfile.read(str(wav), dtype="float32", always_2d=True)
            samples = audio[:, 0]
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not reopen %s to re-score cues: %s", wav, exc)
            return {}

    channel = samples
    extractor = _extractor(model, threads)
    references = dict(enrolled)

    changes: dict[int, str] = {}
    for index, cue in enumerate(cues):
        # Below about a second and a half the vector is mostly the start of a
        # word, and scoring it against anything is guesswork.
        if cue.end_ms - cue.start_ms < 1500:
            continue
        piece = channel[
            int(cue.start_ms / 1000 * sample_rate) : int(cue.end_ms / 1000 * sample_rate)
        ]
        if len(piece) < sample_rate:
            continue
        try:
            vector = embed_samples(extractor, piece, sample_rate)
        except Exception:  # pragma: no cover
            continue

        scores = {
            name: cosine(vector, reference) for name, reference in references.items()
        }
        if not scores:
            continue

        best = max(scores, key=scores.get)
        current = scores.get(cue.speaker, 0.0)
        if best != cue.speaker and scores[best] >= threshold() and scores[best] - current >= margin:
            changes[index] = best
    return changes


def identify(
    cluster_vectors: dict[int, list[float]],
    enrolled: list[tuple[str, list[float]]],
    minimum: float | None = None,
) -> dict[int, str]:
    """Name the clusters that clearly match somebody enrolled.

    A cluster matching nobody is left out, and keeps its "Speaker 2" label. Two
    clusters are never given the same name: if one recording produced two
    clusters for one person, naming both would claim a conversation happened
    between somebody and themselves, so the better match wins and the other
    stays a placeholder.
    """
    if not cluster_vectors or not enrolled:
        return {}
    minimum = threshold() if minimum is None else minimum

    scored: list[tuple[float, int, str]] = []
    for speaker, vector in cluster_vectors.items():
        for name, enrolled_vector in enrolled:
            similarity = cosine(vector, enrolled_vector)
            if similarity >= minimum:
                scored.append((similarity, speaker, name))

    scored.sort(reverse=True)
    names: dict[int, str] = {}
    taken: set[str] = set()
    for _similarity, speaker, name in scored:
        if speaker in names or name in taken:
            continue
        names[speaker] = name
        taken.add(name)
    return names
