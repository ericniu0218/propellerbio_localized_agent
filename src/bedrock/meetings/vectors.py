"""Cosine similarity over voice prints, in plain Python.

Every speaker decision in this package ends in the same question: do these two
vectors point the same way? Clusters merge when their average voices are close
enough (`diarize.consolidate`), and a cluster gets a name when it is close
enough to somebody enrolled (`enrollment.identify`).

This used to be numpy, which put the *arithmetic* behind the same optional
dependency as the *models*. Those are not the same thing. Running diarization
needs sherpa-onnx and a few hundred megabytes of ONNX weights, and a machine
without them genuinely cannot identify speakers. Averaging a list of floats
needs nothing, and a machine without numpy could still do every bit of it — but
did not, because the import sat one guard too high. CI, which installs without
the speaker extra, could not run these functions at all.

The numbers here are small enough that it does not matter: a voice print is 192
floats, and a meeting has tens of clusters, not thousands. The model that
produced the vectors costs orders of magnitude more than comparing them.
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence

Vector = Sequence[float]


def dot(a: Vector, b: Vector) -> float:
    """Dot product, over the shorter of the two if they disagree in length.

    They should never disagree — one extractor produces every vector in a
    meeting — but truncating is better than a stray IndexError deep inside
    clustering, where the only visible symptom would be a transcript with no
    speaker names and nothing in the log to say why.
    """
    return float(sum(x * y for x, y in zip(a, b)))


def norm(vector: Vector) -> float:
    return sqrt(float(sum(x * x for x in vector)))


def cosine(a: Vector, b: Vector) -> float:
    """How alike two voices are, from -1 to 1. Zero when either has no length.

    A zero vector means the extractor was handed silence. It matches nothing,
    which is the honest answer: refusing to name a speaker is always available,
    and a wrong name on a transcript is worse than no name.
    """
    denominator = norm(a) * norm(b)
    return dot(a, b) / denominator if denominator else 0.0


def centroid(vectors: Iterable[Vector]) -> list[float]:
    """The average voice of a group, normalised to unit length.

    Normalised so that `dot()` of two centroids *is* their cosine similarity,
    which is what the merge loop in `consolidate` compares against its
    threshold. An all-zero average is returned as-is rather than divided by
    zero; `cosine` then scores it against everything at 0.
    """
    rows = [list(v) for v in vectors]
    if not rows:
        return []
    width = min(len(row) for row in rows)
    averaged = [sum(row[i] for row in rows) / len(rows) for i in range(width)]
    length = norm(averaged)
    return [x / length for x in averaged] if length else averaged
