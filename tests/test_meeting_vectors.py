"""The arithmetic underneath every speaker decision.

These are tiny functions, but two things depend on them being right: whether
two clusters are judged the same person, and whether a cluster is close enough
to an enrolled voice print to be given that person's name. A wrong name on a
transcript is worse than no name, so the edges are tested rather than assumed.
"""

from __future__ import annotations

from math import isclose

from bedrock.meetings.vectors import centroid, cosine, dot, norm

X = [1.0, 0.0, 0.0]
Y = [0.0, 1.0, 0.0]


def test_a_vector_is_identical_to_itself():
    assert isclose(cosine(X, X), 1.0)


def test_perpendicular_voices_score_zero():
    assert isclose(cosine(X, Y), 0.0)


def test_opposite_directions_score_minus_one():
    assert isclose(cosine(X, [-1.0, 0.0, 0.0]), -1.0)


def test_length_does_not_change_similarity():
    """Voice prints are not unit vectors; a louder sample must not read as a
    better match than a quiet one."""
    assert isclose(cosine(X, [7.0, 0.0, 0.0]), 1.0)


def test_silence_matches_nothing_rather_than_dividing_by_zero():
    """An all-zero vector means the extractor was handed silence. Scoring it at
    0 keeps it below every threshold, which is the honest answer."""
    assert cosine(X, [0.0, 0.0, 0.0]) == 0.0
    assert cosine([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0.0


def test_a_centroid_is_the_average_direction():
    assert [isclose(v, e) for v, e in zip(centroid([X, Y]), [0.5**0.5, 0.5**0.5, 0.0])] == [True] * 3


def test_a_centroid_comes_back_unit_length():
    """So that dot() of two centroids is their cosine similarity - which is
    what consolidate() compares against its merge threshold."""
    assert isclose(norm(centroid([[3.0, 4.0, 0.0], [0.0, 5.0, 0.0]])), 1.0)


def test_the_centroid_of_one_vector_is_that_vector_normalised():
    assert [isclose(v, e) for v, e in zip(centroid([[3.0, 4.0]]), [0.6, 0.8])] == [True, True]


def test_the_centroid_of_nothing_is_nothing():
    assert centroid([]) == []


def test_a_centroid_that_cancels_out_is_not_divided_by_zero():
    assert centroid([X, [-1.0, 0.0, 0.0]]) == [0.0, 0.0, 0.0]


def test_two_centroids_dotted_together_are_their_cosine():
    a, b = centroid([[2.0, 1.0, 0.0]]), centroid([[1.0, 2.0, 0.0]])
    assert isclose(dot(a, b), cosine([2.0, 1.0, 0.0], [1.0, 2.0, 0.0]))


def test_mismatched_lengths_truncate_instead_of_raising():
    """One extractor produces every vector in a meeting, so this should never
    happen. If it does, a slightly wrong score beats an IndexError thrown from
    inside a clustering loop, where the only symptom would be a transcript with
    no speaker names and nothing in the log to explain it."""
    assert dot([1.0, 1.0, 1.0], [1.0, 1.0]) == 2.0
    narrowed = centroid([[1.0, 1.0, 1.0], [1.0, 1.0]])
    assert [isclose(v, 0.5**0.5) for v in narrowed] == [True, True]
