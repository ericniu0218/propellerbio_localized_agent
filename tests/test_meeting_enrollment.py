"""Voice prints: saving one, and using it to name a speaker.

The matching tests are mostly about refusing. A meeting that attributes a
decision to the wrong colleague is worse than one that attributes it to
nobody — the second is obviously incomplete, the first looks finished and is
wrong — so every rule here leans towards leaving a placeholder alone.
"""

from __future__ import annotations

from bedrock.meetings.enrollment import VoicePrintStore, identify

# Three vectors: two nearly identical, one clearly different. Cosine similarity
# is all `identify` uses, so plain numbers are enough to pin the behaviour.
ALICE = [1.0, 0.0, 0.0]
ALICE_AGAIN = [0.98, 0.05, 0.0]
BOB = [0.0, 1.0, 0.0]


# --- The store ----------------------------------------------------------------


def test_a_voice_print_round_trips(tmp_path):
    store = VoicePrintStore(tmp_path / "v.sqlite")
    saved = store.save("maya", "p1", "Priya", ALICE, 10.0)
    assert saved.name == "Priya"
    assert [v.name for v in store.list("maya")] == ["Priya"]
    assert store.vectors("maya") == [("Priya", ALICE)]


def test_the_vector_is_never_exposed_to_the_browser(tmp_path):
    """A voice print is personal data. The page needs the name, not the maths."""
    store = VoicePrintStore(tmp_path / "v.sqlite")
    public = store.save("maya", "p1", "Priya", ALICE, 10.0).public()
    assert set(public) == {"id", "name", "createdAt", "seconds"}


def test_voice_prints_are_scoped_to_their_owner(tmp_path):
    store = VoicePrintStore(tmp_path / "v.sqlite")
    store.save("maya", "p1", "Priya", ALICE, 10.0)
    assert store.list("someone-else") == []
    assert store.vectors("someone-else") == []


def test_re_enrolling_replaces_rather_than_duplicates(tmp_path):
    store = VoicePrintStore(tmp_path / "v.sqlite")
    store.save("maya", "p1", "Priya", ALICE, 10.0)
    store.save("maya", "p1", "Priya", BOB, 12.0)
    assert len(store.list("maya")) == 1
    assert store.vectors("maya") == [("Priya", BOB)]


def test_deleting_a_voice_print_reports_whether_it_existed(tmp_path):
    store = VoicePrintStore(tmp_path / "v.sqlite")
    store.save("maya", "p1", "Priya", ALICE, 10.0)
    assert store.delete("maya", "p1") is True
    assert store.delete("maya", "p1") is False


def test_one_person_cannot_delete_anothers_voice_print(tmp_path):
    store = VoicePrintStore(tmp_path / "v.sqlite")
    store.save("maya", "p1", "Priya", ALICE, 10.0)
    assert store.delete("intruder", "p1") is False
    assert len(store.list("maya")) == 1


# --- Matching -----------------------------------------------------------------


def test_a_clear_match_gets_the_name():
    assert identify({0: ALICE_AGAIN}, [("Priya", ALICE)], minimum=0.55) == {0: "Priya"}


def test_a_speaker_matching_nobody_keeps_its_placeholder():
    assert identify({0: BOB}, [("Priya", ALICE)], minimum=0.55) == {}


def test_the_best_match_wins_when_two_names_are_close():
    names = identify({0: ALICE_AGAIN}, [("Priya", ALICE), ("Ravi", [0.7, 0.7, 0.0])], minimum=0.5)
    assert names == {0: "Priya"}


def test_one_person_is_never_given_to_two_speakers():
    """If a recording split one person into two clusters, naming both would
    claim a conversation happened between somebody and themselves."""
    names = identify({0: ALICE, 1: ALICE_AGAIN}, [("Priya", ALICE)], minimum=0.5)
    assert list(names.values()) == ["Priya"]
    assert len(names) == 1


def test_nothing_enrolled_means_nothing_named():
    assert identify({0: ALICE}, [], minimum=0.5) == {}


def test_no_speakers_means_nothing_named():
    assert identify({}, [("Priya", ALICE)], minimum=0.5) == {}


def test_a_zero_vector_cannot_match_anything():
    """Guards the division rather than returning a similarity of nan."""
    assert identify({0: [0.0, 0.0, 0.0]}, [("Priya", ALICE)], minimum=0.1) == {}


def test_the_threshold_is_what_decides():
    assert identify({0: [0.7, 0.7, 0.0]}, [("Priya", ALICE)], minimum=0.9) == {}
    assert identify({0: [0.7, 0.7, 0.0]}, [("Priya", ALICE)], minimum=0.5) == {0: "Priya"}


# --- Re-scoring individual cues ----------------------------------------------


class FakeCue:
    def __init__(self, start_ms, end_ms, speaker):
        self.start_ms, self.end_ms, self.speaker = start_ms, end_ms, speaker


def test_nothing_is_rescored_without_enrolled_voices(tmp_path):
    from bedrock.meetings.enrollment import rescore_cues

    assert rescore_cues(tmp_path / "x.wav", [FakeCue(0, 5000, "Priya")], [], tmp_path / "m.onnx") == {}


def test_nothing_is_rescored_without_cues(tmp_path):
    from bedrock.meetings.enrollment import rescore_cues

    assert rescore_cues(tmp_path / "x.wav", [], [("Priya", ALICE)], tmp_path / "m.onnx") == {}


def test_a_missing_file_is_not_an_error(tmp_path):
    """A meeting is not worth losing because the working copy was cleaned up."""
    from bedrock.meetings.enrollment import rescore_cues

    assert rescore_cues(
        tmp_path / "gone.wav", [FakeCue(0, 5000, "Priya")], [("Priya", ALICE)], tmp_path / "m.onnx"
    ) == {}


# --- The model's length ceiling ----------------------------------------------
#
# These use a fake extractor rather than the real ONNX model, so they run on a
# machine with no models installed and in CI. What they pin is the arithmetic
# that decides how much audio reaches the model, which is exactly where the bug
# was: the model itself behaved correctly, by raising.


class FakeExtractor:
    """Records how much audio each call was given, and refuses too much.

    The ceiling is real: TitaNet's ONNX export is built for 12,288 frames at a
    10ms hop and raises above that, which is what shipped as "voice enrolment
    does not work".
    """

    CEILING_SAMPLES = 12_288 * 160  # frames x hop, in samples at 16 kHz

    def __init__(self):
        self.lengths: list[int] = []

    def create_stream(self):
        return self

    def accept_waveform(self, sample_rate, samples):
        self.lengths.append(len(samples))
        self._last = len(samples)

    def input_finished(self):
        pass

    def compute(self, stream):
        if self._last > self.CEILING_SAMPLES:
            raise RuntimeError(
                "Attempting to broadcast an axis by a dimension other than 1. "
                f"12288 by {self._last // 160}"
            )
        return [1.0, 0.0, 0.0]


class FakeTurn:
    def __init__(self, start_ms, end_ms, speaker):
        self.start_ms, self.end_ms, self.speaker = start_ms, end_ms, speaker


def test_audio_longer_than_the_model_accepts_is_truncated():
    from bedrock.meetings.enrollment import MAX_EMBED_S, embed_samples

    extractor = FakeExtractor()
    embed_samples(extractor, [0.0] * (16_000 * 300), 16_000)  # five minutes
    assert extractor.lengths == [int(MAX_EMBED_S * 16_000)]


def test_short_audio_is_passed_through_whole():
    from bedrock.meetings.enrollment import embed_samples

    extractor = FakeExtractor()
    embed_samples(extractor, [0.0] * 16_000, 16_000)
    assert extractor.lengths == [16_000]


def test_the_speaker_who_talked_most_still_gets_a_vector(monkeypatch):
    """The regression. Three turns of a minute each totalled 180 seconds, went
    to the model in one call, raised, and were swallowed by a warning - so the
    person who spoke most in the meeting was the one who could never be given
    their enrolled name. It only ever worked on short test clips."""
    import bedrock.meetings.enrollment as enrollment

    extractor = FakeExtractor()
    monkeypatch.setattr(enrollment, "ENROLLMENT_AVAILABLE", True)
    monkeypatch.setattr(enrollment, "_extractor", lambda model, threads: extractor)

    turns = [FakeTurn(i * 60_000, (i + 1) * 60_000, 0) for i in range(3)]
    vectors = enrollment.cluster_embeddings(
        "ignored.wav", turns, "model.onnx", samples=[0.0] * (16_000 * 180)
    )

    assert 0 in vectors, "the dominant speaker must get a vector"
    assert len(extractor.lengths) == 3, "each turn is embedded on its own"
    assert max(extractor.lengths) <= FakeExtractor.CEILING_SAMPLES


def test_one_unusable_turn_does_not_cost_the_speaker_their_vector(monkeypatch):
    """Embedding per turn means a turn the model chokes on pulls the average
    instead of destroying the whole profile."""
    import bedrock.meetings.enrollment as enrollment

    class OneBadTurn(FakeExtractor):
        def compute(self, stream):
            if len(self.lengths) == 2:  # the second turn only
                raise RuntimeError("model said no")
            return [1.0, 0.0, 0.0]

    extractor = OneBadTurn()
    monkeypatch.setattr(enrollment, "ENROLLMENT_AVAILABLE", True)
    monkeypatch.setattr(enrollment, "_extractor", lambda model, threads: extractor)

    turns = [FakeTurn(i * 10_000, (i + 1) * 10_000, 0) for i in range(3)]
    vectors = enrollment.cluster_embeddings(
        "ignored.wav", turns, "model.onnx", samples=[0.0] * (16_000 * 30)
    )
    assert 0 in vectors
