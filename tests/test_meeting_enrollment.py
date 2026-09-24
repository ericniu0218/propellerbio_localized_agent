"""Voice prints: saving one, and using it to name a speaker.

The matching tests are mostly about refusing. A meeting that attributes a
decision to the wrong colleague is worse than one that attributes it to
nobody — the second is obviously incomplete, the first looks finished and is
wrong — so every rule here leans towards leaving a placeholder alone.
"""

from __future__ import annotations

from app.meetings.enrollment import VoicePrintStore, identify

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
