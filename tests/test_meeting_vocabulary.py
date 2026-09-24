"""Repairing words whisper spelled phonetically.

Most of these tests are about *not* changing things. Rewriting words inside a
record of what somebody said is the kind of help that can do real damage, so
the guards matter more than the corrections do.
"""

from __future__ import annotations

from app.meetings.vocabulary import correct, correct_texts, parse_vocabulary

VOCAB = "KRAS, EGFR, G12C, Priya, Ravi, sotorasib, cohort, interim analysis"


# --- The corrections it should make --------------------------------------------


def test_the_mistake_seen_in_a_real_run_is_repaired():
    """Whisper wrote "CRAS" for "KRAS" on one run of the test audio, one edit
    away, which is exactly what this is for."""
    texts, report = correct_texts(
        ["let's start with the enrollment numbers for the CRAS trial"], VOCAB
    )
    assert "KRAS trial" in texts[0]
    assert report.total == 1


def test_the_repair_uses_the_configured_spelling_not_the_heard_casing():
    texts, _ = correct_texts(["the kras trial"], "KRAS")
    # Already an exact case-insensitive match, so it is left alone rather than
    # re-cased — this repairs misspellings, it is not a style pass.
    assert texts == ["the kras trial"]


def test_a_repair_is_reported_so_a_reader_knows_the_transcript_was_edited():
    _, report = correct_texts(["the CRAS trial and the CRAS cohort"], VOCAB)
    described = report.describe()
    assert "CRAS→KRAS" in described and "×2" in described


def test_it_works_on_a_whole_vtt_without_touching_the_timings():
    vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nThe CRAS trial is enrolling.\n"
    out, report = correct(vtt, VOCAB)
    assert "00:00:01.000 --> 00:00:04.000" in out
    assert "KRAS trial" in out
    assert report.total == 1


# --- The corrections it must NOT make ------------------------------------------


def test_an_inflection_is_left_alone():
    """With "cohort" configured, "cohorts" is right as it stands. Replacing it
    would corrupt the sentence."""
    texts, report = correct_texts(["we split the cohorts evenly"], VOCAB)
    assert texts == ["we split the cohorts evenly"]
    assert report.total == 0


def test_an_unrelated_word_is_left_alone():
    texts, report = correct_texts(["the grass was wet and the crowd was large"], VOCAB)
    assert report.total == 0, report.describe()
    assert texts == ["the grass was wet and the crowd was large"]


def test_a_multi_word_term_is_never_used_for_repair():
    """Deciding where a phrase begins in a stream of words invites exactly the
    mistake these guards exist to prevent."""
    assert "interim analysis" not in parse_vocabulary(VOCAB)


def test_a_distant_word_is_not_dragged_into_a_term():
    """"Boston" must not become "KRAS" no matter how the scores land."""
    texts, _ = correct_texts(["the site coordinator in Boston left"], VOCAB)
    assert "Boston" in texts[0]


def test_very_short_tokens_are_never_touched():
    """Two-letter words rhyme with everything."""
    texts, report = correct_texts(["we go to it"], "Ravi, KRAS")
    assert report.total == 0
    assert texts == ["we go to it"]


def test_no_vocabulary_means_no_changes_at_all():
    texts, report = correct_texts(["the CRAS trial"], None)
    assert texts == ["the CRAS trial"]
    assert report.total == 0
    assert report.describe() == ""


def test_an_exact_match_is_not_counted_as_a_repair():
    _, report = correct_texts(["the KRAS trial and EGFR screen"], VOCAB)
    assert report.total == 0


# --- Parsing the setting -------------------------------------------------------


def test_vocabulary_accepts_commas_semicolons_and_newlines():
    assert parse_vocabulary("KRAS; EGFR\nPriya, Ravi") == ["KRAS", "EGFR", "Priya", "Ravi"]


def test_blank_and_tiny_entries_are_dropped():
    assert parse_vocabulary("KRAS, , a, ok") == ["KRAS"]


# --- Shape is what separates a fumbled name from a real word -------------------


def test_a_capitalised_word_opening_a_sentence_gets_no_extra_latitude():
    """Every sentence starts with a capital, so it is no evidence at all.
    "Rain" is two edits from "Ravi" and must survive."""
    texts, report = correct_texts(["Rain delayed the site visit"], "Ravi, KRAS")
    assert texts == ["Rain delayed the site visit"], report.describe()


def test_a_lowercase_real_word_is_held_to_the_strict_rate():
    """This is the "grass"/"KRAS" case: two edits apart, 67% similar."""
    texts, _ = correct_texts(["the grass was wet"], "KRAS")
    assert texts == ["the grass was wet"]


def test_an_all_capitals_fumble_is_repaired_at_the_looser_rate():
    """"SOTORASIB" misheard as "SOTORASIC" — one edit, and shaped like a name,
    so the similarity rate that protects ordinary words does not block it."""
    texts, _ = correct_texts(["SOTORASIC was dosed"], "sotorasib")
    assert "sotorasib was dosed" == texts[0]


def test_a_two_edit_miss_on_a_short_name_is_deliberately_left_alone():
    """"PREA" really is whisper fumbling "Priya", and this does not fix it.
    Allowing two edits on a five-letter name also turns "Rain" into "Ravi",
    and putting a colleague's name into a sentence they are not in is worse
    than leaving a misspelling for a human to notice. The decoder hint is what
    gets this one right most of the time."""
    texts, report = correct_texts(["PREA owns it"], "Priya")
    assert texts == ["PREA owns it"]
    assert report.total == 0


def test_a_token_with_digits_is_treated_as_a_name():
    texts, _ = correct_texts(["the G12D cohort"], "G12C")
    assert "G12C" in texts[0]


def test_the_looser_rate_still_will_not_reach_a_distant_term():
    """Latitude on similarity, never on edit distance."""
    texts, _ = correct_texts(["BOSTON is the site"], "Priya, KRAS, sotorasib")
    assert "BOSTON" in texts[0]


# --- Common words are never a mis-hearing -------------------------------------


def test_a_common_word_is_never_repaired_into_a_name():
    """The one that got through and corrupted a real transcript: "Luke" is a
    single edit and 75% similar to "like", and three spoken "like"s became
    "Luke" — in a meeting that really did have a Luke in it, so the result read
    as plausible and was wrong."""
    texts, report = correct_texts(["I was like, let's do a fake meeting"], "Luke, KRAS")
    assert texts == ["I was like, let's do a fake meeting"]
    assert report.total == 0


def test_the_name_itself_still_survives_untouched():
    texts, _ = correct_texts(["Hey Luke, come here"], "Luke")
    assert "Luke" in texts[0]


def test_capitals_beat_the_common_word_guard():
    """Whisper writes an unfamiliar name in capitals, and "LUKE" mid-sentence
    is not the adverb."""
    texts, _ = correct_texts(["and then LUKI said no"], "Luke")
    assert "Luke said no" in texts[0]


def test_other_common_short_words_near_names_are_safe():
    for word, term in [("well", "Will"), ("man", "Max"), ("name", "Dane"), ("class", "Clara")]:
        texts, _ = correct_texts([f"the {word} was there"], term)
        assert word in texts[0], f"{word} was corrupted into {term}"
