"""Harvesting the vocabulary from the company's own words.

Most of these are about what must NOT become a term. A vocabulary padded with
common words biases the decoder towards them, which makes transcripts worse —
and the failure is silent, which is what makes it worth guarding carefully.
"""

from __future__ import annotations

from bedrock.server.meetings.vocabulary_source import from_meetings, harvest, suggest

TRANSCRIPT = (
    "Good morning. Enrollment for the KRAS trial is 148 of 200. "
    "The G12C cohort showed a partial response. Because of that, we continue. "
    "Priya will update the protocol. Monday is the deadline. "
    "KRAS again, and the G12C readout, and Priya owns it."
)


def terms(text: str, minimum: int = 2) -> list[str]:
    return [t for t, _ in harvest([text], minimum=minimum)]


def test_gene_symbols_are_found():
    assert "KRAS" in terms(TRANSCRIPT)


def test_alphanumeric_codes_are_found():
    assert "G12C" in terms(TRANSCRIPT)


def test_a_colleagues_name_is_found():
    assert "Priya" in terms(TRANSCRIPT)


def test_a_word_that_only_ever_starts_a_sentence_is_not_a_term():
    """Otherwise every common word that opens a sentence becomes vocabulary."""
    assert "Because" not in terms(TRANSCRIPT)


def test_calendar_noise_is_not_a_term():
    """A vocabulary containing "Monday" steers the decoder for no benefit."""
    assert "Monday" not in terms(TRANSCRIPT)


def test_a_term_seen_once_is_not_enough():
    assert terms("The BRD-1192 compound was dosed.", minimum=2) == []


def test_a_term_seen_twice_qualifies():
    assert "BRD-1192" in terms("We dosed BRD-1192. Then BRD-1192 again.", minimum=2)


def test_ordinary_lowercase_words_are_never_terms():
    assert terms("the enrollment was behind the target and the target moved") == []


def test_frequency_orders_the_result():
    found = harvest(["KRAS KRAS KRAS EGFR EGFR"], minimum=2)
    assert [t for t, _ in found][:2] == ["KRAS", "EGFR"]


def test_meetings_are_harvested_from_transcript_and_summary():
    found = from_meetings([{"transcript": TRANSCRIPT, "summary": "KRAS and G12C again."}])
    assert "KRAS" in found and "G12C" in found


def test_a_meeting_with_no_transcript_is_harmless():
    assert from_meetings([{"title": "empty"}]) == []


def test_configured_terms_come_first_and_are_never_dropped(monkeypatch):
    """Whisper weights the start of its prompt most heavily, and somebody typed
    the configured list on purpose."""
    monkeypatch.setenv("MEETING_VOCABULARY", "sotorasib, adagrasib")
    result = suggest([{"transcript": TRANSCRIPT}])
    assert result[:2] == ["sotorasib", "adagrasib"]
    assert "KRAS" in result


def test_a_configured_term_is_not_suggested_twice(monkeypatch):
    monkeypatch.setenv("MEETING_VOCABULARY", "KRAS")
    assert suggest([{"transcript": TRANSCRIPT}]).count("KRAS") == 1
