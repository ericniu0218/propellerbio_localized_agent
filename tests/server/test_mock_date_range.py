from bedrock.server.models.mock_fixtures import fake_reply
import json


def _resolve(request_text: str, reference_iso: str) -> dict:
    prompt = (
        f"The current date and time is {reference_iso}. Extract the "
        "date/time range this request refers to. Respond with JSON: "
        '{"from": "<ISO 8601>", "to": "<ISO 8601>"}.\n\n'
        f"Request: {request_text}"
    )
    return json.loads(fake_reply(prompt, "mock-model"))


def test_yesterday_resolves_to_the_full_previous_day():
    result = _resolve("what happened yesterday?", "2026-01-15T09:00:00+00:00")
    assert result["from"].startswith("2026-01-14T00:00:00")
    assert result["to"].startswith("2026-01-15T00:00:00")


def test_today_resolves_from_midnight_to_reference_time():
    result = _resolve("what happened today?", "2026-01-15T09:00:00+00:00")
    assert result["from"].startswith("2026-01-15T00:00:00")
    assert result["to"] == "2026-01-15T09:00:00+00:00"


def test_last_week_resolves_to_the_full_prior_calendar_week():
    # 2026-01-15 is a Thursday; this week started Monday 2026-01-12.
    result = _resolve("what happened last week?", "2026-01-15T09:00:00+00:00")
    assert result["from"].startswith("2026-01-05T00:00:00")  # prior Monday
    assert result["to"].startswith("2026-01-12T00:00:00")  # this week's Monday


def test_named_weekday_in_the_past_resolves_to_that_single_day():
    # Reference is Thursday 2026-01-15; "last Tuesday" is 2026-01-13.
    result = _resolve("what happened last Tuesday?", "2026-01-15T09:00:00+00:00")
    assert result["from"].startswith("2026-01-13T00:00:00")
    assert result["to"].startswith("2026-01-14T00:00:00")


def test_unrecognized_phrasing_falls_back_to_a_bounded_week_not_nothing():
    result = _resolve("what happened during the thing?", "2026-01-15T09:00:00+00:00")
    assert result["from"].startswith("2026-01-08T09:00:00")
    assert result["to"] == "2026-01-15T09:00:00+00:00"
