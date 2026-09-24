import json

import pytest

from bedrock.json_reply import decode_json_reply


def test_strict_json_reply_accepts_plain_and_fenced_json():
    expected = {"steps": []}

    assert decode_json_reply('{"steps": []}') == expected
    assert decode_json_reply('```json\n{"steps": []}\n```') == expected


def test_strict_json_reply_rejects_surrounding_prose():
    with pytest.raises(json.JSONDecodeError):
        decode_json_reply('Here is the result: {"steps": []}')


def test_permissive_json_reply_preserves_meeting_prose_and_fence_support():
    expected = {"decisions": []}

    assert decode_json_reply(
        'Here is the result: {"decisions": []} done.',
        allow_surrounding_text=True,
    ) == expected
    assert decode_json_reply(
        'Result:\n```json\n{"decisions": []}\n```',
        allow_surrounding_text=True,
    ) == expected


def test_permissive_json_reply_reports_a_missing_object():
    with pytest.raises(ValueError, match="no JSON object"):
        decode_json_reply("nothing structured here", allow_surrounding_text=True)
