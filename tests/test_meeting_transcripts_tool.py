import hashlib
import hmac

import httpx
import pytest

from app.tools.meeting_transcripts import (
    DatacenterConnectorError,
    MeetingTranscriptSearchArgs,
    MeetingTranscriptSearchTool,
)

SECRET = "shared-secret"
BASE_URL = "http://transcript-store.internal:8080"

MEETING = {
    "meetingId": "meeting-1",
    "topic": "Weekly sync",
    "hostEmail": "host@example.com",
    "startTime": 1_700_000_000_000,
    "endTime": 1_700_000_600_000,
    "transcript": "WEBVTT\n\nhello",
    "format": "vtt",
}


def _signature_valid(request: httpx.Request, secret: str) -> bool:
    timestamp = request.headers["X-Vault-Timestamp"]
    path_with_query = request.url.raw_path.decode()
    message = f"{timestamp}.GET.{path_with_query}.".encode()
    expected = f"v1={hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()}"
    return request.headers["X-Vault-Signature"] == expected


def _transport(meetings: list[dict], expected_secret: str = SECRET) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if not _signature_valid(request, expected_secret):
            return httpx.Response(401, json={"error": "bad signature"})
        assert request.url.path == "/transcripts"
        return httpx.Response(200, json={"transcripts": meetings})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_returns_matching_transcripts_with_citations():
    tool = MeetingTranscriptSearchTool(
        base_url=BASE_URL, signing_secret=SECRET, transport=_transport([MEETING])
    )
    result = await tool.run(
        MeetingTranscriptSearchArgs(
            from_time="2026-01-01T00:00:00Z", to_time="2026-01-02T00:00:00Z", user_id="alice"
        )
    )
    assert result.citations == ["meeting:meeting-1"]
    assert "Weekly sync" in result.content
    assert "hello" in result.content
    assert "1 meeting transcript" in result.summary


@pytest.mark.asyncio
async def test_no_matches_returns_empty_result_not_an_error():
    tool = MeetingTranscriptSearchTool(base_url=BASE_URL, signing_secret=SECRET, transport=_transport([]))
    result = await tool.run(
        MeetingTranscriptSearchArgs(
            from_time="2026-01-01T00:00:00Z", to_time="2026-01-02T00:00:00Z", user_id="alice"
        )
    )
    assert result.citations == []
    assert "No meeting transcripts" in result.summary


@pytest.mark.asyncio
async def test_caps_results_at_the_configured_max_meetings():
    many_meetings = [{**MEETING, "meetingId": f"meeting-{i}"} for i in range(25)]
    tool = MeetingTranscriptSearchTool(
        base_url=BASE_URL, signing_secret=SECRET, transport=_transport(many_meetings)
    )
    result = await tool.run(
        MeetingTranscriptSearchArgs(
            from_time="2026-01-01T00:00:00Z", to_time="2026-01-02T00:00:00Z", user_id="alice"
        )
    )
    assert len(result.citations) == 10  # config/budgets.yaml meeting_transcripts.max_meetings


@pytest.mark.asyncio
async def test_reports_unconfigured_instead_of_failing_when_env_is_missing():
    tool = MeetingTranscriptSearchTool(base_url="", signing_secret="")
    result = await tool.run(
        MeetingTranscriptSearchArgs(
            from_time="2026-01-01T00:00:00Z", to_time="2026-01-02T00:00:00Z", user_id="alice"
        )
    )
    assert "not configured" in result.summary
    assert result.citations == []


@pytest.mark.asyncio
async def test_wrong_secret_is_rejected_by_the_receiver():
    tool = MeetingTranscriptSearchTool(
        base_url=BASE_URL,
        signing_secret="wrong-secret",
        transport=_transport([MEETING], expected_secret=SECRET),
    )
    with pytest.raises(DatacenterConnectorError):
        await tool.run(
            MeetingTranscriptSearchArgs(
                from_time="2026-01-01T00:00:00Z", to_time="2026-01-02T00:00:00Z", user_id="alice"
            )
        )


@pytest.mark.asyncio
async def test_never_reaches_out_when_the_transport_would_be_a_real_network_call():
    """Guards the air-gap property directly: if this tool ever gets rewritten
    to call something other than the injected transport (e.g. a real
    httpx.AsyncHTTPTransport hitting the public internet), this test's
    MockTransport-only setup should be the only thing that makes network
    calls possible in the test suite at all."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"transcripts": []})

    tool = MeetingTranscriptSearchTool(
        base_url=BASE_URL, signing_secret=SECRET, transport=httpx.MockTransport(handler)
    )
    await tool.run(
        MeetingTranscriptSearchArgs(
            from_time="2026-01-01T00:00:00Z", to_time="2026-01-02T00:00:00Z", user_id="alice"
        )
    )
    assert call_count == 1
