"""Redaction, consent and access control on a recorded meeting.

The access tests matter most. Until this existed, anyone who could reach the
retrieval tool could read any meeting, including ones they were not in — in a
company whose meetings discuss patients and unpublished results.
"""

from __future__ import annotations

import httpx
import pytest

from app.compliance.rbac import RBACFilter, StaticPermissionSource
from app.meetings.governance import (
    Consent,
    capture_consent,
    groups_for,
    may_read,
    redact_record,
)
from app.tools.datacenter_client import (
    DatacenterConfig,
    DatacenterTranscriptClient,
    MeetingTranscript,
    transcripts_path,
)
from app.tools.meeting_transcripts import MeetingTranscriptsArgs, MeetingTranscriptsTool

SECRET = "test-secret"


def transcript(meeting_id: str, groups: list[str] | None = None) -> MeetingTranscript:
    return MeetingTranscript(
        meeting_id=meeting_id,
        topic=f"Meeting {meeting_id}",
        host_email="maya",
        start_time=1_790_000_000_000,
        end_time=1_790_000_600_000,
        transcript="WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nEnrollment is 148 of 200.\n",
        access_groups=groups or [],
    )


# --- Redaction ----------------------------------------------------------------


def test_identifiers_are_removed_from_the_transcript(monkeypatch):
    monkeypatch.setenv("MEETING_REDACT_PHI", "true")
    fields, summary = redact_record(
        {"transcript": "Patient MRN 4457821 called from 415-555-0134.", "decisions": []}
    )
    assert "4457821" not in fields["transcript"]
    assert "415-555-0134" not in fields["transcript"]
    assert summary.applied and {"MRN", "PHONE"} <= set(summary.findings)


def test_redaction_covers_everything_derived_from_the_transcript(monkeypatch):
    """A patient identifier taken out of the transcript but left in an action
    item has not been taken out."""
    monkeypatch.setenv("MEETING_REDACT_PHI", "true")
    fields, summary = redact_record(
        {
            "transcript": "no identifiers here",
            "summary": "Follow up with MRN 4457821.",
            "actions": ["Call 415-555-0134 about the consent form"],
            "findings": ["dob 01/02/1980 on the intake form"],
        }
    )
    assert "4457821" not in fields["summary"]
    assert "415-555-0134" not in fields["actions"][0]
    assert "01/02/1980" not in fields["findings"][0]
    assert summary.applied


def test_ordinary_meeting_content_is_untouched(monkeypatch):
    monkeypatch.setenv("MEETING_REDACT_PHI", "true")
    fields, summary = redact_record({"transcript": "Enrollment is 148 of 200 patients."})
    assert fields["transcript"] == "Enrollment is 148 of 200 patients."
    assert not summary.applied


def test_redaction_can_be_turned_off_deliberately(monkeypatch):
    """Off by configuration, never by omission."""
    monkeypatch.setenv("MEETING_REDACT_PHI", "false")
    fields, summary = redact_record({"transcript": "MRN 4457821"})
    assert "4457821" in fields["transcript"]
    assert not summary.applied


def test_non_string_list_items_survive_redaction(monkeypatch):
    monkeypatch.setenv("MEETING_REDACT_PHI", "true")
    fields, _ = redact_record({"actions": [{"text": "structured"}, "MRN 4457821"]})
    assert fields["actions"][0] == {"text": "structured"}
    assert "4457821" not in fields["actions"][1]


# --- Consent ------------------------------------------------------------------


def test_consent_records_who_said_it_and_when():
    consent = capture_consent({"id": "u1", "name": "Maya Chen"}, True, ["Luke", "Priya"])
    record = consent.to_record()
    assert record["confirmed"] is True
    assert record["by"] == "u1" and record["byName"] == "Maya Chen"
    assert record["participants"] == ["Luke", "Priya"]
    assert record["at"]


def test_consent_is_stored_as_a_claim_not_a_proof():
    """It is one employee's statement about a room. Storing it as anything
    stronger would misrepresent what we actually know."""
    record = capture_consent({"id": "u1", "name": "Maya"}, True).to_record()
    assert "not independently verified" in record["basis"]


def test_unconfirmed_consent_is_still_recorded():
    """That somebody declined to confirm is itself worth knowing later."""
    assert capture_consent({"id": "u1", "name": "Maya"}, False).to_record()["confirmed"] is False


def test_blank_participant_names_are_dropped():
    consent = capture_consent({"id": "u1", "name": "Maya"}, True, ["Luke", "  ", ""])
    assert consent.participants == ["Luke"]


# --- Access groups ------------------------------------------------------------


def test_a_meeting_in_a_project_belongs_to_that_project():
    assert groups_for("proj-42") == ["project:proj-42"]


def test_an_explicit_group_wins():
    assert groups_for("proj-42", ["clinical-ops"]) == ["clinical-ops"]


def test_a_meeting_with_no_groups_stays_readable_by_everyone():
    """Meetings recorded before access control existed must not vanish."""
    assert may_read({}, set())
    assert may_read({"accessGroups": []}, set())


def test_a_grouped_meeting_needs_a_matching_group():
    meeting = {"accessGroups": ["project:kras"]}
    assert may_read(meeting, {"project:kras"})
    assert not may_read(meeting, {"project:egfr"})


def test_general_is_readable_by_anyone():
    assert may_read({"accessGroups": ["general"]}, set())


# --- The tool actually withholds ----------------------------------------------


class _Client:
    def __init__(self, meetings):
        self.meetings = meetings
        self.groups_seen = None

    async def fetch_range(self, from_iso, to_iso, groups=None):
        self.groups_seen = groups
        return self.meetings


def _tool(meetings, grants=None):
    source = StaticPermissionSource()
    for user, group in (grants or []):
        source.grant(user, group)
    client = _Client(meetings)
    return MeetingTranscriptsTool(client=client, rbac=RBACFilter(source)), client


async def test_a_meeting_you_are_not_entitled_to_is_withheld():
    tool, _ = _tool([transcript("m1", ["project:kras"])])
    result = await tool.run(MeetingTranscriptsArgs(query="enrollment", user_id="outsider"))
    assert "not readable by you" in result.summary
    assert result.content == ""


async def test_a_meeting_in_your_group_comes_back():
    tool, _ = _tool([transcript("m1", ["project:kras"])], grants=[("maya", "project:kras")])
    result = await tool.run(MeetingTranscriptsArgs(query="enrollment", user_id="maya"))
    assert "148 of 200" in result.content


async def test_an_ungrouped_meeting_is_still_returned():
    tool, _ = _tool([transcript("m1")])
    result = await tool.run(MeetingTranscriptsArgs(query="enrollment", user_id="anyone"))
    assert "148 of 200" in result.content


async def test_without_an_identity_only_open_meetings_come_back():
    """Refusing to guess is the safe direction: the alternative is handing
    somebody a meeting they were not in because the caller forgot to say who
    it was acting for."""
    tool, _ = _tool([transcript("open"), transcript("closed", ["project:kras"])])
    result = await tool.run(MeetingTranscriptsArgs(query="enrollment"))
    assert "Meeting open" in result.citations[0]
    assert all("closed" not in citation for citation in result.citations)


async def test_the_groups_are_sent_to_the_connector_not_just_filtered_here():
    """Client-side filtering alone is not a boundary — by the time this process
    has the meeting, it has already been handed over."""
    tool, client = _tool([transcript("m1")], grants=[("maya", "project:kras")])
    await tool.run(MeetingTranscriptsArgs(query="enrollment", user_id="maya"))
    assert client.groups_seen and "project:kras" in client.groups_seen


# --- The groups are inside the signature --------------------------------------


def test_groups_are_part_of_the_signed_path():
    """An access claim that could be edited in flight would be no control."""
    without = transcripts_path("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z")
    with_groups = transcripts_path("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", ["clinical"])
    assert "groups=clinical" in with_groups
    assert with_groups != without


def test_a_request_without_groups_signs_exactly_as_it_always_did():
    """So a connector that predates access control keeps working."""
    assert transcripts_path("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", []) == (
        transcripts_path("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z")
    )


async def test_the_reference_connector_withholds_server_side(tmp_path, monkeypatch):
    """The half that matters: a connector that hands over a meeting the caller
    may not read has already leaked it."""
    import hashlib
    import json

    monkeypatch.setenv("DATACENTER_SIGNING_SECRET", SECRET)
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    from contrib.datacenter_connector.main import app as connector

    for meeting_id, groups in [("open", []), ("closed", ["project:kras"])]:
        record = {
            "meetingId": meeting_id,
            "topic": meeting_id,
            "hostEmail": "maya",
            "startTime": 1_790_000_000_000,
            "endTime": 1_790_000_600_000,
            "transcript": "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nHello.\n",
            "format": "vtt",
            "accessGroups": groups,
        }
        digest = hashlib.sha256(meeting_id.encode()).hexdigest()
        (tmp_path / f"{digest}.json").write_text(json.dumps(record))

    client = DatacenterTranscriptClient(
        config=DatacenterConfig(base_url="http://transcripts.internal", signing_secret=SECRET),
        transport=httpx.ASGITransport(app=connector),
    )
    outsider = await client.fetch_range(
        "2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z", ["project:egfr"]
    )
    assert [m.meeting_id for m in outsider] == ["open"]

    insider = await client.fetch_range(
        "2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z", ["project:kras"]
    )
    assert sorted(m.meeting_id for m in insider) == ["closed", "open"]
