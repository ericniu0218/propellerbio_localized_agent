"""Filing a recorded meeting where the assistant already knows to look.

`app/tools/meeting_transcripts.py` answers "what did we decide on Tuesday?" by
asking the datacenter connector for a date range. It does not know or care
whether a transcript came from Zoom or from somebody's microphone — it takes
what the connector returns. So the cheapest way to make recorded meetings
answerable was to deliver them through the same contract endpoint the Zoom
relay uses, in the same JSON shape, rather than building a second retrieval
path and teaching the agent to consult both.

The meeting id is prefixed `recording:` so provenance survives. Everything
downstream treats it as an opaque string, but a person reading storage can see
at a glance which meetings Zoom captured and which this deployment recorded
itself — and the two have genuinely different properties, since only Zoom's
have per-speaker attribution.

Delivery failure is reported, never fatal. By the time this runs the meeting is
already saved in the workspace and readable; losing an hour of transcription
because the connector was briefly down would be the worse outcome.
"""

from __future__ import annotations

import logging
from datetime import datetime

from bedrock.server.meetings.pipeline import TranscriptionOutcome
from bedrock.server.tools.datacenter_client import (
    DatacenterError,
    DatacenterNotConfigured,
    DatacenterTranscriptClient,
    MeetingTranscript,
)

logger = logging.getLogger(__name__)

MEETING_ID_PREFIX = "recording:"


async def deliver_recording(
    record: dict,
    outcome: TranscriptionOutcome,
    started_at: datetime,
    owner: str,
) -> str | None:
    """Push one recorded meeting to the datacenter. Returns a note, or None.

    A returned string is shown to the person who made the recording, so it says
    what it means for them — whether the meeting can be asked about in chat —
    rather than quoting an HTTP status.
    """
    if outcome.is_empty:
        return None

    start_ms = int(started_at.timestamp() * 1000)
    transcript = MeetingTranscript(
        meeting_id=f"{MEETING_ID_PREFIX}{record['id']}",
        topic=record.get("title") or "Untitled meeting",
        host_email=owner,
        start_time=start_ms,
        end_time=start_ms + max(outcome.duration_ms, 0),
        # The stored transcript, not `outcome.vtt` — the stored one has been
        # through redaction. Sending the raw one would put the identifiers back
        # into the datacenter that were just taken out of the workspace, which
        # is the failure that makes redaction theatre rather than a control.
        transcript=record.get("transcript") or outcome.vtt,
        format="vtt",
        access_groups=list(record.get("accessGroups") or []),
    )

    try:
        client = DatacenterTranscriptClient()
    except DatacenterNotConfigured:
        # An ordinary state, not an error: a deployment can run the workspace
        # without a datacenter connector at all. Say plainly what that costs.
        return (
            "Saved to this workspace only. No datacenter connector is "
            "configured (DATACENTER_BASE_URL), so the assistant cannot yet "
            "search this meeting by date in chat."
        )
    except DatacenterError as exc:
        logger.warning("Datacenter misconfigured, meeting kept locally: %s", exc)
        return f"Saved to this workspace only — the datacenter is misconfigured: {exc}"

    try:
        await client.store(transcript)
    except DatacenterError as exc:
        logger.warning("Delivering meeting %s failed: %s", transcript.meeting_id, exc)
        return (
            "Saved to this workspace, but it could not be sent to the "
            f"datacenter, so date-based search in chat will not find it yet: {exc}"
        )

    return None
