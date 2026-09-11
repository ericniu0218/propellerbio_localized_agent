"""Search Zoom meeting transcripts held in the company's own datacenter.

Meeting audio is captured and transcribed via Zoom, then pushed into this
company's on-prem storage by a separate internet-facing relay
(zoom-transcript-vault) the moment Zoom produces it — never stored there,
never touched by this backend except at query time.

This tool calls the datacenter's own storage *directly*, over the internal
network, using the same signed HTTP contract zoom-transcript-vault pushes
transcripts through (see that repo's README, "Datacenter connector
contract", endpoint 3). It deliberately never calls zoom-transcript-vault
itself: that service lives on the public internet, and this system is
air-gapped (docs/ARCHITECTURE.md section 1, `app/compliance/airgap_check.py`)
— any outbound call to the internet at runtime, including one routed
through a "relay", is exactly the call the air gap exists to make
impossible. If this ever needs to change, that's a constraint change to
raise explicitly, not something to quietly work around here.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import ClassVar
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

from app.config import load_budgets_config
from app.tools.base import Tool, ToolResult

_SIGNATURE_VERSION = "v1"


class MeetingTranscriptSearchArgs(BaseModel):
    from_time: str  # ISO 8601, e.g. "2026-01-02T00:00:00Z"
    to_time: str  # ISO 8601
    user_id: str


class DatacenterConnectorError(RuntimeError):
    """Raised when the datacenter connector is unreachable or returns an
    error. Deliberately not swallowed into a ToolResult — this should
    surface as a tool failure the orchestrator can retry/report, not a
    quiet "no results found."."""


def _sign(secret: str, method: str, path_with_query: str, body: str) -> tuple[str, str]:
    """Matches zoom-transcript-vault's src/connector.ts signedHeaders exactly:
    HMAC-SHA256(secret, f"{timestamp}.{method}.{path_with_query}.{body}"),
    binding the method and path into the signature so one signed request
    can't be replayed against a different meeting id or date range."""
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}.{method}.{path_with_query}.{body}".encode()
    signature = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return timestamp, f"{_SIGNATURE_VERSION}={signature}"


class MeetingTranscriptSearchTool(Tool):
    name: ClassVar[str] = "meeting_transcripts"
    description: ClassVar[str] = (
        "Search company Zoom meeting transcripts by date/time range "
        "(e.g. 'what happened last Tuesday afternoon'). Returns each "
        "matching meeting's transcript with a citation (meeting id, topic, "
        "start time). Queries the datacenter's own storage directly over "
        "the internal network — never the public internet."
    )
    args_model: ClassVar[type[BaseModel]] = MeetingTranscriptSearchArgs

    def __init__(
        self,
        base_url: str | None = None,
        signing_secret: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("DATACENTER_BASE_URL", "")).rstrip("/")
        self._signing_secret = signing_secret or os.environ.get("DATACENTER_SIGNING_SECRET", "")
        self._transport = transport

    async def run(self, args: MeetingTranscriptSearchArgs) -> ToolResult:
        if not self._base_url or not self._signing_secret:
            return ToolResult(
                summary=(
                    "Meeting transcript search is not configured "
                    "(DATACENTER_BASE_URL / DATACENTER_SIGNING_SECRET missing)."
                ),
                content="",
                citations=[],
            )

        max_meetings = load_budgets_config().meeting_transcripts["max_meetings"]
        # Built once, used for both the signature and the actual request, so
        # there's no risk of httpx re-encoding the query differently between
        # what we signed and what we sent.
        path_with_query = f"/transcripts?{urlencode({'from': args.from_time, 'to': args.to_time})}"
        timestamp, signature = _sign(self._signing_secret, "GET", path_with_query, "")

        async with httpx.AsyncClient(
            base_url=self._base_url, transport=self._transport, timeout=15.0
        ) as client:
            try:
                response = await client.get(
                    path_with_query,
                    headers={"X-Vault-Timestamp": timestamp, "X-Vault-Signature": signature},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise DatacenterConnectorError(
                    f"Could not reach the datacenter's transcript store: {exc}"
                ) from exc

        meetings = response.json().get("transcripts", [])[:max_meetings]
        if not meetings:
            return ToolResult(
                summary=(
                    f"No meeting transcripts found between {args.from_time} and {args.to_time}."
                ),
                content="",
                citations=[],
            )

        citations = [f"meeting:{m['meetingId']}" for m in meetings]
        content = "\n\n".join(
            f"[meeting:{m['meetingId']}] {m['topic']} "
            f"(host: {m['hostEmail']}, started: {m['startTime']}):\n{m['transcript']}"
            for m in meetings
        )
        return ToolResult(
            summary=f"Found {len(meetings)} meeting transcript(s) between {args.from_time} and {args.to_time}.",
            content=content,
            citations=citations,
        )
