"""Signed HTTP client for the customer's datacenter transcript connector.

This is the other half of the contract documented in zoom-transcript-vault's
README ("Datacenter connector contract", endpoint 3). That repo is a relay on
the public internet: it captures Zoom transcripts and pushes them into the
customer's own datacenter. It never holds transcript content.

**This client must talk to the datacenter directly, over the internal
network — never to the Worker.** Routing a question through the Worker would
mean this air-gapped system making an outbound call to the public internet to
answer it, which defeats the air gap the whole deployment exists to maintain.
`assert_internal_endpoint` is a cheap guard against that specific
misconfiguration, in the same spirit as app/compliance/airgap_check.py.

Signing scheme, which must match the Worker's `signedHeaders` byte for byte:

    HMAC-SHA256(secret, f"{timestamp_ms}.{METHOD}.{path_with_query}.{body}")

hex-encoded, sent as `X-Vault-Timestamp` and `X-Vault-Signature: v1=<hex>`.
The method and path-with-query are inside the signed message on purpose: without
them, a valid signature for one date range could be replayed against another.
Because the *exact* path string is signed, this module builds the query string
itself and sends that same string, rather than letting the HTTP library
re-encode it — `%3A` and `:` produce different signatures.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from urllib.parse import quote, urlsplit

import httpx

logger = logging.getLogger(__name__)

# Fetching a wide date range means pulling every transcript in it across the
# network. These bound a single call so one vague question can't drag tens of
# megabytes through the datacenter connector.
DEFAULT_TIMEOUT_S = 30.0
MAX_MEETINGS_PER_CALL = 200


class DatacenterError(RuntimeError):
    """Raised when the datacenter connector is unreachable or refuses a request."""


class DatacenterNotConfigured(DatacenterError):
    """Raised when no datacenter connector has been configured for this deployment."""


@dataclass(frozen=True)
class MeetingTranscript:
    meeting_id: str
    topic: str
    host_email: str
    start_time: int  # unix ms
    end_time: int  # unix ms
    transcript: str
    format: str = "vtt"
    access_groups: list[str] = field(default_factory=list)
    """Who may read this meeting. An additive extension to the contract in
    zoom-transcript-vault's README: a connector that does not know the field
    ignores it, and a reader that gets no field treats the meeting as readable
    by everyone, which is how every meeting stored before this behaved."""


def sign_request(secret: str, timestamp_ms: int, method: str, path_with_query: str, body: str) -> str:
    """Hex HMAC-SHA256 over the Worker's exact message layout."""
    message = f"{timestamp_ms}.{method.upper()}.{path_with_query}.{body}"
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def signed_headers(secret: str, method: str, path_with_query: str, body: str) -> dict[str, str]:
    timestamp_ms = int(time.time() * 1000)
    signature = sign_request(secret, timestamp_ms, method, path_with_query, body)
    return {
        "Content-Type": "application/json",
        "X-Vault-Timestamp": str(timestamp_ms),
        "X-Vault-Signature": f"v1={signature}",
    }


def transcripts_path(from_iso: str, to_iso: str, groups: list[str] | None = None) -> str:
    """The exact path+query that gets both signed and sent.

    Values are percent-encoded the way the Worker's `encodeURIComponent` does,
    so an ISO timestamp's colons arrive as `%3A` — see the module docstring.

    `groups` names what the caller may read, and is appended only when present
    so a request without it signs byte-for-byte as it always did. It is inside
    the signed path on purpose: an access claim that could be edited in flight
    would be no access control at all.
    """
    path = f"/transcripts?from={quote(from_iso, safe='')}&to={quote(to_iso, safe='')}"
    if groups:
        path += f"&groups={quote(','.join(sorted(groups)), safe='')}"
    return path


# Hosts that mean someone has pointed this at the public relay instead of the
# internal datacenter. Not an exhaustive "is this the internet" test — it
# catches the one misconfiguration the architecture specifically warns about.
_PUBLIC_HOST_MARKERS = (".workers.dev", ".cloudflare.com", ".pages.dev")


def assert_internal_endpoint(base_url: str) -> None:
    host = (urlsplit(base_url).hostname or "").lower()
    if any(host.endswith(marker) for marker in _PUBLIC_HOST_MARKERS):
        raise DatacenterError(
            f"DATACENTER_BASE_URL points at a public host ({host}). This system is "
            "air-gapped and must query the datacenter connector directly over the "
            "internal network, not the public Zoom relay. Set AGENT_PACK_ALLOW_PUBLIC_"
            "DATACENTER=true only if you have deliberately deployed the connector there."
        )


@dataclass(frozen=True)
class DatacenterConfig:
    base_url: str
    signing_secret: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(cls) -> "DatacenterConfig":
        base_url = os.environ.get("DATACENTER_BASE_URL", "").rstrip("/")
        secret = os.environ.get("DATACENTER_SIGNING_SECRET", "")
        if not base_url or not secret:
            raise DatacenterNotConfigured(
                "No meeting transcript datacenter is configured. Set "
                "DATACENTER_BASE_URL and DATACENTER_SIGNING_SECRET to the internal "
                "connector described in zoom-transcript-vault's README."
            )
        if os.environ.get("AGENT_PACK_ALLOW_PUBLIC_DATACENTER", "").lower() != "true":
            assert_internal_endpoint(base_url)
        return cls(
            base_url=base_url,
            signing_secret=secret,
            timeout_s=float(os.environ.get("DATACENTER_TIMEOUT_S", DEFAULT_TIMEOUT_S)),
        )


class DatacenterTranscriptClient:
    """Reads transcripts from the customer's own infrastructure, and writes the
    ones this deployment recorded itself.

    Reading is the original job: zoom-transcript-vault puts Zoom's transcripts
    into the datacenter and `meeting_transcripts` reads them back. Writing was
    added for `app/meetings/`, which records meetings nobody else captured — a
    microphone in a room, or the audio of an online call. Those are delivered
    through endpoint 1 of the same contract, in the same shape the Worker uses,
    so a recorded meeting and a Zoom meeting are indistinguishable to every
    reader downstream and retrieval needed no new code.
    """

    def __init__(
        self,
        config: DatacenterConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config or DatacenterConfig.from_env()
        # Test seam: lets the suite run a request against an in-process ASGI
        # connector while still exercising real URL handling and real headers.
        self._transport = transport

    async def fetch_range(
        self, from_iso: str, to_iso: str, groups: list[str] | None = None
    ) -> list[MeetingTranscript]:
        """Every transcript whose meeting overlaps [from_iso, to_iso] that the
        caller's `groups` allow it to read."""
        path = transcripts_path(from_iso, to_iso, groups)
        headers = signed_headers(self._config.signing_secret, "GET", path, "")

        try:
            async with httpx.AsyncClient(
                timeout=self._config.timeout_s, transport=self._transport
            ) as client:
                # The full URL is passed as one pre-encoded string so httpx
                # sends exactly the path that was signed.
                response = await client.get(f"{self._config.base_url}{path}", headers=headers)
        except httpx.HTTPError as exc:
            raise DatacenterError(
                f"Could not reach the datacenter transcript connector: {exc}"
            ) from exc

        if response.status_code == 401:
            raise DatacenterError(
                "The datacenter connector rejected our signature (401). Check that "
                "DATACENTER_SIGNING_SECRET matches the secret the connector holds."
            )
        if response.status_code >= 400:
            raise DatacenterError(
                f"The datacenter connector returned {response.status_code}."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise DatacenterError("The datacenter connector returned invalid JSON.") from exc

        records = payload.get("transcripts") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise DatacenterError(
                "The datacenter connector's response had no 'transcripts' list."
            )

        return [
            _to_transcript(record)
            for record in records[:MAX_MEETINGS_PER_CALL]
            if isinstance(record, dict) and record.get("transcript")
        ]

    async def store(self, transcript: MeetingTranscript) -> None:
        """Hand one transcript to the datacenter (contract endpoint 1).

        The body is serialised once and both signed and sent as that exact
        string. Re-serialising it — letting httpx take a dict and encode it
        itself — can reorder keys or change separators, and the signature is
        over bytes, so the connector would reject every request with a 401.
        """
        body = json.dumps(
            {
                "meetingId": transcript.meeting_id,
                "topic": transcript.topic,
                "hostEmail": transcript.host_email,
                "startTime": transcript.start_time,
                "endTime": transcript.end_time,
                "transcript": transcript.transcript,
                "format": transcript.format,
                "accessGroups": transcript.access_groups,
            },
            separators=(",", ":"),
        )
        path = "/transcripts"
        headers = signed_headers(self._config.signing_secret, "POST", path, body)

        try:
            async with httpx.AsyncClient(
                timeout=self._config.timeout_s, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._config.base_url}{path}", content=body, headers=headers
                )
        except httpx.HTTPError as exc:
            raise DatacenterError(
                f"Could not reach the datacenter transcript connector: {exc}"
            ) from exc

        if response.status_code == 401:
            raise DatacenterError(
                "The datacenter connector rejected our signature (401). Check that "
                "DATACENTER_SIGNING_SECRET matches the secret the connector holds."
            )
        if response.status_code >= 400:
            raise DatacenterError(
                f"The datacenter connector returned {response.status_code} storing "
                f"meeting {transcript.meeting_id}."
            )


    async def forget(self, meeting_id: str) -> str | None:
        """Ask the datacenter to delete one transcript.

        Returns None when the copy is gone, or a plain sentence saying why it
        is not. Never raises: this is called while deleting a meeting, and
        failing to reach the datacenter must not leave the workspace holding a
        recording the employee has already been told is deleted.

        DELETE is an extension, not part of the connector contract, so a
        datacenter that has not implemented it answers 404 or 405. That is
        reported as a fact about their system rather than as our error, because
        somebody there has to remove it by hand.
        """
        path = f"/transcripts/{meeting_id}"
        headers = signed_headers(self._config.signing_secret, "DELETE", path, "")

        try:
            async with httpx.AsyncClient(
                timeout=self._config.timeout_s, transport=self._transport
            ) as client:
                response = await client.delete(
                    f"{self._config.base_url}{path}", headers=headers
                )
        except httpx.HTTPError as exc:
            logger.warning("Could not reach the datacenter to delete %s: %s", meeting_id, exc)
            return (
                "Deleted here, but the datacenter could not be reached, so its "
                "copy of the transcript is still there."
            )

        if response.status_code in (404, 405, 501):
            return (
                "Deleted here. This datacenter does not support deleting "
                "transcripts, so ask whoever runs it to remove their copy."
            )
        if response.status_code == 401:
            return (
                "Deleted here, but the datacenter rejected our signature, so "
                "its copy of the transcript is still there."
            )
        if response.status_code >= 400:
            return (
                f"Deleted here, but the datacenter returned "
                f"{response.status_code}, so its copy is still there."
            )
        return None


def _to_transcript(record: dict) -> MeetingTranscript:
    # `format` is absent from the Worker's own /transcripts response even
    # though the connector contract documents it, so it is defaulted rather
    # than required — a caller shouldn't have to know which side answered.
    return MeetingTranscript(
        meeting_id=str(record.get("meetingId", "")),
        topic=str(record.get("topic") or "Untitled meeting"),
        host_email=str(record.get("hostEmail", "")),
        start_time=int(record.get("startTime") or 0),
        end_time=int(record.get("endTime") or 0),
        transcript=str(record.get("transcript") or ""),
        format=str(record.get("format") or "vtt"),
        access_groups=[str(g) for g in (record.get("accessGroups") or [])],
    )
