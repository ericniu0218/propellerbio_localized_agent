"""Reference implementation of the datacenter transcript connector.

zoom-transcript-vault's README specifies three HTTP endpoints that "the
customer's engineering team implements on whatever they run internally" — and
until one exists, neither side of that contract can be tested. This is a
working one: small enough to read in a sitting, complete enough to run the
whole path end to end on a laptop.

It is deliberately *not* production infrastructure. It stores transcripts as
files on local disk, has no replication, no retention policy, and no
authentication beyond the shared-secret signature the contract specifies. A
real deployment replaces it with the customer's object store or database. What
it is for: running the loop locally, and serving as an executable spec that a
customer's team can read instead of interpreting prose.

    pip install fastapi uvicorn
    DATACENTER_SIGNING_SECRET=dev-secret \
    TRANSCRIPT_DIR=./transcripts \
    uvicorn contrib.datacenter_connector.main:app --port 8080

Endpoints, per the contract:
  POST /transcripts               store one meeting        (called by the Worker)
  GET  /transcripts/{meeting_id}  one transcript back      (called by the Worker)
  GET  /transcripts?from=&to=     every overlapping meeting (called by the AI)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Requests older than this are refused even with a valid signature, so a
# captured request can't be replayed later. The signature covers the timestamp,
# so an attacker cannot simply update it.
MAX_CLOCK_SKEW_MS = 5 * 60 * 1000

app = FastAPI(title="Datacenter Transcript Connector (reference)")


def _secret() -> str:
    secret = os.environ.get("DATACENTER_SIGNING_SECRET")
    if not secret:
        raise HTTPException(503, "DATACENTER_SIGNING_SECRET is not configured.")
    return secret


def _storage_dir() -> Path:
    directory = Path(os.environ.get("TRANSCRIPT_DIR", "./transcripts"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _path_with_query(request: Request) -> str:
    """The request target exactly as sent — that string is what was signed.

    Read from the raw ASGI scope, not `request.url.path`, which is
    percent-*decoded*: the client signed `from=...T00%3A00%3A00Z`, so verifying
    against a decoded `:` would fail every signature. `raw_path` carries the
    original bytes; it excludes the query string, which is appended here.
    """
    scope = request.scope
    raw_path = scope.get("raw_path")
    path = raw_path.decode("latin-1") if raw_path else scope["path"]
    # Some servers include the query in raw_path; don't append it twice.
    if "?" in path:
        return path
    query = scope.get("query_string", b"").decode("latin-1")
    return f"{path}?{query}" if query else path


async def verify_signature(request: Request, body: str) -> None:
    timestamp = request.headers.get("x-vault-timestamp", "")
    signature = request.headers.get("x-vault-signature", "")
    if not timestamp or not signature.startswith("v1="):
        raise HTTPException(401, "Missing or malformed vault signature headers.")

    try:
        sent_at = int(timestamp)
    except ValueError:
        raise HTTPException(401, "Invalid vault timestamp.") from None
    if abs(int(time.time() * 1000) - sent_at) > MAX_CLOCK_SKEW_MS:
        raise HTTPException(401, "Vault timestamp is outside the accepted window.")

    message = f"{timestamp}.{request.method.upper()}.{_path_with_query(request)}.{body}"
    expected = hmac.new(_secret().encode(), message.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature[len("v1=") :]):
        raise HTTPException(401, "Vault signature verification failed.")


def _record_path(meeting_id: str) -> Path:
    # Meeting ids come from Zoom and can contain '/' and other path characters,
    # so they are hashed rather than trusted as filenames.
    digest = hashlib.sha256(meeting_id.encode()).hexdigest()
    return _storage_dir() / f"{digest}.json"


@app.get("/health")
async def health() -> JSONResponse:
    """Unsigned: says the service is up, reveals nothing about what it holds."""
    return JSONResponse({"ok": True})


@app.post("/transcripts")
async def store_transcript(request: Request) -> JSONResponse:
    body = (await request.body()).decode()
    await verify_signature(request, body)

    try:
        record = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body.") from None

    meeting_id = record.get("meetingId")
    if not meeting_id or not record.get("transcript"):
        raise HTTPException(400, "meetingId and transcript are required.")

    _record_path(meeting_id).write_text(json.dumps(record))
    return JSONResponse({"ok": True})


@app.get("/transcripts/{meeting_id:path}")
async def get_transcript(meeting_id: str, request: Request) -> JSONResponse:
    await verify_signature(request, "")
    path = _record_path(meeting_id)
    if not path.exists():
        raise HTTPException(404, "No transcript held for that meeting.")
    record = json.loads(path.read_text())
    return JSONResponse(
        {"transcript": record["transcript"], "format": record.get("format", "vtt")}
    )


@app.delete("/transcripts/{meeting_id:path}")
async def forget_transcript(meeting_id: str, request: Request) -> JSONResponse:
    """Remove one transcript. An extension to the contract, not part of it.

    The workspace calls this when somebody deletes a recorded meeting, so that
    "delete" means the copy here goes too. A datacenter that has not
    implemented it answers 404 or 405 and the workspace says plainly that the
    local copy is gone and this one is not — which is the honest outcome, and
    better than a delete button that quietly leaves a transcript behind.

    Deleting something that is already absent succeeds. The caller asked for it
    to be gone; it is gone.
    """
    await verify_signature(request, "")
    path = _record_path(meeting_id)
    existed = path.exists()
    if existed:
        path.unlink()
    return JSONResponse({"ok": True, "deleted": existed})


@app.get("/transcripts")
async def search_transcripts(request: Request) -> JSONResponse:
    await verify_signature(request, "")

    from_iso = request.query_params.get("from")
    to_iso = request.query_params.get("to")
    if not from_iso or not to_iso:
        raise HTTPException(400, 'Provide "from" and "to" ISO 8601 query parameters.')
    try:
        start = _parse_iso_ms(from_iso)
        end = _parse_iso_ms(to_iso)
    except ValueError:
        raise HTTPException(400, "Invalid ISO 8601 date range.") from None

    # Who is asking, and what they may read. The caller states its identity and
    # groups in the signed request, so they cannot be altered in flight without
    # invalidating the signature. This is the server-side half of access
    # control: a connector that returns meetings the caller may not read has
    # already leaked them, whatever the caller does with them afterwards.
    groups = {
        group.strip()
        for group in (request.query_params.get("groups") or "").split(",")
        if group.strip()
    }

    # "Every transcript whose meeting *overlaps* the range" — a meeting that
    # started before the window but ran into it still counts.
    matches = []
    withheld = 0
    for path in sorted(_storage_dir().glob("*.json")):
        record = json.loads(path.read_text())
        if not (record.get("startTime", 0) <= end and record.get("endTime", 0) >= start):
            continue
        # A meeting with no access groups is readable by everyone: that is how
        # every meeting stored before access control existed behaves, and
        # hiding those would look exactly like data loss.
        allowed = record.get("accessGroups") or []
        if allowed and "general" not in allowed and not (groups & set(allowed)):
            withheld += 1
            continue
        matches.append(record)

    matches.sort(key=lambda record: record.get("startTime", 0))
    return JSONResponse({"transcripts": matches, "withheld": withheld})


def _parse_iso_ms(value: str) -> int:
    from datetime import datetime, timezone

    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)
