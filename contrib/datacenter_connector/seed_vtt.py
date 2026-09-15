"""Load a Zoom transcript file into the connector, for trying this without Zoom.

Zoom lets you download a meeting's transcript by hand from the web portal
(Recordings > the meeting > Audio Transcript > Download). That file is the exact
same VTT the webhook would have delivered, so dropping one in here exercises the
whole retrieval and question-answering path using a real meeting, before any of
the Cloudflare or Zoom setup exists.

    python contrib/datacenter_connector/seed_vtt.py meeting.vtt \
        --topic "Phase II weekly sync" --date 2026-09-12

Requires the connector to be running, and DATACENTER_SIGNING_SECRET to match it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


def sign(secret: str, timestamp_ms: int, method: str, path: str, body: str) -> str:
    import hmac

    message = f"{timestamp_ms}.{method.upper()}.{path}.{body}"
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtt", type=Path, help="transcript file downloaded from Zoom")
    parser.add_argument("--topic", required=True, help="the meeting title")
    parser.add_argument("--date", help="meeting date, YYYY-MM-DD (default: today)")
    parser.add_argument("--host", default="unknown@example.com", help="host email")
    parser.add_argument("--minutes", type=int, default=60, help="meeting length")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DATACENTER_BASE_URL", "http://127.0.0.1:8080"),
    )
    args = parser.parse_args()

    secret = os.environ.get("DATACENTER_SIGNING_SECRET")
    if not secret:
        print("DATACENTER_SIGNING_SECRET is not set — it must match the connector.", file=sys.stderr)
        return 2
    if not args.vtt.exists():
        print(f"No such file: {args.vtt}", file=sys.stderr)
        return 2

    transcript = args.vtt.read_text(errors="replace")
    if "-->" not in transcript:
        print(
            f"{args.vtt} doesn't look like a VTT transcript (no timestamp lines found).",
            file=sys.stderr,
        )
        return 2

    if args.date:
        try:
            start = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print("--date must be YYYY-MM-DD.", file=sys.stderr)
            return 2
    else:
        start = datetime.now(timezone.utc)

    start_ms = int(start.timestamp() * 1000)
    # A meeting id derived from the file's content, so loading the same
    # transcript twice updates one record instead of creating a duplicate.
    meeting_id = "manual-" + hashlib.sha256(transcript.encode()).hexdigest()[:16]

    record = {
        "meetingId": meeting_id,
        "topic": args.topic,
        "hostEmail": args.host,
        "startTime": start_ms,
        "endTime": start_ms + args.minutes * 60_000,
        "transcript": transcript,
        "format": "vtt",
    }
    body = json.dumps(record)
    timestamp_ms = int(time.time() * 1000)

    req = request.Request(
        f"{args.base_url.rstrip('/')}/transcripts",
        data=body.encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Vault-Timestamp": str(timestamp_ms),
            "X-Vault-Signature": f"v1={sign(secret, timestamp_ms, 'POST', '/transcripts', body)}",
        },
    )
    try:
        with request.urlopen(req, timeout=15) as response:
            response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        print(f"Connector refused the transcript ({exc.code}): {detail}", file=sys.stderr)
        if exc.code == 401:
            print("Check that DATACENTER_SIGNING_SECRET matches the connector's.", file=sys.stderr)
        return 1
    except error.URLError as exc:
        print(f"Could not reach the connector at {args.base_url}: {exc.reason}", file=sys.stderr)
        print("Is it running? See contrib/datacenter_connector/README.md", file=sys.stderr)
        return 1

    words = len(transcript.split())
    print(f'Loaded "{args.topic}" ({start.strftime("%Y-%m-%d")}, ~{words:,} words) as {meeting_id}.')
    print("Ask about it with the meeting_transcripts tool, or:")
    print(f'  curl -s "{args.base_url.rstrip("/")}/health" >/dev/null  # connector is up')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
