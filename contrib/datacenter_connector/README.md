# Reference datacenter transcript connector

A working implementation of the three HTTP endpoints that
[zoom-transcript-vault](https://github.com/Bedrock-Technical-Consulting/zoom-transcript-vault)
specifies but leaves to "the customer's engineering team." Until one existed,
neither side of that contract could be tested.

**This is not production infrastructure.** It stores transcripts as files on
local disk, with no replication, retention policy, or access control beyond the
shared-secret signature. A real deployment replaces it with the customer's own
object store or database. Its value is that it runs the whole path on a laptop,
and gives a customer's team an executable spec to read instead of prose.

## Run it

```bash
pip install fastapi uvicorn

DATACENTER_SIGNING_SECRET=dev-secret \
TRANSCRIPT_DIR=./transcripts \
uvicorn contrib.datacenter_connector.main:app --port 8080
```

Then point the agent pack at it:

```bash
export DATACENTER_BASE_URL=http://127.0.0.1:8080
export DATACENTER_SIGNING_SECRET=dev-secret
```

## Endpoints

| Endpoint | Called by | Purpose |
|---|---|---|
| `POST /transcripts` | the vault Worker | store one meeting |
| `GET /transcripts/{meeting_id}` | the vault Worker | read one back |
| `GET /transcripts?from=&to=` | this repo's AI | every meeting overlapping a range |

All three require a valid `X-Vault-Signature`. Requests whose timestamp is more
than five minutes old are refused even with a valid signature, so a captured
request cannot be replayed later.

## What a real implementation must get right

- **Verify against the raw, still-encoded request path.** The client signs
  `from=...T00%3A00%3A00Z`. A framework that hands you a percent-*decoded* path
  will fail every signature. See `_path_with_query` for how this reads the raw
  ASGI scope rather than the parsed URL.
- **Compare signatures in constant time** (`hmac.compare_digest`).
- **Reject stale timestamps**, or a captured request is replayable forever.
- **Don't trust meeting ids as filenames.** Zoom ids can contain `/` and `..`;
  this implementation hashes them.
- **"Overlaps the range" is not "starts in the range."** A meeting that began
  before the window and ran into it still counts.
