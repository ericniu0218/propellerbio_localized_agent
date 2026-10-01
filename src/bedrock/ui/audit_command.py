"""The `/audit` command: read back the local action journal.

Every tracked action writes twice - an *intent* before it starts and an
*outcome* after it finishes - so this feed shows attempts, not just successes.
That is the whole point of it. A row with an intent and no outcome is an
action whose result was never confirmed: the app may have been closed, or the
work may still be going on in another process. This command says exactly that
and refuses to guess, because inventing a "failed" would be worse than
admitting the gap.

What it is not: an authenticated approval ledger, or tamper-proof storage.
The journal is a SQLite file owned by this OS account, and the account owner
can change it. `/audit export` writes the same rows as JSONL for review
elsewhere.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from bedrock.ui.argparsing import CommandParser, CommandUsageError, split_arguments
from bedrock.ui.rendering import as_cell, as_code, as_code_cell, safe_line

AUDIT_HELP = """\
**`/audit` — what this client did on this machine**

| Command | What it does |
|---|---|
| `/audit` | The first page of recorded actions |
| `/audit --after 120 --limit 50` | Continue from a sequence number |
| `/audit export "/approved/review/actions.jsonl"` | Write the whole journal as JSONL |

Each action records an **intent** before it runs and an **outcome** after it
finishes. An intent with no outcome means the result was never confirmed —
possibly still running, never failed-by-assumption.

Folds, digests, approvals, run tagging and literature reviews are recorded
here — including, for a review, the external hosts it was authorised to
contact (the literature search, the automatic retraction check), never the
question itself.

Export writes metadata only: identifiers, hashes, decisions and bounded error
codes. No file contents, prompts, approval previews or credentials are stored.
"""

_STATE_NOTES = {
    "intent": "about to start",
    "succeeded": "confirmed",
    "failed": "confirmed failure",
    "cancelled": "confirmed cancellation",
    "unknown": "outcome never confirmed",
}

_DETAIL_ORDER = (
    "run_id",
    "project_id",
    "artifact_id",
    "session_id",
    "task_id",
    "decision",
    "source_kind",
    "source_bytes",
    "source_sha256",
    "manifest_sha256",
    "error_code",
    "actor_source",
)


async def run_audit_command(args: str, *, services: Any, notify) -> None:
    """Handle one ``/audit ...`` invocation."""
    text = args.strip()
    verb, _, rest = text.partition(" ")
    try:
        if verb in {"help", "?"}:
            await notify(AUDIT_HELP, "info")
        elif verb == "export":
            await _export(rest.strip(), services=services, notify=notify)
        else:
            await _events(text, services=services, notify=notify)
    except CommandUsageError as exc:
        await notify(f"{safe_line(exc)}\n\nType `/audit help` for the options.", "error")
    except services.errors as exc:
        await notify(safe_line(exc, limit=400), "error")


async def _events(text: str, *, services: Any, notify) -> None:
    parser = CommandParser("/audit")
    parser.add_argument("--after", type=int, default=0)
    parser.add_argument("--limit", type=int, default=25)
    options = parser.parse_args(split_arguments(text))

    history = await services.history()
    events = await asyncio.to_thread(history.events, after=options.after, limit=options.limit)
    await notify(format_events(events, after=options.after), "info")


async def _export(text: str, *, services: Any, notify) -> None:
    parser = CommandParser("/audit export")
    parser.add_argument("destination")
    options = parser.parse_args(split_arguments(text))

    destination = Path(options.destination).expanduser()
    history = await services.history()
    count = await asyncio.to_thread(history.export, destination)
    await notify(
        f"Exported {count} recorded action(s) to {as_code(destination)}. "
        "Metadata only — no file contents, prompts or approval previews.",
        "info",
    )


def format_events(events: list[dict], after: int = 0, prefix: str = "/audit") -> str:
    """``prefix`` is how the reader continues: the slash command in the app,
    ``bedrock-biotech audit`` on the command line."""
    if not events:
        return (
            "_No recorded actions after sequence "
            f"{after}._ Folds, digests, approvals, run tagging and literature "
            "reviews are recorded here."
        )
    lines = [
        f"**Local action history** — {len(events)} event(s) after sequence {after}",
        "",
        "| # | Recorded (UTC) | Action | State | Subject | Detail |",
        "|---|---|---|---|---|---|",
    ]
    for event in events:
        state = str(event.get("state", ""))
        lines.append(
            "| {sequence} | {recorded} | {action} | {state} | {subject} | {details} |".format(
                sequence=event.get("sequence", "—"),
                recorded=as_cell(event.get("recorded_at", "")),
                action=as_cell(event.get("action", "")),
                state=f"{as_cell(state)} — {_STATE_NOTES.get(state, state)}",
                subject=as_code_cell(event.get("subject_id") or "—"),
                details=_format_details(event.get("details") or {}),
            )
        )
    last = events[-1].get("sequence")
    lines.extend(
        [
            "",
            f"Continue with `{prefix} --after {last}`.",
            "",
            "*Local history for this OS account. An intent with no outcome was never "
            "confirmed; it is not evidence that the action failed or stopped.*",
        ]
    )
    return "\n".join(lines)


def _format_details(details: dict) -> str:
    ordered = [key for key in _DETAIL_ORDER if key in details]
    ordered.extend(sorted(key for key in details if key not in _DETAIL_ORDER))
    parts = []
    for key in ordered:
        value = details[key]
        if key.endswith("sha256") and isinstance(value, str) and len(value) > 12:
            value = f"{value[:12]}…"
        parts.append(f"{as_cell(key, limit=32)}={as_cell(value, limit=48)}")
    return ", ".join(parts) if parts else "—"
