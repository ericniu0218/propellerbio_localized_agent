"""Search the company's own Zoom meeting transcripts.

Completes the loop that zoom-transcript-vault opens: that service captures
transcripts from Zoom and pushes them into the customer's datacenter, and this
tool is the consumer its README describes — the air-gapped side reading them
back over the internal network. See app/tools/datacenter_client.py for the
signed transport and why it must not go through the public relay.

The connector only supports retrieval by date range, but people ask questions
by *topic* ("what did we decide about enrollment?"). So this tool pulls the
meetings in a window and does the relevance ranking locally: transcripts are
parsed into speaker-attributed cues, grouped into short overlapping windows of
conversation, and scored against the question. Only the best windows travel on
to the agent, because a month of meetings is far larger than any context
budget — the same reason rag_search returns top-k chunks instead of documents.

Scoring is deterministic token overlap, matching KeywordOverlapIndex in
rag_search.py. That is not semantic search; it is the same honest placeholder,
and it swaps for embeddings behind the same interface when the RAG ingest
pipeline exists.

**No per-user access control.** rag_search filters its hits through RBACFilter;
this tool cannot, because the connector contract in zoom-transcript-vault has no
notion of who may read which meeting — it takes a date range and returns
everything in it. So any user who can reach this tool can read any recorded
meeting, including ones they were not in. That is a contract-level gap, not
something this tool can fix locally: the connector needs to accept an identity
and filter server-side (or transcripts need per-meeting access groups at
ingest). Until then, deploy this only where every user is entitled to every
meeting, and see docs/MEETINGS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import ClassVar

from pydantic import BaseModel

from app.config import load_budgets_config
from app.tools.base import Tool, ToolResult
from app.tools.datacenter_client import (
    DatacenterError,
    DatacenterNotConfigured,
    DatacenterTranscriptClient,
    MeetingTranscript,
)
from app.tools.vtt import Cue, parse_vtt, speakers

DEFAULT_LOOKBACK_DAYS = 30

# A single cue is one short utterance, which is rarely an answer on its own.
# Windows overlap so a decision spanning a window boundary still scores.
WINDOW_CUES = 6
WINDOW_STRIDE = 3

# Question words carry no signal and match every meeting ever recorded.
_STOPWORDS = frozenset(
    """a an and are as at be been but by can did do does for from had has have
    how i if in is it its me my of on or our says that the their them then there
    these they this to us was we were what when where which who why will with
    would you your about said say get got make made just like know think really
    going go want need see look come take use""".split()
)


class MeetingTranscriptsArgs(BaseModel):
    query: str
    from_time: str | None = None  # ISO 8601; defaults to lookback_days ago
    to_time: str | None = None  # ISO 8601; defaults to now
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    top_k: int | None = None


@dataclass(frozen=True)
class Excerpt:
    meeting: MeetingTranscript
    cues: list[Cue]
    score: int

    @property
    def citation(self) -> str:
        date = _utc_date(self.meeting.start_time)
        return f"{self.meeting.topic} ({date}) @{self.cues[0].timestamp}"

    def render(self) -> str:
        body = "\n".join(cue.render() for cue in self.cues)
        return f"[{self.citation}]\n{body}"


def _utc_date(unix_ms: int) -> str:
    return datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def _windows(cues: list[Cue]) -> list[list[Cue]]:
    if len(cues) <= WINDOW_CUES:
        return [cues] if cues else []
    return [
        cues[start : start + WINDOW_CUES]
        for start in range(0, len(cues) - WINDOW_CUES + 1, WINDOW_STRIDE)
    ]


def rank_excerpts(
    query: str, meetings: list[MeetingTranscript], top_k: int
) -> tuple[list[Excerpt], bool]:
    """Best-matching conversation windows across every meeting.

    Returns ``(excerpts, matched)``. When nothing matches the question, falls
    back to the opening of each meeting — that is where the agenda gets stated,
    so "catch me up on this week" still gets a useful answer instead of an
    empty one. ``matched`` tells the caller which of the two happened, so the
    agent never presents a fallback as if it were a hit.
    """
    query_tokens = _tokens(query)
    scored: list[Excerpt] = []

    for meeting in meetings:
        cues = parse_vtt(meeting.transcript)
        for window in _windows(cues):
            overlap = query_tokens & _tokens(" ".join(cue.text for cue in window))
            if overlap:
                scored.append(Excerpt(meeting=meeting, cues=window, score=len(overlap)))

    if scored:
        scored.sort(key=lambda e: (-e.score, e.meeting.start_time, e.cues[0].start_ms))
        return _dedupe(scored)[:top_k], True

    openings = [
        Excerpt(meeting=meeting, cues=cues[:WINDOW_CUES], score=0)
        for meeting in meetings
        if (cues := parse_vtt(meeting.transcript))
    ]
    return openings[:top_k], False


def _dedupe(excerpts: list[Excerpt]) -> list[Excerpt]:
    """Drop overlapping windows from the same meeting, keeping the best.

    Adjacent windows share cues by design, so without this the top-k is often
    the same thirty seconds of one meeting reported several times.
    """
    kept: list[Excerpt] = []
    for excerpt in excerpts:
        span = range(excerpt.cues[0].start_ms, excerpt.cues[-1].end_ms + 1)
        if any(
            other.meeting.meeting_id == excerpt.meeting.meeting_id
            and other.cues[0].start_ms in span
            for other in kept
        ):
            continue
        kept.append(excerpt)
    return kept


class MeetingTranscriptsTool(Tool):
    name: ClassVar[str] = "meeting_transcripts"
    description: ClassVar[str] = (
        "Search transcripts of the company's own Zoom meetings for what was "
        "said, decided, or assigned. Use for questions about meetings, calls, "
        "standups, or who said what. Returns excerpts with the meeting title, "
        "date, and timestamp as citations. Defaults to the last 30 days; pass "
        "from_time/to_time (ISO 8601) to widen or narrow the window."
    )
    args_model: ClassVar[type[BaseModel]] = MeetingTranscriptsArgs

    def __init__(self, client: DatacenterTranscriptClient | None = None) -> None:
        # Constructed lazily: the registry builds every tool at import time, and
        # a deployment with no transcript connector must not fail to start.
        self._client = client

    def _get_client(self) -> DatacenterTranscriptClient:
        if self._client is None:
            self._client = DatacenterTranscriptClient()
        return self._client

    async def run(self, args: MeetingTranscriptsArgs) -> ToolResult:
        now = datetime.now(timezone.utc)
        from_iso = args.from_time or _iso(now - timedelta(days=max(args.lookback_days, 1)))
        to_iso = args.to_time or _iso(now)
        top_k = args.top_k or load_budgets_config().rag["top_k_chunks"]

        try:
            meetings = await self._get_client().fetch_range(from_iso, to_iso)
        except DatacenterNotConfigured as exc:
            return ToolResult(summary=str(exc), content="", citations=[])
        except DatacenterError as exc:
            return ToolResult(
                summary=f"Meeting transcripts are unavailable: {exc}",
                content="",
                citations=[],
            )

        if not meetings:
            return ToolResult(
                summary=f"No meetings were recorded between {from_iso} and {to_iso}.",
                content="",
                citations=[],
            )

        excerpts, matched = rank_excerpts(args.query, meetings, top_k)
        if not excerpts:
            return ToolResult(
                summary=(
                    f"{len(meetings)} meeting(s) were found in this window, but none "
                    "had a readable transcript."
                ),
                content="",
                citations=[],
            )

        max_chars = load_budgets_config().rag["max_chunk_tokens"] * 4  # ~4 chars/token
        content = "\n\n".join(_trim(excerpt.render(), max_chars) for excerpt in excerpts)
        citations = [excerpt.citation for excerpt in excerpts]

        if matched:
            summary = (
                f"Found {len(excerpts)} relevant excerpt(s) across {len(meetings)} "
                f"meeting(s) between {from_iso} and {to_iso}."
            )
        else:
            # Said plainly so the agent doesn't present the agenda of an
            # unrelated meeting as an answer to the question asked.
            summary = (
                f"Nothing in the {len(meetings)} meeting(s) between {from_iso} and "
                f"{to_iso} matched this question directly. Showing how each meeting "
                "opened instead, which is usually where the agenda is stated."
            )

        people = sorted({s for e in excerpts for s in speakers(e.cues)})
        if people:
            summary += f" Speakers quoted: {', '.join(people)}."

        return ToolResult(summary=summary, content=content, citations=citations)


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"
