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

Scoring is hybrid: keyword overlap plus semantic similarity from a small
static embedding model (see app/meetings/semantic.py). Keyword alone missed
every meeting that used a different word for the same thing — "recruitment"
scored zero against "enrollment" — and meaning alone would blur "G12C" into
"G12D", which in a biotech company is the worse mistake. Added together, the
exact term and the synonym each find their own case. Without an embedding
model installed this falls back to keyword overlap, which is what it always
did.

**Access control.** Every meeting carries access groups, and this tool returns
only the ones the asking user may read. The groups travel on the datacenter
contract as an additive field, so a connector that predates them keeps working
and a meeting stored without them stays readable by everyone — which is what
keeps meetings captured before this from vanishing.

Filtering here is the second of two. A connector that hands over meetings the
caller may not read has already leaked them into this process, so the reference
connector in `contrib/` filters server-side on the identity in the signed
request, and this filters again on what comes back. Meetings that Zoom captured
through zoom-transcript-vault carry no groups yet: that relay has not been
taught to set them, so they remain readable by everyone until it is. See
docs/MEETINGS.md.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import ClassVar

from pydantic import BaseModel

from bedrock.server.config import load_budgets_config
from bedrock.server.tools.base import Tool, ToolResult
from bedrock.server.compliance.rbac import RBACFilter, get_rbac_filter
from bedrock.server.meetings.governance import GENERAL as GENERAL_GROUP
from bedrock.server.meetings.semantic import (
    MIN_SEMANTIC_MATCH,
    SEMANTIC_WEIGHT,
    load as load_semantic_model,
    similarities,
)
from bedrock.server.tools.datacenter_client import (
    DatacenterError,
    DatacenterNotConfigured,
    DatacenterTranscriptClient,
    MeetingTranscript,
)

# The VTT reader lives in bedrock-private-ai, the employee terminal client, so
# both products parse transcripts with the same code rather than two copies
# that drift apart (requirements.txt pins the commit). The import is guarded
# like RDKit's in chemistry_tools.py: without that package the server still
# starts, and only this tool reports that it cannot read transcripts.
try:
    from bedrock.meetings.transcript import Cue, parse_vtt, speakers

    _READER_AVAILABLE = True
except ImportError:
    _READER_AVAILABLE = False

READER_MISSING = (
    "Meeting transcript search is unavailable: the transcript reader from "
    "bedrock-private-ai is not installed. Install requirements.txt, which "
    "pins it, to enable this tool."
)

logger = logging.getLogger(__name__)

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
    user_id: str | None = None
    """Who is asking. Without it the tool can only return meetings that are
    readable by everyone — refusing to guess is the safe direction, because the
    alternative is handing somebody a meeting they were not in."""
    from_time: str | None = None  # ISO 8601; defaults to lookback_days ago
    to_time: str | None = None  # ISO 8601; defaults to now
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    top_k: int | None = None


@dataclass(frozen=True)
class Excerpt:
    meeting: MeetingTranscript
    cues: list[Cue]
    score: float

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

    # Every window, scored two ways. Keyword overlap alone missed any meeting
    # that used a different word for the same thing — "recruitment" scored zero
    # against "enrollment", which a clinical team says interchangeably. Meaning
    # alone would blur "G12C" into "G12D", which in a biotech company is worse.
    # So both, added: the exact term and the synonym each find their own case.
    candidates: list[tuple[MeetingTranscript, list[Cue], float, str]] = []
    for meeting in meetings:
        cues = parse_vtt(meeting.transcript)
        for window in _windows(cues):
            text = " ".join(cue.text for cue in window)
            candidates.append(
                (meeting, window, float(len(query_tokens & _tokens(text))), text)
            )

    # A question made only of stopwords — "what did we do about it" — says
    # nothing specific, but embeds to a perfectly ordinary vector and will sit
    # somewhere near any passage of English. Keyword overlap rejects it by
    # construction; semantic scoring has to be told, or every contentless
    # question comes back with a confident answer to something else.
    model = load_semantic_model() if query_tokens else None
    if model is not None and candidates:
        try:
            scores = similarities(model, query, [text for _m, _w, _s, text in candidates])
            # A passage counts as an answer if it shares a word with the
            # question, or if it is *clearly* about the same thing. Anything
            # weaker scores zero and falls through to the "nothing matched"
            # path, which is what keeps the fallback honest.
            candidates = [
                (
                    meeting,
                    window,
                    keyword + (similarity * SEMANTIC_WEIGHT if similarity >= MIN_SEMANTIC_MATCH else 0.0),
                    text,
                )
                for (meeting, window, keyword, text), similarity in zip(candidates, scores)
            ]
        except Exception:  # never lose a search to the ranker
            logger.warning("Semantic ranking failed; using keyword overlap", exc_info=True)

    scored = [
        Excerpt(meeting=meeting, cues=window, score=score)
        for meeting, window, score, _text in candidates
        if score > 0
    ]

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

    def __init__(
        self,
        client: DatacenterTranscriptClient | None = None,
        rbac: "RBACFilter | None" = None,
    ) -> None:
        # Constructed lazily: the registry builds every tool at import time, and
        # a deployment with no transcript connector must not fail to start.
        self._client = client
        self._rbac = rbac or get_rbac_filter()

    def _get_client(self) -> DatacenterTranscriptClient:
        if self._client is None:
            self._client = DatacenterTranscriptClient()
        return self._client

    def _visible_to(
        self, user_id: str | None, meetings: list[MeetingTranscript]
    ) -> list[MeetingTranscript]:
        """The meetings this user may read.

        A meeting carrying no access groups is readable by everyone. That is
        deliberate and it is the compatibility seam: every meeting recorded
        before access control existed has no groups, and making them all
        invisible overnight would look exactly like data loss. A deployment
        that wants the strict reading sets MEETING_ACCESS_GROUPS and backfills
        the meetings it already holds.

        With no `user_id`, only ungrouped meetings come back. Callers that know
        who is asking should say so; a caller that does not must not be handed
        a restricted meeting on the grounds that it forgot to mention itself.
        """
        groups = self._rbac.groups_for(user_id) if user_id else set()
        return [
            meeting
            for meeting in meetings
            if not meeting.access_groups
            or GENERAL_GROUP in meeting.access_groups
            or (groups & set(meeting.access_groups))
        ]

    async def run(self, args: MeetingTranscriptsArgs) -> ToolResult:
        # Checked before fetching: there is no point pulling transcripts
        # across the network that nothing here can read.
        if not _READER_AVAILABLE:
            return ToolResult(summary=READER_MISSING, content="", citations=[])

        now = datetime.now(timezone.utc)
        from_iso = args.from_time or _iso(now - timedelta(days=max(args.lookback_days, 1)))
        to_iso = args.to_time or _iso(now)
        top_k = args.top_k or load_budgets_config().rag["top_k_chunks"]

        try:
            groups = sorted(self._rbac.groups_for(args.user_id)) if args.user_id else None
            meetings = await self._get_client().fetch_range(from_iso, to_iso, groups)
        except DatacenterNotConfigured as exc:
            return ToolResult(summary=str(exc), content="", citations=[])
        except DatacenterError as exc:
            return ToolResult(
                summary=f"Meeting transcripts are unavailable: {exc}",
                content="",
                citations=[],
            )

        # Filtered here, after fetching and before ranking. The connector
        # contract takes a date range and returns everything in it, so this is
        # the first point that knows who is asking. Filtering client-side alone
        # is not a security boundary — a connector that hands over meetings the
        # caller may not read has already leaked them to this process — so the
        # reference connector filters too, and this is the second of the two.
        visible = self._visible_to(args.user_id, meetings)
        withheld = len(meetings) - len(visible)
        meetings = visible

        if not meetings:
            if withheld:
                return ToolResult(
                    summary=(
                        f"{withheld} meeting(s) in this window are not readable by "
                        "you. Ask whoever ran the meeting to share it."
                    ),
                    content="",
                    citations=[],
                )
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
