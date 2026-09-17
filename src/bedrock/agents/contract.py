"""The frozen boundary between understanding a meeting and acting on it.

This file exists so two things can be built at once without either one waiting
for the other, and it is the only module both halves import.

*Left of the line* (``bedrock/meetings/``): reading a transcript and working out
what was asked for. It produces ``Ask`` values and knows nothing about calendars,
mail servers or browsers.

*Right of the line* (``bedrock/agents/``): turning an ``Ask`` into a ``Plan``,
getting it approved, and running it against the outside world. It consumes
``Ask`` values and never parses a transcript.

Three rules make the seam hold:

1. **Everything here is frozen and stdlib-only.** No model, no I/O, no
   configuration. A dataclass cannot drift under either side.
2. **Evidence travels with the ask.** An ``Ask`` carries the verbatim quote it
   came from, checked by ``meetings/citations.py``. Nothing acts on a request
   nobody can see was made - the same rule that governs timestamps.
3. **Nothing executes without a Grant.** A ``Plan`` is inert. It describes what
   would happen in words a person can read, and only an explicit ``Grant``
   turns it into an ``Outcome``. Reading the plan is how someone decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AskKind(str, Enum):
    """What someone asked for. Deliberately short: each entry has to earn a
    real executor on the other side of the line, and an unclassifiable request
    is more useful as a note than as a wrong category."""

    SCHEDULE = "schedule"       # "let's meet next Tuesday"
    REMIND = "remind"           # "someone ping me before the DSMB call"
    PREPARE = "prepare"         # "pull up the enrolment numbers"
    CREATE = "create"           # "draft the protocol amendment"
    LOOK_UP = "look_up"         # "find out what the FDA guidance says"


class Risk(str, Enum):
    """How much a step can cost if it is wrong.

    READ_ONLY can run before anyone looks at it. REACHES_PEOPLE and
    CHANGES_THINGS never run unattended, because an email cannot be unsent and
    a deleted file is not a draft.
    """

    READ_ONLY = "read_only"
    CHANGES_THINGS = "changes_things"
    REACHES_PEOPLE = "reaches_people"


@dataclass(frozen=True)
class Evidence:
    """Where an ask came from, in the speaker's own words.

    ``timestamp`` is whatever ``citations.resolve_timestamp`` returned, so it is
    ``None`` when the quote could not be located - which is allowed. An ask with
    an unverifiable quote may still be shown to a person; it may never be
    executed unattended.
    """

    quote: str
    speaker: str | None = None
    timestamp: str | None = None

    @property
    def is_located(self) -> bool:
        return self.timestamp is not None


@dataclass(frozen=True)
class Ask:
    """One request detected in a meeting, before anyone decides what to do.

    ``text`` is the readable paraphrase. ``subject`` is the thing acted on (a
    file name, a topic, a person). ``when_text`` is the time expression exactly
    as spoken - "next Tuesday", "before the DSMB" - and stays a string here on
    purpose: resolving it to a date needs the meeting's own date and the
    company's calendar, which is the executor's business, not the reader's.
    """

    kind: AskKind
    text: str
    evidence: Evidence
    owner: str | None = None
    subject: str | None = None
    when_text: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class Step:
    """One thing a tool would do, described before it does it."""

    tool: str
    summary: str
    risk: Risk
    arguments: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    """What would happen, in a form a person can read and refuse.

    A plan is always produced, even when it cannot run. ``blocked_reason`` set
    means the steps are a description of what was wanted, not an offer.
    """

    ask: Ask
    steps: tuple[Step, ...] = ()
    blocked_reason: str | None = None

    @property
    def needs_approval(self) -> bool:
        return any(step.risk is not Risk.READ_ONLY for step in self.steps)


@dataclass(frozen=True)
class Grant:
    """Permission for one plan, from one person, this once.

    Scoped to a single plan by design. A standing grant is a different feature
    with a different risk profile, and it does not belong in the first version.
    """

    plan_id: str
    granted_by: str
    steps_allowed: tuple[str, ...]


@dataclass(frozen=True)
class Outcome:
    """What actually happened, for the journal and for the person.

    ``artifacts`` are local paths produced by the run. ``failed`` carries the
    plain-English reason, because a step that did not run must say so rather
    than be silently dropped from the notes.
    """

    plan_id: str
    completed: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    failed: str | None = None
