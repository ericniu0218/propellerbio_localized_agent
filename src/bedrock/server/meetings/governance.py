"""Who may read a meeting, who agreed to be recorded, and what gets removed.

Three separate obligations that all land on the same record, so they live
together rather than being scattered through the pipeline.

**Redaction.** `src/bedrock/server/compliance/phi_redaction.py` has existed since the start and
was applied on no path at all — `docs/server/MEETINGS.md` said so in its own "known
gaps". A recorded meeting is the likeliest place in this product for a patient
to be named out loud, because people say things in a room that they would never
type into a form. Redaction runs over the transcript and over everything
derived from it before any of it is stored or delivered.

The original audio is deliberately *not* rewritten. Redacting a transcript is
reversible in the sense that matters — somebody can re-transcribe from the
recording if a redaction turns out to be wrong — and destroying the only copy
of what was said to protect it is the worse mistake. What that means is the
audio is now the sensitive artefact, and `MEETING_RECORDING_DIR` needs the
retention policy in `retention.py`.

**Consent.** The UI has always warned that everyone should know they are being
recorded. A warning is not a record. Several US states require every party to
agree, and "the person who pressed Record says they told everyone" is the
honest thing to store — it is a claim by a named employee at a known time, not
proof, and it is labelled as such.

**Access.** Until now any user who could reach the retrieval tool could read
any meeting, including ones they were not in. Every meeting now carries access
groups, and both the workspace and the retrieval tool filter on them. A meeting
with no groups is readable by everyone, which keeps existing data working and
is why `default_groups` exists for deployments that want the opposite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bedrock.server.compliance.phi_redaction import redact_phi

# Fields on a stored meeting that hold words somebody said, or a model's
# rendering of them. Everything here is redacted together, because a patient
# name removed from the transcript but left in an action item is not removed.
# `transcriptEnglish` is the same speech translated: an identifier scrubbed
# from the original and left in the English has not been scrubbed. `title`
# is written by the model from the notes when nobody named the meeting.
REDACTABLE_TEXT = ("transcript", "transcriptEnglish", "summary", "enhancedNotes", "title")
REDACTABLE_LISTS = ("decisions", "actions", "findings", "deadlines", "questions")

# The group every meeting gets when a deployment has not configured any. Named
# to match `RBACFilter`, which already treats "general" as "visible to all".
GENERAL = "general"


def redaction_enabled() -> bool:
    """On by default. A deployment turns it off deliberately, not by omission."""
    return os.environ.get("MEETING_REDACT_PHI", "true").lower() != "false"


def default_groups() -> list[str]:
    raw = os.environ.get("MEETING_ACCESS_GROUPS", "").strip()
    return [group.strip() for group in raw.split(",") if group.strip()] or [GENERAL]


@dataclass
class RedactionSummary:
    applied: bool = False
    findings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.applied:
            return ""
        kinds = ", ".join(sorted(set(self.findings)))
        return f"redacted {len(self.findings)} item(s): {kinds}"


def redact_record(fields: dict) -> tuple[dict, RedactionSummary]:
    """Remove structured identifiers from every field derived from speech.

    Returns a new dict; the caller decides what to do with the summary. The
    regex baseline catches SSNs, MRNs, emails, phone numbers and dates of
    birth. It does not catch a patient's name said out loud — that needs the
    clinical NER hook `phi_redaction.py` already exposes, and until one is
    deployed the meeting page has to say so rather than imply the transcript is
    clean.
    """
    summary = RedactionSummary()
    if not redaction_enabled():
        return dict(fields), summary

    out = dict(fields)
    for key in REDACTABLE_TEXT:
        value = out.get(key)
        if isinstance(value, str) and value:
            result = redact_phi(value)
            if result.redacted:
                summary.applied = True
                summary.findings += result.findings
                out[key] = result.text
    for key in REDACTABLE_LISTS:
        values = out.get(key)
        if isinstance(values, list) and values:
            cleaned = []
            for item in values:
                if not isinstance(item, str):
                    cleaned.append(item)
                    continue
                result = redact_phi(item)
                if result.redacted:
                    summary.applied = True
                    summary.findings += result.findings
                cleaned.append(result.text)
            out[key] = cleaned
    return out, summary


@dataclass(frozen=True)
class Consent:
    """What the person who pressed Record says about the room.

    Deliberately a claim, not a proof. It records who asserted it and when, so
    that a later question about a meeting has somebody's name against it.
    """

    confirmed: bool
    by: str
    by_name: str
    at: str
    participants: list[str] = field(default_factory=list)

    def to_record(self) -> dict:
        return {
            "confirmed": self.confirmed,
            "by": self.by,
            "byName": self.by_name,
            "at": self.at,
            "participants": list(self.participants),
            "basis": "asserted by the person recording; not independently verified",
        }


def capture_consent(user: dict, confirmed: bool, participants: list[str] | None = None) -> Consent:
    return Consent(
        confirmed=bool(confirmed),
        by=user.get("id", ""),
        by_name=user.get("name", ""),
        at=datetime.now(timezone.utc).isoformat(),
        participants=[p.strip() for p in (participants or []) if p.strip()],
    )


def consent_required() -> bool:
    """Whether a recording may start without somebody confirming the room knows.

    Off by default: turning it on is a policy decision for a deployment, and a
    hard block that surprises somebody mid-meeting is worse than a warning.
    """
    return os.environ.get("MEETING_REQUIRE_CONSENT", "false").lower() == "true"


def groups_for(project_id: str | None, explicit: list[str] | None = None) -> list[str]:
    """Which access groups a new meeting should carry.

    A meeting recorded into a project belongs to that project's group, which is
    the behaviour people expect: a meeting about a programme is readable by the
    people on that programme. Everything else falls back to the deployment
    default.
    """
    if explicit:
        return [group.strip() for group in explicit if group.strip()]
    if project_id:
        return [f"project:{project_id}"]
    return default_groups()


def may_read(meeting: dict, user_groups: set[str]) -> bool:
    """Whether a user in these groups may read this meeting.

    A meeting with no groups recorded is readable by everyone. That is on
    purpose: meetings captured before access control existed must not become
    invisible, and silently hiding somebody's own meeting is a worse failure
    than showing one they could already see yesterday. A deployment that wants
    the strict reading sets MEETING_ACCESS_GROUPS and backfills.
    """
    groups = set(meeting.get("accessGroups") or [])
    if not groups or GENERAL in groups:
        return True
    return bool(groups & user_groups)
