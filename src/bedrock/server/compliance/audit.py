"""Audit sink: every event on the bus is written to the append-only audit_log
table as it happens. Registered on the EventBus in bedrock.server.main at
startup. See docs/server/ARCHITECTURE.md section 8.

Three things it does beyond copying events into rows:

* **One row per model reply, not per token.** A streamed reply arrives as
  hundreds of ``agent.token`` events. Each used to be its own row - a SELECT and
  an INSERT+COMMIT per token, on the event loop, holding every other request up
  for the length of an answer. Tokens are now gathered per (session, agent,
  task) and written once, when the reply is complete, as an ``agent.output`` row
  carrying the whole text, its length and its SHA-256. Nothing is lost: the
  same text, one row. If a run ends with a reply still open (a failure,
  a cancel), what arrived is written, marked ``incomplete``.
* **PHI is redacted before it is written.** Every string in a row's ``detail``
  goes through ``phi_redaction.redact_phi``; ``detail["redacted"]`` lists what
  was found. Identity fields are the exception (``IDENTITY_FIELDS``): the
  approver named in a signature record is the record - an email address there
  is who signed, not patient data. The architecture document always said redaction ran before
  anything was logged; until now nothing did. The results and history tables
  keep the text as it was - raw PHI stays scoped to the task that needed it.
* **Off the event loop.** The database write runs in a worker thread, awaited
  in turn, so one run's rows keep their order. Each row also records its
  event's ``seq`` (``detail["event_seq"]``), the session's own ordering, for
  when rows from concurrent runs commit a moment apart.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict

from bedrock.server.compliance.phi_redaction import redact_phi
from bedrock.server.db import get_database
from bedrock.server.schemas.enums import AgentRole, EventType
from bedrock.server.schemas.events import Event

AGENT_OUTPUT = "agent.output"
"""Audit-only action: one model reply, assembled from its agent.token events."""

_TERMINAL = (EventType.TASK_COMPLETED, EventType.TASK_FAILED)
IDENTITY_FIELDS = frozenset({
    "approved_by", "user_id", "owner", "session_id", "task_id", "agent_id",
    "call_id", "result_ref", "event_id",
})
_open_replies: dict[tuple[str, str, str | None], list[str]] = {}
_session_users: OrderedDict[str, str] = OrderedDict()
_SESSION_USER_CACHE = 1000


def reset() -> None:
    """Forget open replies and cached users (tests; see bedrock.server.state)."""
    _open_replies.clear()
    _session_users.clear()


def _agent_from_payload(payload: dict) -> AgentRole | None:
    role = payload.get("role") or payload.get("reviewed_by") or payload.get("created_by")
    if role in (r.value for r in AgentRole):
        return AgentRole(role)
    return None


def _redacted(value, found: list[str]):
    if isinstance(value, str):
        result = redact_phi(value)
        found.extend(result.findings)
        return result.text
    if isinstance(value, dict):
        return {k: v if k in IDENTITY_FIELDS else _redacted(v, found) for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v, found) for v in value]
    return value


async def _user_for(session_id: str) -> str:
    # The cache is only touched here, on the event loop; the lookup itself runs
    # in a worker thread.
    user = _session_users.get(session_id)
    if user is None:
        user = await asyncio.to_thread(get_database().get_session_user, session_id)
        _session_users[session_id] = user
        if len(_session_users) > _SESSION_USER_CACHE:
            _session_users.popitem(last=False)
    return user


async def _write(session_id: str, action: str, detail: dict, *, task_id=None, agent=None, model_name=None) -> None:
    found: list[str] = []
    detail = _redacted(detail, found)
    if found:
        detail["redacted"] = sorted(set(found))
    user_id = await _user_for(session_id)
    await asyncio.to_thread(
        get_database().write_audit,
        session_id=session_id,
        user_id=user_id,
        action=action,
        detail=detail,
        task_id=task_id,
        agent=agent,
        model_name=model_name,
    )


async def _write_reply(session_id: str, key: tuple[str, str, str | None], role: AgentRole | None, *, incomplete: bool) -> None:
    text = "".join(_open_replies.pop(key, []))
    agent_id, task_id = key[1], key[2]
    detail = {
        "agent_id": agent_id,
        "role": role.value if role else None,
        "task_id": task_id,
        "text": text,
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    if incomplete:
        detail["incomplete"] = True
    await _write(session_id, AGENT_OUTPUT, detail, task_id=task_id, agent=role)


async def audit_event_sink(event: Event) -> None:
    payload = event.payload
    if event.type == EventType.AGENT_TOKEN:
        key = (event.session_id, payload["agent_id"], payload.get("task_id"))
        _open_replies.setdefault(key, []).append(payload.get("text", ""))
        if payload.get("done"):
            await _write_reply(event.session_id, key, _agent_from_payload(payload), incomplete=False)
        return

    await _write(
        event.session_id,
        event.type.value,
        dict(payload) | {"event_seq": event.seq},
        task_id=payload.get("task_id"),
        agent=_agent_from_payload(payload),
        model_name=payload.get("model_name"),
    )
    if event.type in _TERMINAL:
        for key in [k for k in _open_replies if k[0] == event.session_id]:
            await _write_reply(event.session_id, key, None, incomplete=True)
