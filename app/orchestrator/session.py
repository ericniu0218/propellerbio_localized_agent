from __future__ import annotations

from uuid import uuid4

from app.db import get_database
from app.events import get_event_bus
from app.schemas.enums import EventType
from app.schemas.events import SessionStartedPayload


def new_session_id() -> str:
    return f"sess_{uuid4().hex[:12]}"


async def start_session(user_id: str) -> str:
    session_id = new_session_id()
    get_database().create_session(session_id, user_id)
    await get_event_bus().publish_typed(
        session_id, EventType.SESSION_STARTED, SessionStartedPayload(session_id=session_id, user_id=user_id)
    )
    return session_id
