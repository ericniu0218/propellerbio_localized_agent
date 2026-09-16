"""Client for the biotech agent-pack backend's REST + event-stream API.

Unlike ``inference/openai_compatible.py`` this is *not* the generic
OpenAI chat-completions wire format — the agent pack has real sessions,
multiple agents working a request, human-approval gates, and a live event
stream, so it needs its own small client rather than shoehorning that into
the single-shot ``stream_chat`` shape. See ``bedrock.agentpack_app`` for
how the UI uses this.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from typing import AsyncIterator

import httpx

from bedrock.agentpack.models import AgentEvent
from bedrock.inference.base import InferenceError

INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0
BACKOFF_MULTIPLIER = 2.0


class AgentPackClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        # ``transport`` lets tests substitute a fake server, same pattern as
        # ``OpenAICompatibleBackend``. ``sleep`` lets tests replace the
        # reconnect backoff delay without actually waiting — real callers
        # never pass it.
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )
        self._sleep = sleep or asyncio.sleep

    async def check_connection(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=5.0)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def create_session(self, user_id: str) -> str:
        try:
            response = await self._client.post("/sessions", json={"user_id": user_id})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InferenceError(f"Could not start a session: {_describe(exc)}") from exc
        return response.json()["session_id"]

    async def send_message(self, session_id: str, text: str) -> str:
        try:
            response = await self._client.post(
                "/messages", json={"session_id": session_id, "message": text}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InferenceError(f"Could not send your message: {_describe(exc)}") from exc
        return response.json()["task_id"]

    async def get_result(self, result_ref: str) -> str:
        try:
            response = await self._client.get(f"/results/{result_ref}")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InferenceError(f"Could not fetch the answer: {_describe(exc)}") from exc
        return response.json()["content"]

    async def approve(self, task_id: str, decision: str, approved_by: str, comment: str | None = None) -> None:
        try:
            response = await self._client.post(
                "/approvals",
                json={
                    "task_id": task_id,
                    "decision": decision,
                    "approved_by": approved_by,
                    "comment": comment,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InferenceError(f"Could not record your decision: {_describe(exc)}") from exc

    async def events(self, session_id: str) -> AsyncIterator[AgentEvent]:
        """Streams events for a session (`GET /sessions/{id}/events`, SSE).

        Runs until cancelled — callers drive this from a background worker
        for the lifetime of the session, same pattern as
        ``openai_compatible.OpenAICompatibleBackend.stream_chat``.

        Reconnects on a dropped connection with exponential backoff and
        jitter, sending back whatever `id:` the server last gave us as
        `Last-Event-ID` so the backend can resume rather than replay
        everything (see app/events/bus.py on the backend for the other
        half of this contract). A disconnect never means the underlying
        job stopped — the backend keeps running it regardless of whether
        anyone is listening — so reconnecting and catching up is always
        the right response, never restarting anything.

        Emits two synthetic event types of its own, never sent by the
        backend, so the UI can show a "reconnecting…" state:
        `client.reconnecting` (payload: `attempt`, `delay_s`) each time a
        drop is detected, and `client.reconnected` once a new connection
        is actually established afterward. A permanent-looking failure
        (a 4xx response — e.g. the session no longer exists) is not
        retried; it raises `InferenceError` same as before, since no
        amount of reconnecting fixes that.
        """
        last_event_id: str | None = None
        attempt = 0
        reconnecting = False

        while True:
            try:
                headers = {"Last-Event-ID": last_event_id} if last_event_id is not None else {}
                async with self._client.stream(
                    "GET", f"/sessions/{session_id}/events", headers=headers
                ) as response:
                    if response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            "server error", request=response.request, response=response
                        )
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(errors="replace")
                        raise InferenceError(f"Could not connect to the live updates: {body[:200]}")

                    if reconnecting:
                        reconnecting = False
                        attempt = 0
                        yield AgentEvent(type="client.reconnected", session_id=session_id, payload={})

                    event_type: str | None = None
                    data_lines: list[str] = []
                    async for line in response.aiter_lines():
                        if line == "":
                            if event_type is not None and data_lines:
                                event = _parse_event(event_type, "".join(data_lines))
                                if event is not None:
                                    yield event
                            event_type, data_lines = None, []
                            continue
                        if line.startswith("event:"):
                            event_type = line[len("event:"):].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:"):].strip())
                        elif line.startswith("id:"):
                            last_event_id = line[len("id:"):].strip()
                    # The server closing a normally-infinite stream (e.g. a
                    # restart) is a drop too, not successful completion —
                    # fall through to the reconnect path below.
            except InferenceError:
                raise  # permanent (4xx) — not retried
            except httpx.HTTPError:
                pass  # connection-level failure or 5xx — retry below

            attempt += 1
            reconnecting = True
            delay = min(MAX_BACKOFF_S, INITIAL_BACKOFF_S * (BACKOFF_MULTIPLIER ** (attempt - 1)))
            delay = delay / 2 + random.uniform(0, delay / 2)  # equal jitter
            yield AgentEvent(
                type="client.reconnecting",
                session_id=session_id,
                payload={"attempt": attempt, "delay_s": delay},
            )
            await self._sleep(delay)

    async def close(self) -> None:
        await self._client.aclose()


def _parse_event(event_type: str, data: str) -> AgentEvent | None:
    """`data` is the backend's full `Event` envelope as JSON — see
    `contracts/types.ts` `Event`: {event_id, seq, type, session_id,
    timestamp, payload}. We only need `session_id` and the inner `payload`
    here; `seq` is read separately, off the SSE `id:` line, since that's
    the field the transport-level id is meant to carry."""
    try:
        envelope = json.loads(data)
    except json.JSONDecodeError:
        return None
    return AgentEvent(
        type=event_type,
        session_id=envelope.get("session_id", ""),
        payload=envelope.get("payload", {}),
        event_id=str(envelope.get("event_id", "")),
    )


def _describe(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "connection refused. Is the AI system running?"
    if isinstance(exc, httpx.TimeoutException):
        return "the server took too long to respond."
    return str(exc) or exc.__class__.__name__
