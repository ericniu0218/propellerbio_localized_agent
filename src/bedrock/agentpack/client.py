"""Client for the biotech agent-pack backend's REST + event-stream API.

Unlike ``inference/openai_compatible.py`` this is *not* the generic
OpenAI chat-completions wire format — the agent pack has real sessions,
multiple agents working a request, human-approval gates, and a live event
stream, so it needs its own small client rather than shoehorning that into
the single-shot ``stream_chat`` shape. See ``bedrock.agentpack_app`` for
how the UI uses this.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from bedrock.agentpack.models import AgentEvent
from bedrock.inference.base import InferenceError


class AgentPackClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # ``transport`` lets tests substitute a fake server, same pattern as
        # ``OpenAICompatibleBackend``.
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

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
        """
        try:
            async with self._client.stream("GET", f"/sessions/{session_id}/events") as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise InferenceError(f"Could not connect to the live updates: {body[:200]}")

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
        except httpx.HTTPError as exc:
            raise InferenceError(f"Lost the live connection: {_describe(exc)}") from exc

    async def close(self) -> None:
        await self._client.aclose()


def _parse_event(event_type: str, data: str) -> AgentEvent | None:
    """`data` is the backend's full `Event` envelope as JSON — see
    `contracts/types.ts` `Event`: {event_id, type, session_id, timestamp,
    payload}. We only need `session_id` and the inner `payload` here."""
    try:
        envelope = json.loads(data)
    except json.JSONDecodeError:
        return None
    return AgentEvent(
        type=event_type,
        session_id=envelope.get("session_id", ""),
        payload=envelope.get("payload", {}),
    )


def _describe(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "connection refused. Is the AI system running?"
    if isinstance(exc, httpx.TimeoutException):
        return "the server took too long to respond."
    return str(exc) or exc.__class__.__name__
