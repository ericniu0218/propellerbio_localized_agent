"""AI provider boundary. Production never falls back to a simulated answer."""
import asyncio
from typing import Protocol

from bedrock.server.workspace.store import now


class AIProvider(Protocol):
    async def execute(self, store, owner, run, context): ...


class ModelRouter:
    @staticmethod
    def tier(mode):
        return {"research": "director", "agent": "director", "analyze": "specialist"}.get(mode)


class MockAIProvider:
    async def execute(self, store, owner, run, context):
        steps = ["Read attached sources", "Apply project context", "Prepare a source-linked brief", "Save result"]
        run.update(steps=[{"title": s, "status": "queued"} for s in steps], progressKind="steps")
        for i, title in enumerate(steps):
            run.update(status="running", currentStep=title, role=["Assistant", "Researcher", "Planner", "Assistant"][i])
            run["steps"][i]["status"] = "running"
            store.put(owner, "runs", run)
            await asyncio.sleep(1.2)
            run["steps"][i]["status"] = "done"
        sources = context["artifacts"]
        parts = ["DEMO ANALYSIS — no language model was called.", "", "Bottom line"]
        if sources:
            parts += [f"I prepared a file profile from {len(sources)} attached source(s). The figures below are computed from the files; a real model is needed for an interpretation of your question.", "", "Key findings / key numbers"]
            for artifact in sources:
                parts += [f"{artifact['name']} [source:{artifact['id']}]", artifact.get("profile", "No numeric profile available.")]
        else:
            parts += ["Attach a file to get its computed profile, or explore the fictional project and meeting examples."]
        parts += ["", "Open questions / risks", "This demonstration does not establish scientific, clinical, or safety conclusions."]
        if context["memory"]:
            parts += ["", "Project context used"] + [m["title"] for m in context["memory"]]
        run.update(answer="\n".join(parts), citations=[a["id"] for a in sources], status="completed", currentStep="Result ready", finishedAt=now())


def context_blocks(context, prompt: str | None = None):
    """A run's files and memories as material for the agents, kept apart from
    the question (bedrock.server.orchestrator.context) instead of pasted into
    it: the Runner classifies the prompt, only the drafting agent reads these,
    fenced as data and cut to fit its window.

    A saved literature review is not cut from the end like other files: the
    sections that answer `prompt` are chosen, within the drafting agent's
    budget, and the block says which were included (`chat_lifecycle.report_context`)."""
    from bedrock.server.schemas.api import ContextBlock
    from bedrock.server.workspace.chat_lifecycle import is_report, report_context

    blocks = []
    for a in context["artifacts"]:
        text = (a.get("profile", "") + "\n" + a.get("text", "")).strip()
        title = a["name"]
        if is_report(a):
            text, shown, _ = report_context(a.get("text", ""), prompt or "", _report_budget_chars(len(context["artifacts"])))
            title = f"{a['name']} (sections: {', '.join(shown)})"
        blocks.append(ContextBlock(id=a["id"], title=title, kind="artifact", text=text))
    blocks += [
        ContextBlock(id=m["id"], title=f"User-approved context: {m['title']}", kind="memory", text=m["content"][:2000])
        for m in context["memory"]
    ]
    return blocks


def _report_budget_chars(artifacts: int) -> int:
    """A saved report's share of the drafting agent's context budget, in
    characters (about 3.5 per token; the orchestrator re-checks in tokens)."""
    from bedrock.server.orchestrator.context import context_budget

    return int(context_budget("specialist") * 3.5 / max(1, artifacts))


def history_turns(messages):
    """The chat's own messages as conversation history for the agents."""
    from bedrock.server.schemas.api import HistoryTurn

    return [
        HistoryTurn(turn_id=m.get("id", ""), role="user" if m["role"] == "user" else "assistant",
                    content=m.get("text", ""), created_at="")
        for m in messages
        if m.get("text")
    ]


STALL_SECONDS = 600.0
"""No event from the agents for this long, while not waiting for a person,
means the request is stalled: a model server that has gone away mid-stream
sends nothing at all. Tokens count as activity, so slow generation is not a
stall. `BEDROCK_CHAT_STALL_SECONDS` overrides it."""

DRAFT_ROLES = ("director", "specialist")
PROGRESS_SAVE_SECONDS = 5.0


class StalledRun(RuntimeError):
    pass


class LocalLLMProvider:
    async def execute(self, store, owner, run, context):
        import os
        import time

        from bedrock.server.events import get_event_bus
        from bedrock.server.orchestrator.session import start_session
        from bedrock.server.orchestrator.loop import handle_message
        from bedrock.server.task_store.store import get_task_store
        from bedrock.server.workspace.chat_lifecycle import EMPTY, explain_failure

        stall = float(os.environ.get("BEDROCK_CHAT_STALL_SECONDS") or STALL_SECONDS)
        # One orchestrator session per run, never shared between a chat's two
        # lanes (a quick reply and an agent can run at once): events are read
        # per session, and a shared one would credit one lane's answer to the
        # other. Continuity comes from passing the chat's messages as history.
        session = await start_session(owner)
        bus = get_event_bus()
        queue = bus.subscribe(session)
        task = asyncio.create_task(handle_message(
            session, owner, run["prompt"], force_tier=ModelRouter.tier(run["mode"]),
            context=context_blocks(context, run["prompt"]), history=history_turns(context["history"]),
        ))
        run.update(sessionId=session, progressKind="unknown", currentStep="Waiting for the model",
                   lastProgressAt=now(), partial="")
        store.put(owner, "runs", run)
        last_event = last_saved = time.monotonic()
        try:
            while not task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), .3)
                except asyncio.TimeoutError:
                    # A finished task is never a stall, however long ago its
                    # last event was; nor is a person taking time to review.
                    if not task.done() and run["status"] != "waiting" and time.monotonic() - last_event > stall:
                        raise StalledRun()
                    continue
                last_event = time.monotonic()
                p = event.payload
                if event.type.value in ("agent.token", "usage.updated"):
                    # What the drafting agent has written so far, kept so a
                    # stream that breaks is not lost; saved every few seconds,
                    # never per token.
                    if event.type.value == "agent.token" and p.get("role") in DRAFT_ROLES and p.get("text"):
                        run["partial"] = run.get("partial", "") + p["text"]
                        if run["status"] == "queued":
                            run.update(status="running", currentStep="Writing the answer")
                    if time.monotonic() - last_saved > PROGRESS_SAVE_SECONDS:
                        run["lastProgressAt"] = now()
                        store.put(owner, "runs", run)
                        last_saved = time.monotonic()
                    continue
                if event.type.value == "agent.started":
                    run["role"] = {"director": "Planner", "specialist": "Researcher", "runner": "Assistant"}.get(p["agent"]["role"], "Assistant")
                    run.update(status="running" if run["status"] == "queued" else run["status"],
                               currentStep=p["agent"].get("current_task_title") or "Working")
                    if p["agent"]["role"] in DRAFT_ROLES:
                        run["partial"] = ""  # a new draft starts; an earlier step's text is not the answer
                elif event.type.value == "tool.called":
                    run["currentStep"] = "Using " + p["tool_name"]
                elif event.type.value == "approval.required":
                    draft = get_task_store().get_result(p["result_ref"]) if p.get("result_ref") else None
                    run.update(status="waiting", approvalTaskId=p["task_id"], preview=p.get("preview", ""),
                               draft=draft or p.get("preview", ""), reviewReason=p.get("reason", ""),
                               currentStep="Waiting for your review")
                elif event.type.value == "task.failed":
                    kind, message = explain_failure(p.get("error"))
                    run.update(status="failed", error=message, errorKind=kind, errorDetail=str(p.get("error", ""))[:500])
                elif event.type.value == "task.completed":
                    run["answer"] = get_task_store().get_result(p["result_ref"]) or ""
                    if run["status"] == "waiting":
                        run["status"] = "running"  # approved: the answer is being posted
                run["lastProgressAt"] = now()
                store.put(owner, "runs", run)
                last_saved = time.monotonic()
            await task
            if run["status"] != "failed":
                if not (run.get("answer") or "").strip():
                    # Never post a placeholder as if it were the model's answer.
                    run.update(status="failed", errorKind=EMPTY[0], error=EMPTY[1])
                else:
                    run.update(status="completed", currentStep="Result ready", finishedAt=now())
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            bus.unsubscribe(session, queue)
