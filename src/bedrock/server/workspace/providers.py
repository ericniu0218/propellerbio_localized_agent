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


def context_blocks(context):
    """A run's files and memories as material for the agents, kept apart from
    the question (bedrock.server.orchestrator.context) instead of pasted into
    it: the Runner classifies the prompt, only the drafting agent reads these,
    fenced as data and cut to fit its window."""
    from bedrock.server.schemas.api import ContextBlock

    blocks = [
        ContextBlock(id=a["id"], title=a["name"], kind="artifact",
                     text=(a.get("profile", "") + "\n" + a.get("text", "")).strip())
        for a in context["artifacts"]
    ]
    blocks += [
        ContextBlock(id=m["id"], title=f"User-approved context: {m['title']}", kind="memory", text=m["content"][:2000])
        for m in context["memory"]
    ]
    return blocks


def history_turns(messages):
    """The chat's own messages as conversation history for the agents."""
    from bedrock.server.schemas.api import HistoryTurn

    return [
        HistoryTurn(turn_id=m.get("id", ""), role="user" if m["role"] == "user" else "assistant",
                    content=m.get("text", ""), created_at="")
        for m in messages
        if m.get("text")
    ]


class LocalLLMProvider:
    async def execute(self, store, owner, run, context):
        from bedrock.server.events import get_event_bus
        from bedrock.server.orchestrator.session import start_session
        from bedrock.server.orchestrator.loop import handle_message
        from bedrock.server.task_store.store import get_task_store
        # One orchestrator session per run, never shared between a chat's two
        # lanes (a quick reply and an agent can run at once): events are read
        # per session, and a shared one would credit one lane's answer to the
        # other. Continuity comes from passing the chat's messages as history.
        session = await start_session(owner)
        bus = get_event_bus()
        queue = bus.subscribe(session)
        task = asyncio.create_task(handle_message(
            session, owner, run["prompt"], force_tier=ModelRouter.tier(run["mode"]),
            context=context_blocks(context), history=history_turns(context["history"]),
        ))
        run.update(sessionId=session, progressKind="unknown", currentStep="Contacting local agents")
        try:
            while not task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), .3)
                except asyncio.TimeoutError:
                    continue
                if event.type.value in ("agent.token", "usage.updated"):
                    # Nothing on the page changes per token; saving the run
                    # for each one rewrote it hundreds of times per answer.
                    continue
                p = event.payload
                if event.type.value == "agent.started":
                    run["role"] = {"director": "Planner", "specialist": "Researcher", "runner": "Assistant"}.get(p["agent"]["role"], "Assistant")
                    run["currentStep"] = p["agent"].get("current_task_title") or "Working"
                elif event.type.value == "tool.called":
                    run["currentStep"] = "Using " + p["tool_name"]
                elif event.type.value == "approval.required":
                    run.update(status="waiting", approvalTaskId=p["task_id"], preview=p.get("preview", ""), currentStep="Waiting for your review")
                elif event.type.value == "task.failed":
                    run.update(status="failed", error=p["error"])
                elif event.type.value == "task.completed":
                    run["answer"] = get_task_store().get_result(p["result_ref"]) or "No result was produced."
                store.put(owner, "runs", run)
            await task
            if run["status"] != "failed":
                run.update(status="completed", currentStep="Result ready", finishedAt=now())
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            bus.unsubscribe(session, queue)
