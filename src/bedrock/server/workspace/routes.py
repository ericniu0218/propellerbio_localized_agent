"""Additive workspace API. Every record read is scoped to a server identity."""
import asyncio
import base64
import os
import re
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bedrock.server.config import get_settings
from bedrock.server.workspace.store import WorkspaceStore, demo_mode, now, uid
from bedrock.server.workspace.files import extract, profile, transcript_summary, SENSITIVE
from bedrock.server.workspace.providers import MockAIProvider, LocalLLMProvider, ModelRouter

router = APIRouter(prefix="/workspace/api", tags=["workspace"])
KINDS = {"chats", "projects", "memory", "meetings", "tasks", "preferences"}
jobs = {}
_chat_locks: dict[str, "asyncio.Lock"] = {}


def _chat_lock(chat_id):
    # A quick reply and a longer agent run can be in flight on the same chat
    # at once; this only serializes the read-append-write of chat.messages
    # itself, so neither one clobbers the other's appended message.
    lock = _chat_locks.get(chat_id)
    if lock is None:
        lock = _chat_locks[chat_id] = asyncio.Lock()
    return lock


def _run_category(mode):
    return "ask" if mode in {"ask", "auto"} else "agent"


def store():
    return WorkspaceStore()


def issue_session(response, owner, name):
    dbstore = store()
    token = uid() + uid()
    with dbstore.db() as db:
        db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (token, owner, name, time.time() + 8 * 3600))
    response.set_cookie("bedrock_workspace", token, httponly=True, samesite="lax", secure=os.getenv("APP_BASE_URL", "http://localhost").startswith("https:"), max_age=8 * 3600)


def identity(request: Request):
    with store().db() as db:
        row = db.execute("SELECT owner,name FROM sessions WHERE token=? AND expires>?",
            (request.cookies.get("bedrock_workspace", ""), time.time())).fetchone()
    if not row:
        raise HTTPException(401, "Please sign in to continue.")
    return {"id": row[0], "name": row[1]}


def require_object(s, owner, object_id, kind=None):
    obj = s.get(owner, object_id, kind)
    if obj is None:
        raise HTTPException(404, "This item is unavailable or you do not have access.")
    return obj


class RecordInput(BaseModel):
    data: dict


class UploadInput(BaseModel):
    name: str = Field(max_length=240)
    base64: str = Field(max_length=28_000_000)
    chatId: str | None = None
    projectId: str | None = None
    meeting: bool = False


class RunInput(BaseModel):
    chatId: str
    prompt: str = Field(min_length=1, max_length=20000)
    mode: Literal["auto", "ask", "research", "analyze", "agent"] = "auto"


class FollowupInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)


@router.get("/config")
def config():
    from bedrock.server.config import get_settings
    settings = get_settings()
    # Mirrors execute_run: a demo with a real model configured uses the model.
    if demo_mode() and settings.mode != "real":
        ai = "Demo provider"
    elif demo_mode():
        ai = "Local model (demo data)"
    elif settings.mode == "real":
        ai = "Local agents — real model backend"
    else:
        ai = "Local agents — mock model backend (deterministic, no live model)"
    zoom_connected = bool(os.getenv("DATACENTER_BASE_URL") and os.getenv("DATACENTER_SIGNING_SECRET"))
    connectors = [{"name": name, "status": "Not configured"} for name in ["Shared Drive", "Lab System", "Internal Database", "AlphaFold", "Semantic Search"]]
    connectors.insert(2, {"name": "Zoom", "status": "Connected" if zoom_connected else "Not configured"})
    return {"demo": demo_mode(), "sso": settings.sso_enabled, "ai": ai,
        "devLogin": not demo_mode() and not settings.sso_enabled,
        "auth": "Keycloak" if settings.sso_enabled else "Development identity",
        "connectors": connectors}


@router.post("/demo-login")
def demo_login(response: Response):
    if not demo_mode():
        raise HTTPException(403, "Development login is disabled.")
    issue_session(response, "demo-maya", "Maya Chen")
    store().seed("demo-maya")
    return {"ok": True}


class DevLoginInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@router.post("/dev-login")
def dev_login(body: DevLoginInput, response: Response):
    # A named identity for real (non-demo) mode before SSO is configured —
    # no fake data is seeded (WorkspaceStore.seed no-ops outside demo mode),
    # so this is a real, empty workspace for whoever signs in.
    from bedrock.server.config import get_settings
    if demo_mode():
        raise HTTPException(403, "Use the demo login while running in demo mode.")
    if get_settings().sso_enabled:
        raise HTTPException(403, "Single sign-on is configured; use the company login instead.")
    name = body.name.strip()
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "user"
    issue_session(response, "dev-" + slug, name)
    return {"ok": True}


@router.post("/logout")
def logout(request: Request, response: Response):
    with store().db() as db:
        db.execute("DELETE FROM sessions WHERE token=?", (request.cookies.get("bedrock_workspace", ""),))
    response.delete_cookie("bedrock_workspace")
    return {"ok": True}


@router.get("/state")
def state(user=Depends(identity)):
    s = store()
    s.seed(user["id"])
    result = {kind: s.list(user["id"], kind) for kind in KINDS | {"runs", "artifacts"}}
    # Text is fetched only for preview/context; keep metadata responses bounded.
    for a in result["artifacts"]:
        a.pop("text", None)
    # Runs saved before events stopped being stored may still carry up to 80;
    # the page never reads them, and this is polled every couple of seconds.
    for r in result["runs"]:
        r.pop("events", None)
        _reconcile(s, user["id"], r)
    result.update(user=user, audit=s.audit_list(user["id"]), config=config())
    return result


@router.post("/records/{kind}")
def create_record(kind: str, body: RecordInput, user=Depends(identity)):
    if kind not in KINDS:
        raise HTTPException(400, "Unknown item type.")
    s = store()
    data = dict(body.data)
    data.pop("id", None)
    if len(str(data)) > 300_000:
        raise HTTPException(413, "This record is too large. Upload a file instead.")
    if data.get("projectId"):
        require_object(s, user["id"], data["projectId"], "projects")
    if kind == "memory" and SENSITIVE.search(data.get("content", "")):
        raise HTTPException(422, "Potential sensitive information detected. Remove it before saving to memory.")
    result = s.put(user["id"], kind, data)
    s.audit(user["id"], "created " + kind, result["id"])
    return result


@router.patch("/records/{kind}/{object_id}")
def update_record(kind: str, object_id: str, body: RecordInput, user=Depends(identity)):
    if kind not in KINDS:
        raise HTTPException(400, "Unknown item type.")
    s = store()
    obj = require_object(s, user["id"], object_id, kind)
    patch = {k: v for k, v in body.data.items() if k not in {"id", "ownerId", "createdAt", "organizationId"}}
    if patch.get("projectId"):
        require_object(s, user["id"], patch["projectId"], "projects")
    if kind == "chats" and "artifactIds" in patch:
        for a in patch["artifactIds"]:
            require_object(s, user["id"], a, "artifacts")
    if kind == "memory" and SENSITIVE.search(patch.get("content", "")):
        raise HTTPException(422, "Remove sensitive information before saving to memory.")
    obj.update(patch)
    result = s.put(user["id"], kind, obj)
    s.audit(user["id"], "updated " + kind, object_id)
    return result


@router.delete("/records/{kind}/{object_id}")
def delete_record(kind: str, object_id: str, user=Depends(identity)):
    if kind not in KINDS:
        raise HTTPException(400, "Unknown item type.")
    s = store()
    require_object(s, user["id"], object_id, kind)
    if kind == "chats" and any(r["chatId"] == object_id and r["status"] in {"running", "waiting", "queued"} for r in s.list(user["id"], "runs")):
        raise HTTPException(409, "Stop this chat’s active task before deleting it.")
    s.delete(user["id"], object_id)
    s.audit(user["id"], "deleted " + kind, object_id)
    return {"ok": True}


@router.post("/uploads")
async def upload(body: UploadInput, user=Depends(identity)):
    s = store()
    chat = require_object(s, user["id"], body.chatId, "chats") if body.chatId else None
    if body.projectId:
        require_object(s, user["id"], body.projectId, "projects")
    try:
        raw = base64.b64decode(body.base64, validate=True)
        if len(raw) > 20_000_000:
            raise HTTPException(413, "Please split files larger than 20 MB for this workspace preview.")
        text, parse_status = await asyncio.to_thread(extract, body.name, raw)
        summary = await asyncio.to_thread(profile, body.name, text)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "We could not read this file. Try a text or CSV export.")
    object_id = uid()
    sensitive = bool(SENSITIVE.search(text))
    s.blobs.joinpath(object_id).write_bytes(raw)
    artifact = s.put(user["id"], "artifacts", {"id": object_id, "name": Path(body.name).name, "size": len(raw),
        "type": Path(body.name).suffix.lower(), "text": text[:300000], "truncated": len(text) > 300000,
        "profile": summary, "parseStatus": parse_status, "scanStatus": "review" if sensitive else "no obvious matches",
        "projectId": body.projectId, "chatId": body.chatId, "scope": "Project" if body.projectId else "Personal"})
    if chat:
        chat.setdefault("artifactIds", []).append(object_id)
        s.put(user["id"], "chats", chat)
    if body.meeting and text and not sensitive:
        s.put(user["id"], "meetings", {"title": Path(body.name).stem, "artifactId": object_id,
            "projectId": body.projectId, "transcript": text[:100000], **transcript_summary(text), "demo": False})
    s.audit(user["id"], "uploaded file", object_id)
    return {k: v for k, v in artifact.items() if k != "text"}


@router.get("/artifacts/{object_id}")
def artifact_preview(object_id: str, user=Depends(identity)):
    return require_object(store(), user["id"], object_id, "artifacts")


@router.get("/artifacts/{object_id}/download")
def download(object_id: str, user=Depends(identity)):
    s = store()
    artifact = require_object(s, user["id"], object_id, "artifacts")
    return FileResponse(s.blobs / object_id, filename=artifact["name"], media_type="application/octet-stream")


@router.post("/artifacts/{object_id}/redact")
def redact(object_id: str, user=Depends(identity)):
    s = store()
    artifact = require_object(s, user["id"], object_id, "artifacts")
    artifact["text"] = SENSITIVE.sub("[REDACTED]", artifact.get("text", ""))
    artifact["profile"] = profile(artifact["name"], artifact["text"])
    artifact["scanStatus"] = "redacted by rule; review still recommended"
    s.put(user["id"], "artifacts", artifact)
    s.audit(user["id"], "redacted context copy (original retained)", object_id)
    return {"ok": True}


def build_context(s, owner, chat):
    # In priority order, and each once: the chat's own attachments (a report
    # attached with "Ask about it in a chat" is what the question is about),
    # then the project's files, then the person's own library. Only the first
    # eight are sent, so the order decides which are left out.
    ordered = list(dict.fromkeys(chat.get("artifactIds", [])))
    everything = s.list(owner, "artifacts")
    if chat.get("projectId"):
        ordered += [a["id"] for a in everything if a.get("projectId") == chat["projectId"]]
    # Personal-library files (uploaded from the Memory tab: no chat, no
    # project) are available everywhere, the same way Personal-scope memory
    # already is below — otherwise a file saved to "your" memory is visible
    # to no chat at all.
    ordered += [a["id"] for a in everything if not a.get("chatId") and not a.get("projectId")]
    artifacts, missing = [], []
    for artifact_id in dict.fromkeys(ordered):
        artifact = s.get(owner, artifact_id, "artifacts")
        if artifact is None:
            # Deleted after it was attached. Said on the run, not a 404 that
            # reads as though the chat itself had gone.
            missing.append(artifact_id)
        elif artifact.get("scanStatus") != "review" and artifact.get("text"):
            artifacts.append(artifact)
    memory = [m for m in s.list(owner, "memory") if m.get("scope") == "Personal" or (m.get("scope") == "Project" and m.get("projectId") == chat.get("projectId") and chat.get("projectId"))]
    return {"artifacts": artifacts[:8], "memory": memory[:12], "history": chat.get("messages", [])[-40:],
            "missing": missing, "leftOut": [a["name"] for a in artifacts[8:]]}


def _context_note(context) -> str | None:
    notes = []
    if context.get("missing"):
        notes.append(f"{len(context['missing'])} attached file(s) are no longer in the workspace and were not used.")
    if context.get("leftOut"):
        notes.append("Only the first 8 files were used; left out: " + ", ".join(context["leftOut"][:5]) + ".")
    return " ".join(notes) or None


RECONCILE_AFTER_SECONDS = 30


def _reconcile(s, owner, run):
    """A run that says it is in progress, with no job behind it, was cut off by
    a server restart. Say so, instead of showing it as working for ever."""
    from bedrock.server.workspace.chat_lifecycle import RESTARTED

    from datetime import datetime, timezone

    if run.get("status") not in {"queued", "running", "waiting"} or run["id"] in jobs:
        return
    # /state runs in a worker thread: a run created a moment ago may not have
    # its job registered yet. Only one old enough to have one is an orphan.
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(run.get("createdAt", ""))).total_seconds()
    except ValueError:
        age = RECONCILE_AFTER_SECONDS
    if age >= RECONCILE_AFTER_SECONDS:
        run.update(status="stopped", errorKind=RESTARTED[0], error=RESTARTED[1], finishedAt=now())
        s.put(owner, "runs", run)


async def _post_partial(s, owner, run):
    """What the model had written before the request failed, posted as a
    clearly incomplete answer rather than thrown away."""
    partial = (run.get("partial") or "").strip()
    if not partial:
        return
    async with _chat_lock(run["chatId"]):
        chat = s.get(owner, run["chatId"], "chats")
        if chat is None:
            return
        chat.setdefault("messages", []).append({"id": uid(), "role": "assistant", "text": partial, "runId": run["id"],
                                                "incomplete": True, "demo": demo_mode()})
        s.put(owner, "chats", chat)


async def execute_run(owner, run, context):
    import logging

    from bedrock.inference import InferenceError
    from bedrock.server.workspace.chat_lifecycle import STALLED, explain_failure
    from bedrock.server.workspace.providers import StalledRun

    log = logging.getLogger("bedrock.server.workspace")
    s = store()
    try:
        # The real orchestrator handles AGENT_PACK_MODE=mock/real itself
        # (deterministic fixture model vs. a live one) — that's independent
        # of whether this falls back to the workspace's own canned
        # MockAIProvider text. demo-zoom.sh, for one, wants the real
        # orchestrator (tool calls, real citations) with AGENT_PACK_MODE=mock
        # for a deterministic fixture model client — not the canned text.
        # So: only fall back to canned text in a demo (DEMO_MODE=true) that
        # has no real model configured; AGENT_PACK_MODE=real always means a
        # real model is configured and should be used, demo or not.
        provider = MockAIProvider() if (demo_mode() and get_settings().mode != "real") else LocalLLMProvider()
        await provider.execute(s, owner, run, context)
        if run["status"] == "completed":
            async with _chat_lock(run["chatId"]):
                chat = s.get(owner, run["chatId"], "chats")
                if chat is not None:  # the chat may have been deleted while this ran
                    chat.setdefault("messages", []).append({"id": uid(), "role": "assistant", "text": run["answer"],
                        "runId": run["id"], "citations": run.get("citations", []), "demo": demo_mode()})
                    s.put(owner, "chats", chat)
        elif run["status"] == "failed":
            await _post_partial(s, owner, run)
    except asyncio.CancelledError:
        run.update(status="cancelled", currentStep="Stopped", error="Stopped. Your message has been kept; choose "
                   "Retry to ask again.", errorKind="cancelled", finishedAt=now())
        await _post_partial(s, owner, run)
    except StalledRun:
        minutes = round(float(os.environ.get("BEDROCK_CHAT_STALL_SECONDS") or 600) / 60, 1)
        run.update(status="failed", errorKind=STALLED[0], error=STALLED[1].format(minutes=f"{minutes:g}"),
                   finishedAt=now())
        await _post_partial(s, owner, run)
    except Exception as exc:
        # Kept for diagnosis without the question or any file text: the kind of
        # error and its message, which name a server or a model, not research.
        log.warning("chat run %s failed: %s: %s", run["id"], type(exc).__name__, str(exc)[:300])
        kind, message = explain_failure(str(exc), unavailable=isinstance(exc, InferenceError)
                                        and getattr(exc, "unavailable", False))
        run.update(status="failed", error=message, errorKind=kind, errorDetail=f"{type(exc).__name__}: {str(exc)[:300]}",
                   finishedAt=now())
        await _post_partial(s, owner, run)
    finally:
        run.setdefault("finishedAt", now())
        s.put(owner, "runs", run)
        s.audit(owner, "analysis " + run["status"], run["id"])
        jobs.pop(run["id"], None)


def _update_followup(owner, run_id, followup_id, **fields):
    s = store()
    run = s.get(owner, run_id, "runs")
    if not run:
        return
    for f in run.get("followups", []):
        if f["id"] == followup_id:
            f.update(fields)
    s.put(owner, "runs", run)


async def execute_followup(owner, run_id, followup_id, context, prompt, mode):
    try:
        if demo_mode():
            await asyncio.sleep(1.1)
            sources = context["artifacts"]
            parts = ["DEMO FOLLOW-UP — no language model was called."]
            parts.append(f"Referring back to the {len(sources)} source(s) already used in this analysis." if sources
                else "No sources are attached to this analysis to refer back to.")
            parts.append("This demonstration does not establish scientific, clinical, or safety conclusions.")
            _update_followup(owner, run_id, followup_id, answer="\n\n".join(parts), status="completed")
            return
        from bedrock.server.events import get_event_bus
        from bedrock.server.orchestrator.session import start_session
        from bedrock.server.orchestrator.loop import handle_message
        from bedrock.server.task_store.store import get_task_store
        from bedrock.server.workspace.providers import context_blocks, history_turns
        session = await start_session(owner)
        bus = get_event_bus()
        queue = bus.subscribe(session)
        # The analysis so far is the chat's history; its files go as material.
        task = asyncio.create_task(handle_message(
            session, owner, prompt, force_tier=ModelRouter.tier(mode),
            context=context_blocks(context, prompt), history=history_turns(context["history"]),
        ))
        from bedrock.server.workspace.chat_lifecycle import EMPTY, explain_failure

        answer, failed = None, None
        try:
            while not task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), .3)
                except asyncio.TimeoutError:
                    continue
                p = event.payload
                if event.type.value == "task.completed":
                    answer = get_task_store().get_result(p["result_ref"]) or ""
                elif event.type.value == "task.failed":
                    failed = explain_failure(p.get("error"))[1]
            await task
        finally:
            bus.unsubscribe(session, queue)
        if failed or not (answer or "").strip():
            _update_followup(owner, run_id, followup_id, answer=failed or EMPTY[1], status="failed")
        else:
            _update_followup(owner, run_id, followup_id, answer=answer, status="completed")
    except Exception as exc:
        _update_followup(owner, run_id, followup_id, answer=explain_failure(str(exc))[1], status="failed")


@router.post("/runs/{object_id}/followups")
async def create_followup(object_id: str, body: FollowupInput, user=Depends(identity)):
    s = store()
    run = require_object(s, user["id"], object_id, "runs")
    if run.get("status") not in {"completed", "failed"}:
        raise HTTPException(409, "Wait for this agent to finish before asking a follow-up.")
    if SENSITIVE.search(body.prompt):
        raise HTTPException(422, "Your question may contain sensitive identifiers. Remove them before sending in this preview.")
    chat = require_object(s, user["id"], run["chatId"], "chats")
    context = build_context(s, user["id"], chat)
    entry = {"id": uid(), "prompt": body.prompt, "answer": None, "status": "running", "createdAt": now()}
    run.setdefault("followups", []).append(entry)
    s.put(user["id"], "runs", run)
    s.audit(user["id"], "asked a follow-up", run["id"])
    asyncio.create_task(execute_followup(user["id"], run["id"], entry["id"], context, body.prompt, run.get("mode", "ask")))
    return entry


@router.post("/runs")
async def start_run(body: RunInput, user=Depends(identity)):
    s = store()
    chat = require_object(s, user["id"], body.chatId, "chats")
    if SENSITIVE.search(body.prompt):
        raise HTTPException(422, "Your request may contain sensitive identifiers. Remove them before sending in this preview.")
    category = _run_category(body.mode)
    async with _chat_lock(chat["id"]):
        # A quick reply and a longer agent task run in separate lanes, so
        # asking a question no longer has to wait for Create Agent to finish —
        # only two runs of the *same* lane on one chat are refused.
        if any(_run_category(r["mode"]) == category and r["status"] in {"queued", "running", "waiting"} for r in s.list(user["id"], "runs") if r["chatId"] == chat["id"]):
            raise HTTPException(409, "This chat already has a task like that running. Wait for it to finish, or start another chat to work in parallel.")
        context = build_context(s, user["id"], chat)
        run = s.put(user["id"], "runs", {"chatId": body.chatId, "prompt": body.prompt, "title": body.prompt[:70], "mode": body.mode,
            "projectId": chat.get("projectId"), "status": "queued", "currentStep": "Preparing request", "steps": [],
            "progressKind": "unknown", "role": "Planner" if body.mode in {"research", "agent"} else "Researcher", "demo": demo_mode(),
            "sourceIds": [a["id"] for a in context["artifacts"]], "contextNote": _context_note(context)})
        chat = require_object(s, user["id"], body.chatId, "chats")
        chat.setdefault("messages", []).append({"id": uid(), "role": "user", "text": body.prompt})
        if chat.get("title") == "New chat":
            chat["title"] = body.prompt[:54]
        s.put(user["id"], "chats", chat)
    s.audit(user["id"], "started analysis", run["id"])
    jobs[run["id"]] = asyncio.create_task(execute_run(user["id"], run, context))
    return run


@router.post("/runs/{object_id}/retry")
async def retry(object_id: str, user=Depends(identity)):
    """Ask the same question again. The question is already in the chat, so it
    is not added a second time; the old run records which run replaced it, so a
    double click cannot start two."""
    s = store()
    old = require_object(s, user["id"], object_id, "runs")
    if old.get("status") not in {"failed", "cancelled", "stopped"}:
        raise HTTPException(409, "Only a request that failed or was stopped can be retried.")
    chat = require_object(s, user["id"], old["chatId"], "chats")
    async with _chat_lock(chat["id"]):
        old = require_object(s, user["id"], object_id, "runs")
        if old.get("retriedBy"):
            raise HTTPException(409, "This request has already been retried.")
        category = _run_category(old["mode"])
        if any(_run_category(r["mode"]) == category and r["status"] in {"queued", "running", "waiting"}
               for r in s.list(user["id"], "runs") if r["chatId"] == chat["id"]):
            raise HTTPException(409, "This chat already has a request running.")
        context = build_context(s, user["id"], chat)
        # The question being retried is already the chat's last turn: it goes
        # as the prompt, not a second time as history.
        history = context["history"]
        if history and history[-1].get("role") == "user" and history[-1].get("text") == old["prompt"]:
            context["history"] = history[:-1]
        run = s.put(user["id"], "runs", {"chatId": chat["id"], "prompt": old["prompt"], "title": old.get("title"),
            "mode": old["mode"], "projectId": chat.get("projectId"), "status": "queued", "currentStep": "Preparing request",
            "steps": [], "progressKind": "unknown", "role": old.get("role"), "demo": demo_mode(), "retryOf": object_id,
            "sourceIds": [a["id"] for a in context["artifacts"]], "contextNote": _context_note(context)})
        old["retriedBy"] = run["id"]
        s.put(user["id"], "runs", old)
    s.audit(user["id"], "retried analysis", run["id"])
    jobs[run["id"]] = asyncio.create_task(execute_run(user["id"], run, context))
    return run


@router.post("/runs/{object_id}/cancel")
async def cancel(object_id: str, user=Depends(identity)):
    # async: Task.cancel must run on the event loop, not in FastAPI's thread pool.
    s = store()
    run = require_object(s, user["id"], object_id, "runs")
    if object_id in jobs:
        jobs[object_id].cancel()
    else:
        run.update(status="cancelled", currentStep="Stopped")
        s.put(user["id"], "runs", run)
    return {"ok": True}


class Decision(BaseModel):
    decision: Literal["approve", "reject"]


@router.post("/runs/{object_id}/decision")
async def decide(object_id: str, body: Decision, user=Depends(identity)):
    # async: resolving an asyncio future from a worker thread is not thread-safe.
    from bedrock.server.compliance.approvals import NotPending, UnknownTask, resolve_approval
    from bedrock.server.schemas.enums import ApprovalDecision
    s = store()
    run = require_object(s, user["id"], object_id, "runs")
    if run.get("status") != "waiting":
        raise HTTPException(409, "This review is no longer pending.")
    try:
        # The same signature record /approvals writes, naming the signed-in
        # person; it used to release the gate with no record at all.
        await resolve_approval(
            run.get("approvalTaskId", ""),
            ApprovalDecision(body.decision),
            signer=user["id"],
            signer_verified=get_settings().sso_enabled and not demo_mode(),
        )
    except (NotPending, UnknownTask):
        raise HTTPException(409, "This review is no longer pending.")
    s.audit(user["id"], "human " + body.decision, object_id)
    return {"ok": True}
