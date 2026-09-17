"""Additive workspace API. Every record read is scoped to a server identity."""
import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.workspace.store import WorkspaceStore, demo_mode, now, uid
from app.workspace.files import extract, profile, transcript_summary, SENSITIVE
from app.workspace.providers import MockAIProvider, LocalLLMProvider

router = APIRouter(prefix="/workspace/api", tags=["workspace"])
KINDS = {"chats", "projects", "memory", "meetings", "tasks", "preferences"}
jobs = {}


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


@router.get("/config")
def config():
    from app.config import get_settings
    settings = get_settings()
    return {"demo": demo_mode(), "sso": settings.sso_enabled,
        "ai": "Demo provider" if demo_mode() else "Local agents (availability checked when requested)",
        "auth": "Keycloak" if settings.sso_enabled else "Development identity",
        "connectors": [{"name": name, "status": "Not configured"} for name in ["Shared Drive", "Lab System", "Zoom", "Internal Database", "AlphaFold", "Semantic Search"]]}


@router.post("/demo-login")
def demo_login(response: Response):
    if not demo_mode():
        raise HTTPException(403, "Development login is disabled.")
    issue_session(response, "demo-maya", "Maya Chen")
    store().seed("demo-maya")
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
    ids = set(chat.get("artifactIds", []))
    if chat.get("projectId"):
        ids.update(a["id"] for a in s.list(owner, "artifacts") if a.get("projectId") == chat["projectId"])
    artifacts = [require_object(s, owner, a, "artifacts") for a in ids]
    artifacts = [a for a in artifacts if a.get("scanStatus") != "review" and a.get("text")]
    memory = [m for m in s.list(owner, "memory") if m.get("scope") == "Personal" or (m.get("scope") == "Project" and m.get("projectId") == chat.get("projectId") and chat.get("projectId"))]
    return {"artifacts": artifacts[:8], "memory": memory[:12], "history": chat.get("messages", [])[-8:]}


async def execute_run(owner, run, context):
    s = store()
    try:
        if demo_mode():
            provider = MockAIProvider()
        else:
            from app.config import get_settings
            if get_settings().mode != "real":
                raise RuntimeError("Real mode is selected but local model serving is not configured.")
            provider = LocalLLMProvider()
        await provider.execute(s, owner, run, context)
        if run["status"] == "completed":
            chat = require_object(s, owner, run["chatId"], "chats")
            chat.setdefault("messages", []).append({"id": uid(), "role": "assistant", "text": run.get("answer", "No answer returned."),
                "runId": run["id"], "citations": run.get("citations", []), "demo": demo_mode()})
            s.put(owner, "chats", chat)
    except asyncio.CancelledError:
        run.update(status="cancelled", currentStep="Stopped", finishedAt=now())
    except Exception as exc:
        run.update(status="failed", error="The analysis could not finish. Check the provider configuration and retry.", finishedAt=now())
    finally:
        s.put(owner, "runs", run)
        s.audit(owner, "analysis " + run["status"], run["id"])
        jobs.pop(run["id"], None)


@router.post("/runs")
async def start_run(body: RunInput, user=Depends(identity)):
    s = store()
    chat = require_object(s, user["id"], body.chatId, "chats")
    if any(r["chatId"] == chat["id"] and r["status"] in {"queued", "running", "waiting"} for r in s.list(user["id"], "runs")):
        raise HTTPException(409, "This chat is already running a task. Start another chat to work in parallel.")
    context = build_context(s, user["id"], chat)
    if SENSITIVE.search(body.prompt):
        raise HTTPException(422, "Your request may contain sensitive identifiers. Remove them before sending in this preview.")
    run = s.put(user["id"], "runs", {"chatId": body.chatId, "prompt": body.prompt, "title": body.prompt[:70], "mode": body.mode,
        "projectId": chat.get("projectId"), "status": "queued", "currentStep": "Preparing request", "steps": [],
        "progressKind": "unknown", "role": "Planner" if body.mode in {"research", "agent"} else "Researcher", "demo": demo_mode(),
        "sourceIds": [a["id"] for a in context["artifacts"]]})
    chat.setdefault("messages", []).append({"id": uid(), "role": "user", "text": body.prompt})
    if chat.get("title") == "New chat":
        chat["title"] = body.prompt[:54]
    s.put(user["id"], "chats", chat)
    s.audit(user["id"], "started analysis", run["id"])
    jobs[run["id"]] = asyncio.create_task(execute_run(user["id"], run, context))
    return run


@router.post("/runs/{object_id}/cancel")
def cancel(object_id: str, user=Depends(identity)):
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
def decide(object_id: str, body: Decision, user=Depends(identity)):
    from app.compliance.approval_gates import get_approval_gate
    from app.schemas.enums import ApprovalDecision
    s = store()
    run = require_object(s, user["id"], object_id, "runs")
    gate = get_approval_gate()
    if run.get("status") != "waiting" or not gate.is_pending(run.get("approvalTaskId", "")):
        raise HTTPException(409, "This review is no longer pending.")
    gate.resolve(run["approvalTaskId"], ApprovalDecision(body.decision))
    s.audit(user["id"], "human " + body.decision, object_id)
    return {"ok": True}
