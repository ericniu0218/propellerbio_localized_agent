"""The workspace's Research tab: the literature review behind /workspace/api/research.

These drive the real routes through FastAPI's test client, with the review's
settings pointed at a temporary directory and its model server replaced by the
scripted one `tests/litreview/test_service.py` uses. The papers come from the
fixture corpus through the same searcher seam the real literature search uses,
so nothing here touches the network.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bedrock.config import PROFILES
from bedrock.inference import Scheduler
from bedrock.server.main import app
from bedrock.server.research import routes as research
from tests.litreview.fixtures import QUESTION
from tests.litreview.test_service import ScriptedBackend, _corpus_file, models_yaml, settings  # noqa: F401 - fixtures

COOKIE = "bedrock_workspace"
STATIC = Path(research.__file__).resolve().parent.parent / "static" / "workspace"


@pytest.fixture(autouse=True)
def workspace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DATA_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("DEMO_MODE", "false")
    research._status_cache.update(at=0.0, body=None)
    return tmp_path / "workspace"


@pytest.fixture
def review_env(settings, tmp_path, monkeypatch):  # noqa: F811 - the fixture above
    """Reviews read the fixture corpus and talk to a scripted model server."""
    from bedrock.litreview.sources import literature
    from bedrock.litreview.sources.corpus import CorpusSearcher

    settings.integrity_checks = "manual"  # the automatic retraction check goes to PubMed
    corpus = _corpus_file(tmp_path)
    backend = ScriptedBackend()
    scheduler = Scheduler(backend, PROFILES["workstation"])

    async def fake_scheduler():
        return scheduler

    monkeypatch.setattr(research, "review_settings", lambda: settings)
    monkeypatch.setattr(research, "review_scheduler", fake_scheduler)
    monkeypatch.setattr(literature.LiteratureSearcher, "from_settings",
                        classmethod(lambda cls, s, **kw: CorpusSearcher(corpus)))
    return backend


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def login(client, name="Alice"):
    client.cookies.clear()
    assert client.post("/workspace/api/dev-login", json={"name": name}).status_code == 200


def wait_for(client, review_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        review = client.get(f"/workspace/api/research/runs/{review_id}").json()
        if review["status"] not in ("queued", "running"):
            return review
        time.sleep(0.1)
    raise AssertionError(f"review did not finish: {review}")


def test_the_research_tab_is_in_the_workspace_below_meetings():
    html = (STATIC / "index.html").read_text()
    assert html.index('data-view="meetings"') < html.index('data-view="research"') < html.index('data-view="knowledge"')
    js = (STATIC / "app.js").read_text()
    assert 'if (view === "research") return renderResearchView(body, force);' in js
    assert 'api("/research" + path' in js


def test_the_page_needs_a_signed_in_person(client):
    client.cookies.clear()
    assert client.get("/workspace/api/research/runs").status_code == 401
    assert client.post("/workspace/api/research/runs", json={"question": QUESTION}).status_code == 401


def test_status_says_when_the_model_is_ready(client, review_env):
    login(client)
    body = client.get("/workspace/api/research/status").json()
    assert body["available"] is True
    assert body["model"]["ready"] is True
    assert [s["name"] for s in body["sources"]] == ["europepmc", "biorxiv", "pubmed"]


def test_status_explains_a_model_server_that_lacks_the_models(client, review_env):
    review_env.models = ["something-else"]
    login(client)
    model = client.get("/workspace/api/research/status").json()["model"]
    assert model["ready"] is False
    assert "ollama pull one-model" in model["reason"]


def test_a_review_runs_end_to_end_with_the_model_and_writes_a_checked_report(client, review_env):
    login(client)
    started = client.post("/workspace/api/research/runs", json={"question": QUESTION})
    assert started.status_code == 200, started.text
    review = wait_for(client, started.json()["id"])

    assert review["status"] == "completed", review
    assert review["integrityPassed"] is True
    assert review["stats"]["evidence_units"] > 0
    assert review["stats"]["sentences_kept"] > 0
    assert review_env.calls, "the model server was used"

    html = client.get(f"/workspace/api/research/runs/{review['id']}/report?format=html")
    assert html.status_code == 200 and "<html" in html.text.lower() and '<base target="_blank">' in html.text
    md = client.get(f"/workspace/api/research/runs/{review['id']}/report?format=md&download=1")
    assert md.status_code == 200 and "attachment" in md.headers["content-disposition"]
    assert client.get(f"/workspace/api/research/runs/{review['id']}/report?format=json").json()["question"]

    listed = client.get("/workspace/api/research/runs").json()
    assert [r["id"] for r in listed] == [review["id"]]


def test_a_review_without_a_model_never_calls_the_server(client, review_env):
    login(client)
    started = client.post("/workspace/api/research/runs", json={"question": QUESTION, "useModel": False})
    review = wait_for(client, started.json()["id"])
    assert review["status"] == "completed", review
    assert review_env.calls == []
    assert review["stats"]["papers_collected"] > 0


def test_the_report_is_saved_to_a_project_so_its_chats_use_it(client, review_env):
    login(client)
    project = client.post("/workspace/api/records/projects", json={"data": {"name": "Psoriasis"}}).json()
    started = client.post("/workspace/api/research/runs", json={"question": QUESTION, "projectId": project["id"]})
    review = wait_for(client, started.json()["id"])

    saved = client.post(f"/workspace/api/research/runs/{review['id']}/save", json={"projectId": project["id"]})
    assert saved.status_code == 200, saved.text
    artifact = saved.json()
    assert artifact["projectId"] == project["id"] and artifact["name"].endswith(".md")

    state = client.get("/workspace/api/state").json()
    assert artifact["id"] in {a["id"] for a in state["artifacts"]}

    from bedrock.server.workspace.routes import build_context, store

    chat = client.post("/workspace/api/records/chats", json={"data": {"title": "t", "projectId": project["id"], "messages": [], "artifactIds": []}}).json()
    s = store()
    owner = client.get("/workspace/api/state").json()["user"]["id"]
    context = build_context(s, owner, s.get(owner, chat["id"], "chats"))
    assert any(a["id"] == artifact["id"] and "psoriasis" in a["text"].lower() for a in context["artifacts"])


def test_a_report_can_start_a_chat(client, review_env):
    login(client)
    review = wait_for(client, client.post("/workspace/api/research/runs", json={"question": QUESTION}).json()["id"])
    chat = client.post("/workspace/api/records/chats", json={"data": {"title": "t", "messages": [], "artifactIds": []}}).json()
    artifact = client.post(f"/workspace/api/research/runs/{review['id']}/save", json={"chatId": chat["id"]}).json()
    chats = client.get("/workspace/api/state").json()["chats"]
    assert artifact["id"] in next(c for c in chats if c["id"] == chat["id"])["artifactIds"]


def test_people_see_only_their_own_reviews(client, review_env):
    login(client, "Alice")
    review = wait_for(client, client.post("/workspace/api/research/runs", json={"question": QUESTION}).json()["id"])
    login(client, "Bob")
    assert client.get("/workspace/api/research/runs").json() == []
    assert client.get(f"/workspace/api/research/runs/{review['id']}").status_code == 404
    assert client.get(f"/workspace/api/research/runs/{review['id']}/report").status_code == 404


def test_an_air_gapped_server_refuses_to_search(client, review_env, monkeypatch):
    monkeypatch.setattr(research, "_airgapped", lambda: True)
    login(client)
    refused = client.post("/workspace/api/research/runs", json={"question": QUESTION, "search": True})
    assert refused.status_code == 409 and "air-gapped" in refused.json()["detail"]
    assert client.get("/workspace/api/research/status").json()["searchAvailable"] is False


def test_a_question_with_no_papers_and_no_search_says_what_to_do(client, review_env):
    login(client)
    started = client.post("/workspace/api/research/runs", json={"question": QUESTION, "search": False})
    review = wait_for(client, started.json()["id"])
    assert review["status"] == "failed"
    assert "Search the literature" in review["error"]


def test_a_review_that_stops_partway_can_be_resumed(client, review_env):
    review_env.fail_after = 3  # the model server goes away mid-run
    login(client)
    review = wait_for(client, client.post("/workspace/api/research/runs", json={"question": QUESTION}).json()["id"])
    assert review["status"] == "stopped", review
    assert review["canResume"] is True

    review_env.fail_after = None
    resumed = client.post(f"/workspace/api/research/runs/{review['id']}/resume")
    assert resumed.status_code == 200, resumed.text
    finished = wait_for(client, review["id"])
    assert finished["status"] == "completed", finished


def test_removing_a_review_only_removes_it_from_the_list(client, review_env):
    login(client)
    review = wait_for(client, client.post("/workspace/api/research/runs", json={"question": QUESTION}).json()["id"])
    assert client.delete(f"/workspace/api/research/runs/{review['id']}").json() == {"ok": True}
    assert client.get("/workspace/api/research/runs").json() == []
