"""The workspace's Research tab: the literature review behind /workspace/api/research.

These drive the real routes through FastAPI's test client, with the review's
settings pointed at a temporary directory and its model server replaced by the
scripted one `tests/litreview/test_service.py` uses. The papers come from the
fixture corpus through the same searcher seam the real literature search uses,
so nothing here touches the network.
"""

from __future__ import annotations

import re
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
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
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
def no_figure_downloads(monkeypatch):
    """A deck review may fetch figure images; these tests never reach the network."""
    from bedrock.litreview import service

    def keep_as_chosen(figures, *args, **kwargs):
        return figures

    monkeypatch.setattr(service.figure_files, "download_figures", keep_as_chosen)


@pytest.fixture
def requests_seen(monkeypatch):
    """The ReviewRequest each started or resumed review was given."""
    seen = []
    original = research._start

    def spy(owner, review, request):
        seen.append(request)
        return original(owner, review, request)

    monkeypatch.setattr(research, "_start", spy)
    return seen


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
    started = client.post("/workspace/api/research/runs", json={"question": QUESTION, "deck": True})
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
    deck = client.get(f"/workspace/api/research/runs/{review['id']}/report?format=pptx&download=1")
    assert deck.status_code == 200 and deck.content[:2] == b"PK" and ".pptx" in deck.headers["content-disposition"]
    assert deck.headers["content-type"] == PPTX

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


def test_the_powerpoint_option_reaches_the_review_and_is_remembered(client, review_env, no_figure_downloads, requests_seen):
    login(client)
    started = client.post("/workspace/api/research/runs", json={"question": QUESTION, "deck": True})
    assert started.status_code == 200, started.text
    assert requests_seen[-1].deck is True and started.json()["deck"] is True
    review = wait_for(client, started.json()["id"])
    assert review["status"] == "completed" and review["deck"] is True

    plain = client.post("/workspace/api/research/runs", json={"question": QUESTION}).json()
    assert requests_seen[-1].deck is False and plain["deck"] is False
    wait_for(client, plain["id"])  # a review left running would stop the next test starting one


def test_a_review_started_without_the_powerpoint_option_has_no_deck_to_download(client, review_env):
    login(client)
    review = wait_for(client, client.post("/workspace/api/research/runs", json={"question": QUESTION}).json()["id"])
    assert review["status"] == "completed" and review["deck"] is False

    refused = client.get(f"/workspace/api/research/runs/{review['id']}/report?format=pptx&download=1")
    assert refused.status_code == 404
    assert refused.json()["detail"] == "This review was started without the PowerPoint option."
    assert client.get(f"/workspace/api/research/runs/{review['id']}/report?format=html").status_code == 200


def test_the_deck_is_drawn_from_the_run_figures_and_the_signed_in_name(client, review_env, no_figure_downloads, monkeypatch):
    from bedrock.litreview.report import slides

    login(client, "Alice")
    review = wait_for(client, client.post("/workspace/api/research/runs", json={"question": QUESTION, "deck": True}).json()["id"])
    assert review["status"] == "completed", review

    drawn = {}
    original = slides.to_pptx

    def spy(report, source="", **kwargs):
        drawn.update(kwargs)
        return original(report, source, **kwargs)

    monkeypatch.setattr(slides, "to_pptx", spy)
    deck = client.get(f"/workspace/api/research/runs/{review['id']}/report?format=pptx")
    assert deck.status_code == 200 and deck.content[:2] == b"PK"
    assert deck.headers["content-type"] == PPTX
    assert drawn["author"] == "Alice"
    assert re.fullmatch(r"\d{1,2} [A-Z][a-z]+ \d{4}", drawn["date"])
    assert isinstance(drawn["figures"], list)


def test_a_review_saved_before_the_option_existed_still_has_its_deck(client, review_env):
    from bedrock.server.workspace.routes import store

    login(client)
    review = wait_for(client, client.post("/workspace/api/research/runs", json={"question": QUESTION}).json()["id"])
    owner = client.get("/workspace/api/state").json()["user"]["id"]
    record = store().get(owner, review["id"], research.KIND)
    del record["deck"]
    store().put(owner, research.KIND, record)

    old = client.get(f"/workspace/api/research/runs/{review['id']}")
    assert old.status_code == 200 and "deck" not in old.json()
    deck = client.get(f"/workspace/api/research/runs/{review['id']}/report?format=pptx&download=1")
    assert deck.status_code == 200 and deck.content[:2] == b"PK" and deck.headers["content-type"] == PPTX


def test_resuming_a_review_keeps_the_powerpoint_option(client, review_env, no_figure_downloads, requests_seen):
    review_env.fail_after = 3
    login(client)
    review = wait_for(client, client.post("/workspace/api/research/runs", json={"question": QUESTION, "deck": True}).json()["id"])
    assert review["status"] == "stopped" and review["deck"] is True

    review_env.fail_after = None
    resumed = client.post(f"/workspace/api/research/runs/{review['id']}/resume")
    assert resumed.status_code == 200, resumed.text
    assert requests_seen[-1].deck is True and resumed.json()["deck"] is True
    finished = wait_for(client, review["id"])
    assert finished["status"] == "completed" and finished["deck"] is True
    assert client.get(f"/workspace/api/research/runs/{review['id']}/report?format=pptx").status_code == 200


def test_the_research_form_offers_the_powerpoint_and_hides_the_button_when_it_was_not_asked_for():
    js = (STATIC / "app.js").read_text()
    assert "Also make a target-proposal PowerPoint" in js and "deck: deckIn.checked" in js
    assert "d.deck !== false ? el(" in js and "format=pptx" in js
