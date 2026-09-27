"""The determinism harness cannot pass by accident.

Fresh mode must see every request reach the server: a client that cannot
report its wire calls is refused, and a reply that arrived without a call
fails the cell. A server that answers differently is recorded as diverging,
with every distinct reply kept. The recorded-output control and cached replay
are separate rows. The "server" here is `httpx.MockTransport` behind the real
backend, scheduler and client - no network, no model.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path

import httpx
import pytest

from bedrock.config import PROFILES
from bedrock.inference import Scheduler
from bedrock.inference.openai_compatible import OpenAICompatibleBackend
from bedrock.litreview.bench.determinism import (
    MANIFEST_KEYS,
    FreshModeUnavailable,
    RecordingClient,
    build_payloads,
    load_payloads,
    replay_control,
    run_cell,
)
from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.llm import FakeLLMClient, SchedulerClient
from bedrock.litreview.llm.cache import CachedClient, CallCache
from bedrock.litreview.runner import answer_question
from tests.litreview.fixtures import QUESTION, corpus, fake_client

PINNED = Path(__file__).parents[2] / "evals" / "litreview" / "determinism" / "payloads.jsonl"
MODELS = {"relevance": "m", "extract": "m"}


def sse(text: str) -> httpx.Response:
    body = f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\ndata: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def server(replies):
    """A mock model server: `/models` lists one model; completions come from `replies`."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if request.url.path == "/api/tags":
            return httpx.Response(404)
        return sse(next(replies))
    return handler


def payloads(n=3):
    return [{"id": f"p{i}", "sha256": f"{i}", "job": "relevance", "prompt": f"prompt {i}", "schema": None,
             "max_tokens": 16} for i in range(n)]


async def cell(tmp_path, replies, *, repeats=3, condition="c", n=3):
    backend = OpenAICompatibleBackend("http://gpu:8000/v1", transport=httpx.MockTransport(server(replies)))
    scheduler = Scheduler(backend, PROFILES["workstation"])
    loop = asyncio.get_running_loop()
    return await asyncio.to_thread(
        run_cell, payloads(n), lambda rep: SchedulerClient(scheduler, loop, context_window=12288),
        models=MODELS, repeats=repeats, mode="fresh", condition=condition, out_dir=tmp_path)


async def test_a_server_that_repeats_itself_passes_with_every_call_observed(tmp_path):
    result = await cell(tmp_path, itertools.repeat("same reply"))
    assert result.repeatable and result.attempts == 9 and result.failures == 0
    records = [json.loads(line) for line in (tmp_path / "c" / "attempts.jsonl").read_text().splitlines()]
    assert all(r["backend_calls"] == 1 and r["status"] == "ok" for r in records)
    assert all(r["raw_sha256"] and r["generation_context_sha256"] for r in records)
    row = json.loads((tmp_path / "c" / "summary.json").read_text())
    assert row["row"] == "fresh_inference" and row["fully_pinned_deployment"] is False


async def test_a_server_that_varies_is_recorded_as_diverging_and_every_reply_is_kept(tmp_path):
    result = await cell(tmp_path, (f"reply {i % 2}" for i in itertools.count()), repeats=2, n=1)
    assert not result.repeatable and result.first_divergent == ["p0"]
    assert len(list((tmp_path / "c" / "outputs").iterdir())) == 2


async def test_a_condition_is_never_overwritten(tmp_path):
    await cell(tmp_path, itertools.repeat("x"), repeats=1, n=1)
    with pytest.raises(FileExistsError):
        await cell(tmp_path, itertools.repeat("x"), repeats=1, n=1)


def test_a_client_that_cannot_report_its_calls_cannot_run_fresh_mode(tmp_path):
    with pytest.raises(FreshModeUnavailable):
        run_cell(payloads(1), lambda rep: FakeLLMClient(replies=["x"]), models=MODELS, repeats=1, mode="fresh",
                 condition="c", out_dir=tmp_path)


def test_replies_that_never_reached_the_server_fail_the_cell(tmp_path):
    """A client that says it reports calls but answers from somewhere else -
    a warm cache, say - produces identical replies and must still fail."""

    class Silent(FakeLLMClient):
        def add_request_observer(self, observer):
            return True  # attached, and never called

    result = run_cell(payloads(2), lambda rep: Silent(responder=lambda p, s: "cached"), models=MODELS, repeats=3,
                      mode="fresh", condition="c", out_dir=tmp_path)
    assert result.no_call_failures == 6 and not result.repeatable
    assert json.loads((tmp_path / "c" / "summary.json").read_text())["repeatable_in_this_cell"] is False


def test_cached_replay_is_its_own_row(tmp_path):
    cache = CallCache(tmp_path / "cache")
    result = run_cell(payloads(2), lambda rep: CachedClient(FakeLLMClient(replies=["a", "b"]), cache), models=MODELS,
                      repeats=3, mode="replay", condition="replay", out_dir=tmp_path / "exp")
    assert result.repeatable and result.row()["row"] == "cached_replay"


def test_a_complete_manifest_is_the_only_fully_pinned_one(tmp_path):
    full = {k: "x" for k in MANIFEST_KEYS}
    partial = dict(full, gpu_type="unknown")
    for name, manifest, pinned in (("full", full, True), ("partial", partial, False)):
        r = run_cell(payloads(1), lambda rep: CachedClient(FakeLLMClient(replies=["a"]), CallCache(tmp_path / name)),
                     models=MODELS, repeats=1, mode="replay", condition=name, out_dir=tmp_path / "exp",
                     manifest=manifest)
        assert r.row()["fully_pinned_deployment"] is pinned
    assert r.row()["manifest_missing"] == ["gpu_type"]


# --- pinned payloads and the rendering control ------------------------------------------------------


def test_the_pinned_payloads_are_intact_and_representative():
    pinned = load_payloads(PINNED)
    assert len(pinned) >= 20
    features = {f for p in pinned for f in p["features"]}
    assert {"negation", "numeric_outcome", "long_context", "mixed_species"} <= features
    assert {p["job"] for p in pinned} == {"relevance", "extract"} and all(p["schema"] for p in pinned)


def test_an_edited_payload_is_refused(tmp_path):
    line = json.loads(PINNED.read_text().splitlines()[0])
    line["prompt"] += " (edited)"
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(line) + "\n")
    with pytest.raises(ValueError, match="pinned hash"):
        load_payloads(path)


def test_payloads_are_rendered_the_same_way_every_time():
    papers = corpus()
    a = build_payloads([("fixture", QUESTION, papers)], relevance_per_corpus=2, extraction_per_corpus=1)
    b = build_payloads([("fixture", QUESTION, list(reversed(papers)))], relevance_per_corpus=2, extraction_per_corpus=1)
    assert a == b and len(a) == 3


def test_recorded_replies_render_to_identical_bytes_twice(tmp_path):
    def settings_for(d):
        s = load_settings(model_set="laptop", env={}, check_endpoint=False)
        s.cache_dir, s.out_dir, s.db_path = d / "cache", d / "out", d / "db.sqlite3"
        return s

    recorder = RecordingClient(fake_client())
    answer_question(QUESTION, settings=settings_for(tmp_path / "rec"), client=recorder, papers=corpus(),
                    db=Database(tmp_path / "rec.sqlite3"))
    result = replay_control(QUESTION, corpus(), recorder.replies, settings_for=settings_for, version="fake-1",
                            work_dir=tmp_path)
    assert result["row"] == "recorded_output_rendering" and result["identical"]
    assert result["runs"][0]["unrecorded_requests"] == 0
