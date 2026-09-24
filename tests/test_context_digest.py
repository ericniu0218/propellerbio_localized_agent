"""The digest layer, and specifically its promise that the model is optional.

The rules digest must always be there; the model may only ever improve it, and
must never be able to take the store down with it.
"""

import pytest

from bedrock.context import ContextStore, ingest
from bedrock.context.adapters import Source
from bedrock.config import PROFILES
from bedrock.context.digest import DIGEST_REPLY_TOKENS, DIGEST_WINDOW_SHARE, parse_reply, summarize
from bedrock.inference import Priority, RequestOptions, Scheduler, estimate_tokens
from bedrock.inference.base import InferenceError

from tests.conftest import FakeBackend

PROSE = "\n\n".join(
    [
        "# Stability Protocol",
        "Samples are equilibrated at ambient temperature before loading. " * 8,
        "## Storage",
        "Store aliquots at minus eighty degrees until assay. " * 8,
    ]
)


@pytest.fixture
def store(tmp_path) -> ContextStore:
    return ContextStore(tmp_path / "context")


def test_rules_digest_needs_no_model_at_all(store: ContextStore):
    artifact = ingest(Source.from_text(PROSE, name="p.md", suffix=".md"), store=store)

    assert artifact.digest.generated_by == "rules"
    assert artifact.digest.gist
    assert artifact.digest_text()


def test_gist_and_overview_do_not_repeat_each_other(store: ContextStore):
    artifact = ingest(Source.from_text(PROSE, name="p.md", suffix=".md"), store=store)
    assert artifact.digest.gist not in artifact.digest.overview


async def test_model_pass_upgrades_and_persists_the_digest(store: ContextStore):
    artifact = ingest(Source.from_text(PROSE, name="p.md", suffix=".md"), store=store)
    backend = FakeBackend(reply="GIST: A protocol for sample stability.\n- Equilibrate first\n- Store at -80C")

    digest = await summarize(artifact, backend, "test-model", store=store)

    assert digest.gist == "A protocol for sample stability."
    assert digest.key_points == ["Equilibrate first", "Store at -80C"]
    assert digest.generated_by == "test-model"
    # And it was written back, so everyone gets the better one from now on.
    assert store.get(artifact.id).digest.gist == "A protocol for sample stability."


async def test_a_dead_backend_leaves_the_stored_digest_intact(store: ContextStore):
    artifact = ingest(Source.from_text(PROSE, name="p.md", suffix=".md"), store=store)

    class DeadBackend:
        async def stream_chat(self, model, messages):
            raise InferenceError("connection refused")
            yield  # pragma: no cover - makes this an async generator

    digest = await summarize(artifact, DeadBackend(), "test-model", store=store)

    assert digest.generated_by == "rules"
    assert digest.gist == artifact.digest.gist


async def test_an_unparseable_reply_falls_back_rather_than_storing_junk(store: ContextStore):
    artifact = ingest(Source.from_text(PROSE, name="p.md", suffix=".md"), store=store)
    digest = await summarize(artifact, FakeBackend(reply="   "), "test-model", store=store)
    assert digest.generated_by == "rules"


async def test_the_model_is_given_the_overview_not_the_whole_file(store: ContextStore):
    """A digest prompt must stay bounded, or summarising a big file costs more
    than reading it would have."""
    big = "\n\n".join(f"Paragraph {i}. " + "filler words " * 50 for i in range(400))
    artifact = ingest(Source.from_text(big, name="big.md", suffix=".md"), store=store)
    backend = FakeBackend(reply="GIST: Filler.\n- A point")

    await summarize(artifact, backend, "test-model")

    prompt = backend.requests[0][1][0]["content"]
    assert estimate_tokens(prompt) <= PROFILES["laptop"].context_window * DIGEST_WINDOW_SHARE
    assert len(big) > 100_000


async def test_the_digest_prompt_is_sized_to_the_profile_window(store: ContextStore):
    big = "\n\n".join(f"Paragraph {i}. " + "filler words " * 50 for i in range(400))
    artifact = ingest(Source.from_text(big, name="big.md", suffix=".md"), store=store)
    prompts = {}
    for name in ("laptop", "workstation"):
        backend = FakeBackend(reply="GIST: Filler.\n- A point")
        await summarize(artifact, Scheduler(backend, PROFILES[name]), "test-model")
        prompts[name] = backend.requests[0][1][0]["content"]
        assert estimate_tokens(prompts[name]) <= PROFILES[name].context_window * DIGEST_WINDOW_SHARE
    assert len(prompts["workstation"]) > len(prompts["laptop"])


async def test_the_model_pass_is_a_user_job_with_a_short_reply_and_no_thinking(store: ContextStore):
    artifact = ingest(Source.from_text(PROSE, name="p.md", suffix=".md"), store=store)
    backend = FakeBackend(reply="GIST: A protocol.\n- Equilibrate first")

    await summarize(artifact, Scheduler(backend, PROFILES["laptop"]), "test-model")

    assert backend.request_options == [RequestOptions(max_tokens=DIGEST_REPLY_TOKENS, reasoning=False)]


async def test_the_model_pass_waits_for_a_streaming_chat_reply(store: ContextStore):
    import asyncio

    artifact = ingest(Source.from_text(PROSE, name="p.md", suffix=".md"), store=store)
    backend = FakeBackend(reply="GIST: A protocol.\n- Equilibrate first")
    backend.hold_reply = asyncio.Event()
    scheduler = Scheduler(backend, PROFILES["laptop"])

    chat = asyncio.create_task(scheduler.complete("test-model", [{"role": "user", "content": "hi"}], priority=Priority.CHAT))
    await asyncio.sleep(0.01)
    job = asyncio.create_task(summarize(artifact, scheduler, "test-model"))
    await asyncio.sleep(0.01)
    assert len(backend.requests) == 1  # only the chat has reached the server

    backend.hold_reply.set()
    await chat
    digest = await job
    assert digest.gist == "A protocol."
    assert len(backend.requests) == 2


def test_parse_reply_tolerates_the_formats_small_models_actually_produce():
    assert parse_reply("GIST: A thing.\n- one\n- two") == ("A thing.", ["one", "two"])
    # Bullets with asterisks, and a preamble the model was told not to write.
    gist, points = parse_reply("Sure! Here you go:\n* one\n* two")
    assert gist == "Sure! Here you go:"
    assert points == ["one", "two"]
    assert parse_reply("") == ("", [])
