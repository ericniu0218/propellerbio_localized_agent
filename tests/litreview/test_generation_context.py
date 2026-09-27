"""The call cache replays a reply only to a call that would have been made the same way.

Before `generation_context.v1` the key held the prompt, model, schema,
temperature and reply limit - and not the seed, the thinking policy or the
server. The audit's probe: a client with seed 1 was handed seed 0's cached
reply and the server was never asked. Each test below changes one part of
the context and checks that the server *is* asked.
"""

from __future__ import annotations

import json

from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.llm.cache import CachedClient, CallCache
from bedrock.inference import ModelInfo
from tests.litreview.test_scheduler_client import ScriptedBackend, in_worker, make


class _Endpoint(ScriptedBackend):
    def __init__(self, *args, base_url: str = "http://gpu-a:8000/v1", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = base_url


async def _ask(cache_dir, backend=None, prompt="grade this", **client_kwargs) -> tuple[str, _Endpoint]:
    backend = backend or _Endpoint(reply="fresh")
    client, _ = make(backend, **client_kwargs)
    cached = CachedClient(client, CallCache(cache_dir))
    reply = await in_worker(lambda: cached.complete(prompt, model="m", schema={"type": "object"}, max_tokens=64))
    return reply, backend


async def test_the_audit_probe_a_different_seed_reaches_the_server(tmp_path):
    first, _ = await _ask(tmp_path, _Endpoint(reply="seed zero says"), seed=0)
    again, backend = await _ask(tmp_path, _Endpoint(reply="seed one says"), seed=1)
    assert (first, again) == ("seed zero says", "seed one says")
    assert len(backend.calls) == 1, "the seed-1 client asked the server instead of replaying seed 0"


async def test_an_identical_context_is_replayed_without_a_call(tmp_path):
    await _ask(tmp_path, seed=0)
    reply, backend = await _ask(tmp_path, _Endpoint(reply="would differ"), seed=0)
    assert reply == "fresh" and backend.calls == []


async def test_the_thinking_policy_is_part_of_the_identity(tmp_path):
    await _ask(tmp_path, model_options={"m": {"thinking": False}})
    _, backend = await _ask(tmp_path, model_options={"m": {}})
    assert len(backend.calls) == 1


async def test_declared_revision_template_and_deployment_are_each_part_of_it(tmp_path):
    await _ask(tmp_path, model_options={"m": {"revision": "r1", "chat_template": "t1", "deployment": "d1"}})
    for changed in ({"revision": "r2", "chat_template": "t1", "deployment": "d1"},
                    {"revision": "r1", "chat_template": "t2", "deployment": "d1"},
                    {"revision": "r1", "chat_template": "t1", "deployment": "d2"}):
        _, backend = await _ask(tmp_path, model_options={"m": changed})
        assert len(backend.calls) == 1, changed


async def test_new_weights_behind_the_same_name_are_asked_again(tmp_path):
    await _ask(tmp_path, _Endpoint(models=[ModelInfo(id="m", digest="aaaaaaaaaaaaaaaa")]))
    _, backend = await _ask(tmp_path, _Endpoint(models=[ModelInfo(id="m", digest="bbbbbbbbbbbbbbbb")]))
    assert len(backend.calls) == 1


async def test_another_server_is_another_namespace(tmp_path):
    await _ask(tmp_path, _Endpoint(base_url="http://gpu-a:8000/v1"))
    _, backend = await _ask(tmp_path, _Endpoint(base_url="http://laptop:11434/v1"))
    assert len(backend.calls) == 1


async def test_a_vllm_name_does_not_claim_to_pin_weights():
    client, _ = make(_Endpoint(models=[ModelInfo(id="m")]), seed=5)
    context = await in_worker(lambda: client.generation_context("m"))
    assert context["weights"] is None and context["revision"] is None
    assert context["seed"] == 5 and context["schema"] == "generation_context.v1"
    assert context["reasoning"] == "server_default" and context["other_sampling"] == "server_default"


async def test_the_context_is_stable_and_asks_for_digests_once():
    class Counting(_Endpoint):
        listed = 0

        async def list_models(self):
            Counting.listed += 1
            return await super().list_models()

    client, _ = make(Counting())
    first = await in_worker(lambda: client.generation_context("m"))
    second = await in_worker(lambda: client.generation_context("m"))
    assert first == second and Counting.listed == 1


def test_a_schema_change_misses(tmp_path):
    fake = FakeLLMClient(replies=["one", "two"])
    cached = CachedClient(fake, CallCache(tmp_path))
    assert cached.complete("p", model="a", schema={"type": "object"}) == "one"
    assert cached.complete("p", model="a", schema={"type": "array"}) == "two"


def test_key_order_does_not_matter():
    a = CallCache.key("p", "m", {"schema": {"x": 1, "y": 2}, "context": {"seed": 0, "model": "m"}})
    b = CallCache.key("p", "m", {"context": {"model": "m", "seed": 0}, "schema": {"y": 2, "x": 1}})
    assert a == b


def test_the_audit_probe_with_clients_that_cannot_identify_themselves(tmp_path):
    """The audit's own probe: two bare clients, seeds 0 and 1, neither able to
    say so. Neither may be answered from the other's entry."""

    class Seeded:
        model_version = "same-model"

        def __init__(self, seed):
            self.seed, self.calls = seed, 0

        def complete(self, prompt, **kwargs):
            self.calls += 1
            return f"seed={self.seed}"

    cache = CallCache(tmp_path)
    a, b = Seeded(0), Seeded(1)
    assert CachedClient(a, cache).complete("synthetic", model="same-model") == "seed=0"
    assert CachedClient(b, cache).complete("synthetic", model="same-model") == "seed=1"
    assert b.calls == 1


def test_an_unidentified_client_does_not_share_entries_with_an_identified_one(tmp_path):
    class Bare:
        model_version = "fake-1"

        def __init__(self, reply):
            self.reply, self.calls = reply, 0

        def complete(self, prompt, **kwargs):
            self.calls += 1
            return self.reply

    CachedClient(FakeLLMClient(replies=["identified"]), CallCache(tmp_path)).complete("p", model="a")
    bare = Bare("unidentified")
    assert CachedClient(bare, CallCache(tmp_path)).complete("p", model="a") == "unidentified" and bare.calls == 1
    assert CachedClient(bare, CallCache(tmp_path)).generation_context("a") == {
        "schema": "generation_context.v1", "unknown": "Bare"}


def test_old_entries_stay_on_disk_and_are_replayed_only_when_asked(tmp_path):
    """An entry written before contexts were in the key does not say which
    seed or server wrote it: replayable on request, not matched by default."""
    params = {"schema": None, "temperature": 0.0, "max_tokens": 2048, "model": "a"}
    old_key = CallCache.key("p", "fake-1", params)
    CallCache(tmp_path).put(old_key, "p", "from the old cache", "fake-1")
    old_file = CallCache(tmp_path)._path(old_key)
    before = old_file.read_bytes()

    legacy = CachedClient(FakeLLMClient(replies=["never asked"]), CallCache(tmp_path), legacy_read=True)
    assert legacy.complete("p", model="a") == "from the old cache" and legacy.legacy_hits == 1

    default = FakeLLMClient(replies=["asked again"])
    assert CachedClient(default, CallCache(tmp_path)).complete("p", model="a") == "asked again"
    assert len(default.prompts) == 1
    assert old_file.read_bytes() == before, "the old entry is left exactly as it was"


def test_entries_record_the_context_they_were_written_under(tmp_path):
    cached = CachedClient(FakeLLMClient(replies=["r"]), CallCache(tmp_path))
    cached.complete("p", model="a")
    [entry] = list(tmp_path.rglob("*.json"))
    assert json.loads(entry.read_text())["context"]["schema"] == "generation_context.v1"


def test_the_fake_client_has_a_context_like_the_live_one():
    context = FakeLLMClient(version="fake-9").generation_context("m")
    assert context == {"schema": "generation_context.v1", "client": "fake", "model": "m",
                       "version": "fake-9", "normalization": "strip_thinking.v1"}
