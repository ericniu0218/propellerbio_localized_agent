"""Remember every model call, so the same question is never asked twice.

This is one of the three mechanisms that answer the client's "I get a different
answer every time" complaint. The other two are `temperature=0` and freezing
the paper set.

The cache key covers the prompt, the model version, the per-call parameters
(schema, temperature, reply limit) and the client's *generation context*:
seed, thinking policy, weights digest where the server reports one, declared
revision/template/deployment, which server, and how the reply was normalised
(`SchedulerClient.generation_context`). That has a useful property: editing a
prompt changes the key, so a prompt change automatically invalidates exactly
the entries it should and nothing else - and so does a different seed or a
different server, which before `generation_context.v1` silently replayed each
other's answers.

What the cache proves, and what it does not. A hit replays a stored reply to
an identical call; it says nothing about whether the server would write the
same reply again. That is a separate, deployment-specific question, measured
only by calls that actually reach the server (see the benchmark's
determinism mode), never by comparing two cached runs.

Entries written before the context was part of the key are not deleted and
are not matched: a call now names more than they recorded, and treating them
as equivalent would be a guess. `CachedClient(legacy_read=True)` replays them
explicitly when that is what is wanted.

Stored as one JSON file per entry under a content-addressed path. Not a
database: this has to be readable and deletable by hand on an air-gapped
machine by somebody debugging at 6pm, and `rm` is a better tool for that than
a migration.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from bedrock.litreview.llm.client import GENERATION_CONTEXT_VERSION


class CallCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(prompt: str, model_version: str, params: dict[str, Any]) -> str:
        payload = json.dumps(
            {"prompt": prompt, "model": model_version, "params": params},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # Shard by the first two characters: a project can produce tens of
        # thousands of calls, and one flat directory that size is painful on
        # every filesystem and unusable in Finder.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str, *, count: bool = True) -> str | None:
        path = self._path(key)
        if not path.exists():
            self.misses += count
            return None
        try:
            response = json.loads(path.read_text(encoding="utf-8"))["response"]
        except (ValueError, KeyError, OSError):
            # A truncated entry from an interrupted write is a miss, never a
            # crash. Losing a cache entry costs one model call; crashing costs
            # the whole run.
            self.misses += count
            return None
        self.hits += count
        return response

    def put(
        self, key: str, prompt: str, response: str, model_version: str, *, context: dict[str, Any] | None = None
    ) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "model": model_version,
            "prompt": prompt,
            "response": response,
        }
        if context is not None:
            record["context"] = context
        # Write to a temp file and rename, so a crash mid-write cannot leave a
        # half-written entry that looks valid.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class CachedClient:
    """Wraps any LLMClient with the cache. Stages use this, not the raw client.

    The key uses the version of the model actually called (for Ollama, the
    weights digest), so one client serving two models never mixes their
    answers, and a re-pulled model with new weights never replays old ones.
    """

    def __init__(self, inner, cache: CallCache, *, read: bool = True, legacy_read: bool = False) -> None:
        self._inner = inner
        self._cache = cache
        self._read = read
        """False re-asks the model and overwrites the entry. This bypasses the
        cache for one client; it is not by itself an experiment, because the
        overwritten entry is lost - the determinism mode keeps each
        repetition in its own cache directory for that reason."""
        self._legacy_read = legacy_read
        """True also replays an entry written before generation contexts were
        part of the key, when the full key misses. Off by default: such an
        entry does not say which seed, thinking policy or server wrote it."""
        self.legacy_hits = 0

    def version_of(self, model: str) -> str:
        fn = getattr(self._inner, "version_of", None)
        return fn(model) if fn else self._inner.model_version

    def generation_context(self, model: str) -> dict[str, Any]:
        """The inner client's context, or an explicit unknown that names the
        client type. An unknown context is never read back (`complete`): a
        client that cannot say which seed or server it uses cannot be told
        apart from one with a different seed or server."""
        fn = getattr(self._inner, "generation_context", None)
        if fn is None:
            return {"schema": GENERATION_CONTEXT_VERSION, "unknown": type(self._inner).__name__}
        return fn(model)

    def complete(self, prompt: str, *, model: str, schema=None, temperature=0.0, max_tokens=2048) -> str:
        params = {
            "schema": schema,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": model,
        }
        version = self.version_of(model)
        context = self.generation_context(model)
        key = CallCache.key(prompt, version, {**params, "context": context})
        if self._read and "unknown" not in context:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            if self._legacy_read:
                cached = self._cache.get(CallCache.key(prompt, version, params), count=False)
                if cached is not None:
                    self.legacy_hits += 1
                    return cached

        response = self._inner.complete(
            prompt, model=model, schema=schema, temperature=temperature, max_tokens=max_tokens
        )
        self._cache.put(key, prompt, response, version, context=context)
        return response

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    @property
    def cache(self) -> CallCache:
        return self._cache

    @property
    def inner(self):
        return self._inner
