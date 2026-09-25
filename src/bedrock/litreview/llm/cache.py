"""Remember every model call, so the same question is never asked twice.

This is one of the three mechanisms that answer the client's "I get a different
answer every time" complaint. The other two are `temperature=0` and freezing
the paper set.

The cache key covers the prompt, the model version and every sampling
parameter. That has a useful property: editing a prompt changes the key, so a
prompt change automatically invalidates exactly the entries it should and
nothing else. During prompt iteration you want fresh answers, and you get them
without anybody remembering to clear a cache.

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

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        except (ValueError, KeyError, OSError):
            # A truncated entry from an interrupted write is a miss, never a
            # crash. Losing a cache entry costs one model call; crashing costs
            # the whole run.
            self.misses += 1
            return None

    def put(self, key: str, prompt: str, response: str, model_version: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "model": model_version,
            "prompt": prompt,
            "response": response,
        }
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

    def __init__(self, inner, cache: CallCache, *, read: bool = True) -> None:
        self._inner = inner
        self._cache = cache
        self._read = read
        """False re-asks the model and overwrites the entry. The determinism
        benchmark uses this to prove that a fresh run reproduces the cached one."""

    def version_of(self, model: str) -> str:
        fn = getattr(self._inner, "version_of", None)
        return fn(model) if fn else self._inner.model_version

    def complete(self, prompt: str, *, model: str, schema=None, temperature=0.0, max_tokens=2048) -> str:
        params = {
            "schema": schema,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": model,
        }
        version = self.version_of(model)
        key = CallCache.key(prompt, version, params)
        if self._read:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        response = self._inner.complete(
            prompt, model=model, schema=schema, temperature=temperature, max_tokens=max_tokens
        )
        self._cache.put(key, prompt, response, version)
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
