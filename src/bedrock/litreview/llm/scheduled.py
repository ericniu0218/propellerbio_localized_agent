"""The one live model client: litreview's `LLMClient`, served by bedrock's `Scheduler`.

Why this exists at all. A review makes hundreds of model calls - one per paper
to grade relevance, one per chunk to extract from, one per report section - to
the same server the employee is chatting with. On a laptop that server works on
one request at a time, so a review that talked to it directly would sit in front
of the employee's next message for as long as it took. Bedrock's scheduler is
what stops that: every call here goes in at `Priority.USER_JOB`, behind the
chat and ahead of background work such as conversation titles. This is also
bedrock's written model-call rule - nothing outside `bedrock.inference` calls
`stream_chat` - and `tests/test_model_call_guard.py` fails if it is broken.

The awkward part is that the two sides disagree about threads. The pipeline is
ordinary blocking Python: a stage loops over papers and calls `complete`. The
scheduler is asyncio, running on the app's event loop alongside the UI. So this
client is **blocking, and must be called from a worker thread**: it hands each
call to the loop with `run_coroutine_threadsafe` and waits for the answer.
`service.run_review` is what sets that up correctly; nothing else should build
one of these by hand.

The exceptions below are the other half of the bridge. `LLMError` already means
"this one call failed, skip this item and carry on", and stages catch it in
exactly that way. That is the right answer for a malformed reply, and the wrong
answer for a server that has gone away: a report finished while the server was
down would come out silently hollow - fewer findings, thinner sections, no
warning that anything was missing. So an unreachable server and a cancelled run
raise `ReviewInterrupted`, which stages do *not* catch, and the run stops where
it is. Every stage's work is already saved, so it can be resumed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import threading
from typing import Any

from bedrock.inference import InferenceError, Priority, RequestOptions
from bedrock.inference.tokens import estimate_tokens
from bedrock.litreview.llm.client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    GENERATION_CONTEXT_VERSION,
    NORMALIZATION_VERSION,
    LLMError,
    strip_thinking,
)


class ReviewInterrupted(RuntimeError):
    """The run stopped before its report, and can be resumed from where it got to.

    Deliberately *not* an `LLMError`. Stages catch `LLMError` to skip one paper
    or one chunk and keep going, which is right when one reply was unusable and
    wrong when the server has gone. This must end the run.
    """


class ReviewCancelled(ReviewInterrupted):
    """The employee stopped the review, or quit the app while it was running."""


class ModelServerUnavailable(ReviewInterrupted):
    """The model server is unreachable, or did not answer in time."""


class PromptTruncated(ReviewInterrupted):
    """The server read fewer prompt tokens than the prompt can possibly hold.

    A server whose context window is smaller than the prompt does not refuse
    it: Ollama keeps the *end* and drops the start, which is where the
    instructions are, and the model then answers a question it was never
    shown. Measured on this project's own laptop setup (Ollama 0.34, 16 GB
    Mac, no `OLLAMA_CONTEXT_LENGTH`): a 4,096-token window read 2,050 tokens of
    a 14,000-token prompt and answered wrongly. Every later call would be cut
    the same way, so this stops the run rather than skipping one item.
    """


MAX_CHARACTERS_PER_TOKEN = 7.0
"""No tokenizer a review runs on packs English, numbers and markup into fewer
tokens than one per ~6 characters (`bedrock.inference.tokens` bounds its own
learned ratio at 6). A server that reports reading fewer tokens than
`len(prompt) / 7` did not read the whole prompt."""

MIN_CHECKED_CHARACTERS = 2_000
"""Short prompts are mostly per-message wrapping tokens; they are not checked."""


def truncated(prompt: str, prompt_tokens: int | None) -> bool:
    """True when a reported prompt count proves the server dropped part of it."""
    if not prompt_tokens or len(prompt) < MIN_CHECKED_CHARACTERS:
        return False
    return prompt_tokens < len(prompt) / MAX_CHARACTERS_PER_TOKEN


class SchedulerClient:
    """litreview's `LLMClient`, served by bedrock's `Scheduler`.

    Blocking: call it from a worker thread. `loop` is the event loop the
    scheduler runs on, which is usually the app's.
    """

    def __init__(
        self,
        scheduler,
        loop: asyncio.AbstractEventLoop,
        *,
        model_options: dict[str, dict[str, Any]] | None = None,
        seed: int = 0,
        context_window: int = 12288,
        context_windows: dict[str, int] | None = None,
        cancel: threading.Event | None = None,
        priority: Priority = Priority.USER_JOB,
    ) -> None:
        self._scheduler = scheduler
        self._loop = loop
        self._options = model_options or {}
        self._seed = seed
        self._window = context_window
        # Per model, as `service.preflight` established them: the smaller of
        # what the server reports and what was configured.
        self._windows = dict(context_windows or {})
        self._cancel = cancel
        self._priority = priority
        self._last_model = ""
        self._digests: dict[str, str] | None = None
        self._roots: dict[str, str] = {}

    # ---- The LLMClient protocol ---------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        schema: dict[str, Any] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        self._check_cancelled()

        # Checked here rather than left to the server, because a server whose
        # window is too small does not say so: it silently drops the start of
        # the prompt, and the model answers a question it was never shown.
        #
        # `max_tokens` is never quietly shrunk to make something fit. It is part
        # of the call cache key, so a smaller one would look like a different
        # call and re-ask the server for everything already answered - and a
        # short extraction reply is truncated JSON, which is a failure anyway.
        window = self._windows.get(model) or self._window
        needed = estimate_tokens(prompt) + max_tokens
        if needed > window:
            raise LLMError(
                f"prompt plus a {max_tokens}-token reply needs ~{needed} tokens; "
                f"window is {window}"
            )

        thinking = self._options.get(model, {}).get("thinking")
        options = RequestOptions(
            max_tokens=max_tokens,
            temperature=temperature,
            json_schema=schema,
            # `thinking: false` in models.yaml means "turn it off"; a model set
            # that says nothing (MedGemma, which cannot turn it off) leaves the
            # server's own default, and `strip_thinking` removes the trace.
            reasoning=False if thinking is False else None,
            seed=self._seed,
        )
        self._last_model = model
        reported: list = []
        raw = self._run(
            self._scheduler.complete(
                model,
                [{"role": "user", "content": prompt}],
                priority=self._priority,
                options=options,
                on_usage=reported.append,
            )
        )
        read = reported[-1].prompt_tokens if reported else None
        if truncated(prompt, read):
            raise PromptTruncated(
                f"the model server read only {read:,} tokens of a prompt of about "
                f"{estimate_tokens(prompt):,} ({len(prompt):,} characters) sent to {model!r}: its context "
                "window is smaller than this review was told, so the start of every prompt - the "
                "instructions - is being dropped. Set the server's window to at least "
                f"{window:,} tokens and resume: for Ollama, OLLAMA_CONTEXT_LENGTH={window} "
                "(or the Ollama app's Settings -> Context length); for vLLM, --max-model-len. "
                "Nothing from this call was kept."
            )
        return strip_thinking(raw)

    @property
    def model_version(self) -> str:
        return self.version_of(self._last_model) if self._last_model else "scheduled"

    def version_of(self, model: str) -> str:
        """`name@digest[:12]` when the server reports a weight hash, else the name.

        A re-pulled tag with new weights is a different model, and a report has
        to be able to say which one wrote it. The format matches the one the
        source's `OllamaClient` recorded, so old runs stay comparable. Servers
        that report no digest (vLLM) just give the name back.
        """
        digest = self._digest(model)
        root = self._roots.get(model) if self._roots else None
        # A role alias (bedrock's agent server: "specialist") says nothing about
        # the weights; the server's `root` does, so the record names both.
        name = f"{model}={root}" if root else model
        return f"{name}@{digest}" if digest else name

    # Identity keys an operator may declare per model in models.yaml. They
    # are recorded as declared - the server is not asked to confirm them.
    DECLARED = ("revision", "tokenizer", "chat_template", "deployment")

    def generation_context(self, model: str) -> dict[str, Any]:
        """Everything this client knows that decides what `model` writes, apart
        from the prompt and the per-call settings the cache key already holds.

        Part of the call-cache key, so a reply is replayed only to a call that
        would have been made the same way: same seed, same thinking policy,
        same weights where the server reports them, same server. What is not
        known is written as None, never assumed: a vLLM model *name* does not
        pin weights, so a vLLM context has `weights: None` unless the operator
        declares a `revision` in models.yaml. Stable for the client's
        lifetime - it asks the server for digests once, like `version_of`.
        """
        options = self._options.get(model, {})
        backend = getattr(self._scheduler, "backend", None)
        base_url = getattr(backend, "base_url", None)
        return {
            "schema": GENERATION_CONTEXT_VERSION,
            "model": model,
            "weights": self._digest(model) or None,
            **{key: (str(options[key]) if options.get(key) is not None else None) for key in self.DECLARED},
            "backend": type(backend).__name__ if backend is not None else None,
            "endpoint_sha256": hashlib.sha256(str(base_url).encode("utf-8")).hexdigest()[:16] if base_url else None,
            "seed": self._seed,
            "reasoning": "off" if options.get("thinking") is False else "server_default",
            "other_sampling": "server_default",
            "normalization": NORMALIZATION_VERSION,
        }

    def add_request_observer(self, observer) -> bool:
        """Ask the backend to report each wire attempt (payload and reply
        hashes; see `OpenAICompatibleBackend.add_request_observer`). False when
        the backend cannot. For audits and the determinism experiment."""
        add = getattr(getattr(self._scheduler, "backend", None), "add_request_observer", None)
        if add is None:
            return False
        add(observer)
        return True

    # ---- Bridging the worker thread to the event loop ------------------------

    def _digest(self, model: str) -> str:
        """The weight hashes the server reports, read once per client.

        A digest is provenance, not the run: a server that will not list its
        models leaves the label as the bare name rather than ending the review,
        which is what the source's Ollama client did. If the server really is
        gone, the next `complete` says so and the run stops there. Being
        cancelled still propagates - a stopped review stops here too.
        """
        if self._digests is None:
            self._digests = {}
            try:
                for info in self._run(self._scheduler.backend.list_models()):
                    if info.digest:
                        self._digests[info.id] = str(info.digest)[:12]
                    if getattr(info, "root", None):
                        self._roots[info.id] = str(info.root)
            except ReviewCancelled:
                raise
            except Exception:  # noqa: BLE001 - see the docstring
                pass
        return self._digests.get(model, "")

    def _check_cancelled(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise ReviewCancelled("the review was stopped")

    def _run(self, coroutine):
        """Run one scheduler coroutine on its loop and wait here for the answer."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            # Waiting on the loop from the loop's own thread never returns.
            # Closing the coroutine keeps Python from also warning that it was
            # never awaited, which would hide this message.
            coroutine.close()
            raise RuntimeError(
                "SchedulerClient blocks, so it must be called from a worker thread, "
                "not from the event loop the scheduler runs on. Use service.run_review."
            )
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result()
        except concurrent.futures.CancelledError as exc:
            raise ReviewCancelled("the review was stopped") from exc
        except InferenceError as exc:
            if self._cancel is not None and self._cancel.is_set():
                raise ReviewCancelled("the review was stopped") from exc
            if getattr(exc, "unavailable", False):
                raise ModelServerUnavailable(str(exc)) from exc
            # Anything else is one bad reply: the stage skips that item, as before.
            raise LLMError(str(exc)) from exc
