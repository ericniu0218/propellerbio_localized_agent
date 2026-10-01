"""`/review`, shared by both terminal application modes.

A literature review is the same job in both modes: the pipeline runs on this
machine, on a worker thread, and every model call goes through a bedrock
`Scheduler` at `USER_JOB` (`bedrock.litreview.llm.scheduled`). What differs is
where the model is:

* In the default mode the review's endpoint is usually the chat's own
  (`api_base`), so the review shares the chat's scheduler and queues behind the
  employee's messages - on a laptop that is the whole point.
* In agent-pack mode nothing on this machine serves models. The review's
  endpoint is the agent server's inference API (`agent_pack_api_base` + `/v1`,
  see `bedrock.litreview.config.resolve_endpoint` and
  `bedrock.server.api.routes_inference`), so the review opens its own
  connection to it, the first time one is started. Opening it starts nothing
  locally: it is an HTTP client.

The agent server's conversational `/messages` endpoint is never used for a
review. It routes each message through a tier classifier and reformats the
answer, and a review needs the opposite: a named model per job, temperature 0,
a seed, and JSON-schema output returned exactly as the model wrote it.
"""

from __future__ import annotations

import asyncio
import threading
import time

from textual import work

from bedrock.config import Settings
from bedrock.ui.argparsing import CommandUsageError
from bedrock.ui.rendering import markdown_text, safe_line
from bedrock.ui.review_command import REVIEW_STALE_INSTALL, parse_review_command, run_review_job
from bedrock.ui.widgets import StatusBar


class ReviewCommandsMixin:
    """Own `/review`: argument checking, one run at a time, stop and quit."""

    settings: Settings

    def _init_review_commands(self) -> None:
        # One literature review at a time, independent of a running fold: they
        # use different machines. The event is how `/review stop` and quitting
        # reach a run that is already inside the pipeline's worker thread.
        self._review_running = False
        self._review_cancel = threading.Event()
        self._own_review_scheduler = None
        # The status bar's live position for a running review, and the timer
        # that redraws it once a second. See `_review_tick`.
        self._review_started: float | None = None
        self._review_position: tuple[str, int, int, str] | None = None
        self._review_clock = None

    # ---- Where the review's model calls go ------------------------------------

    def shared_review_scheduler(self, review_settings):
        """The app's own scheduler, when the review's model server is the one
        it already talks to; None otherwise. The default mode overrides this."""
        return None

    def review_scheduler(self, review_settings):
        """The scheduler for this review's model calls. See the module docstring."""
        shared = self.shared_review_scheduler(review_settings)
        if shared is not None:
            return shared
        current = self._own_review_scheduler
        if current is not None and getattr(current.backend, "base_url", None) == review_settings.base_url:
            return current
        from bedrock.inference import Scheduler
        from bedrock.litreview.service import review_backend

        if current is not None:
            # The endpoint changed in the config file since the last review.
            # Nothing is running on it (one review at a time), so close it.
            asyncio.get_running_loop().create_task(current.backend.close())
        self._own_review_scheduler = Scheduler(review_backend(review_settings), self.settings.profile)
        return self._own_review_scheduler

    # ---- The command ------------------------------------------------------------
    #
    # A review is hundreds of model calls, so every one of them goes in at
    # ``USER_JOB``: behind the chat, ahead of background work. Its notices are
    # not saved into any conversation; the run is recorded in litreview's own
    # database, and ``/reviews`` finds it. Local history (``/audit``) records
    # that it ran and which outside hosts it was allowed to contact - never
    # the question (``bedrock.litreview.journal``).

    async def start_review(self, args: str) -> None:
        """Validate the arguments here, so a typo never starts a long job."""
        try:
            command = parse_review_command(args)
        except CommandUsageError as exc:
            await self.post_notice(str(exc), kind="error")
            return
        except ImportError as exc:
            # The review pipeline is imported lazily (see below), so a virtual
            # environment left behind by an older checkout - one without
            # pydantic, lxml, PyYAML or rapidfuzz - fails here, on the first
            # `/review` rather than at start-up. Raw, that is a traceback over
            # the whole screen for something a reinstall fixes.
            await self.post_notice(REVIEW_STALE_INSTALL.format(detail=markdown_text(str(exc))), kind="error")
            return
        if command.stop:
            await self.stop_review()
            return
        if self._review_running:
            await self.post_notice(
                "A literature review is already running. `/review stop` stops it.", kind="error"
            )
            return
        # Imported here, not at the top: the pipeline pulls in pydantic and the
        # whole of sources/ and stages/, which is 115 ms of start-up for a
        # feature most sessions never use. See ui/review_command.py.
        try:
            from bedrock.litreview.config import ConfigError as LitReviewConfigError
        except ImportError as exc:  # as above
            await self.post_notice(REVIEW_STALE_INSTALL.format(detail=markdown_text(str(exc))), kind="error")
            return

        try:
            settings = await asyncio.to_thread(self._review_settings, command.request.model_set)
        except LitReviewConfigError as exc:
            await self.post_notice(markdown_text(str(exc)), kind="error")
            return
        self._review_running = True
        self._review_cancel = threading.Event()
        self.literature_review(command.request, settings)

    async def stop_review(self) -> None:
        if not self._review_running:
            await self.post_notice("No literature review is running.", kind="error")
            return
        self._review_cancel.set()
        await self.post_notice(
            "Stopping the literature review before its next model call. "
            "It can be resumed with `/review --resume <run-id>`."
        )

    def _review_settings(self, model_set: str | None):
        """litreview's settings, read off the UI thread (they touch disk)."""
        from bedrock.litreview.config import load_settings as load_review_settings

        return load_review_settings(self.settings, model_set=model_set)

    def _review_tick(self, stage: str, done: int, total: int, detail: str = "") -> None:
        """Called (on the UI thread) as each item of a stage finishes.

        Draws straight away so a new count is never up to a second stale, and
        `_tick_review_clock` keeps redrawing on a timer so the elapsed time
        goes on moving *between* ticks. Both halves matter: a single extraction
        call can take minutes, and a counter frozen for minutes looks exactly
        like a hung process, which is what this exists to rule out.
        """
        self._review_position = (stage, done, total, detail)
        self._tick_review_clock()

    def _tick_review_clock(self) -> None:
        from bedrock.litreview.progress import format_tick

        if self._review_started is None or self._review_position is None:
            return
        stage, done, total, detail = self._review_position
        elapsed = time.monotonic() - self._review_started
        self.query_one(StatusBar).review_progress = format_tick(stage, done, total, detail, elapsed)

    def _clear_review_progress(self) -> None:
        self._review_started = None
        self._review_position = None
        if self._review_clock is not None:
            self._review_clock.stop()
            self._review_clock = None
        try:
            self.query_one(StatusBar).review_progress = None
        except Exception:  # noqa: BLE001 - the app may be shutting down
            pass

    # Not exclusive: the flag above is what keeps it to one review, and an
    # exclusive worker would cancel a long run if that ever failed.
    @work(group="litreview")
    async def literature_review(self, request, settings) -> None:
        self._review_started = time.monotonic()
        self._review_position = ("starting", 0, 0, "")
        self._review_clock = self.set_interval(1.0, self._tick_review_clock)
        # Drawn once here as well: the first stage that ticks may be minutes
        # away (papers are collected and gated first), and the bar should say
        # a review is running from the moment one is.
        self._tick_review_clock()
        try:
            try:
                # As for /digest: a build without `bedrock.provenance` still
                # runs a review, just unrecorded (ui/services.py). A journal
                # that exists but cannot be opened stops it before it starts.
                history = await self.services.optional_history()
            except self.services.errors as exc:
                await self.post_notice(
                    safe_line(f"Local history is unavailable ({exc}); the review was not started.", limit=600),
                    kind="error",
                )
                return
            scheduler = self.review_scheduler(settings) if request.use_model else None
            await run_review_job(
                request,
                settings=settings,
                scheduler=scheduler,
                cancel=self._review_cancel,
                notify=self.post_notice,
                tick=lambda *a: self.call_from_thread(self._review_tick, *a),
                progress=lambda message: self.call_from_thread(self.notify, message),
                history=history,
                actor_id=self.settings.user_id,
                # A review that shares the chat's connection uses the chat's
                # `request_timeout`; one that opens its own has it raised to
                # `REVIEW_TIMEOUT_FLOOR`, so only the first is worth warning about.
                timeout_floor_applied=self.shared_review_scheduler(settings) is None,
            )
        finally:
            self._review_running = False
            # However the run ended - finished, stopped, or failed - the bar
            # must not keep showing a position that is no longer moving.
            self._clear_review_progress()

    async def close_review_commands(self) -> None:
        """On quit, before any backend closes: a review waiting on a model call
        reads the event and stops where it is, so the run stays resumable,
        rather than seeing the connection disappear underneath it."""
        self._review_cancel.set()
        own, self._own_review_scheduler = self._own_review_scheduler, None
        if own is not None:
            await own.backend.close()
