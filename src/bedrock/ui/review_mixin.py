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

from textual import work

from bedrock.config import Settings
from bedrock.ui.argparsing import CommandUsageError
from bedrock.ui.rendering import markdown_text
from bedrock.ui.review_command import parse_review_command, run_review_job


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
    # database, and ``/reviews`` finds it.

    async def start_review(self, args: str) -> None:
        """Validate the arguments here, so a typo never starts a long job."""
        try:
            command = parse_review_command(args)
        except CommandUsageError as exc:
            await self.post_notice(str(exc), kind="error")
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
        from bedrock.litreview.config import ConfigError as LitReviewConfigError

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

    # Not exclusive: the flag above is what keeps it to one review, and an
    # exclusive worker would cancel a long run if that ever failed.
    @work(group="litreview")
    async def literature_review(self, request, settings) -> None:
        try:
            scheduler = self.review_scheduler(settings) if request.use_model else None
            await run_review_job(
                request,
                settings=settings,
                scheduler=scheduler,
                cancel=self._review_cancel,
                notify=self.post_notice,
                progress=lambda message: self.call_from_thread(self.notify, message),
            )
        finally:
            self._review_running = False

    async def close_review_commands(self) -> None:
        """On quit, before any backend closes: a review waiting on a model call
        reads the event and stops where it is, so the run stays resumable,
        rather than seeing the connection disappear underneath it."""
        self._review_cancel.set()
        own, self._own_review_scheduler = self._own_review_scheduler, None
        if own is not None:
            await own.backend.close()
