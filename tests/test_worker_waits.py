"""Deterministic coverage of the worker cancellation race seen on CI."""
import asyncio

import pytest
from textual.app import App
from textual.worker import WorkerCancelled, WorkerFailed

from tests.workers import wait_for_app_workers


def observe_wait(worker, monkeypatch):
    captured = asyncio.Event()
    original = worker.wait

    async def wait():
        captured.set()
        return await original()

    monkeypatch.setattr(worker, "wait", wait)
    return captured


async def test_wait_follows_cancelled_refresh_and_its_replacement(monkeypatch):
    app = App()
    async with app.run_test():
        old = app.run_worker(asyncio.Event().wait(), group="history")
        captured = observe_wait(old, monkeypatch)
        waiter = asyncio.create_task(wait_for_app_workers(app))
        await asyncio.wait_for(captured.wait(), 2)

        release = asyncio.Event()
        replacement = app.run_worker(release.wait(), group="history", exclusive=True)
        replacement_captured = observe_wait(replacement, monkeypatch)
        await asyncio.wait_for(replacement_captured.wait(), 2)
        assert old.is_cancelled
        assert not waiter.done()
        release.set()
        await waiter
        assert replacement.is_finished


async def test_original_wait_reproduces_ci_cancellation(monkeypatch):
    app = App()
    async with app.run_test():
        old = app.run_worker(asyncio.Event().wait(), group="history")
        captured = observe_wait(old, monkeypatch)
        waiter = asyncio.create_task(app.workers.wait_for_complete())
        await asyncio.wait_for(captured.wait(), 2)
        replacement = app.run_worker(asyncio.sleep(0), group="history", exclusive=True)
        with pytest.raises(WorkerCancelled):
            await waiter
        await replacement.wait()


@pytest.mark.parametrize("group", ["history", "chat"])
async def test_wait_still_reports_real_worker_errors(group, monkeypatch):
    release = asyncio.Event()

    async def fail():
        await release.wait()
        raise RuntimeError("real failure")

    app = App()
    async with app.run_test():
        worker = app.run_worker(fail(), group=group, exit_on_error=False)
        captured = observe_wait(worker, monkeypatch)
        waiter = asyncio.create_task(wait_for_app_workers(app))
        await asyncio.wait_for(captured.wait(), 2)
        release.set()
        with pytest.raises(WorkerFailed, match="real failure"):
            await waiter


async def test_wait_rejects_unexpected_chat_cancellation(monkeypatch):
    app = App()
    async with app.run_test():
        worker = app.run_worker(asyncio.Event().wait(), group="chat")
        captured = observe_wait(worker, monkeypatch)
        waiter = asyncio.create_task(wait_for_app_workers(app))
        await asyncio.wait_for(captured.wait(), 2)
        worker.cancel()
        with pytest.raises(WorkerCancelled):
            await waiter


async def test_wait_reports_a_stuck_refresh():
    app = App()
    async with app.run_test():
        app.run_worker(asyncio.Event().wait(), group="history")
        with pytest.raises(TimeoutError, match="deadline"):
            await wait_for_app_workers(app, timeout=0.02)
