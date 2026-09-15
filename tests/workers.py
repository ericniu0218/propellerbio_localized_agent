"""Wait for finite app work, including replacement sidebar refreshes."""
import asyncio

from textual.app import App
from textual.worker import WorkerCancelled


async def wait_for_app_workers(
    app: App, *, cancelled_groups: frozenset[str] = frozenset(), timeout: float = 10
) -> None:
    """Ignore only expected cancellation; propagate errors and wait for successors.

    Refresh workers are exclusive: a newer refresh routinely cancels an older
    one. Chat cancellation must be explicitly allowed by a test exercising Esc
    or /forget. This helper is not for agent-pack's permanent event listener.
    """
    allowed = {"history", "history-poll", "connection"} | cancelled_groups
    deadline = asyncio.get_running_loop().time() + timeout
    while workers := list(app.workers):
        tasks = [asyncio.create_task(worker.wait()) for worker in workers]
        try:
            _, pending = await asyncio.wait(
                tasks, timeout=max(0, deadline - asyncio.get_running_loop().time())
            )
            if pending:
                raise TimeoutError("App workers did not settle before the test deadline")
            for worker, task in zip(workers, tasks):
                try:
                    task.result()
                except WorkerCancelled:
                    if worker.group not in allowed:
                        raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # A completed worker may have started titles or replaced a refresh.
        await asyncio.sleep(0)
