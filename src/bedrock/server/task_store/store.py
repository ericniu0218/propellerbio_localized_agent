"""Task graph + reference store used by the orchestrator and agents.

Agents pass *references* (task IDs, result_refs) to each other; only the
code that actually needs content calls `get_result`. This is the mechanism
behind the token-efficiency requirement in docs/ARCHITECTURE.md section 6.
Persistence is delegated to app.db.Database (SQLite in mock/test mode,
Postgres in real mode); this module adds the task-graph-shaped API the
orchestrator wants plus an in-process cache for hot reads within a request.
"""

from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

from bedrock.server.db import Database, get_database
from bedrock.server.schemas.enums import TaskStatus
from bedrock.server.schemas.tasks import TaskNode, TaskPlan


class TaskStore:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._plans: dict[str, TaskPlan] = {}

    # -- plans / graph --------------------------------------------------------
    def save_plan(self, plan: TaskPlan) -> None:
        self._plans[plan.session_id] = plan
        for node in plan.nodes:
            self._db.upsert_task(plan.session_id, node)

    def get_plan(self, session_id: str) -> TaskPlan | None:
        return self._plans.get(session_id)

    def get_task(self, task_id: str) -> TaskNode | None:
        return self._db.get_task(task_id)

    def get_task_session_id(self, task_id: str) -> str | None:
        return self._db.get_task_session_id(task_id)

    def update_task(self, session_id: str, node: TaskNode) -> None:
        self._db.upsert_task(session_id, node)
        plan = self._plans.get(session_id)
        if plan:
            for i, n in enumerate(plan.nodes):
                if n.task_id == node.task_id:
                    plan.nodes[i] = node
                    break

    def ready_tasks(self, session_id: str) -> list[TaskNode]:
        """Tasks whose dependencies are all completed and that are still pending."""
        nodes = self._db.list_tasks(session_id)
        done_ids = {n.task_id for n in nodes if n.status == TaskStatus.COMPLETED}
        return [
            n
            for n in nodes
            if n.status == TaskStatus.PENDING and set(n.depends_on) <= done_ids
        ]

    # -- references -------------------------------------------------------------
    def store_result(self, session_id: str, content: str, content_type: str = "text/plain") -> str:
        return self._db.store_result(session_id, content, content_type)

    def get_result(self, result_ref: str) -> str | None:
        return self._db.get_result(result_ref)

    @staticmethod
    def new_task_id() -> str:
        return f"task_{uuid4().hex[:10]}"


@lru_cache
def get_task_store() -> TaskStore:
    return TaskStore(get_database())
