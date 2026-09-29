"""Forget every process-wide object the server has built, so it can be built again.

The server keeps one of several things per process - settings, the database
engine, the event bus, the approval gate, the model and tool registries, the
meeting job store - each made on first use and cached. That is right for a
running server and wrong for a test that changes a setting and wants the next
request to see it. ``reset_server_state`` drops them all; the next use rebuilds
them from the current environment.

Only for tests and tools that start a server twice in one process. Calling it in
a running server would orphan in-flight work (a pending approval, a running
recording job).
"""

from __future__ import annotations


def reset_server_state() -> None:
    from bedrock.server import config
    from bedrock.server.auth import state_store
    from bedrock.server.compliance import approval_gates, audit, rbac
    from bedrock.server.db import base, database
    from bedrock.server.events import bus
    from bedrock.server.meetings import routes as meeting_routes
    from bedrock.server.models import hub as model_hub
    from bedrock.server.models import registry as model_registry
    from bedrock.server.task_store import store as task_store
    from bedrock.server.tools import registry as tool_registry

    for cached in (
        config.get_settings,
        config.load_models_config,
        config.load_budgets_config,
        database.get_database,
        bus.get_event_bus,
        task_store.get_task_store,
        approval_gates.get_approval_gate,
        model_registry.get_model_registry,
        model_hub.get_model_hub,
        tool_registry.get_tool_registry,
        rbac.get_rbac_filter,
        state_store.get_state_store,
        meeting_routes.job_store,
        meeting_routes.recording_jobs,
    ):
        cached.cache_clear()
    audit.reset()
    base.dispose_engine()
