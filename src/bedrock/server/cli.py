"""``bedrock-server``: run the agent-pack server, or check that it could run.

    bedrock-server [--host H] [--port P] [--reload]
    bedrock-server check

**Always one worker.** The event stream's history, the human-approval gates and
the running jobs live in this process's memory (``bedrock.server.events.bus``,
``compliance.approval_gates``, ``meetings.jobs``). With two workers, an approval
sent to one process would never reach the request waiting in the other, and a
reconnecting event stream would find no history - silently. So ``--workers`` is
refused unless it is 1. Scaling out needs a shared bus first (see
``docs/server/DEPLOYMENT.md``).

``check`` reports everything a deployment needs, all at once, and exits 1 if
anything is missing: settings, the data directory, the model servers named in
``models.yaml`` (real mode), what the meeting recorder is missing, and single
sign-on. It never starts the server.
"""

from __future__ import annotations

import argparse
import sys

ONE_WORKER = (
    "bedrock-server runs one worker: the event stream, approvals and jobs live in "
    "one process's memory, and a second worker would silently split them. "
    "See docs/server/DEPLOYMENT.md."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bedrock-server", description="Run the Bedrock agent-pack server.")
    parser.add_argument("command", nargs="?", choices=["run", "check"], default="run")
    parser.add_argument("--host", help="address to listen on (default: [server] host, or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (default: [server] port, or 8000)")
    parser.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    parser.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        from bedrock.server.config import get_settings
    except ImportError as exc:  # the [server] extra is not installed
        print(f"The server is not installed ({exc.name} is missing). "
              'Install it with: pip install "bedrock-private-ai[server]"', file=sys.stderr)
        return 2

    if args.workers != 1:
        parser.error(ONE_WORKER)
    if args.command == "check":
        return check()

    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "bedrock.server.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        workers=1,
    )
    return 0


def check() -> int:
    """Print every problem a deployment would hit; exit 1 if there are any."""
    import asyncio

    from bedrock.server.config import get_settings, load_models_config, private_dir
    from bedrock.server.meetings.config import RecordingConfig, readiness_issues

    settings = get_settings()
    problems: list[str] = []
    print(f"mode: {settings.mode}")
    try:
        private_dir(settings.data_dir)
        print(f"data directory: {settings.data_dir}")
    except OSError as exc:
        problems.append(f"The data directory {settings.data_dir} cannot be created: {exc}")
    print(f"database: {settings.database_url.split('@')[-1]}")

    models = load_models_config()
    if settings.mode == "real":
        problems += asyncio.run(_unreachable_models(models))
    else:
        print("models: mock mode, no model server is called")

    meeting_issues = readiness_issues(RecordingConfig.from_env())
    if meeting_issues:
        print("meeting recorder: not ready (optional - everything else works without it):")
    for issue in meeting_issues:
        print(f"  - {issue}")
    print("single sign-on: " + ("Keycloak configured" if settings.sso_enabled else "not configured (development sign-in)"))

    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


async def _unreachable_models(models) -> list[str]:
    from bedrock.inference.openai_compatible import OpenAICompatibleBackend

    problems = []
    for role in ("director", "specialist", "runner"):
        config = models.for_role(role)
        backend = OpenAICompatibleBackend(base_url=config.endpoint, api_key=config.api_key, timeout=5.0)
        try:
            reachable = await backend.check_connection()
        finally:
            await backend.close()
        print(f"{role}: {config.model_name} at {config.endpoint} - {'reachable' if reachable else 'NOT reachable'}")
        if not reachable:
            problems.append(f"The {role} model server at {config.endpoint} did not answer.")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
