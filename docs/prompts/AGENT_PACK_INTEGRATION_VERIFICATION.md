# Agent-pack integration verification

Branch: `feature/agent-pack-server`, based on `main` at `c5a1361`.
The owner confirmed D3, D10, D16, D21 and D28 on 2026-09-28 and authorized
pushing the branch and opening a pull request. The decisions are recorded in
[the plan](AGENT_PACK_INTEGRATION.md#2-decisions-where-the-codebases-overlap-and-which-version-wins).

## Validation

The implementation handoff at `6a9db58` recorded these Python 3.13 results;
the saved final test output was verified before publication:

| Install | Result |
| --- | --- |
| Full extras | 2,231 passed |
| Terminal only (`dev,api`) | 1,985 passed |
| Minimal server (`dev,server`) | 2,224 passed, 7 skipped |

A fresh pre-publication run on 2026-09-28 passed **311 tests**:

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/server tests/test_agentpack_server_features.py \
  tests/test_import_boundaries.py tests/test_model_call_guard.py \
  tests/test_server_package_data.py tests/test_scheduler.py \
  tests/test_openai_backend.py
```

`git diff --check origin/main...HEAD` passed. Fetching the remote confirmed
that `main` had not advanced beyond `c5a1361`.

The implementation handoff also recorded a wheel-install smoke test covering
mock chat, SSE, approvals, audit aggregation, static assets and private data
paths; terminal-client checks covering features, history, context, tier
selection and cancellation; and real Ollama checks covering token usage and
budget exhaustion. These smoke tests were not repeated during publication.

## Known flake and outstanding checks

`tests/test_queue_execution.py::test_switching_chats_pauses_the_queue_and_returning_needs_a_resume`
is a pre-existing terminal timing flake, reported at about one run in eight
under load, including in the terminal-only environment. This branch does not
change that test. A failure there needs a rerun before it is classified as a
server regression; the flake is not fixed or hidden by this change.

Docker build: not run because Docker is not installed. Browser walk-through:
not run because no browser is connected to the automation environment.
Remaining manual checks are demo login, CSV upload, concurrent Ask and Create
Agent runs, regulatory approval and the recorder readiness message.

The full local environment uses Python 3.13 without rapidocr; CI uses Python
3.12 and includes recorder extras. CI results remain necessary to verify that
platform and dependency combination.

## Review and deployment notes

The imported history is preserved, with the WAV removed before import. The
source-to-import commit map is retained locally at
`/Users/li/bedrock-agentpack-commit-map.txt`.

The integration is one PR, with separate commits for the history import,
packaging, scheduler, runtime, compliance, memory, terminal client, tools and
documentation. No source-repository changes are required.

The legacy agent API remains unauthenticated, and terminal sign-in is a
follow-up. Orchestration, event replay and approval gates remain single-process
and partly in memory. Deployment requires the documented trusted-network
boundary and one worker; see [deployment guidance](../server/DEPLOYMENT.md).
