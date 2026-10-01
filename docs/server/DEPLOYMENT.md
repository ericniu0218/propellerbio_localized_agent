# Deploying the Bedrock server (`bedrock-server`)

For IT. The server is the multi-agent system, the scientist workspace and the
meeting recorder (formerly the separate `bedrock-agent-pack` repository). It
runs on the company's server, next to the model servers. Employees never
install it: their terminal app (`bedrock`) talks to it with
`BEDROCK_BACKEND=agentpack` and `AGENT_PACK_API_BASE=http://<server>:8000`.

## Install

```bash
pip install "bedrock-private-ai[server,postgres,recorder,speakers]"   # add ,chem for RDKit/Biopython tools
bedrock-server check        # everything missing, all at once; exit 1 on problems
bedrock-server              # listens on 127.0.0.1:8000 by default
```

`website` is an alias for `bedrock-server`. For a local browser workspace, run
`website`, then open http://localhost:8000. It uses the same mode and settings;
the default is mock mode with demo sign-in.

Or with Docker, from the repository root:

```bash
docker build -f docker/server/Dockerfile -t bedrock-server .          # --build-arg EXTRAS=... to change extras
docker compose -f docker/server/compose.yml up --build                # mock mode, with Postgres
docker compose -f docker/server/compose.yml --profile real up --build # plus the vLLM model servers
```

A finished literature review downloads from the Research tab as HTML, PDF,
JSON and - when the review was started with the deck option - PowerPoint. The
report PDF needs nothing beyond the `server` extra (reportlab, with its font in
the package). **Slides as PDF** are the PowerPoint converted by LibreOffice on
this server, so they need LibreOffice Impress installed
(`apt install libreoffice-impress`, or `--build-arg SLIDES_PDF=1` for the
Docker image); the button appears only when it is found (`soffice` on PATH, or
`BEDROCK_SOFFICE=/path/to/soffice`). Nothing in either conversion leaves the
server.

Python 3.11 or 3.12. On 3.13+ the recorder's screen reading
(`rapidocr-onnxruntime`) does not install; everything else works and the
recorder falls back to telling speakers apart by voice.

**One worker.** The event stream, approvals and running jobs live in the
process's memory. `bedrock-server` refuses `--workers`; do not run it behind
several uvicorn/gunicorn workers.

## Settings

Environment variables win over the `[server]` table of
`~/.config/bedrock/config.toml` (of the account running the server), which wins
over defaults. The agent pack's variable names are unchanged.

| Variable | `[server]` key | Default |
|---|---|---|
| `AGENT_PACK_MODE` | `mode` | `mock` (fixture agents, no model) — set `real` |
| `BEDROCK_SERVER_DATA_DIR` | `data_dir` | `~/.local/share/bedrock/server` (Docker: `/var/lib/bedrock`) |
| `DATABASE_URL` | `database_url` | `sqlite:///<data_dir>/agent_pack.db`; Postgres: `postgresql+psycopg2://...` |
| `BEDROCK_SERVER_CONFIG_DIR` | `config_dir` | `~/.config/bedrock/server` — whole-file overrides of `models.yaml`, `budgets.yaml` |
| `BEDROCK_SERVER_HOST`, `BEDROCK_SERVER_PORT` | `host`, `port` | `127.0.0.1`, `8000` |
| `BEDROCK_SERVER_KNOWLEDGE_DIR` | `knowledge_dir` | none — the company documents `rag_search` reads (see below) |
| `BEDROCK_SERVER_LITERATURE_DIR` | `literature_dir` | none — a local literature mirror for `literature_search` |
| — | `[server.connectors.<name>]` | none — systems of record for `eln_lims`, same shape as the terminal app's `[connectors.<name>]` |
| `AIRGAP_CHECK` | `airgap_check` | `true` — real mode refuses to start if the internet is reachable |
| `BEDROCK_SERVER_INFERENCE_TOKEN` | — (a secret: environment only) | unset = in real mode the inference API refuses everyone; in mock mode it serves only this machine |
| `BEDROCK_SERVER_INFERENCE_ROLES` | `inference_roles` | `runner,specialist,director` — the roles the inference API serves |
| `BEDROCK_SERVER_INFERENCE_API` | `inference_api` | `true` — `false` switches `/v1` off |
| `BEDROCK_SERVER_ALLOW_MODEL_DOWNLOAD` | — | `false` — real mode sets `HF_HUB_OFFLINE=1` |
| `DEMO_MODE` | — | follows the mode: mock → demo data and demo sign-in, real → none |
| `KEYCLOAK_BASE_URL`, `KEYCLOAK_REALM`, `KEYCLOAK_CLIENT_ID`, `KEYCLOAK_CLIENT_SECRET`, `APP_BASE_URL` | same, lower-case | unset = development sign-in |
| `WORKSPACE_DATA_DIR`, `MEETING_RECORDING_DIR` | — | under `data_dir` |
| `MEETING_*`, `DATACENTER_*` | — | as documented in `MEETINGS.md` |

**Models** live in `models.yaml` (packaged in `bedrock/server/defaults/`):
`director`, `specialist`, `runner` and `notes` (meeting summaries), each with a
model name, an OpenAI-compatible endpoint (vLLM, SGLang or Ollama), its context
length and `max_parallel`. The `${VAR:-default}` entries read
`DIRECTOR_LLM_URL`, `DIRECTOR_MODEL_NAME`, `DIRECTOR_MAX_PARALLEL` and the
like, and `MEETING_SUMMARY_MODEL/ENDPOINT/CONTEXT`. Roles on the same endpoint
share one queue. Every call reports its token counts, and per-task budgets
(`budgets.yaml`) stop a request that runs over.

## The inference API (literature reviews from the terminal)

A scientist's `/review` (and `bedrock-litreview ask`) runs the literature
review pipeline on their own machine. In agent-pack mode their machine has no
model, so each of the review's model calls - hundreds per review, each a
narrow structured job - goes to this server's **inference API**, never to the
conversational `/messages`:

    GET  /v1/models             roles served, each with its window and capabilities
    POST /v1/chat/completions   one call to one named role

Models are named by role (`runner`, `specialist`, `director`); each entry
reports the weights behind it as `root`, and the client records them on every
review. Every call waits in the role's model-server queue at `USER_JOB`
priority (the same queue the agents use), and is logged as metadata only -
role, sizes, status, duration - never its text.

What IT sets:

- **A token.** Set `BEDROCK_SERVER_INFERENCE_TOKEN` to a long random value
  and give each laptop the same value as its `api_key`
  (`api_key = "..."` in `~/.config/bedrock/config.toml`, or `BEDROCK_API_KEY`).
  Without a token a real-mode server refuses every inference call: behind a
  reverse proxy on the same host every caller looks local, so "local only"
  would not be a boundary. (A mock/development server without a token serves
  only callers on the same machine.)
- **Windows that are true.** A review needs at least 12,288 tokens per call.
  Each role's `context_length` in `models.yaml` must not exceed the
  `--max-model-len` its vLLM was started with; the API reports the smaller of
  the two, and refuses a call that would not fit rather than let vLLM truncate
  it.
- **Capabilities.** `supports_json_schema` and `supports_seed` (default
  `true`, as vLLM and Ollama do both) per role. Set either to `false` for an
  engine that ignores it: the API then refuses such a request, and the review
  says which model cannot do what, instead of silently getting free-form text.

The API refuses any parameter it does not pass on (by name, HTTP 400), and in
mock mode it still calls the roles' real model servers - fixture agents cannot
honour a JSON schema. `GET /health` lists `inference_api` among its features,
so a laptop can tell an older server apart. On a laptop,
`bedrock-litreview doctor` checks the whole path: server, key, models,
windows, structured output.

## Migrating an existing agent-pack deployment

The data used to live in the working directory. To keep it where it is, set:

```bash
DATABASE_URL=sqlite:////path/to/old/agent_pack.db   # or your Postgres URL, unchanged
WORKSPACE_DATA_DIR=/path/to/old/data/workspace
MEETING_RECORDING_DIR=/path/to/old/data/recordings
DEMO_MODE=false     # or true, whichever the deployment ran - it used to default to true
```

`DEMO_MODE` selects `data/workspace/demo` or `data/workspace/live`. It used to
default to `true` even in real mode, so an old real deployment may have its
workspace under `demo/`. Either set `DEMO_MODE=true` to keep reading it, or
move `demo/` to `live/`.

Replace `uvicorn app.main:app` with `bedrock-server` in service files.

## Company documents for `rag_search`

Put the documents (text, Markdown, CSV, JSON, logs) in a folder, point
`knowledge_dir` at it, and build the index:

```bash
bedrock-server index
```

Rebuild after the folder changes. Before the index exists, the folder is read
directly with a simpler keyword ranking. An optional `access.toml` in the
folder restricts files to access groups (first match wins; unlisted files are
`general`):

```toml
"clinical/*" = ["clinical"]
"QA-*.txt" = ["qa", "clinical"]
```

A real server with no `knowledge_dir` searches nothing. The packaged fictional
samples are only used in mock mode.

## Sign-in

With the four `KEYCLOAK_*` variables set, the workspace shows "Log in with
company account". Register `APP_BASE_URL/auth/callback` as a redirect URI on
the Keycloak client. Signing in opens the workspace; the legacy page at
`/legacy` uses `/auth/login?next=/legacy`.

**The REST API the terminal app uses (`/sessions`, `/messages`, `/approvals`,
the event stream) has no sign-in yet** — the terminal app cannot sign in.
Keep the server on the internal network. An approval sent there records the
name the client gives, marked `signer_verified: false`; one given in the
workspace under Keycloak is marked `true`.

## Air gap

- Real mode checks at start that the internet is unreachable (`AIRGAP_CHECK`).
- Nothing downloads at runtime. Stage these on the server first:
  - whisper.cpp (`whisper-cli`, `whisper-vad-speech-segments`) and ffmpeg on
    `PATH` (the Docker image has ffmpeg; mount whisper.cpp);
  - the speech and voice-activity models in `MEETING_MODEL_DIR`
    (`ggml-large-v3-turbo.bin`, `ggml-silero-v5.1.2.bin`);
  - the meeting-search embedding model: download `minishlab/potion-base-8M`
    once on a connected machine, copy the folder over and set
    `MEETING_TEXT_MODEL` to its path. Without it, meeting search ranks by
    keywords.
- `DATACENTER_BASE_URL` must be the customer's internal connector, never the
  public Zoom relay (refused unless `AGENT_PACK_ALLOW_PUBLIC_DATACENTER=true`).

## Checking a deployment

`bedrock-server check` prints the mode, data directory, database, whether each
model server answers (real mode), what the meeting recorder is missing
(optional) and whether single sign-on is configured.
