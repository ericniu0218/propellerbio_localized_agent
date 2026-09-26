# Literature review: integration and deploy-readiness review

*Audit of branch `litreview/deploy-readiness-audit` (HEAD `97b9383`), 25 September 2026.
Read-only: no code, tests or config were changed to produce it. Written by an AI
model (Claude Opus) acting as auditor; no person has reviewed it.*

## Summary

1. **Verdict: the integration is sound, but scientists cannot deploy it on their own yet.** The wiring, the scheduler bridge, the packaging and the guarantees are correct and tested; the guarantee-carrying code is byte-identical to the old repository. A clean wheel install runs a review.
2. **Blocker 1, documentation:** `INSTALL.md` never mentions the review, and its laptop advice (an 8,192 window) and example server address (`https://something.company.com`) both make `/review` refuse to start.
3. **Blocker 2, IT setup:** the `office` model set names two placeholder models at one server address, and model names can only be changed by a file on each scientist's machine. Company host names also need `allowed_hosts`.
4. **Also:** four small bugs a scientist will hit: a dragged-in corpus file, a mistyped `--resume` id, the app's 120-second timeout, and a silent empty search. (Three are now fixed and the timeout is documented. See the status note above the fix list.) Tests: 1,474 passed, 2 skipped (the 2 API tests, which pass when the `api` extra is installed).
5. **Capabilities: 51 new since the move, 26 carried over, 18 opportunities not yet wired** (journal, workspaces, connectors, drop folder, Stage 9 chat, semantic retrieval and others).

## How this was checked

- Read, in order: root `CLAUDE.md`, `src/bedrock/litreview/CLAUDE.md`,
  `docs/prompts/LITREVIEW_INTEGRATION.md` (the integration plan, used as the
  checklist), `docs/litreview/README.md`, `INSTALL.md`, `README.md`.
- "Before" is the old repository's local clone at
  `~/Localized-Airgap-LLM-for-literature-review`, plus the verbatim-copy commit
  `ef42055`. "After" is this branch.
- Built a wheel from this checkout, installed it into a throwaway virtual
  environment (dependencies from PyPI, no FastAPI), and ran the scientist
  commands from outside the repository with `HOME` pointed at a scratch folder.
  Nothing touched the real home directory. No model server was started or
  called, and no live literature search was run. A server that is down was
  simulated with `BEDROCK_API_BASE=http://localhost:11439/v1` (nothing listens
  there).

## Part 1: Integration check

### Scorecard

| # | Check | Result |
|---|---|---|
| 1 | Entry points (`bedrock-litreview`, `/review`, `/reviews`, API extra) | **PASS** (stale help text noted) |
| 2 | Packaging: runtime data files in the wheel | **PASS** (no guard against regressions) |
| 3 | Fresh install from the wheel, then a no-model review | **PASS** |
| 4 | Model-call rule (`SchedulerClient`, `USER_JOB`, guard test) | **PASS** (import rule is untested) |
| 5 | Configuration (`[litreview]`, env, profile window, model sets) | **PARTIAL** |
| 6 | Data locations, test isolation, `/forget` and privacy | **PARTIAL** |
| 7 | Privacy: host allowlist, redirects, no telemetry, search on request only | **PASS** |
| 8 | First-run check (`doctor.py`) | **PARTIAL** |
| 9 | Tests | **PASS** (API tests skipped in the dev venv and in CI) |
| 10 | Documentation a scientist would follow | **FAIL** |
| 11 | Error paths a scientist will hit | **PARTIAL** |

The guarantee-carrying code is untouched by the move. I re-ran the plan's
Phase 8 byte-identity check against the old repository (`dc5c012`): after the
import rewrite, `llm/prompts`, `llm/cache.py`, `llm/prompts.py`, `knowledge`,
`stages`, `extract`, `report/guard.py`, `report/invariants.py`, `models.py`,
`facets.py`, `lexical.py`, `negation.py`, `ingest.py` and `parsing` are all
byte-identical. Only `api.py`, `bench/__main__.py`, `bench/run.py`, `cli.py`,
`config.py`, `db.py`, `llm/__init__.py`, `llm/client.py`, `report/layout.py`,
`runner.py` and `sources/http.py` differ, plus the new `service.py`,
`llm/scheduled.py` and `defaults/`. That matches the plan exactly.

### 1. Entry points - PASS

| What | Evidence |
|---|---|
| `bedrock-litreview` script | `pyproject.toml:52`; present in the built wheel's `entry_points.txt` |
| `python -m bedrock.litreview` | `src/bedrock/litreview/__main__.py`; ran from the fresh install, exit 0 |
| `/review` and `/litreview` (default mode) | `src/bedrock/app.py:283-284` -> `start_review` `app.py:382-409`, worker `app.py:429-441` |
| `/review stop` | `src/bedrock/ui/review_command.py:101-102`, `app.py:389-390`, `app.py:411-419` |
| `/reviews`, `/reviews verify` (both modes) | `src/bedrock/ui/local_commands.py:79-80`, `130-143`; `review_command.py:381-506` |
| `/review` in agent-pack mode | Refused with a pointer: `src/bedrock/agentpack_app.py:81-86`, `276-280` |
| Help and command list | `src/bedrock/ui/commands.py:31-34`, `63-71`, `120-122`; agent-pack help `agentpack_app.py:100-101` |
| Optional `api` extra | `pyproject.toml:34`; app factory `src/bedrock/litreview/api.py:129-215` |

`bedrock-litreview --help` and `search`, `ask`, `verify-run`, `runs` and
`rate --help` all exit 0 from the fresh install. `python -m
bedrock.litreview.bench --help` also works. With FastAPI added to the throwaway
venv, `create_app()` builds and exposes `/ask`, `/reports/{report_id}` and
`/reports/{report_id}/ratings`.

Problems found:
- **Stale help text in `cli.py`.** `--out` says `(default: out/)` at
  `cli.py:166`. The real default is `~/.local/share/bedrock/litreview/reports/`
  (`runner.py:345`, `config.py:263`). `--sources` says
  `(default: config/pipeline.yaml)` at `cli.py:172`; the file is now
  `defaults/pipeline.yaml`. `verify-run --db` says the benchmark uses
  `data/bench.sqlite3` at `cli.py:178`; it is now `<data_dir>/bench.sqlite3`
  (`bench/run.py:214`). Small fix.
- **Dead constant.** `AGENTPACK_NOTICE` at `review_command.py:62-65` is never
  used; agent-pack mode has its own copy (`agentpack_app.py:81-86`). Both tell
  the user to start without `BEDROCK_BACKEND=agentpack`, but `INSTALL.md:105-108`
  sets agent-pack mode with `backend = "agentpack"` in `config.toml`, so the
  instruction does not match how a scientist got into that mode. Small fix.

### 2. Packaging - PASS

Every file the review reads through a path relative to the package:

| Read at | File(s) | In the wheel? |
|---|---|---|
| `config.py:43`, `111` | `defaults/models.yaml`, `pipeline.yaml`, `report.yaml` | Yes |
| `llm/prompts.py:22`, `53` | `llm/prompts/*.md` (9 files) | Yes, all 9 |
| `knowledge/__init__.py:33` | `knowledge/entities.yaml` | Yes |
| `extract/verify.py:214` | `knowledge/meaning_words.yaml` | Yes |
| `runner.py:108` | every `knowledge/*.yaml` (hashed onto the run) | Yes |

The globs are at `pyproject.toml:61-63`. The wheel built from this checkout
(582,593 bytes) holds all 14 data files and 67 `litreview` Python modules.
`evals/litreview/` is deliberately not in the wheel (`config.py:45-48`); only
the benchmark reads it. In an installed wheel, `EVALS_DIR` points at a path
that does not exist, which is harmless because scientists never run the
benchmark.

Risk: nothing stops this regressing. CI installs in editable mode
(`.github/workflows/test.yml:34`), which reads files straight from `src/`, so a
new prompt or YAML file outside the three globs would pass every test and break
every real install. The plan's optional `dist/installed_smoke.py` was not
added. Small fix: one test that lists every non-`.py` file under
`src/bedrock/litreview` and checks that each matches a `package-data` glob.

### 3. Fresh install as a scientist would do it - PASS

Free disk space was 8.4 GB, so this was run for real.

1. Built the wheel: `pip wheel . --no-deps` (the dev venv has no setuptools, so
   the build had to fetch it). The build left a `build/` folder in the
   checkout, which I removed.
2. New venv, `pip install <wheel>`: all dependencies from PyPI, 78 MB, exit 0.
3. From outside the repository with `HOME` pointed at a scratch folder:

| Command | Result |
|---|---|
| `bedrock-litreview --help` and every subcommand's `--help` | exit 0 |
| `bedrock-litreview ask "If I want to prevent psoriasis, what should I target?" --no-model --corpus evals/litreview/corpora/psoriasis-prevention-targets/papers.jsonl --out <scratch>/out` | exit 0 in 0.46 s. 108 papers collected, 84 pass the exact-entity gate, 16 planned for reading. Wrote `report.html`, `report.md`, `report.json` and `run.json` |
| Same question again, no `--corpus` | exit 0; "reusing the frozen set of 108". **Same report id** (`61e35796584704f6`) |
| Same again, no `--out` | Report under `~/.local/share/bedrock/litreview/reports/…` |
| `runs` | Both runs listed with their report id |
| `verify-run <run>` | `OK`, exit 0 |
| `rate 61e35796584704f6 4 --rater jd --comment test` | Rating stored, exit 0. Works **without** FastAPI installed |
| `ask … --models office2` | `unknown model set 'office2'; known: ['laptop', 'medgemma', 'office']`, exit 1 |

The data directory was created `drwx------` and the database `-rw-------`.
The throwaway venv was deleted afterwards.

### 4. The model-call rule - PASS

- The only live client is `SchedulerClient` (`llm/scheduled.py:65-211`). Every
  call goes to `scheduler.complete(...)` at `self._priority`
  (`scheduled.py:133-140`), which defaults to `Priority.USER_JOB`
  (`scheduled.py:81`). `make_client` never passes another priority
  (`llm/__init__.py:63-71`), and without a scheduler it refuses
  (`llm/__init__.py:72-76`).
- `service.run_review` is the only route in: it runs `preflight`, then builds
  the client on the running loop (`service.py:129-138`). The CLI
  (`cli.py:95-109`), the app (`review_command.py:330-336`), the API
  (`api.py:177-183`) and the benchmark (`bench/__main__.py:76-77`) all go
  through the scheduler.
- Only `llm/scheduled.py:38-39` and `service.py:33`, `192` import
  `bedrock.inference`. No file under `litreview/llm/` imports an HTTP client.
- `tests/test_model_call_guard.py:23`, `38-47` walks all of `src/bedrock`, so
  it covers `litreview`. It passes.

Gap: the guard only looks for `stream_chat` calls. Nothing tests the
litreview-specific rules: "only `llm/scheduled.py` and `service.py` import
`bedrock.inference`", and "no HTTP client under `litreview/llm/`". Both hold
today. Small fix: one `ast` test in `tests/litreview/`.

### 5. Configuration - PARTIAL

What works:
- `Settings.litreview` is parsed from the `[litreview]` table
  (`src/bedrock/config/settings.py:193`, `283`, `332`, `367`) and validated only
  when a review runs, so a typo never stops the app (`config.py:280-293`).
  Unknown keys are refused with the allowed list (`config.py:58`, `287-292`).
- Model set precedence: `--models`, then `BEDROCK_LITREVIEW_MODELS`, then
  `model_set`, then the machine profile (`config.py:225-231`); `office` on the
  `server` profile, else `laptop` (`config.py:61-62`).
- Server, key, timeout and window come from bedrock, never restated
  (`config.py:256`, `259`).
- `preflight` refuses a window below 12,288 and names both numbers and the fix
  (`service.py:71-81`). Run for real with the default `laptop` profile:
  "a literature review needs a context window of at least 12,288 tokens, and
  this machine is set to 8,192 … Set `context_window = 12288` … start Ollama
  with OLLAMA_CONTEXT_LENGTH=12288 …". Exit 1, before any model call.
- Whole-file overrides in `~/.config/bedrock/litreview/` (`config.py:103-118`),
  and the run records which file it read (`runner.py:95-137`). Knowledge tables
  cannot be overridden (`runner.py:106-108`, test
  `tests/litreview/test_bedrock_settings.py:133`).
- Temperature must be 0.0 (`config.py:241-245`).

What does not:
- **The `office` model set cannot run on one vLLM server as written.** It names
  two models, `qwen3-8b-instruct` and `qwen3-30b-a3b-instruct`
  (`defaults/models.yaml:18-20`), at one `api_base`. `preflight`'s own message
  says "a vLLM server offers only the model it was started with"
  (`service.py:99-100`). So the default on the `server` profile fails
  preflight unless IT runs a router in front of two servers, or serves the
  models under exactly these names. The names are placeholders, not real
  Hugging Face ids.
- **Model names can only be changed per machine.** The override lives in each
  user's `~/.config/bedrock/litreview/models.yaml`. The `[litreview]` table can
  pick a set but cannot name models (`config.py:58`). For an office rollout,
  IT must copy a file onto every scientist's machine, or change the packaged
  default. Big-ish fix, and a product decision: either ship the real office
  names in `defaults/models.yaml` once IT fixes them, or let `[litreview]`
  carry the names.
- **Company host names are refused by default.** `assert_local_url` accepts
  only `localhost`, private IPs and `.local`, `.internal`, `.lan`, `.home.arpa`
  (`config.py:69`, `72-89`). A server at `https://ai.propellerbio.com` is
  refused unless `allowed_hosts` names it. The message is clear (verified), but
  `INSTALL.md:95` tells the scientist the address "looks like
  `https://something.company.com`", so the documented setup fails for reviews
  while chat works. Deliberate safety rule (plan D9); needs documenting and
  ideally a first-run hint.
- **In the app, the review uses the app's 120-second timeout.** The CLI, API
  and benchmark raise it to at least 600 s (`service.py:177-202`). `/review`
  uses the app's shared scheduler and backend (`app.py:432-436`), so it keeps
  `request_timeout` (default 120 s, `settings.py:291`). Streaming reads time out
  per gap between bytes, so this mostly bites when the server loads the review
  model and reads a 4,300-token prompt on a slow laptop. The result is
  `ModelServerUnavailable`, and the stop notice then says to raise
  `request_timeout` (`review_command.py:256-260`). Preflight does not warn
  first. Small fix: have `/review`'s preflight warn on the `laptop` profile
  when `request_timeout < 600`.

### 6. Data locations and privacy of stored data - PARTIAL

- Default data directory `~/.local/share/bedrock/litreview/`
  (`config.py:50`), override by `BEDROCK_LITREVIEW_DIR` or `data_dir`
  (`config.py:247-251`). Database, cache and reports are derived from it
  (`config.py:262-264`).
- Created private: `private_dir` (`config.py:121-130`), called at the start of
  every run (`service.py:126`). The database is chmod 0600 (`db.py:72-75`).
  Cache files are written with `mkstemp` (0600, `llm/cache.py:77`). Report
  files are 0644 inside 0700 directories. The parent blocks access, but the
  docstring promises 0600 files (`config.py:125-126`). And `ask --out
  <elsewhere>` writes world-readable reports wherever the user points it.
- **Tests never write to the real home.** `private_bedrock_directories`
  redirects `DEFAULT_DATA_DIR` and `DEFAULT_CONFIG_DIR` (`tests/conftest.py:255-256`),
  and `tests/litreview/conftest.py:22-24` hides the developer's `config.toml`.
  I checked this by running `tests/litreview` and `tests/test_review_command.py`
  with `BEDROCK_LITREVIEW_DIR` and `BEDROCK_LITREVIEW_CONFIG_DIR` exported to
  scratch folders: nothing was written there. One gap: the fixtures do not
  clear `BEDROCK_*` variables, so with `BEDROCK_API_BASE=https://example.com/v1`
  exported, `test_cli_ask_with_a_model_goes_through_the_scheduler_and_runs_lists_it`
  (`tests/litreview/test_sources.py:451`) fails. `BEDROCK_LITREVIEW_MODELS` and
  `BEDROCK_CONTEXT_WINDOW` did not break anything. Small fix: `delenv` the
  `BEDROCK_*` and `BEDROCK_LITREVIEW_*` variables in the autouse fixture.
- **`/forget` and reviews.** Review notices are never saved into a
  conversation (`review_command.py:22-24`), so `/forget` stays complete for
  chat. But **there is no way to delete a review**. `db.py` has no delete for a
  question, run or report (only `save_papers` replaces a question's paper list,
  `db.py:115`). A sensitive research question stays in `litreview.sqlite3`,
  the model-call cache (which holds full prompts) and `reports/` until someone
  deletes files by hand, and no document tells them how. That conflicts with
  the spirit of the root privacy rule that `/forget` is complete. Medium fix:
  `/reviews forget <run-id>` and `bedrock-litreview forget`, deleting the run,
  its artifacts, its report files, and the question and papers once no run
  uses them. The cache is keyed by prompt hash, so either clear it for that
  run's calls or document that it is kept.
- Unlike bedrock's conversations (`settings.py:59-68`), the litreview directory
  ignores `XDG_DATA_HOME`. This matches bedrock's other data directories
  (`settings.py:31-34`), so it is consistent, just worth knowing.

### 7. Privacy - PASS

- **Search only when asked.** `service._searcher` builds the internet searcher
  only for `search=True` (`service.py:158-174`). The CLI searches only in
  `search` and `ask --search` (`cli.py:208-226`), the app only with
  `/review --search` (`review_command.py:108`, `129`), and the API only on
  `"search": true` (`api.py:167-175`). Tested by
  `tests/litreview/test_service.py:282` and the HTTP test.
- **Host allowlist, redirects included.** `SOURCE_HOSTS = {"www.ebi.ac.uk",
  "eutils.ncbi.nlm.nih.gov"}` (`sources/http.py:40`), enforced by an httpx
  request hook that fires on every hop (`http.py:48-55`, `69-73`). All three
  source clients use `SourceHTTP` (`europepmc.py:79`, `pubmed.py:161`,
  `literature.py:197-201`). The requests are anonymous (`http.py:35-37`).
- **The app says what leaves** when a search starts (`review_command.py:57-60`,
  `173-174`).
- **No telemetry.** The only hard-coded URLs are the two source APIs, links
  written into reports (`doi.org`, `pubmed.ncbi.nlm.nih.gov`,
  `europepmc.org/article`), and Open Targets in `bench/corpus.py:38`.
  `report.html` loads nothing external: no `<script>`, `<link>` or `<img>`,
  only hyperlinks (checked on a generated report).
- **Model endpoint check.** A public model server or an Ollama `-cloud` model
  is refused for reviews (`config.py:72-100`, `273-276`).

Notes, not failures: `bench/corpus.py:61` posts disease ids to
`api.platform.opentargets.org` outside the allowlist. It is developer-only
benchmark tooling (`python -m bedrock.litreview.bench build-corpus`), but it
ships in the wheel. The endpoint check covers reviews only, not chat; that is
a known follow-up (plan §8.6).

### 8. First-run check (`src/bedrock/doctor.py`) - PARTIAL

What it does: when the profile's window is below 12,288, it adds a line saying
`/review` needs `context_window >= 12,288` and `OLLAMA_CONTEXT_LENGTH=12288`
(`doctor.py:33`, `121-133`; tested in `tests/test_doctor.py`).

What a scientist will still trip on:
- It runs **only when no `config.toml` exists** (`doctor.py:52-53`). A
  scientist whose IT pre-installed a config never sees the window warning.
- It runs only in the default mode. In agent-pack mode (`doctor.py:57-69`) it
  says nothing about reviews, which is correct because they do not run there.
- It never mentions **which models to pull** (`qwen3:8b` for `laptop`,
  `medgemma1.5:4b`), `request_timeout` for in-app reviews, or `allowed_hosts`
  for a company host name.
- It prints `api_base = "https://<the real address>"` as the fix
  (`doctor.py:78-84`), which litreview will then refuse unless the host is
  allowed.
- There is no re-runnable check. `bedrock-biotech status` is the equivalent
  for AlphaFold and PyMOL. Small-to-medium fix: `bedrock-litreview doctor` (or
  `/reviews check`) that loads settings and runs `preflight` without starting a
  run. That would give the scientist every problem at once instead of one per
  attempt.

### 9. Tests - PASS

| Run | Result |
|---|---|
| `.venv/bin/pytest tests/litreview -q` | **276 passed, 2 skipped** in 3.3 s |
| `.venv/bin/pytest -q` (whole of bedrock) | **1,474 passed, 2 skipped** in 4 min 14 s, exit 0 |
| The 2 skipped tests (`test_sources.py:462`, `:498`), run in the throwaway venv with the `api` extra, against the installed wheel | **2 passed** |

The 2 skips are `test_http_ask_searches_only_when_asked` and
`test_http_ask_says_503_when_the_machine_cannot_run_a_review`, both skipped
with "could not import 'fastapi.testclient'". The dev venv does not have the
`api` extra, and neither does CI (`.github/workflows/test.yml:34` installs
`.[dev]` only). So **CI never runs the HTTP API tests**. Small fix: install
`.[dev,api]` in CI.

The coordinator reported that the dev `.venv` was missing pyyaml, pydantic,
lxml and rapidfuzz: it was built before they were added to `pyproject.toml`,
and they re-installed it before these runs. That is a real hazard for every
developer who updates an existing checkout (see item 10). Also, a missing
dependency makes `/review` raise `ModuleNotFoundError` from
`parse_review_command`, which imports `bedrock.litreview.service`
(`review_command.py:96`). That call is inside `start_review` (`app.py:385`),
outside the `try` that catches only `CommandUsageError`, so it would surface
as an unhandled exception in the app.

### 10. Documentation a scientist would follow - FAIL

The developer and README documentation is good and accurate: no stale
`litreview ask`, `--profile`, `python -m litreview`, `config/*.yaml` or bare
`evals/` path was found in `README.md`, `INSTALL.md`,
`docs/litreview/README.md`, `src/bedrock/litreview/CLAUDE.md`,
`evals/litreview/README.md` or the two `.claude/agents/litreview-*.md` files.
The failure is that the **scientist's path from install to a first report is
not written down anywhere a scientist would look**:

1. **`INSTALL.md` never mentions the literature review.** The non-engineer
   walkthrough has no `/review`, no `bedrock-litreview`, no 12,288 window, no
   `ollama pull qwen3:8b`, no `allowed_hosts`, no `request_timeout`, and no
   troubleshooting rows for the preflight messages (`INSTALL.md:151-159`).
2. **`INSTALL.md`'s laptop advice blocks reviews.** It recommends
   `OLLAMA_CONTEXT_LENGTH=8192`, or 4096 (`INSTALL.md:120-140`). Preflight
   refuses both. The README says a review needs a 16 GB machine
   (`README.md:251-253`); `INSTALL.md` is written for 8 GB.
3. **`INSTALL.md`'s server address example would be refused** by the review
   (`INSTALL.md:95`, `113`; see item 5).
4. **`INSTALL.md`'s agent-pack setup disables `/review`** (`INSTALL.md:103-108`),
   and the refusal message names an environment variable the scientist never
   set (item 1).
5. **`docs/litreview/README.md` "Getting started" is for developers**
   (`docs/litreview/README.md:628-690`): venv, `pip install -e ".[dev]"`,
   `pytest`. The HTTP instructions say `pip install -e ".[api]"`
   (`README.md:299`, `docs/litreview/README.md:685`), which does not
   apply to a `uv tool install`. That route needs something like `uv tool
   install --force "git+…bedrock-private-ai.git[api]"` (IT-facing, low
   priority).
6. **Where papers come from is unclear for an air-gapped scientist.** Every
   example uses `--corpus papers.jsonl`. The file comes from `bedrock-litreview
   search` on a networked machine (`docs/litreview/README.md:509-511`), and
   `INSTALL.md` says nothing about this.
7. **How to open the report.** The finish notice prints the path
   (`review_command.py:241`), but nothing says to open `report.html` in a
   browser, and the app does not open it the way `/pymol` opens PyMOL.
8. **Updating an existing developer checkout.** Nothing says to re-run
   `pip install -e ".[dev]"` after pulling. That is exactly how the dev venv
   ended up without the review's four new dependencies. `INSTALL.md:145-149`
   covers `uv tool install` users (re-running it re-resolves dependencies),
   but `README.md`'s Quick start (`README.md:27-45`) does not cover updating.
9. **CLI help text** is stale in three places (item 1).
10. Cosmetic: docstrings inside the verbatim-copied modules still say
    `config/pipeline.yaml`, `config/report.yaml` or `litreview ask`
    (`stages/s5_extract.py:15`, `stages/s7_write.py:5`, `stages/s8_render.py:5`,
    `sources/query.py:6`, `sources/base.py:9`, `report/invariants.py:22`).
    Leave these until the byte-identity check is retired: editing them breaks
    the Phase 8 comparison.

Suggested fix (small, docs only): an "Literature review" section in
`INSTALL.md` that covers: needs a 16 GB machine or the office server; add
`context_window = 12288` and `request_timeout = 600`; add `[litreview]
allowed_hosts = ["ai.propellerbio.com"]` when the server has a company name;
`ollama pull qwen3:8b`; start Ollama with `OLLAMA_CONTEXT_LENGTH=12288`; how to
get `papers.jsonl`; open `report.html` in a browser. Plus troubleshooting
rows for the four preflight messages.

### 11. Error paths a scientist will hit - PARTIAL

Run for real where it was safe (no server, no network). Otherwise read from
the code.

| Situation | What the scientist sees | How checked | Verdict |
|---|---|---|---|
| Model server down | "the model server at http://localhost:11439/v1 is not answering: Could not list models: connection refused. Is the inference server running? Start it, or point `api_base` at the office server." exit 1, before the run starts (`service.py:83-89`) | Run (CLI) | Good |
| Window too small (default laptop) | The 12,288 message quoted in item 5, exit 1 (`service.py:71-81`) | Run (CLI); app test `tests/test_review_command.py:410` | Good |
| Model not pulled | "…does not have 'qwen3:8b', which the `laptop` model set needs. On a laptop running Ollama: `ollama pull qwen3:8b` … On the office server (vLLM), ask IT…" (`service.py:91-101`) | Code; `test_service.py:129`, `test_review_command.py:428` | Good; for the `office` set it still suggests `ollama pull qwen3-30b-a3b-instruct`, which is not an Ollama tag |
| Company host name | "refusing model endpoint 'https://ai.propellerbio.com/v1' … add it to `allowed_hosts` in the [litreview] table …" exit 1 | Run (CLI) | Clear message; undocumented setup step |
| Server dies mid-run | `ModelServerUnavailable` stops the run instead of producing a hollow report (`scheduled.py:205-209`). CLI prints the exact resume command (`cli.py:116-121`); app names `/review --resume <id>` (`review_command.py:246-261`) | `test_service.py:207` | Good |
| Slow laptop, app, default 120 s timeout | Stop notice advising `request_timeout` (above) | Code | Warn before, not after (item 5) |
| No network with `--search` | Each request retries 5 times with backoff (`http.py:89-109`), about 15 s per query; sources run in parallel but each runs its queries in turn (`literature.py:239-257`), so the error arrives after about 15 s times the number of queries the question produces. A failed query or source is recorded and the others carry on. If all fail: "every source failed: europepmc: …; pubmed: …" (`literature.py:217-220`); CLI exit 1, app error notice, API 502 | Code; offline tests in `test_sources.py` (no live search run) | Good |
| Empty search result (sources answer, zero hits) | Not an error. The run finishes with "papers: collected 0" and a report that says "Compiled from 0 papers…", every section "No evidence…", exit 0 | Run with an empty corpus file (same `collect` path) | Honest but quiet: a scientist can mistake it for "no literature exists". Small fix: a warning notice when 0 papers were collected or 0 passed the gate |
| New question with no papers | "no papers for this question yet — pass --corpus FILE, or --search" (`cli.py:113-115`; app `review_command.py:341-347`) | Run (CLI) | Good |
| Vague question | "Could not find a disease, pathogen, drug or target in the question. Name it explicitly, e.g. …" | Run (CLI) | Good |
| Missing corpus file | "ask: [Errno 2] No such file or directory: '/nonexistent.jsonl'" | Run (CLI) | Acceptable, terse. In the app the "Literature review started" notice comes first, because `start_review` does not check that the file exists |
| **Corpus file dragged onto the terminal** | Dropping a file pastes it as an `@` reference (`widgets.py:140-153`, `attachments.py:208-232`), so `/review … --corpus <drop>` becomes `--corpus @/Users/…/papers.jsonl`. `parse_review_command` keeps the `@` (`review_command.py:128`), and the run fails with "No such file or directory: '@/…'" | Parsed with the real `dropped_references` and `parse_review_command` | **Wrong**: drag-and-drop is how INSTALL-level users give a file. Small fix: `removeprefix("@")` on `--corpus`, as natural-language `/fold` already does (`commands.py:194`) |
| **Resume with a mistyped run id (CLI)** | **Silently starts a new run** and finishes, exit 0. `runner.py:184` treats an unknown id as "no run" | Run (CLI) | **Wrong**: the scientist believes the old run resumed |
| **Resume with a mistyped run id (app)** | `/review --resume typo123` (no question) gives **"empty question"** | Simulated through `service.run_review` | **Wrong**: does not say the id is unknown |
| `verify-run` unknown id | "cannot be verified: ValueError: run 'nosuchrun' has no finished report", exit 1 | Run (CLI) | Good |
| Quitting the app mid-review | Cancel event set before the backend closes, so the run stays resumable (`app.py:1464-1467`) | `test_review_command.py:363` | Good |
| Second `/review` while one runs | Refused: "A literature review is already running…" (`app.py:392-396`) | `test_review_command.py:335` | Good |

Small fix for both resume cases: in `service.run_review` (or `runner.py:184`),
raise `ValueError(f"no review run {id!r}; /reviews lists them")` when a
`resume` id is given and `db.get_run` returns `None`.

---

## Part 2: What the literature review gained by moving into the main app

Every item below was checked in the code on this branch. **NEW** means it did
not exist in the old repository (`dc5c012`). **CARRIED OVER** means it existed
there and survived the move; those are listed separately and briefly.
**OPPORTUNITY** means the main app has the piece but the review does not use it
yet.

### Counts

| Tag | Count |
|---|---|
| NEW since the move | **51** (one of them, A3, only partly holds) |
| CARRIED OVER | **26** |
| OPPORTUNITY | **18** |

### A. Running beside the chat

| # | What it is, and why it matters | Evidence | Tag |
|---|---|---|---|
| A1 | Every review call waits behind the scientist's own chat (`Priority.USER_JOB`), so a review of hundreds of calls never makes them wait for more than the one call in flight. **Only inside the app**: `bedrock-litreview` and the API open their own scheduler per process, so a shell review competes with the app's chat at the server | `llm/scheduled.py:81`, `133-140`; the app shares one scheduler, `app.py:137`, `435`, `1379`; `scheduler.py:14-15`; test `test_scheduler_client.py:116`. Per-process schedulers: `service.py:177-199`, `cli.py:108` | NEW |
| A2 | A review call that has started is never cancelled for a chat; the chat simply goes next. No 4,096-token extraction is thrown away half-done | `scheduler.py:18-23` | NEW |
| A3 | Background work (conversation titles) runs at a lower priority than the review, and waits for quiet on a laptop. **Caveat, from reading the code (not tested):** the idle clock only counts chat replies (`scheduler.py:239-240`, `259-260`), and `_dispatch()` runs after every finished call (`scheduler.py:262`). So a title can start in the gap between two review calls. It uses the chat's model, which on a one-model laptop can force a model swap. `README.md:248-250` says titles wait until the review finishes, which is not strictly true | `scheduler.py:24-26`, `235-245`; `titles.py:28-29`, `286` | NEW (partial) |
| A4 | Warns when the chat model differs from the review's model on a one-model machine, and says which `/model` to switch to, so the server does not swap gigabytes of weights on every message | `review_command.py:190-213`, `322-324`; tests `test_review_command.py:264`, `274` | NEW |
| A5 | `/review stop` stops the run before its next model call, and quitting the app does the same before the connection closes, so the run stays resumable | `app.py:411-419`, `1464-1467`; `scheduled.py:181-183`, `203-207`; tests `test_review_command.py:346`, `363`, `test_scheduler_client.py:171`, `183` | NEW |
| A6 | If the model server goes away mid-run, the run **stops** and can be resumed, instead of finishing a silently thinner report. In the old repo every HTTP error was a skipped item | `scheduled.py:48-62`, `205-211`; tests `test_service.py:207`, `test_scheduler_client.py:155`, `164` | NEW |
| A7 | Pipeline warnings arrive as toasts instead of being printed over the terminal screen | `review_command.py:281-297`, `326-352` | NEW |
| A8 | One review at a time, independent of a running `/fold`; `Esc` stops a chat reply but never a review | `app.py:147-151`, `392-396`; test `test_review_command.py:335` | NEW |

### B. Machine profile and context budget

| # | What | Evidence | Tag |
|---|---|---|---|
| B1 | The context window comes from bedrock's machine profile, so one setting (`context_window`, or the profile) serves chat and reviews | `config.py:259`; `config/profile.py:101-134` | NEW |
| B2 | Preflight before a run: window too small, server silent, model not pulled. Each is refused before the first call, with the fix spelled out, rather than failing forty minutes in | `service.py:64-101`; tests `test_service.py:119-145`, `test_review_command.py:410`, `428`; run for real (Part 1, item 11) | NEW |
| B3 | Every call is checked against the window using bedrock's token estimator. A prompt that cannot fit is refused rather than silently truncated by the server, and the reply limit is never shrunk (it is part of the cache key) | `scheduled.py:106-119`; tests `test_scheduler_client.py:129`, `140` | NEW |
| B4 | No per-request context size is sent, so Ollama does not reload the model every time chat and review alternate. The old client sent `num_ctx` on every call | `openai_compatible.py:193-221` (no `num_ctx`); old `litreview/llm/client.py:192` | NEW |
| B5 | The model set follows the machine: `office` on the `server` profile, `laptop` elsewhere | `config.py:61-62`, `225-231`; test `test_bedrock_settings.py:34` | NEW |

### C. The terminal UI

| # | What | Evidence | Tag |
|---|---|---|---|
| C1 | `/review <question>` in the app. The question is taken verbatim, so "What's worth targeting?" works | `review_command.py:90-152`; tests `test_review_command.py:36-86` | NEW |
| C2 | A start notice names the question, the paper source and the model set, and says exactly what leaves the machine when `--search` is used | `review_command.py:57-60`, `169-187` | NEW |
| C3 | A finish receipt: papers collected, on-subject and read, findings verified, sentences kept or removed, integrity result (a failure is shown as an error), and report paths | `review_command.py:216-243` | NEW |
| C4 | An interrupted run names the exact resume command, plus timeout advice after a server outage | `review_command.py:246-261`, `337-340`, `359-373` | NEW |
| C5 | `/reviews`: the 20 most recent runs, in both app modes, with the model server off | `review_command.py:381-449`; `local_commands.py:79-80`, `130-143`; tests `test_review_command.py:199`, `220`, `230` | NEW |
| C6 | `/reviews verify <run-id>` and `--all` re-prove reports from inside the app | `review_command.py:452-506`; test `test_review_command.py:199` | NEW |
| C7 | A typo in the options is an error notice and never starts a long job | `app.py:382-388`; test `test_review_command.py:372` | NEW |
| C8 | Listed in `/help` and the command list; agent-pack mode explains where reviews run | `commands.py:31-34`, `63-71`, `120-122`; `agentpack_app.py:81-86`, `100-101`, `276-280`; test `test_review_command.py:446` | NEW |
| C9 | No start-up cost for scientists who never review: the pipeline is imported only when `/review` is typed (about 115 ms saved) | `review_command.py:26-31`; `app.py:396-400`; `local_commands.py:130-138` | NEW |

### D. The command line

| # | What | Evidence | Tag |
|---|---|---|---|
| D1 | `bedrock-litreview` is installed with bedrock; nothing separate to install | `pyproject.toml:52`; wheel `entry_points.txt` | NEW |
| D2 | `bedrock-litreview runs` lists recent runs and their reports | `cli.py:180-181`, `231-240`; `db.py:170-192` | NEW |
| D3 | On interruption, the exact resume command is printed with the unfinished run's id | `cli.py:116-121`, `137-147` | NEW |
| D4 | Configuration problems are one-line messages with exit 1, never tracebacks | `cli.py:196-202`; run for real (Part 1) | NEW |
| D5 | `runs`, `verify-run`, `rate` and `/reviews` work even when the configured model set does not exist | `cli.py:45-53`; `config.py:201-206`, `233`; tests `test_bedrock_settings.py:200`, `224` | NEW |
| D6 | `search`, `verify-run`, `rate`, `runs` and `ask --no-model` never open a connection to a model server, so they work on the networked machine that has no GPU | `cli.py:19-20`, `104-109`; test `test_service.py:149` | NEW |

### E. The HTTP API

| # | What | Evidence | Tag |
|---|---|---|---|
| E1 | One backend and one scheduler per API process, opened and closed by the app's lifespan; reviews go through the same `run_review` as the terminal | `api.py:129-165`, `177-183` | NEW |
| E2 | "The machine cannot run a review" and "the run stopped" return 503 with the reason, not a 500 | `api.py:192-195`; test `test_sources.py:498` (passes with the `api` extra) | NEW |

### F. Configuration

| # | What | Evidence | Tag |
|---|---|---|---|
| F1 | One place to configure the model server for chat and reviews (`api_base`, `api_key`, `request_timeout`). The old `.env`, `LLM_BASE_URL` and per-profile URLs are gone | `config.py:1-13`, `256`; old `litreview/config.py:4`, `46`, `145` | NEW |
| F2 | A `[litreview]` table in `config.toml`, checked only when a review runs, so a typo never stops the app | `settings.py:193`, `332`, `367`; `config.py:280-293`; test `test_bedrock_settings.py:70` | NEW |
| F3 | Whole-file overrides of `models.yaml`, `pipeline.yaml` and `report.yaml` in `~/.config/bedrock/litreview/`, recorded on each run by path and hash, so IT can change names and layout without a code change | `config.py:103-118`; `runner.py:95-137`; test `test_bedrock_settings.py:101` | NEW |
| F4 | Environment variables `BEDROCK_LITREVIEW_MODELS`, `_DIR`, `_CONFIG_DIR`, `_ALLOWED_HOSTS` | `config.py:118`, `228`, `248`, `297` | NEW |

### G. Install and packaging

| # | What | Evidence | Tag |
|---|---|---|---|
| G1 | **Runs from an installed package.** The old package read `config/*.yaml` from the repository root and did not ship them, so it only worked from a checkout. The defaults are now package data | `pyproject.toml:57-64`; `config.py:42-43`; old `litreview/config.py:25`, old `pyproject.toml` package-data; fresh-install run (Part 1, item 3) | NEW |
| G2 | The one-command `uv tool install` gives scientists `bedrock-litreview` as well as `bedrock` | `INSTALL.md:49`; `pyproject.toml:52` | NEW |
| G3 | The first-run check mentions the 12,288-token requirement | `doctor.py:33`, `121-133`; `tests/test_doctor.py` | NEW |
| G4 | A run records the installed package version when there is no git checkout | `runner.py:141-158` | NEW |

### H. Storage and persistence

| # | What | Evidence | Tag |
|---|---|---|---|
| H1 | Data lives in `~/.local/share/bedrock/litreview/`, not in `data/`, `out/` and `.cache/` inside a source checkout | `config.py:50`, `262-264`; old `litreview/config.py:97-100` | NEW |
| H2 | Private by default: directory 0700, database 0600, cache files 0600. Research questions are sensitive | `config.py:121-130`; `db.py:66-75`; `service.py:126`; `llm/cache.py:77`; test `test_service.py:196`; checked on the fresh install | NEW |
| H3 | Two identical runs both show their report. The report id is read from the run record, because identical runs share one report row | `db.py:181-192`; commit `f762f20` | NEW |

### I. Resume

| # | What | Evidence | Tag |
|---|---|---|---|
| I1 | Resume from inside the app with `/review --resume <run-id>`, without retyping the question | `review_command.py:122-123`, `175-176`; test `test_review_command.py:55` | NEW |

(Resume after a stop, a quit or an outage is A5 and A6. A mistyped id is
mishandled: see Part 1, item 11.)

### J. Provenance, determinism and the inference layer

| # | What | Evidence | Tag |
|---|---|---|---|
| J1 | Bedrock's inference layer gained `RequestOptions.seed` and `ModelInfo.digest`. Reviews pin the seed and record `model@digest` through bedrock's backend, and chat code can use both | `inference/base.py:81`, `127`; `openai_compatible.py:219-220`; `scheduled.py:147-179`; tests `test_openai_backend.py`, `test_scheduler_client.py:206`, `212` | NEW |
| J2 | Proof that the move changed nothing: a run through the real scheduler gives a byte-identical report to the fake-client path on the same papers | `test_service.py:161` | NEW |
| J3 | Every call at temperature 0 with the seed and a JSON schema, checked end to end through the app | `test_service.py:182`; `test_review_command.py:387` | NEW (as a test through the app) |

On the fresh install, two runs of the same question gave the same report id
(`61e35796584704f6`).

### K. Privacy

| # | What | Evidence | Tag |
|---|---|---|---|
| K1 | The searcher may reach only `www.ebi.ac.uk` and `eutils.ncbi.nlm.nih.gov`, **redirects included**, so a redirect cannot carry search terms to a third party | `sources/http.py:40-55`, `69-73`; test in `test_sources.py` (redirect refused, other sources carry on) | NEW |
| K2 | The app tells the scientist what leaves the machine before a search starts | `review_command.py:57-60`, `173-174` | NEW |

### L. Engineering safety nets

| # | What | Evidence | Tag |
|---|---|---|---|
| L1 | Bedrock's model-call guard now covers the review's code | `tests/test_model_call_guard.py:23`, `38-47` | NEW |
| L2 | Tests can never write to the real home directory | `tests/conftest.py:219-257`; `tests/litreview/conftest.py:22-24` | NEW |
| L3 | The review's tests run in CI on every pull request. The old repository had no CI | `.github/workflows/test.yml:39`; no `.github/` in the old clone | NEW |
| L4 | The bridge is tested through a real `Scheduler`: 14 scheduler-client tests, 14 service tests, 15 settings tests, 24 app tests | `tests/litreview/test_scheduler_client.py`, `test_service.py`, `test_bedrock_settings.py`, `tests/test_review_command.py` | NEW |

### CARRIED OVER (existed before, survived the move)

The guarantee-carrying modules are byte-identical after the import rewrite
(Part 1).

1. The question-to-report pipeline, Stages 0-8 (`stages/`, `runner.py`, `facets.py`).
2. Verbatim quote verification (`extract/verify.py`).
3. Per-field provenance: fields not found in the paper are blanked (`extract/provenance.py`).
4. The citation guard on report sentences (`report/guard.py`).
5. Report invariants and the integrity check (`report/invariants.py`, `runner.py:320-325` in the old repo).
6. `verify-run` (`cli.py:242-262`).
7. Negation checks (`negation.py`).
8. The exact-entity gate, near-misses and knowledge tables (`knowledge/`).
9. Gene-aware BM25 ranking (`lexical.py`).
10. No model scores paper quality; the relevance grade comes from a written rule.
11. Temperature 0 enforced (`config.py:241-245`), seed 0.
12. The model-call cache (`llm/cache.py`).
13. Frozen, pinned paper sets per question and per run (`db.py`, `runner.py:225-259`).
14. Versioned prompts, recorded on each finding (`llm/prompts/*.md`).
15. Run records: input hashes, prompt and model versions, code version (`runner.py:95-158`).
16. Resume from the stage reached (`runner.py:184-260`; `--resume`).
17. The paper searcher: Europe PMC, preprints, PubMed, open-access JATS full text, merge and dedupe, loud `SearchFailed` (`sources/`).
18. Search only when asked, anonymous user agent, NCBI contact address and key (`http.py:35-37`, `config.py:270-271`).
19. The model endpoint check and the Ollama cloud-model refusal (`config.py:72-100`; the error text now names bedrock's config).
20. Stripping thinking traces (MedGemma) (`scheduled.py:141`, `llm/client.py`).
21. `report.json`, `report.md` and a self-contained `report.html`, laid out by `report.yaml` (`stages/s8_render.py`).
22. Ratings (`rate`, `POST /reports/{id}/ratings`).
23. The HTTP endpoints and their 422 and 502 codes (`api.py:167-215`).
24. The benchmark, file-judge seam and scorecards (`bench/`, `evals/litreview/`).
25. `--no-model` mode, and the progress callback.
26. A 600-second model timeout: kept for the CLI, API and benchmark by a floor (`service.py:185-197`), **but not for `/review` in the app** (Part 1, item 5).

### OPPORTUNITY (the main app has it; the review does not use it yet)

| # | Opportunity | Why it matters | Wiring it would take | Size |
|---|---|---|---|---|
| O1 | Record reviews and searches in the action journal (`/audit`) | Search is the one step that sends anything off the machine; compliance will ask when and by whom | Add `litreview.review` and `litreview.search` to the closed action set (`provenance/history.py:34`) and to `docs/compliance_mvp/CONTRACT.md`; append intent and outcome in `service.run_review` with run id, report sha and source kind, never content. Plan §8.1 | Medium |
| O2 | Publish a finished report to a shared workspace | Scientists share reports with their team today by copying files | Hand `report.html`/`report.json` to `/workspace publish` as a new artifact type, with the run id and integrity result. Read `docs/SHARED_WORKSPACES.md` first. Plan §8.2 | Medium to big |
| O3 | `--corpus <source>://<id>` through company connectors | A curated paper list in a company system could be reviewed without exporting a file | In `parse_review_command`, accept `parse_source_ref` as `/fold` does (`fold_command.py:28`, `78-81`); fetch through `SourceHub`, stage the JSON-lines file, record the fetch provenance into `run.input_hashes` | Medium |
| O4 | Drop folder built on the `share` connector | Paywalled PDFs a scientist has legally obtained | `dropfolder.py` is a stub; build it on `share` (path-escape guard and provenance already exist) plus the GROBID PDF rung. Plan D22, §8.3 | Big |
| O5 | Drag-and-drop for `--corpus` | Scientists drop files onto the terminal; today that breaks `/review` (Part 1, item 11) | Strip the leading `@` in `parse_review_command` | Small |
| O6 | Stage 9: chat about one finished report | The client's "it can't remember earlier conversation" complaint | Read-only chat tools over one run's frozen evidence. **Not** `/digest` or `@file` on `report.md`: that would bypass every guard (plan §8.4) | Big |
| O7 | Feed finished reports to `/digest` or `/recall` | Tempting for "what did the review say about IL-23?" | Only through O6's guarded tools, for the same reason. Listed so nobody wires it the easy way | n/a |
| O8 | `/review` in agent-pack mode | `INSTALL.md` puts some scientists in that mode | A read-only capability that starts the pipeline but cannot steer it. Plan §8.5 | Medium |
| O9 | Apply the endpoint check to chat | Chat can still be pointed at a public host; reviews cannot | Product decision. Plan §8.6 | Small code, needs a decision |
| O10 | Strip thinking traces from chat replies; carry `finish_reason` in `Usage` | MedGemma traces show up in chat | Plan §8.7 | Small |
| O11 | Authentication for the HTTP API | It has none; localhost or office network only | Wire the Keycloak sign-in the web reference client already has. Concurrent `POST /ask` also shares one SQLite file. Plan §8.8 | Big |
| O12 | Use `max_parallel` on the `server` profile | The scheduler runs 4 calls at once there, but the pipeline makes one blocking call at a time, so a shared GPU is under-used | Fan out Stage 3 relevance calls in a thread pool with results kept in paper order. The cache keys do not change, but this touches verbatim stage code, so benchmark before and after | Medium |
| O13 | Learn token counts from review calls too | The window check (B3) uses a characters-per-token ratio learned only from chat (`scheduler.py:165-168`) | Call `tokens.observe` for `USER_JOB` usage as well | Small |
| O14 | A re-runnable readiness check | `bedrock-biotech status` exists for AlphaFold and PyMOL; reviews have nothing equivalent | `bedrock-litreview doctor` or `/reviews check`: load settings, run `preflight`, report every problem at once | Small to medium |
| O15 | Open a report from the app | `/pymol` opens PyMOL; a review only prints a path | `/reviews open <run-id>` using the platform's open command | Small |
| O16 | Delete a review | `/forget` is complete for chat; reviews cannot be deleted (Part 1, item 6) | `/reviews forget <run-id>` and `bedrock-litreview forget` | Medium |
| O17 | Shell reviews behind the app's chat | A1 holds only inside the app | A cross-process queue, or have `bedrock-litreview ask` hand the job to a running app | Big |
| O18 | **Semantic retrieval (LONG-TERM, blocked on a deployed embedding model)** | Stage 3 ranks by gene-aware BM25 plus small-model grades, so a paper that says "pH" is not found for "acidity". `defaults/models.yaml:37-40` already names an unused `embed` model | The same blocker and plan as `/digest` (`NEXT_STEPS.md` §6); an embedding pass would add recall to Stage 3 without letting a model choose terms | Big |

---

## Fix list, in priority order

"Small" means about 30 lines or fewer and low risk. Nothing here was changed by
this review itself.

**Status after the audit (26 September 2026).** Fixed on the same branch in
commit `25d452d`:
- #1: INSTALL.md gains a literature-review section and troubleshooting rows.
- #5: `--corpus @/path` works.
- #6: an unknown `--resume` id is refused.
- #7: a search or corpus with no papers stops with a message instead of
  writing an empty report.
- #12: CI installs the `api` extra.
- #15: the test fixtures clear `BEDROCK_*` variables.
- #16, docs half: README says to re-install after pulling. The `ImportError`
  catch in the app is not done.
- #17, help-text half: the CLI help no longer names old-repository paths.

Everything else is still open and is tracked, with the audit's accuracy
findings, in [OPEN_ISSUES.md](OPEN_ISSUES.md).

| # | Finding | Where | Fix | Size |
|---|---|---|---|---|
| 1 | `INSTALL.md` has no literature-review path; its 8,192 laptop advice and company-host example both make `/review` refuse; agent-pack setup disables it; no troubleshooting rows | `INSTALL.md:89-159` | Add a "Literature review" section (16 GB or office server, `context_window = 12288`, `request_timeout = 600`, `allowed_hosts`, `ollama pull qwen3:8b`, `OLLAMA_CONTEXT_LENGTH=12288`, where `papers.jsonl` comes from, open `report.html`) and troubleshooting rows for the four preflight messages | Small (docs) |
| 2 | `office` model set: two placeholder names at one `api_base`; names changeable only per machine | `defaults/models.yaml:18-20`; `config.py:58` | IT fixes the served names; then ship them in `defaults/models.yaml`, or allow names in `[litreview]` | Big (decision) |
| 3 | Company host names refused until `allowed_hosts` is set; not in any scientist doc; the first-run check suggests such a host | `config.py:69-89`; `doctor.py:78-84` | Document it (row 1) and have the first-run check mention `allowed_hosts` when `api_base` is not local | Small |
| 4 | `/review` in the app keeps the 120 s timeout (the CLI and API get 600 s) | `app.py:432-436`; `service.py:185-197` | Warn in `/review` preflight on the `laptop` profile when `request_timeout < 600` | Small |
| 5 | A corpus file dragged onto the terminal arrives as `@/path` and fails | `review_command.py:128` | `removeprefix("@")` on `--corpus` | Small |
| 6 | A mistyped `--resume` id silently starts a new run (CLI) or says "empty question" (app) | `runner.py:184` | Refuse an unknown id: "no review run …; `/reviews` lists them" | Small |
| 7 | An empty search result finishes quietly with exit 0 | `runner.py` Stage 1; `review_command.py:354-356` | Warn when 0 papers were collected or 0 passed the gate | Small |
| 8 | No way to delete a review; questions, prompts and reports persist | `db.py` (no delete) | `/reviews forget <run-id>` and `bedrock-litreview forget` | Medium |
| 9 | Report files are 0644 (protected only by the 0700 parent) | `stages/s8_render.py:322-327` | chmod 0600 after writing, or document | Small |
| 10 | The first-run check runs only without a config file, and never mentions models, timeout or `allowed_hosts`; there is no re-runnable check | `doctor.py:52-53`, `121-133` | `bedrock-litreview doctor` / `/reviews check` running `preflight` | Small to medium |
| 11 | A title can run between two review calls (README says titles wait until the review finishes). From code reading, not tested | `scheduler.py:239-240`, `259-262`; `README.md:248-250` | Count `USER_JOB` completions in the idle clock, or correct the README | Small |
| 12 | CI never runs the two HTTP API tests | `.github/workflows/test.yml:34` | `pip install -e ".[dev,api]"` | Small |
| 13 | Nothing checks that `package-data` covers every runtime data file (CI installs editable) | `pyproject.toml:57-64` | A test comparing non-`.py` files under `litreview/` against the globs | Small |
| 14 | The rule "only `llm/scheduled.py` and `service.py` import `bedrock.inference`" is untested | `tests/test_model_call_guard.py` | One `ast` test in `tests/litreview/` | Small |
| 15 | An exported `BEDROCK_API_BASE` breaks one test | `tests/conftest.py:219-257` | `delenv` `BEDROCK_*` variables in the autouse fixture | Small |
| 16 | A stale dev venv (missing the four new dependencies) makes `/review` raise `ModuleNotFoundError` inside the app; no doc says to re-install after pulling | `app.py:385`; `review_command.py:96`; `README.md:27-45` | Catch `ImportError` in `start_review` with a "re-run `pip install -e ".[dev]"`" notice, and add an "Updating" note to the README | Small |
| 17 | Stale CLI help (`out/`, `config/pipeline.yaml`, `data/bench.sqlite3`); unused `AGENTPACK_NOTICE`; agent-pack refusal names only the env var | `cli.py:166`, `172`, `178`; `review_command.py:62-65`; `agentpack_app.py:81-86` | Correct the strings; remove the dead constant; mention `backend = "agentpack"` | Small |
