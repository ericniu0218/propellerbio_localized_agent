# Literature review into bedrock-private-ai: prompt and plan

This file is both the prompt for the agent that does the integration and the
plan the team reviews that work against. Give the agent the whole file.

- **Source:** `Bedrock-Technical-Consulting/Localized-Airgap-LLM-for-literature-review`,
  cloned at `/Users/li/Localized-Airgap-LLM-for-literature-review`, `main` at
  `dc5c012`. It holds the package `litreview`: the question-to-cited-report
  pipeline, the paper searcher and the benchmark. It has 217 tests and needs no
  GPU or network.
- **Target:** `Bedrock-Technical-Consulting/bedrock-private-ai`, cloned at
  `/Users/li/Bedrock private AI`, `origin/main` at `4bc696a`. It is a Textual
  terminal client for privately hosted models.
- **Result:** the pipeline lives on as `bedrock.litreview`. In the terminal app
  it runs beside the chat with `/review`, in a shell with `bedrock-litreview`,
  and over HTTP through the optional FastAPI app.

---

## The prompt

You are moving the literature-review pipeline into bedrock-private-ai. Every
design decision is already made in sections 2–4 of this file. Carry them out and
do not redesign them. Where both codebases had a version of the same feature,
section 2 says which version wins and why.

**Where to work.** Create a git worktree at `/Users/li/bedrock-litreview` on a
new branch, `feature/litreview-integration`, cut from `origin/main` (Phase 0).
Do all the work there. Never edit, stash or reset anything in
`/Users/li/Bedrock private AI`, which holds someone's untracked files. Never
modify the source repository either: you only read it and archive it.

**Read these first, in this order:** the target's `CLAUDE.md` and `AGENTS.md`;
the source's `CLAUDE.md`, whose rules survive the move unchanged; then sections
2–4 of this file. Read a file before you edit it.

**Spend tokens on the new code, not on the move.**
1. Copy files with the Phase 1 and Phase 3 scripts (`git archive`, `cp`). Never
   retype a file that the scripts copy.
2. Make mechanical changes with the `perl` one-liners given here, not by
   opening files one at a time.
3. Hand-edit only the files that each phase lists under **Edit**, and only the
   parts it names.
4. Do not open anything you only copy: the corpora, the sample reports,
   `BENCHMARK_RESULTS.md` (120 KB), the prompt files and the stage modules.

**Rules that do not bend** (the source `CLAUDE.md` gives the reasons):
- Prompt files, knowledge tables, stages, quote verification, provenance, the
  citation guard and the invariants are copied unchanged apart from the import
  rewrite. A prompt's `version` is recorded on every finding and is part of
  every cache key.
- Every model call goes through bedrock's `Scheduler`, by way of
  `SchedulerClient` at `Priority.USER_JOB`. `tests/test_model_call_guard.py`
  must stay green, and nothing under `src/bedrock/litreview/` may talk HTTP to
  a model server.
- Every review call runs at temperature 0.0. Model names come only from
  `defaults/models.yaml` or its override. The literature search runs only when
  someone asks for it.
- The source's FastAPI app is kept as the HTTP front door for the app bedrock
  is going to become (§4.9). This is the decision of bedrock's owners. The
  "THIS IS NOT A WEBSITE" rule in the target's `CLAUDE.md` still governs the
  terminal interface, and the API is not that interface. The API is an
  optional extra, and it runs reviews through the same service and scheduler
  as the terminal and the CLI.
- No test may read or write a real home directory.

**Commits.** Make one commit per phase, using the message given, after that
phase's tests pass. Do not push or open a pull request unless the person
running you asks. The target's `AGENTS.md` forbids pushing to `main` in any
case.

**When the code disagrees with this file** (a function is missing, a path has
moved, a count differs), stop and report the difference instead of working
around it. You may fix obvious mechanical slips that the plan clearly meant to
cover, such as an import the rewrite missed. List them in the hand-off note.

Finish with the hand-off note in section 7.

---

## 1. What moves

Almost everything the source built lands in bedrock, most of it byte for byte:

- **The pipeline.** Stages 0–8, `runner.py` and `facets.py` take a question
  through the exact-entity gate, relevance ranking, the evidence plan, verified
  extraction, scoring, cited writing and rendering. A run resumes from the
  stage it reached, and no model decides what happens next.
- **The guarantees.** Verbatim quote verification (`extract/verify.py`),
  per-field provenance (`extract/provenance.py`), the citation guard
  (`report/guard.py`), report invariants and `verify-run`
  (`report/invariants.py`) and negation checks (`negation.py`). No model scores
  paper quality.
- **Exactness.** The entity table and its matcher (`knowledge/`), and BM25 with
  gene-aware tokens (`lexical.py`).
- **Determinism.** Temperature 0, the model-call cache (`llm/cache.py`), paper
  sets that are frozen and pinned per run (`db.py`), and versioned prompts
  (`llm/prompts/*.md`).
- **The paper searcher.** Europe PMC, preprints, PubMed, open-access JATS full
  text, and merge and dedupe (`sources/`, `parsing/`, `ingest.py`). It runs
  only when asked.
- **Reports.** `report.json`, `report.md` and a self-contained `report.html`
  (`stages/s8_render.py`), laid out by `report.yaml`.
- **The HTTP API.** `api.py` provides `POST /ask`, `GET /reports/{id}` and
  `POST /reports/{id}/ratings`. It is kept as an optional `[api]` extra and is
  the front door for the app bedrock will become.
- **The benchmark.** `bench/` and `evals/` hold the baselines, judge,
  distractors, perturbations, file-judge seam, scorecards and history, plus the
  frozen sample corpora and reports.
- **Documents.** The README, `BENCHMARK_RESULTS.md` (with the 64-step to-do
  list), the rules in `CLAUDE.md`, and two project agents.

Not moved: the vLLM and Ollama HTTP clients (D2), `.env` loading (D7) and
`scripts/dev_setup.sh`. The repo-root `config/` folder moves
into the package.

---

## 2. Where the two codebases overlap, and which version wins

| # | Feature | bedrock-private-ai | litreview | Winner | Why |
|---|---|---|---|---|---|
| D1 | Where code lives | Each feature is `src/bedrock/<feature>/` | Top-level `litreview` package | **bedrock**: `src/bedrock/litreview/` with imports rewritten | One install. The model-call guard walks `src/bedrock`, so it covers the review too. |
| D2 | Talking to the model server | `InferenceBackend` behind `Scheduler`: priorities, one call at a time on a laptop, the chat goes first | Its own sync `OpenAICompatibleClient` and `OllamaClient` calling HTTP directly | **bedrock** transport. litreview's `LLMClient` protocol stays as the seam the stages see, served by the new `SchedulerClient`. | A review makes hundreds of calls to the server the employee is chatting with, and only the scheduler stops them colliding. It is also the target's written model-call rule. |
| D3 | Context window | Set server-wide (`OLLAMA_CONTEXT_LENGTH` / vLLM `--max-model-len`); the profile states it | `num_ctx: 12288` sent per request through Ollama's native API | **bedrock**, plus a litreview minimum (`min_context_window: 12288`) checked before a run | Ollama reloads a model whenever a request asks for a different context size, so alternating chat and review calls would reload it on every switch. Extraction sends about 4,300 prompt tokens plus a 4,096-token reply, which an 8,192 window cannot hold. |
| D4 | Reply length | `profile.reply_limit` (1,024 on a laptop) | Set per call (512–4,096) | **litreview**, checked against the profile's window | A 1,024 cap would cut extraction JSON short, and `max_tokens` is part of the cache key. |
| D5 | Which model | Auto-picks the largest model that fits; `/model` | Pinned per job in `models.yaml` | **litreview** for reviews; chat unchanged | Determinism is a client requirement, and auto-pick differs between machines. |
| D6 | Server address and key | `api_base`, `api_key` and `request_timeout` from `config.toml` or env | `base_url` per profile in YAML, `LLM_BASE_URL`, `.env` | **bedrock** | There should be one place to configure the server. |
| D7 | Settings and knobs | env, then `config.toml`, then defaults | YAML at the repo root, plus `.env` | **bedrock's loader** plus a `[litreview]` table. The YAML stays YAML as package data, with a whole-file override directory. | An installed wheel has no repo root, and IT must be able to change model names and the report layout without a code change. |
| D8 | The word "profile" | Machine profile: `laptop`, `workstation`, `server` | Model set: `office`, `laptop`, `medgemma` | **bedrock** keeps the word. litreview's becomes `model_set`, with the flag `--models`. | One word should have one meaning. |
| D9 | Air-gap endpoint check | None | `assert_local_url` and `assert_local_model` refuse public hosts and Ollama `-cloud` models | **litreview**, applied to reviews | It is strictly safer. Applying it to the chat changes chat behaviour, so that is a follow-up (§8). |
| D10 | Determinism | A temperature option only | Temperature 0 enforced, a seed, the call cache, weight-digest pinning, frozen papers | **litreview**. bedrock gains `RequestOptions.seed` and `ModelInfo.digest` (§4.1). | The client's acceptance test is the same question twice giving a byte-identical report. |
| D11 | Thinking models | `reasoning=False` works on Ollama and vLLM, with an old-vLLM fallback | An `enable_thinking` flag, plus `strip_thinking` | **Both**: bedrock's switch, and litreview's stripper on the reply | MedGemma's thinking trace cannot be switched off. |
| D12 | Parsing JSON replies | `json_reply.decode_json_reply` | `parse_json_reply`, which also strips thinking traces | **litreview's** inside litreview; bedrock's untouched | The stages catch its `LLMError`. |
| D13 | Counting tokens | `inference/tokens.estimate_tokens`, which learns from the server | None | **bedrock** | The window check needs it. |
| D14 | Server outage mid-run | `InferenceError(unavailable=True)` | Any HTTP error becomes `LLMError`, and the stage skips that item | **bedrock's** signal, with new behaviour: stop the run, which can be resumed | Otherwise a report written while the server was down comes out silently hollow. |
| D15 | Keyword ranking | BM25 on SQLite FTS5 for `/digest`, with `[a-z0-9_]+` tokens | BM25 with gene-aware tokens (IL-23 becomes `il23`, Greek letters kept) and ties broken by id | **Each keeps its own**; no merge | bedrock's tokenizer splits "IL-23" and drops "α". |
| D16 | Storage | JSON files and SQLite, directories 0700 and files 0600 | One SQLite database with pinned paper versions | **litreview's** database under **bedrock's** private-file rule | Run pinning and `verify-run` depend on its schema, and research questions are sensitive. |
| D17 | Outbound HTTP | The agent's `web.fetch` requires a proxy, uses an allowlist and does not follow redirects | `SourceHTTP` retries, backs off, respects rate limits, sends an anonymous user agent and follows redirects; it only runs when asked | **litreview's** client plus **bedrock's** host rule: a request may reach only `www.ebi.ac.uk` and `eutils.ncbi.nlm.nih.gov` | A redirect could otherwise carry search terms to another host. The proxy rule was written for URLs a model chooses; this client's URLs are fixed in code, and httpx already honours `HTTPS_PROXY`. |
| D18 | Front ends | Textual terminal app (terminal for now; bedrock will become an app) | A CLI plus a FastAPI app | **All three**: `/review` in the terminal, the `bedrock-litreview` command, and the FastAPI app as an optional `[api]` extra. All three call `service.run_review`. | The terminal is bedrock's interface today, and the HTTP API is the front door for the app it becomes. One shared service means one pipeline, one scheduler and one set of guarantees. |
| D19 | Run records | `/runs` covers AlphaFold manifests and re-hashes files | A runs table, plus `verify-run`, which re-proves quotes and citations | **Both, separately.** `/reviews` copies the way `/runs` works. | They make different guarantees. |
| D20 | Logging | None; Textual owns the screen | Python `logging` | litreview's calls stay, and the app forwards warnings and worse to the job's notices | Nothing may write over the terminal UI. |
| D21 | Tests | `FakeBackend` and autouse private-directory fixtures | `FakeLLMClient` and synthetic fixtures | **Both**, with bedrock's isolation extended to the litreview directories | Stage tests stay fast, and the bridge is tested through a real `Scheduler`. |
| D22 | Drop folder (planned) | The `share` connector, already built, with provenance and a path-escape guard | A `dropfolder.py` stub | Follow-up: build it on `share` (§8) | Reuse what already exists. |

---

## 3. Where everything goes

```
src/bedrock/litreview/              <- litreview/  (unchanged unless a phase below edits it)
  CLAUDE.md, AGENTS.md              <- source CLAUDE.md, edited in Phase 7; AGENTS.md points at it
  defaults/models.yaml              <- config/models.yaml, restructured (§4.4)
  defaults/pipeline.yaml            <- config/pipeline.yaml (only comment paths updated)
  defaults/report.yaml              <- config/report.yaml, verbatim
  config.py                         REWRITTEN: settings come from bedrock (§4.3)
  service.py                        NEW: preflight, run_review, standalone_scheduler (§4.6)
  llm/scheduled.py                  NEW: SchedulerClient (§4.5)
  llm/client.py                     concrete HTTP clients deleted
  llm/__init__.py                   make_client rewritten
  api.py                            FastAPI app kept (optional [api] extra), rebuilt on service (§4.9)
  cli.py                            bedrock-litreview (§4.8)
  db.py                             private file modes; recent_runs() added
  sources/http.py                   host allowlist added
  runner.py, report/layout.py, bench/run.py, bench/__main__.py   small path and setting edits
  everything else                   verbatim: models, stages, extract, report, knowledge,
                                    parsing, sources, bench, llm/prompts, the stub files
src/bedrock/ui/review_command.py    NEW: /review and /reviews (§4.7)
tests/litreview/                    <- tests/  (imports rewritten; a few tests adapted)
evals/litreview/                    <- evals/  (a source checkout only; not in the wheel)
scripts/litreview/judge_answers.py  <- scripts/judge_answers.py
docs/litreview/                     <- README.md, BENCHMARK_RESULTS.md, config/sources.yaml
.claude/agents/litreview-*.md       <- .claude/agents/{benchmark-runner,deployment-auditor}.md
```

Dropped: `.env.example`, `scripts/dev_setup.sh` and the source `pyproject.toml`
and `.gitignore`. The last two are merged into the target's in Phase 1.

---

## 4. New interfaces

### 4.1 Three fields in bedrock

All three are additive with defaults, which the Phase 0 contract in
`docs/roadmaps/LAPTOP_MEMORY.md` allows.

```python
# src/bedrock/inference/base.py
class RequestOptions:            # frozen dataclass; add:
    seed: int | None = None      # sent as "seed"; None leaves it to the server

class ModelInfo:                 # frozen dataclass; add:
    digest: str | None = None    # weights hash when the server reports one (Ollama /api/tags "digest")

# src/bedrock/config/settings.py
class Settings:                  # add, and add "litreview" to `known` in load_settings:
    litreview: dict = field(default_factory=dict)
    # The [litreview] table. Validated by bedrock.litreview.config when a review
    # runs, so a typo there never stops the app from starting (as with connectors).
```

In `openai_compatible.py`, `_payload` sends `seed` when it is set.
`_ollama_sizes` becomes a reader for both size and digest, and `list_models`
fills in `digest`. vLLM reports no digest, so it stays `None` there.

### 4.2 Configuration

| Setting | Env (wins) | `[litreview]` key | Default |
|---|---|---|---|
| Model set | `BEDROCK_LITREVIEW_MODELS` | `model_set` | `office` on the `server` profile, otherwise `laptop` |
| Data (DB, cache, reports) | `BEDROCK_LITREVIEW_DIR` | `data_dir` | `~/.local/share/bedrock/litreview` |
| YAML overrides | `BEDROCK_LITREVIEW_CONFIG_DIR` | none | `~/.config/bedrock/litreview` |
| Extra model hosts that count as local | `BEDROCK_LITREVIEW_ALLOWED_HOSTS` (comma-separated) | `allowed_hosts` (list) | none |
| Contact address sent to NCBI | `SOURCES_CONTACT_EMAIL` | `contact_email` | none |
| PubMed key (a secret, so env only) | `NCBI_API_KEY` | none | none |
| Allow Ollama cloud models | `LLM_ALLOW_CLOUD_MODELS=1` | none | refused |

- The `--models` flag or argument beats every row above.
- A file named `models.yaml`, `pipeline.yaml` or `report.yaml` in the override
  directory replaces the packaged default of the same name completely; the two
  are never merged. The run's `input_hashes` record which file was read.
- The knowledge tables (`knowledge/*.yaml`) cannot be overridden. They decide
  what counts as the exact entity, so a change to them goes through code review
  like any other code.
- Unknown `[litreview]` keys raise `ConfigError` naming the allowed keys.
- These are derived from the data directory: `cache_dir = data_dir/cache/llm`,
  `out_dir = data_dir/reports` and `db_path = data_dir/litreview.sqlite3`.
- These come from bedrock: `base_url = api_base`,
  `context_window = profile.context_window`, and the API key and timeout used
  by the backend.

### 4.3 `bedrock.litreview.config`

Keep `ConfigError`, `assert_local_model`, `ModelSpec` and the `Settings`
methods `model_for`, `model_options` and `knob` verbatim. Keep
`assert_local_url` too, but change its error text to name `allowed_hosts` in
`[litreview]` or `BEDROCK_LITREVIEW_ALLOWED_HOSTS` instead of
`LLM_ALLOWED_HOSTS`. Remove `ROOT`, `.env` loading and `${VAR}` expansion.

```python
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULTS_DIR = PACKAGE_DIR / "defaults"
EVALS_DIR = PACKAGE_DIR.parents[2] / "evals" / "litreview"    # exists in a source checkout only
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "bedrock" / "litreview"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "bedrock" / "litreview"
# Module constants read at call time, so tests can redirect them (Phase 3).

def config_file(name: str, env: Mapping[str, str] | None = None) -> Path:
    """The override in the config dir if it exists, else DEFAULTS_DIR / name."""

def private_dir(path: Path) -> Path:
    """mkdir(parents=True, exist_ok=True, mode=0o700), then chmod 0o700."""

@dataclass
class Settings:              # not frozen: tests assign cache_dir, out_dir, db_path
    model_set: str
    base_url: str
    models: dict[str, ModelSpec]
    jobs: dict[str, str]
    context_window: int
    min_context_window: int
    data_dir: Path; cache_dir: Path; out_dir: Path; db_path: Path
    pipeline: dict[str, Any]
    temperature: float = 0.0; seed: int = 0; max_tokens: int = 2048
    allowed_hosts: tuple[str, ...] = ()
    ncbi_api_key: str = ""; contact_email: str = ""

def load_settings(bedrock_settings=None, *, model_set=None, env=None, check_endpoint=True) -> Settings:
    """bedrock_settings None -> bedrock.config.load_settings(env=env). Reads
    config_file("models.yaml") and config_file("pipeline.yaml"). Keeps the source's
    `temperature must be 0.0` refusal. With check_endpoint, runs assert_local_url on
    base_url and assert_local_model on every model in the set. Touches no disk
    beyond reading."""
```

### 4.4 `defaults/models.yaml`

Rewrite the top of the file as below. Copy the `jobs`, `embed` and `defaults`
blocks from the original unchanged.

```yaml
# Model names for the literature review, and nothing else about the server.
# The server is bedrock's own (`api_base` in ~/.config/bedrock/config.toml, or
# BEDROCK_API_BASE), and every call waits its turn in bedrock's scheduler.
# Model names live here and nowhere else. To change them on one machine, put a
# whole replacement of this file in ~/.config/bedrock/litreview/models.yaml.
#
# Choose a set with `model_set` in [litreview], BEDROCK_LITREVIEW_MODELS or
# --models. Default: `office` on the `server` machine profile, else `laptop`.
# Tiers are split by COST, not skill: relevance runs once per paper, so it may
# use the cheap model; everything else runs tens of times per question.

# Extraction sends about 4,300 prompt tokens and allows a 4,096-token reply.
# The server's window (bedrock's `context_window`, which must match
# OLLAMA_CONTEXT_LENGTH or vLLM --max-model-len) must be at least this.
min_context_window: 12288

model_sets:
  office:      # the client's GPU, served by vLLM
    small:  {name: qwen3-8b-instruct, thinking: false}
    medium: {name: qwen3-30b-a3b-instruct, thinking: false}
  laptop:      # Ollama on a 16 GB machine
    small:  {name: "qwen3:8b", thinking: false}
    medium: {name: "qwen3:8b", thinking: false}
  medgemma:    # MedGemma 1.5 4B via Ollama; produced docs/litreview/BENCHMARK_RESULTS.md. No thinking mode.
    small:  {name: "medgemma1.5:4b"}
    medium: {name: "medgemma1.5:4b"}
```

### 4.5 `llm/scheduled.py`

This is the reference implementation; keep its behaviour.

```python
class ReviewInterrupted(RuntimeError):
    """The run stopped before its report and can be resumed. Deliberately not an
    LLMError: stages catch LLMError to skip one item, and this must end the run."""

class ReviewCancelled(ReviewInterrupted): ...          # the employee stopped it, or quit the app
class ModelServerUnavailable(ReviewInterrupted): ...   # unreachable, or no answer in time

class SchedulerClient:
    """litreview's LLMClient, served by bedrock's Scheduler. Blocking: call it
    from a worker thread; `loop` is the event loop the scheduler runs on."""

    def __init__(self, scheduler, loop, *, model_options, seed, context_window,
                 cancel: threading.Event | None = None, priority=Priority.USER_JOB): ...

    def complete(self, prompt, *, model, schema=None, temperature=0.0, max_tokens=2048) -> str:
        self._check_cancelled()
        needed = estimate_tokens(prompt) + max_tokens
        if needed > self._window:     # never shrink max_tokens: it is part of the cache key
            raise LLMError(f"prompt plus a {max_tokens}-token reply needs ~{needed} tokens; window is {self._window}")
        thinking = self._options.get(model, {}).get("thinking")
        options = RequestOptions(max_tokens=max_tokens, temperature=temperature, json_schema=schema,
                                 reasoning=False if thinking is False else None, seed=self._seed)
        self._last_model = model
        raw = self._run(self._scheduler.complete(
            model, [{"role": "user", "content": prompt}], priority=self._priority, options=options))
        return strip_thinking(raw)

    def version_of(self, model) -> str:
        """`name@digest[:12]` when the server reports a digest, else the name.
        Uses list_models() once. The format matches the source's OllamaClient."""

    @property
    def model_version(self) -> str: ...

    def _run(self, coroutine):
        # 1. If called on self._loop's own thread: close the coroutine and raise
        #    RuntimeError (it would deadlock otherwise).
        # 2. asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()
        # 3. concurrent.futures.CancelledError  -> ReviewCancelled
        #    InferenceError while cancel is set  -> ReviewCancelled
        #    InferenceError(unavailable=True)    -> ModelServerUnavailable
        #    any other InferenceError            -> LLMError (the stage skips that item, as before)
```

`llm/__init__.py`:
`make_client(settings, *, scheduler=None, loop=None, cancel=None, fake=None, cache=True, read_cache=True)`.
If `fake` is given, use it. If both `scheduler` and `loop` are given, build a
`SchedulerClient` with
`model_options={s.name: s.options for s in settings.models.values()}`,
`seed=settings.seed` and `context_window=settings.context_window`. Otherwise
raise `ValueError` saying to use `service.run_review`, or to pass `fake=` in
tests. Wrap the client in `CachedClient(CallCache(settings.cache_dir))`, as
the source does. Export `SchedulerClient` and the three exceptions, and drop
the concrete clients.

### 4.6 `service.py`

This module is the only way into a model-using review. Both the CLI and the app
go through it.

```python
@dataclass(frozen=True)
class ReviewRequest:
    question: str
    corpus: Path | None = None
    search: bool = False
    refresh: bool = False
    resume: str | None = None
    use_model: bool = True
    model_set: str | None = None
    sources: tuple[str, ...] | None = None
    full_text: bool = True
    out_dir: Path | None = None
    papers: list[Paper] | None = None                      # papers posted with the question (HTTP)
    approved_terms: dict[str, list[str]] | None = None     # HTTP only, as in the source

async def preflight(settings, backend) -> None:
    """Raise ConfigError with the fix spelled out:
    - context_window < min_context_window: name both numbers and say to set
      `context_window = 12288` (or BEDROCK_CONTEXT_WINDOW) with
      OLLAMA_CONTEXT_LENGTH=12288 on a 16 GB machine, or to use the
      `workstation` profile or the office server;
    - list_models() fails: say the server at base_url is not answering;
    - a model the jobs need is not listed: name it, give `ollama pull <name>`,
      and say to ask IT when the server is vLLM."""

async def run_review(request, *, settings, scheduler, progress, cancel: threading.Event) -> RunResult:
    """private_dir(settings.data_dir). With a model: await preflight(...), then
    make_client(settings, scheduler=scheduler, loop=<running loop>, cancel=cancel).
    The searcher is CorpusSearcher(corpus) when a corpus is given,
    LiteratureSearcher.from_settings(settings, **overrides) when search is asked
    for, and None otherwise. Then
    `await asyncio.to_thread(answer_question, ...)`, passing `papers` and
    `approved_terms` through. `progress` is called from
    that worker thread. ConfigError, QuestionError, NoPapers, SearchFailed,
    ReviewInterrupted, OSError and ValueError propagate to the caller, which
    renders them."""

@asynccontextmanager
async def standalone_scheduler(bedrock_settings, *, backend=None, timeout_floor=600.0):
    """For the CLI, the benchmark and the HTTP app. Uses `backend` when one is
    given (tests); otherwise create_backend(settings with
    request_timeout >= timeout_floor). Yields Scheduler(backend, profile), then
    closes the backend."""
```

When a run is interrupted, callers name the most recent unfinished run from
`Database.recent_runs()` so the person can resume it.

### 4.7 In the app

| Command | Behaviour |
|---|---|
| `/review <question> [--corpus FILE \| --search] [--refresh] [--resume RUN] [--no-model] [--models SET] [--sources a,b] [--no-full-text]` | Starts a review beside the chat. Only in the default mode (`BedrockApp`). |
| `/review stop` | Stops the running review before its next model call. |
| `/reviews` | The 20 most recent runs. Works in both modes (`LocalCommandsMixin`) and needs no model. |
| `/reviews verify <run-id> \| --all` | Runs `verify_run`, then prints OK or the violations. |

- **Parsing.** The question is the text before the first ` --`, taken verbatim,
  so apostrophes work ("What's…"). The options after it go through
  `CommandParser` and `split_arguments` from `ui/argparsing.py`. `--corpus` and
  `--search` cannot be combined. A question is required unless `--resume` is
  given. `stop` on its own is the stop verb.
- **Running.** One review at a time, with its own flag, independent of a
  running fold. It runs as a `@work(group="litreview")` worker that awaits
  `service.run_review(..., scheduler=self.scheduler)`, so its calls queue behind
  the employee's chat at `USER_JOB`. Esc does not touch it.
- **Notices.** A start notice names the question, the model set and the paper
  source. With `--search` it adds: "search terms and paper identifiers go to
  Europe PMC and NCBI; your question, papers and report stay on this machine".
  Progress goes to toasts through `self.call_from_thread(self.notify, ...)`,
  and a progress callback must never raise into the pipeline. While the job
  runs, warnings and errors from the `bedrock.litreview` logger go to the same
  toasts. A finish notice gives the run id, report id, papers collected, passed
  and read, findings, sentences kept and removed, the integrity result (FAILED
  is an error notice), the paths `render()` returned, and
  "`/reviews verify <run-id>` re-checks it". As with `/fold`, notices are not
  saved into the conversation.
- **Interrupted runs.** On a `ReviewInterrupted`, name the run and how to
  resume it: `/review --resume <run-id>`. For a `ModelServerUnavailable`, also
  mention raising `request_timeout` on slow machines.
- **Quitting.** `action_quit` sets the cancel event before it closes the
  backend, so the run stays resumable.
- **Agent-pack mode.** `/review` there replies that reviews run in the default
  mode or with `bedrock-litreview`.
- **Other commands.** Add rows for these commands to `HELP_TEXT`, and add
  `review`, `litreview` and `reviews` to `KNOWN_COMMANDS`. `litreview` is an
  alias of `review`.
- **Shared module.** `ui/review_command.py` holds the parsing, the job and the
  rendering, in the same way `ui/fold_command.py` does for folds.

### 4.8 Command line (`bedrock-litreview`, also `python -m bedrock.litreview`)

These are the source commands with `--profile` renamed to `--models`: `search`,
`ask`, `verify-run` and `rate`, with the same flags and exit codes. There is one
new subcommand, `runs [--limit N]`, which prints what `/reviews` shows. `ask`
with a model runs `asyncio.run(...)` around `standalone_scheduler`, then
`service.run_review`. `search`, `verify-run`, `rate`, `runs` and
`ask --no-model` never build a backend. The benchmark keeps
`python -m bedrock.litreview.bench`, with `--models` and the same scheduler
wrapper.

### 4.9 HTTP API (`bedrock.litreview.api`, optional `[api]` extra)

Keep the source's endpoints, request bodies and status codes. Change only how a
review reaches the model:

```python
def create_app(settings=None, *, bedrock_settings=None, backend=None):
    """One backend and one Scheduler per server process. The app's lifespan opens
    them with `standalone_scheduler(bedrock_settings, backend=backend)` and closes
    them on shutdown. `backend` is for tests."""
```

- `POST /ask` becomes `async def`. It turns `AskBody` into a `ReviewRequest`
  (question, papers, search, refresh_papers, approved_terms) and awaits
  `service.run_review(..., scheduler=app.state.scheduler)`.
- The existing status codes stay: `QuestionError` and `NoPapers` give 422,
  and `SearchFailed` gives 502. New: a `ConfigError` from preflight, or a
  `ReviewInterrupted`, gives 503 with its message.
- `GET /reports/{id}` and `POST /reports/{id}/ratings` are unchanged, as are
  `set_searcher`, `ask()` and `rate_report()`.
- The search still runs only on `"search": true`.
- Serve it with `uvicorn bedrock.litreview.api:create_app --factory --host 127.0.0.1`.
  It has no authentication yet (§8), so keep it on localhost or the office
  network, as the source README says.

---

## 5. Steps

Every phase ends with its tests passing and a commit. Each Bash call starts a
fresh shell, so the scripts use full paths and set their own variables.

### Phase 0: worktree and baseline

```bash
git -C "/Users/li/Bedrock private AI" fetch origin
git -C "/Users/li/Bedrock private AI" worktree add -b feature/litreview-integration /Users/li/bedrock-litreview origin/main
cd /Users/li/bedrock-litreview && python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]" && pytest -q | tail -1
test -z "$(git -C /Users/li/Localized-Airgap-LLM-for-literature-review status --porcelain)" && git -C /Users/li/Localized-Airgap-LLM-for-literature-review rev-parse --short HEAD
mkdir -p /Users/li/bedrock-litreview/docs/prompts && cp "/Users/li/Bedrock private AI/docs/prompts/LITREVIEW_INTEGRATION.md" /Users/li/bedrock-litreview/docs/prompts/
```

If `/Users/li/bedrock-litreview` already exists on
`feature/litreview-integration` because the person running you created it,
skip the first two commands. Record the baseline count. It was 1,073 on the local `main` at `c6e8fed`, and
`origin/main` adds a few meeting tests. Also record the source commit, which
should be `dc5c012`. If the source has uncommitted changes, stop and ask. There
is no commit in this phase; this file goes into the Phase 1 commit.

### Phase 1: copy the package, data and documents verbatim

Run this as one script:

```bash
set -euo pipefail
cd /Users/li/bedrock-litreview
SRC=/Users/li/Localized-Airgap-LLM-for-literature-review
STAGE=$(mktemp -d)
git -C "$SRC" archive HEAD | tar -x -C "$STAGE"            # tracked files only: no caches, no local runs
mkdir -p src/bedrock/litreview/defaults evals/litreview scripts/litreview docs/litreview .claude/agents
cp -R "$STAGE/litreview/." src/bedrock/litreview/
cp "$STAGE"/config/{models,pipeline,report}.yaml src/bedrock/litreview/defaults/
cp -R "$STAGE/evals/." evals/litreview/
cp "$STAGE/scripts/judge_answers.py" scripts/litreview/
cp "$STAGE/README.md" "$STAGE/BENCHMARK_RESULTS.md" docs/litreview/
cp "$STAGE/config/sources.yaml" docs/litreview/sources.yaml
cp "$STAGE/CLAUDE.md" src/bedrock/litreview/CLAUDE.md
cp "$STAGE/.claude/agents/benchmark-runner.md" .claude/agents/litreview-benchmark-runner.md
cp "$STAGE/.claude/agents/deployment-auditor.md" .claude/agents/litreview-deployment-auditor.md
find src/bedrock/litreview evals/litreview scripts/litreview -name '*.py' -print0 | xargs -0 perl -pi -e '
  s/\bfrom litreview\b/from bedrock.litreview/g;
  s/\bimport litreview\b/import bedrock.litreview/g;
  s/(["\x27])litreview\./$1bedrock.litreview./g;
  s/python -m litreview\b/python -m bedrock.litreview/g;
  s/python -m evals\.run_extract\b/python -m evals.litreview.run_extract/g;
  s{(?<![\w/.])evals/(?!litreview/)}{evals/litreview/}g;'
! grep -rnE '^\s*(from|import) litreview\b' src evals scripts
rm -rf "$STAGE"
```

**Edit**
- `pyproject.toml`:
  - Add dependencies `"pydantic>=2.0"`, `"pyyaml>=6.0"`, `"lxml>=5.0"` (JATS
    full text) and `"rapidfuzz>=3.0"` (quote matching; a slower pure-Python
    fallback exists).
  - Add the optional extra `api = ["fastapi>=0.110", "uvicorn>=0.29"]`. Do not
    install it until Phase 5.
  - Add the script `bedrock-litreview = "bedrock.litreview.cli:main"`.
  - Add `"litreview/llm/prompts/*.md"`, `"litreview/knowledge/*.yaml"` and
    `"litreview/defaults/*.yaml"` to `[tool.setuptools.package-data] bedrock`.
- `.gitignore`: add `evals/litreview/results/*/` and
  `!evals/litreview/results/history.jsonl`.

**Check**
- Run `pip install -e ".[dev]"`.
- Run `python -c "import bedrock.litreview.runner, bedrock.litreview.bench.run, bedrock.litreview.sources.literature"`.
- Run `pytest -q`. It should give the baseline count, all passing, and the
  model-call guard now scans the new package.

**Commit:** `Copy litreview into src/bedrock/litreview (verbatim, imports rewritten)`.
In the body, name the source commit.

### Phase 2: seed and digest in bedrock's inference layer

**Edit:** `inference/base.py` and `inference/openai_compatible.py`, as in §4.1.
Update the module docstring's list of how each option reaches the server.

**Tests** (in `tests/test_openai_backend.py`, following its transport-mock
pattern):
- `seed` is sent when set and absent when `None`.
- `digest` is read from `/api/tags`.
- A server without `/api/tags` leaves `digest` as `None`.

**Commit:** `Inference: RequestOptions.seed and ModelInfo.digest`.

### Phase 3: settings, paths and the ported tests

**Copy the tests** as one script:

```bash
set -euo pipefail
cd /Users/li/bedrock-litreview
STAGE=$(mktemp -d)
git -C /Users/li/Localized-Airgap-LLM-for-literature-review archive HEAD tests | tar -x -C "$STAGE"
mkdir -p tests/litreview && cp -R "$STAGE/tests/." tests/litreview/ && rm -rf "$STAGE"
find tests/litreview -name '*.py' -print0 | xargs -0 perl -pi -e '
  s/\bfrom litreview\b/from bedrock.litreview/g;
  s/\bimport litreview\b/import bedrock.litreview/g;
  s/(["\x27])litreview\./$1bedrock.litreview./g;
  s/python -m litreview\b/python -m bedrock.litreview/g;
  s/\bfrom tests\.fixtures\b/from tests.litreview.fixtures/g;
  s{(?<![\w/.])evals/(?!litreview/)}{evals/litreview/}g;
  s/"--profile", "laptop"/"--models", "laptop"/g;'
find src/bedrock/litreview tests/litreview -name '*.py' -print0 | xargs -0 perl -pi -e 's/\bload_settings\(profile=/load_settings(model_set=/g'
perl -pi -e 's/--profile/--models/g; s/\bargs\.profile\b/args.models/g' src/bedrock/litreview/cli.py src/bedrock/litreview/bench/__main__.py
perl -pi -e 's{litreview/sources/literature\.py}{src/bedrock/litreview/sources/literature.py}g; s{(?<![\w-])litreview (search|ask)\b}{bedrock-litreview $1}g' src/bedrock/litreview/defaults/pipeline.yaml
! grep -rnE '^\s*(from|import) litreview\b|from tests\.fixtures\b' tests/litreview
```

**Edit**
- `src/bedrock/config/settings.py`: the `litreview` field (§4.1).
- `src/bedrock/litreview/config.py`: rewrite it to §4.2 and §4.3.
- `src/bedrock/litreview/defaults/models.yaml`: §4.4.
- `runner.py`:
  - Import `PACKAGE_DIR` and `config_file` instead of `ROOT`.
  - `input_files()` returns `config_file(...)` for the three YAML files plus the
    knowledge tables, so it hashes the files this run actually read.
  - `_rel()` makes paths relative to `PACKAGE_DIR`; anything else stays an
    absolute path.
  - `code_version()` uses `cwd=PACKAGE_DIR` and
    `importlib.metadata.version("bedrock-private-ai")`.
- `report/layout.py`: `load_layout(question_type, path=None)` reads
  `path or config_file("report.yaml")` at call time.
- `bench/run.py`: `CORPORA = EVALS_DIR / "corpora"` and
  `RESULTS = EVALS_DIR / "results"`.
- `db.py`: create the parent directory with `mode=0o700`, and `chmod` the file
  to 0o600 unless it is `:memory:`.
- `cli.py` and `bench/__main__.py`: change the `--models` help text to "model
  set from defaults/models.yaml (office, laptop, medgemma)".
- `tests/conftest.py`, `private_bedrock_directories`: also set
  `bedrock.litreview.config.DEFAULT_DATA_DIR` and `DEFAULT_CONFIG_DIR` under
  its temporary root.
- New `tests/litreview/conftest.py` (autouse): point
  `bedrock.config.settings.DEFAULT_CONFIG_PATH` at a missing file in
  `tmp_path`, because the litreview loader reads bedrock's settings when it is
  given none.
- Test adaptations. These are the only hand edits to ported tests:
  - `test_guard_cache_config.py::test_a_non_zero_temperature_is_refused`:
    write a `models.yaml` override in the new schema to `tmp_path`, then call
    `load_settings(model_set="x", env={"BEDROCK_LITREVIEW_CONFIG_DIR": str(tmp_path)})`.
  - `test_runs.py`: the input-hash keys become `knowledge/entities.yaml` and
    `defaults/{models,pipeline,report}.yaml`.
  - `test_merge.py`: `parents[1]` becomes `parents[2]` (the rewrite already
    fixed the path string).
  - `test_sources.py`: the two CLI tests set `BEDROCK_LITREVIEW_DIR` instead of
    `DATA_DIR` and `OUT_DIR`.

**New tests** (`tests/litreview/test_bedrock_settings.py`):
- The model set is chosen in this order: argument, then env, then the
  `[litreview]` table, then the profile default.
- An unknown table key is refused.
- An override file replaces the packaged file, and the run records its path.
- A public `api_base` is refused, a `*.internal` one is accepted, and
  `allowed_hosts` lets an extra host through.
- Temperature 0.7 is refused.

Also add a case to `tests/test_settings.py`: `[litreview]` is parsed when
present and is `{}` when absent.

**Check:** `pytest -q` passes in full. The HTTP test in
`tests/litreview/test_sources.py` is skipped, because the `api` extra is not
installed until Phase 5.

**Commit:** `Literature review: settings from bedrock, YAML defaults as package data, tests ported`.

### Phase 4: model calls through bedrock's scheduler

**New:** `src/bedrock/litreview/llm/scheduled.py` (§4.5) and
`src/bedrock/litreview/service.py` (§4.6).

**Edit**
- `llm/client.py`: delete `OpenAICompatibleClient` and `OllamaClient`. Keep
  `strip_thinking`, `LLMError`, `LLMClient`, `FakeLLMClient`,
  `parse_json_reply`, `Responder` and the two defaults. Update the module
  docstring to say the one live client is `SchedulerClient`.
- `llm/__init__.py`: §4.5.
- `api.py`: `ask()` is documented as needing `client=`, or use
  `service.run_review`.

**New tests** (`tests/litreview/test_scheduler_client.py`) use a real
`Scheduler` and bedrock's `FakeBackend`, or a small scripted backend in the
test file:
- One call goes out with a single user message, `temperature=0.0`, the seed,
  the schema, the `max_tokens` and `Priority.USER_JOB`, and thinking traces are
  stripped from the reply.
- A call whose prompt plus reply overflows the window raises `LLMError`, and
  nothing is sent.
- A backend `InferenceError` becomes `LLMError`, and one with
  `unavailable=True` becomes `ModelServerUnavailable`.
- Once the cancel event is set, the client raises `ReviewCancelled` and sends
  nothing.
- Calling on the loop's own thread raises `RuntimeError` rather than
  deadlocking.
- `version_of` returns `name@digest12` when a digest is listed and the plain
  name otherwise.
- `thinking: false` becomes `reasoning=False`; an absent flag becomes `None`.

**New tests** (`tests/litreview/test_service.py`):
- `preflight` refuses a window that is too small (the message names 12,288
  and `OLLAMA_CONTEXT_LENGTH`), a missing model (the message names
  `ollama pull <name>`) and a silent server.
- `run_review` over a scripted backend produces a report byte-identical to the
  `FakeLLMClient` path on the same papers. The backend yields
  `tests/litreview/fixtures.py`'s `responder` reply unchanged, in one piece.
  This proves the bridge changes nothing. Model versions are written into the
  report, so use one model name for every tier (a models override) and
  `FakeLLMClient(version=<that name>)`.
- A server that goes down mid-run ends the run with `ModelServerUnavailable`,
  and the run can be resumed from the stage it reached.

**Check:** `pytest -q`, including `tests/test_model_call_guard.py`. Also
`! grep -rn "httpx" src/bedrock/litreview/llm`.

**Commit:** `Literature review: model calls through bedrock's scheduler`.

### Phase 5: command line, benchmark, and the searcher's host rule

**Edit**
- `cli.py`: `ask` goes through `standalone_scheduler` and
  `service.run_review` (§4.8). Add the `runs` subcommand. Update the docstring
  and `prog` to `bedrock-litreview`, and remove the HTTP mention. On
  `ReviewInterrupted`, print the resume command.
- `db.py`: add `recent_runs(limit=20) -> list[Run]`, newest first, and
  `report_id_for(run_id) -> str | None`.
- `api.py`: rebuild `create_app` on `standalone_scheduler` and
  `service.run_review` (§4.9). Change the docstring's serve command to
  `uvicorn bedrock.litreview.api:create_app --factory`.
- `bench/__main__.py`: `run` uses `asyncio.run` around `standalone_scheduler`,
  then `await asyncio.to_thread(evaluate, ...)` with
  `make_client(settings, scheduler=..., loop=...)`. The `--judge-file` path is
  unchanged.
- `sources/http.py`: add
  `SOURCE_HOSTS = frozenset({"www.ebi.ac.uk", "eutils.ncbi.nlm.nih.gov"})` and
  an httpx `event_hooks={"request": [...]}` hook that raises `SourceError` for
  any other host, redirects included. The docstring should say why (D17). Keep
  the user agent and NCBI `tool=litreview` unchanged, because they are
  deliberately anonymous.

**Tests**
- Run `pip install -e ".[dev,api]"`. Then adapt
  `test_http_ask_searches_only_when_asked`:
  - build the app with `create_app(settings, backend=<scripted backend>)`;
  - open the client with `with TestClient(app) as client:`, so the lifespan
    runs;
  - drop `settings.backend = "fake"`;
  - give the settings a 12,288 window and a model set that the backend lists.

  Every existing assertion stays. Add a case where a preflight failure returns
  503.
- Add a CLI `ask` test that runs with a model through a monkeypatched
  `create_backend` in `service` and prints the report path.
- `runs` lists the run.
- A redirect to another host is refused and recorded as a source failure,
  while the other sources carry on.

**Commit:** `Literature review: bedrock-litreview CLI, benchmark on the scheduler, searcher stays on its hosts`.

### Phase 6: the terminal app

**New:** `src/bedrock/ui/review_command.py`, which holds the parsing, the
`run_review_job(...)` coroutine (it never raises and returns True on a finished
report), and the text for the finish, list and verify notices. Use
`markdown_text`, `as_code` and `as_code_path` from `ui/rendering.py` on
anything a person typed.

**Edit**
- `app.py`:
  - Add the imports, and in `__init__` set `self._review_running = False` and
    `self._review_cancel = threading.Event()`.
  - Add `case "review" | "litreview"` to `run_command`.
  - Add `start_review` (it validates the arguments first, so a typo never
    starts a job) and a `literature_review` worker (not exclusive; the flag
    enforces one at a time).
  - Add `stop_review`.
  - `action_quit` sets the cancel event before it closes the backend.
  - Add one line to the module docstring.
- `ui/local_commands.py`: add `case "reviews"`. It reads the database in
  `asyncio.to_thread`, opening and closing its own `Database` in that thread.
- `ui/commands.py`: update `HELP_TEXT` and `KNOWN_COMMANDS` (§4.7).
- `agentpack_app.py`: add one `case` that answers `/review` (§4.7).

**Tests** (`tests/test_review_command.py`, driving the real app headless as the
existing app tests do):
- Parsing handles apostrophes, quoted paths with spaces, `stop`,
  `--resume` without a question, an empty argument (a usage error), and
  `--corpus` together with `--search` (an error).
- `/review … --no-model --corpus <fixture jsonl>` finishes. The report paths in
  the notice exist, and `FakeBackend.requests` is empty.
- A chat message sent while a review is held open gets its reply.
- A second `/review` while one is running is refused.
- `/review stop` sets the event.
- `/reviews` lists the run, and `/reviews verify <id>` says OK.
- On the model path, with a models override naming the fake's model and a
  12,288 window, the first request carries `temperature=0.0`, a `json_schema`
  and `seed=0`. The run ends in a notice, never an exception.
- The `laptop` profile's 8,192 window produces the preflight notice.

**Commit:** `App: /review runs a literature review beside the chat; /reviews lists and verifies them`.

### Phase 7: documents and project agents

**Moved documents.** Rewrite them mechanically once:

```bash
cd /Users/li/bedrock-litreview && perl -pi -e '
  s{\]\((litreview/[^)]*)\)}{](../../src/bedrock/$1)}g;
  s{\]\(config/(models|pipeline|report)\.yaml\)}{](../../src/bedrock/litreview/defaults/$1.yaml)}g;
  s{\]\(evals/([^)]*)\)}{](../../evals/litreview/$1)}g;
  s{\]\(CLAUDE\.md\)}{](../../src/bedrock/litreview/CLAUDE.md)}g;
  s{`litreview/}{`src/bedrock/litreview/}g;
  s{`config/(models|pipeline|report)\.yaml`}{`src/bedrock/litreview/defaults/$1.yaml`}g;
  s{(?<![\w.-])python -m litreview\.bench}{python -m bedrock.litreview.bench}g;
  s{(?<![\w-])litreview (ask|search|verify-run|rate)\b}{bedrock-litreview $1}g;
  s{--profile (office|laptop|medgemma)}{--models $1}g;' docs/litreview/README.md evals/litreview/README.md
```

Then do the following:
- `docs/litreview/README.md`: add this header note, then fix by hand the
  sections "Getting started", "Repository layout" and "Current state". Change
  the HTTP serve command to `uvicorn bedrock.litreview.api:create_app --factory`,
  installed with `pip install -e ".[api]"`.

  > Moved from Localized-Airgap-LLM-for-literature-review at `<sha>`. In
  > bedrock, `litreview` is `bedrock-litreview` (or `/review` in the app),
  > `--profile` is `--models`, `config/*.yaml` is
  > `src/bedrock/litreview/defaults/*.yaml`, `evals/` is `evals/litreview/`,
  > and the HTTP API is `bedrock.litreview.api` (install the `api` extra).
  > See `docs/prompts/LITREVIEW_INTEGRATION.md` §3.

- `docs/litreview/BENCHMARK_RESULTS.md`: add only a header note (with
  `perl -0pi -e 's/\A/…\n\n/'`) saying this is a record of measurements made in
  the source repository, with commands kept exactly as they were run. Do not
  rewrite the body.
- `evals/litreview/README.md`: after the script above, read it once and fix
  what the rules missed.
- `src/bedrock/litreview/CLAUDE.md`:
  - Retitle it "Working in `src/bedrock/litreview/`" and say that the root
    `CLAUDE.md` also applies.
  - Update the paths in "Read these first".
  - Change the commands in the Rules section to `bedrock-litreview …` or
    `/reviews verify`.
  - Rewrite the Conventions bullets for these points:
    - model names live in `defaults/models.yaml` or its override;
    - the only live model client is `SchedulerClient`, which goes through
      bedrock's `Scheduler` at `USER_JOB`;
    - nothing else in the package imports `bedrock.inference` except
      `llm/scheduled.py` and `service.py`;
    - the search is opt-in, through `search`, `ask --search` or
      `/review --search` or `"search": true` over HTTP;
    - tests run with `pytest tests/litreview`;
    - the benchmark runs with `python -m bedrock.litreview.bench run evals/litreview/questions/*.yaml --models laptop`.
  - Update the Current state table.
- `src/bedrock/litreview/AGENTS.md`: two lines pointing to `CLAUDE.md` in the
  same folder.
- Root `CLAUDE.md` and `AGENTS.md`: add an "Also built: literature review"
  paragraph saying:
  - what it is, and that it is a pipeline and not an agent;
  - to read the nested `CLAUDE.md` before changing it;
  - that its calls go through `SchedulerClient` at `USER_JOB`, at temperature
    0, with pinned model names;
  - that it needs a window of at least 12,288 tokens;
  - that its settings are the `[litreview]` table and its data lives in
    `~/.local/share/bedrock/litreview/`;
  - that its HTTP API (`bedrock.litreview.api`, the `[api]` extra) is kept on
    purpose. bedrock is terminal-first for now and will become an app, and the
    "not a website" rule covers the terminal interface, not this API.
- Root `README.md`:
  - Add the four command rows.
  - Add a "Literature review" section: examples; requirements (a 16 GB machine
    or the `workstation` profile, `context_window = 12288` with
    `OLLAMA_CONTEXT_LENGTH=12288`, the models pulled, and `request_timeout` of
    600 or more on slow machines); a `[litreview]` example table; what
    `--search` sends; where files go; `bedrock-litreview`; the HTTP API
    (`pip install -e ".[api]"`, the uvicorn command, localhost only until it
    has authentication).
  - Add rows to the configuration table.
  - Add a `litreview/` line to "How the code is organised".
- `NEXT_STEPS.md`: add a short "Literature review" section that lists §8 and
  points to `docs/litreview/BENCHMARK_RESULTS.md` § Next steps.
- `.claude/agents/litreview-*.md`: change `name:` to `litreview-benchmark-runner`
  and `litreview-deployment-auditor`, and point them at the nested
  `CLAUDE.md`, `defaults/*.yaml`, `docs/litreview/BENCHMARK_RESULTS.md` and the
  new commands.
- Optional: in `doctor.py`, have `ollama_settings(profile)` add one line saying
  that `/review` needs `context_window >= 12288` when the profile's window is
  smaller.

**Commit:** `Docs: literature review in bedrock`.

### Phase 8: verify

1. Run `pytest -q` in full. The model-call guard must pass.
2. These greps must find nothing in `src/` or `tests/`:
   - `grep -rnE "^\s*(from|import) litreview\b"`
   - `grep -rn "ROOT /"`
3. The copied logic is still byte-identical once the Phase 1 rewrite is
   accounted for. The script applies the same rewrite to a fresh archive and
   requires an exact match. It does not use `diff -I`, which macOS's diff does
   not honour for this. Every `CHANGED:` line needs an explanation in the
   hand-off note:

   ```bash
   cd /Users/li/bedrock-litreview && S=$(mktemp -d) && git -C /Users/li/Localized-Airgap-LLM-for-literature-review archive HEAD litreview | tar -x -C "$S" && find "$S" -name '*.py' -print0 | xargs -0 perl -pi -e '
     s/\bfrom litreview\b/from bedrock.litreview/g;
     s/\bimport litreview\b/import bedrock.litreview/g;
     s/(["\x27])litreview\./$1bedrock.litreview./g;
     s/python -m litreview\b/python -m bedrock.litreview/g;
     s/python -m evals\.run_extract\b/python -m evals.litreview.run_extract/g;
     s{(?<![\w/.])evals/(?!litreview/)}{evals/litreview/}g;' && for f in llm/prompts llm/cache.py llm/prompts.py knowledge stages extract report/guard.py report/invariants.py models.py facets.py lexical.py negation.py ingest.py parsing; do diff -r -x __pycache__ "$S/litreview/$f" "src/bedrock/litreview/$f" >/dev/null || echo "CHANGED: $f"; done; rm -rf "$S"
   ```

4. Build the wheel with `pip wheel . --no-deps -w "$(mktemp -d)"` and list it.
   The prompts, knowledge YAML and defaults YAML must be inside. Optionally,
   add these checks to `dist/installed_smoke.py`:
   - the package data is present;
   - `bedrock-litreview --help` exits 0;
   - `ask --no-model` on a two-paper corpus written to the temporary home
     produces `report.html`.
5. Run without a model:

   ```bash
   BEDROCK_LITREVIEW_DIR="$(mktemp -d)" bedrock-litreview ask "If I want to prevent psoriasis, what areas should I target?" --corpus evals/litreview/corpora/psoriasis-prevention-targets/papers.jsonl --no-model
   ```

   It should print a `report.html` path.
6. With a model, if Ollama is available. Skip this and say so if it is not.
   1. Run `ollama pull medgemma1.5:4b`, and start Ollama with
      `OLLAMA_CONTEXT_LENGTH=12288`.
   2. Run the same `ask` with `--models medgemma` and `BEDROCK_CONTEXT_WINDOW=12288`,
      twice. The report id must be identical, and the second run must replay
      the cache.
   3. Run the benchmark on one question:
      `python -m bedrock.litreview.bench run evals/litreview/questions/psoriasis_prevention.yaml --models medgemma --judge-file evals/litreview/judge/answers.jsonl`.
      Compare it with `docs/litreview/BENCHMARK_RESULTS.md`. The transport
      changed, so do not expect identical bytes against old samples; expect
      scores within noise.
   4. Run `uvicorn bedrock.litreview.api:create_app --factory --host 127.0.0.1`
      with the same environment. `POST /ask` the same question: its papers
      are already frozen from step 2. The report id must match step 2, and
      `GET /reports/{id}?format=html` must return the report.
7. In the app:
   1. Run `/review <question> --corpus … --no-model` while a chat reply is
      streaming.
   2. Stop a model review with `/review stop`, then resume it with
      `--resume <id>`.
   3. Check `/reviews` and `/reviews verify`.

---

## 6. Done means

- The full suite passes. The expected total is the Phase 0 baseline, plus the
  ported litreview tests (all but the deleted FastAPI one), plus the new tests.
  Report all three numbers.
- `tests/test_model_call_guard.py` passes, and no review model call bypasses
  `Scheduler`.
- Prompts, knowledge tables, stages, verification, provenance, the guard and
  the invariants are unchanged apart from imports (Phase 8, step 3).
- A fresh wheel install runs a `--no-model` review from package data alone.
- `/review` runs beside the chat, stops, resumes, and never writes over the
  screen.
- On a model, the same question over the same papers gives the same report id
  twice (or the note says why this could not be run).
- The HTTP test passes with the `api` extra installed, and `POST /ask` goes
  through the same `service.run_review` and scheduler as `/review`.
- No test reads or writes a real home directory.

## 7. Hand-off note

End with:
- the files added, changed and deleted;
- the test counts before and after;
- every deviation from this plan, and why;
- any decision you had to make that the plan did not cover;
- the Phase 8 step 6 results, or "not run" with the reason;
- a 3–5 line paragraph for the root `CLAUDE.md` "Current status", if you did
  not already add it.

## 8. Not in this change

1. Record reviews and searches in `/audit`, with new journal actions
   `litreview.review` and `litreview.search`. Search is the one step that sends
   anything off the machine. Read `docs/COMPLIANCE_MVP.md` and
   `docs/compliance_mvp/CONTRACT.md` first.
2. Publish a finished report to a shared workspace (`/workspace publish`).
3. Build the drop folder on bedrock's `share` connector, and add the GROBID PDF
   rung.
4. Build Stage 9 chat: the bedrock chat with narrow, read-only tools over one
   run's frozen evidence. It must not be free chat over `report.md`, which
   would bypass every guard.
5. Support `/review` in agent-pack mode, or add a read-only agent capability
   that starts the pipeline but cannot steer it.
6. Apply the endpoint check (D9) to the chat as well. This is a product
   decision.
7. Use `strip_thinking` on chat replies (MedGemma traces), and carry
   `finish_reason` in `Usage` so a cut-off free-text section is logged as it
   was in the source.
8. Authentication for the HTTP API, once bedrock's company sign-in (Keycloak)
   reaches it; until then it has none. Concurrent `POST /ask` requests also
   share one SQLite file, which a multi-user service must handle (the source's
   Phase 5).
9. The source's to-do list continues in `docs/litreview/BENCHMARK_RESULTS.md`
   § Next steps. Its multi-user and packaging phases now mean bedrock's
   packaging. Archive the source repository, with a pointer here, once this
   merges.
