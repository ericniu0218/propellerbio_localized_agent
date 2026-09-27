"""Does the model server give the same reply twice? Measured, not assumed.

Four different things are easy to confuse, and this module keeps them in
separate rows because only the last one is about the server:

  orchestration        the pipeline's own code, given the same inputs, makes the
                       same choices. Proved by the unit tests, offline.
  cached replay        the call cache returns a stored reply to an identical
                       call. `bench run`'s "same report twice (cached replay)".
  recorded rendering   the same recorded model replies, fed through the pipeline
                       twice, give byte-identical JSON, Markdown and HTML
                       (`replay_control`). If this fails the problem is in the
                       code, not the server.
  fresh inference      the same pinned request, sent to the server again and
                       again with no cache at all, gets the same reply
                       (`run_cell` with mode "fresh"). Deployment-specific: a
                       result holds for the server, model, hardware and
                       settings it was measured on, and nothing else.

Fresh mode refuses to pass by accident. Each repetition gets a new client with
no cache; the backend's request observer must be attached (a client that
cannot report its wire calls cannot run fresh mode) and must see a completed
response for every payload - a repetition that made no call to the server is a
failure, never an identical reply. Every attempt is appended to
`attempts.jsonl` and every distinct reply is kept by hash, so nothing a
repetition returned is overwritten by the next.

Nothing here talks to a model directly: the live client is `SchedulerClient`,
built by the caller through `service`/`make_client`, so every call still
waits its turn at `Priority.USER_JOB`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from bedrock.litreview.facets import facets_for
from bedrock.litreview.ingest import ingest
from bedrock.litreview.models import ExtractionJob, Paper, QuestionSpec
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s3_rank import relevance_request
from bedrock.litreview.stages.s5_extract import extraction_schema, methods_context, plan_chunks, render_prompt
from bedrock.litreview.study_design import design_signals

PAYLOADS_VERSION = "determinism_payloads.v1"
EXPERIMENT_VERSION = "determinism_experiment.v1"
MANIFEST_KEYS = (
    "model", "model_revision", "weights_sha256", "tokenizer", "chat_template", "vllm_version", "python", "torch",
    "cuda", "driver", "gpu_type", "gpu_count", "tensor_parallel", "pipeline_parallel", "quantization", "dtype",
    "attention_backend", "compilation", "cuda_graphs", "launch_command", "seed", "sampling_defaults",
    "structured_output_backend", "speculative_decoding", "prefix_caching", "batch_invariant", "max_num_seqs",
    "concurrency", "request_schedule",
)
"""What a deployment manifest must state before a cell may be called fully pinned."""


def sha256(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


# --- pinned payloads -------------------------------------------------------------------------


def _features(text: str, paper: Paper) -> list[str]:
    out = []
    lowered = text.lower()
    if any(w in lowered for w in ("did not", "no significant", "not significantly", "failed to", "no effect")):
        out.append("negation")
    if any(ch.isdigit() for ch in text) and ("%" in text or "p=" in lowered or "p <" in lowered or "95%" in text):
        out.append("numeric_outcome")
    if "mixed" in design_signals(paper)["labels"]:
        out.append("mixed_species")
    if len(text) >= 5000:
        out.append("long_context")
    return out


def build_payloads(
    questions: Iterable[tuple[str, str, list[Paper]]],
    *,
    relevance_per_corpus: int = 5,
    extraction_per_corpus: int = 5,
    chunk_chars: int = 6000,
    methods_chars: int = 2500,
) -> list[dict[str, Any]]:
    """Representative stage requests, rendered from public frozen corpora with
    no model: relevance prompts for the first papers that exercise negation,
    numbers and mixed species, and extraction prompts for the longest chunks.
    Deterministic: the same corpora give the same file."""
    out: list[dict[str, Any]] = []
    for corpus_id, question, raw_papers in questions:
        papers = sorted(ingest(raw_papers), key=lambda p: p.id)
        spec: QuestionSpec = parse_question(question, None, model=None)
        facets = facets_for(spec)
        scored = []
        for p in papers:
            text = p.head_text()
            if not p.abstract:
                continue
            feats = _features(text, p)
            scored.append((-len(feats), p.id, p, feats))
        for _, pid, paper, feats in sorted(scored)[:relevance_per_corpus]:
            prompt, schema = relevance_request(paper, spec)
            out.append(_payload(corpus_id, "relevance", prompt, schema, 600, pid, None, feats))
        chunks = []
        query = f"{spec.text} {' '.join(t for e in spec.entities for t in e.exact_terms)}"
        for paper in papers:
            if not paper.has_full_text:
                continue
            job = ExtractionJob(paper_id=paper.id, sections=[s for s in ("results",) if paper.sections.get(s)],
                                facets=[f.id for f in facets])
            for section, i, text in plan_chunks(paper, job, query, chunk_chars=chunk_chars, max_chunks=2):
                chunks.append((-len(text), paper.id, section, i, text, paper, job))
        for _, pid, section, i, text, paper, job in sorted(chunks)[:extraction_per_corpus]:
            prompt = render_prompt(spec, facets, job, section, text, methods_context(paper, methods_chars))
            out.append(_payload(corpus_id, "extract", prompt, extraction_schema(facets), 4096, pid,
                                f"{section}:{i}", _features(text, paper)))
    return out


def _payload(corpus, job, prompt, schema, max_tokens, paper_id, chunk, features) -> dict[str, Any]:
    body = {"job": job, "prompt": prompt, "schema": schema, "max_tokens": max_tokens}
    digest = sha256(json.dumps(body, sort_keys=True))
    return {
        "version": PAYLOADS_VERSION, "id": f"{corpus}:{job}:{paper_id}{':' + chunk if chunk else ''}",
        "sha256": digest, **body,
        "source": {"corpus": corpus, "paper_id": paper_id, "chunk": chunk}, "features": sorted(features),
    }


def load_payloads(path: Path) -> list[dict[str, Any]]:
    """Read pinned payloads, refusing any whose content no longer matches its hash."""
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        p = json.loads(line)
        body = {k: p[k] for k in ("job", "prompt", "schema", "max_tokens")}
        if p.get("version") != PAYLOADS_VERSION or sha256(json.dumps(body, sort_keys=True)) != p.get("sha256"):
            raise ValueError(f"{path}:{n}: payload {p.get('id')!r} does not match its pinned hash")
        out.append(p)
    return out


# --- one experiment cell ---------------------------------------------------------------------------


class FreshModeUnavailable(RuntimeError):
    """Fresh mode needs a client whose backend reports its wire calls."""


@dataclass
class CellResult:
    condition: str
    mode: str
    repeats: int
    payloads: int
    attempts: int = 0
    failures: int = 0
    no_call_failures: int = 0
    unique: dict[str, int] = field(default_factory=dict)
    """Payload id -> number of distinct normalised replies."""
    unique_raw: dict[str, int] = field(default_factory=dict)
    first_divergent: list[str] = field(default_factory=list)
    manifest_missing: list[str] = field(default_factory=list)

    @property
    def repeatable(self) -> bool:
        return self.failures == 0 and self.attempts > 0 and all(n == 1 for n in self.unique.values())

    def row(self) -> dict[str, Any]:
        return {
            "row": {"fresh": "fresh_inference", "replay": "cached_replay"}[self.mode],
            "condition": self.condition, "repeats": self.repeats, "payloads": self.payloads,
            "attempts": self.attempts, "failures": self.failures, "no_call_failures": self.no_call_failures,
            "payloads_with_one_reply": sum(1 for n in self.unique.values() if n == 1),
            "max_distinct_replies": max(self.unique.values(), default=0),
            "max_distinct_raw_replies": max(self.unique_raw.values(), default=0),
            "diverging_payloads": self.first_divergent,
            "repeatable_in_this_cell": self.repeatable,
            "fully_pinned_deployment": not self.manifest_missing,
            "manifest_missing": self.manifest_missing,
        }


def run_cell(
    payloads: list[dict[str, Any]],
    make_client: Callable[[int], Any],
    *,
    models: dict[str, str],
    repeats: int,
    mode: str,
    condition: str,
    out_dir: Path,
    manifest: dict[str, Any] | None = None,
) -> CellResult:
    """Send every payload `repeats` times, each repetition through a new client.

    `make_client(rep)` returns the client for repetition `rep`: in fresh mode
    one with no cache, in replay mode one reading an existing cache. Records go
    to `out_dir`, which must not already hold this condition's records.
    """
    if mode not in ("fresh", "replay"):
        raise ValueError(f"mode must be fresh or replay, not {mode!r}")
    out_dir = Path(out_dir)
    cell_dir = out_dir / condition
    if cell_dir.exists():
        raise FileExistsError(f"{cell_dir} already exists: records are never overwritten; pick a new condition name")
    (cell_dir / "outputs").mkdir(parents=True)
    missing = [k for k in MANIFEST_KEYS if (manifest or {}).get(k) in (None, "", "unknown")]
    if manifest is not None:
        (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    result = CellResult(condition=condition, mode=mode, repeats=repeats, payloads=len(payloads),
                        manifest_missing=missing)
    seen: dict[str, list[str]] = defaultdict(list)
    seen_raw: dict[str, set[str]] = defaultdict(set)
    attempts_path = cell_dir / "attempts.jsonl"
    with attempts_path.open("a", encoding="utf-8") as log:
        for rep in range(repeats):
            client = make_client(rep)
            events: list[dict[str, Any]] = []
            if mode == "fresh":
                attach = getattr(client, "add_request_observer", None)
                if attach is None or not attach(events.append):
                    raise FreshModeUnavailable(
                        "fresh mode needs a client whose backend reports each wire call "
                        "(SchedulerClient over OpenAICompatibleBackend); without it a cached or "
                        "missing call could pass as a repeated reply")
            context_fn = getattr(client, "generation_context", None)
            for p in payloads:
                model = models[p["job"]]
                context = context_fn(model) if context_fn else None
                before = len(events)
                started = time.monotonic()
                record: dict[str, Any] = {
                    "version": EXPERIMENT_VERSION, "condition": condition, "mode": mode, "rep": rep,
                    "payload_id": p["id"], "payload_sha256": p["sha256"], "model": model,
                    "generation_context_sha256": sha256(json.dumps(context, sort_keys=True)) if context else None,
                    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                try:
                    reply = client.complete(p["prompt"], model=model, schema=p["schema"], temperature=0.0,
                                            max_tokens=p["max_tokens"])
                except Exception as exc:  # noqa: BLE001 - a failure is a result, recorded as one
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    reply = None
                wire = events[before:]
                responses = [e for e in wire if e.get("event") == "response"]
                record.update(
                    elapsed_s=round(time.monotonic() - started, 3), wire=wire, backend_calls=len(responses),
                    raw_sha256=responses[-1]["raw_sha256"] if responses else None,
                )
                if reply is not None:
                    digest = sha256(reply)
                    record.update(normalized_sha256=digest, status="ok")
                    out = cell_dir / "outputs" / f"{digest}.txt"
                    if not out.exists():
                        out.write_text(reply, encoding="utf-8")
                    if mode == "fresh" and not responses:
                        # The reply came from somewhere other than the server.
                        record.update(status="no_backend_call")
                        result.no_call_failures += 1
                if record.get("status") != "ok":
                    result.failures += 1
                else:
                    seen[p["id"]].append(record["normalized_sha256"])
                    if record["raw_sha256"]:
                        seen_raw[p["id"]].add(record["raw_sha256"])
                result.attempts += 1
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
    for p in payloads:
        hashes = seen.get(p["id"], [])
        result.unique[p["id"]] = len(set(hashes))
        result.unique_raw[p["id"]] = len(seen_raw.get(p["id"], set()))
        if len(set(hashes)) > 1:
            result.first_divergent.append(p["id"])
    (cell_dir / "summary.json").write_text(json.dumps(result.row(), indent=2, sort_keys=True), encoding="utf-8")
    return result


# --- recorded-output rendering control --------------------------------------------------------------


class RecordingClient:
    """Pass calls through and keep every reply, keyed by the exact request."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.replies: dict[str, str] = {}

    @staticmethod
    def key(prompt: str, model: str, schema, temperature, max_tokens) -> str:
        return sha256(json.dumps([prompt, model, schema, temperature, max_tokens], sort_keys=True))

    def complete(self, prompt, *, model, schema=None, temperature=0.0, max_tokens=2048):
        reply = self.inner.complete(prompt, model=model, schema=schema, temperature=temperature, max_tokens=max_tokens)
        self.replies[self.key(prompt, model, schema, temperature, max_tokens)] = reply
        return reply

    @property
    def model_version(self) -> str:
        return getattr(self.inner, "model_version", "recorded")

    def version_of(self, model: str) -> str:
        fn = getattr(self.inner, "version_of", None)
        return fn(model) if fn else self.model_version


class ReplayClient:
    """Serve recorded replies; a request that was not recorded is an error,
    never a new call - this control must not touch a server."""

    def __init__(self, replies: dict[str, str], version: str = "recorded") -> None:
        self.replies = dict(replies)
        self._version = version
        self.misses: list[str] = []

    def complete(self, prompt, *, model, schema=None, temperature=0.0, max_tokens=2048):
        key = RecordingClient.key(prompt, model, schema, temperature, max_tokens)
        if key not in self.replies:
            from bedrock.litreview.llm import LLMError

            self.misses.append(key)
            raise LLMError("replay control: this request was not recorded")
        return self.replies[key]

    @property
    def model_version(self) -> str:
        return self._version

    def version_of(self, model: str) -> str:
        return self._version

    def generation_context(self, model: str) -> dict[str, Any]:
        return {"schema": "generation_context.v1", "client": "replay", "model": model, "version": self._version}


def replay_control(question: str, papers: list[Paper], replies: dict[str, str], *, settings_for, version: str,
                   work_dir: Path) -> dict[str, Any]:
    """Run the pipeline twice on the same recorded replies, in separate
    databases and output folders, and compare every artifact byte for byte.
    Run ids and timestamps are operational and are not compared; nothing in
    the report is excluded."""
    from bedrock.litreview.db import Database
    from bedrock.litreview.runner import answer_question

    outputs = []
    for n in (1, 2):
        d = Path(work_dir) / f"replay{n}"
        settings = settings_for(d)
        client = ReplayClient(replies, version=version)
        r = answer_question(question, settings=settings, client=client, papers=papers, db=Database(d / "db.sqlite3"),
                            out_dir=d / "out", progress=lambda m: None)
        outputs.append({
            "report_id": r.report_id,
            "json": sha256(r.paths["json"].read_bytes()),
            "markdown": sha256(r.paths["md"].read_bytes()),
            "html": sha256(r.paths["html"].read_bytes()),
            "unrecorded_requests": len(client.misses),
        })
    first, second = outputs
    identical = all(first[k] == second[k] for k in ("report_id", "json", "markdown", "html"))
    return {"row": "recorded_output_rendering", "identical": identical and not first["unrecorded_requests"],
            "runs": outputs}
