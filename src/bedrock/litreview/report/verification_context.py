"""What a run was built under, and whether today's checks run under the same.

`verify-run` re-proves a stored report with the code, knowledge tables and
word lists installed *now*. That is the right check - a report that fails
today's rules should say so - but on its own it cannot tell a scientist
whether today's rules are the ones the report was written and checked under.
A run that passes after `meaning_words.yaml` was edited has passed a
different test from the one it passed on the day. This module makes that
visible. It does not change what passes.

Two halves:

  capture   at the start of every invocation of a run (the first and each
            resume), record the logical inputs actually in effect: which
            config file played which role and the hash of its bytes, the
            hash of each prompt file as loaded, the code identity and how it
            was found, a fingerprint of the verifier's own source, and the
            data-model schemas. Stored as the run's `verification_context.v1`
            artifact - a list, one entry per invocation, appended and never
            rewritten except to close the entry's own stage range.

  compare   set a run's record beside the current environment, role by role,
            and name each difference: what it was, what it is, and whether
            it affects how reports are *generated* or how they are *checked*.

What this is not. It is provenance comparison, not an attestation: the
hashes sit in the same local database as the report, so someone able to
rewrite both is outside what it can detect. A difference proves the
environment changed, not that a verdict changed - a comment added to a YAML
file is a difference. And it never reads a file from a path a stored run
names, never checks out or runs old code, and never goes to the network.

Legacy runs (before this artifact) are compared from what they did record:
`Run.input_hashes`, prompt version *labels* and `code_version`. What they
did not record - prompt content, the verifier fingerprint - is reported as
not recorded, never guessed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from bedrock.litreview.config import DEFAULTS_DIR, PACKAGE_DIR, config_file

ARTIFACT = "verification_context.v1"
SCHEMA_VERSION = 1

CONFIG_FILES = ("models.yaml", "pipeline.yaml", "report.yaml")
PROMPTS = ("question", "relevance", "extract", "write_section")

# The source files `verify_run` executes: the checks themselves, the matchers
# and normalisers they call, and the serialiser `check_pinned` compares with.
VERIFIER_SOURCES = (
    "report/invariants.py",
    "report/guard.py",
    "extract/verify.py",
    "extract/provenance.py",
    "extract/normalize.py",
    "knowledge/__init__.py",
    "negation.py",
    "models.py",
    "stages/s7_write.py",
    "stages/s8_render.py",
)

Affects = Literal["generation", "verification", "generation+verification", "unknown"]
Status = Literal["unchanged", "changed", "incomplete"]

# What each logical input is used for. Config files steer how a report is
# written; the knowledge tables and word lists are also what the checks read.
_AFFECTS: dict[str, Affects] = {
    "config/models.yaml": "generation",
    "config/pipeline.yaml": "generation",
    "config/report.yaml": "generation",
    "knowledge/entities.yaml": "generation+verification",
    "knowledge/meaning_words.yaml": "generation+verification",
    "code": "generation+verification",
    "verifier_source": "verification",
}


def affects(role: str) -> Affects:
    if role in _AFFECTS:
        return _AFFECTS[role]
    if role.startswith("prompt/"):
        return "generation"
    if role.startswith("schema/"):
        return "generation+verification"
    if role.startswith("knowledge/"):
        return "generation+verification"
    return "unknown"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- code identity -----------------------------------------------------------------


@dataclass(frozen=True)
class CodeIdentity:
    version: str | None
    method: Literal["git-describe", "package-metadata", "unknown"]
    dirty: bool | None
    """True when git reports uncommitted changes: the version then names a
    commit the running code is *not* exactly, so two equal dirty versions do
    not prove equal code."""


def code_identity() -> CodeIdentity:
    """`git describe --always --dirty` of the package, else its installed
    version, else unknown. Never raises: provenance is best effort, and a run
    must not fail on it."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=PACKAGE_DIR, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            value = out.stdout.strip()
            return CodeIdentity(value, "git-describe", value.endswith("-dirty"))
    except Exception:  # noqa: BLE001
        pass
    try:
        return CodeIdentity(importlib.metadata.version("bedrock-private-ai"), "package-metadata", None)
    except Exception:  # noqa: BLE001
        return CodeIdentity(None, "unknown", None)


# --- the current environment ----------------------------------------------------------


def verifier_fingerprint() -> str:
    """One hash over the verifier's source files as they are on disk now."""
    h = hashlib.sha256()
    for rel in VERIFIER_SOURCES:
        h.update(rel.encode("utf-8") + b"\0")
        data = (PACKAGE_DIR / rel).read_bytes() if (PACKAGE_DIR / rel).is_file() else b"<missing>"
        h.update(_sha256_bytes(data).encode("ascii") + b"\n")
    return h.hexdigest()


def schema_fingerprints() -> dict[str, str]:
    """The JSON schemas of the stored data model: a report is parsed with
    these, so a change to them is a change to how it is read back."""
    from bedrock.litreview.models import EvidenceUnit, Paper, Report, Run

    return {
        f"schema/{cls.__name__}": _sha256_bytes(
            json.dumps(cls.model_json_schema(), sort_keys=True).encode("utf-8")
        )
        for cls in (Report, EvidenceUnit, Paper, Run)
    }


def current_inputs(kb=None, env=None) -> dict[str, dict[str, Any]]:
    """Role -> {"source", "sha256"} for the inputs in effect in this process.

    The config files are the ones `config_file` resolves now (an override if
    there is one); their bytes are hashed as they are on disk, because every
    run reads them afresh. The knowledge table and word lists are hashed as
    *loaded*: they are cached per process, and the checks use the cached
    copy, so that is the one that matters. A `Knowledge` built in memory has
    no file and so no hash; it is recorded as unknown rather than compared
    against the packaged table it did not come from.
    """
    from bedrock.litreview.extract.verify import meaning_words_sha256
    from bedrock.litreview.knowledge import TABLE, load_knowledge

    out: dict[str, dict[str, Any]] = {}
    for name in CONFIG_FILES:
        path = config_file(name, env)
        source = "package" if path.parent.resolve() == DEFAULTS_DIR.resolve() else "override"
        out[f"config/{name}"] = {"source": source, "sha256": _file_sha256(path)}
    kb = kb if kb is not None else load_knowledge()
    sha = getattr(kb, "source_sha256", None)
    out["knowledge/entities.yaml"] = {"source": "package" if sha else "memory", "sha256": sha}
    out["knowledge/meaning_words.yaml"] = {"source": "package", "sha256": meaning_words_sha256()}
    for path in sorted(TABLE.parent.glob("*.yaml")):
        role = f"knowledge/{path.name}"
        if role not in out:
            out[role] = {"source": "package", "sha256": _file_sha256(path)}
    return out


def current_prompts() -> dict[str, dict[str, str]]:
    from bedrock.litreview.llm import load_prompt

    out = {}
    for name in PROMPTS:
        prompt = load_prompt(name)
        out[name] = {"version": prompt.version, "sha256": prompt.sha256}
    return out


def capture(
    *, stage_at_start: int, kb=None, env=None, code: CodeIdentity | None = None,
    generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """This invocation's context, ready to append to a run's artifact.

    No absolute paths: a role and whether it came from the package or an
    override is what a comparison needs, and a path would put a user's home
    directory into anything exported from the run."""
    code = code or code_identity()
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": _now(),
        "stage_at_start": stage_at_start,
        "stage_at_end": None,
        "inputs": current_inputs(kb, env),
        "prompts": current_prompts(),
        "code": {"version": code.version, "method": code.method, "dirty": code.dirty},
        "verifier_sha256": verifier_fingerprint(),
        "schemas": schema_fingerprints(),
        # Per job, the client's generation context (seed, thinking policy,
        # weights where known, declared revision, server) - the same identity
        # the call cache keys on. Recorded, not compared: a later verifier has
        # no model client to compare it with, and needs none.
        "generation": generation,
    }


def load_contexts(db, run_id: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """The run's recorded contexts, or (None, why) when there are none or the
    artifact cannot be read. Malformed is reported, never repaired."""
    body = db.get_artifact(run_id, ARTIFACT)
    if body is None:
        return None, None
    try:
        data = json.loads(body)
    except ValueError:
        return None, f"the {ARTIFACT} artifact is not valid JSON"
    if not isinstance(data, list) or not all(isinstance(c, dict) for c in data) or not data:
        return None, f"the {ARTIFACT} artifact is not a list of contexts"
    return data, None


def append_context(db, run_id: str, context: dict[str, Any]) -> None:
    """Add one invocation's context. Earlier entries are kept as they were."""
    existing, _ = load_contexts(db, run_id)
    db.put_artifact(run_id, ARTIFACT, json.dumps([*(existing or []), context], sort_keys=True))


def close_context(db, run_id: str, stage: int) -> None:
    """Record how far the latest invocation got."""
    existing, _ = load_contexts(db, run_id)
    if not existing:
        return
    existing[-1]["stage_at_end"] = stage
    db.put_artifact(run_id, ARTIFACT, json.dumps(existing, sort_keys=True))


# --- comparison ---------------------------------------------------------------------------

DifferenceKind = Literal[
    "changed",          # recorded and current both known, and they differ
    "source_changed",   # the run read an override and now the package default is in effect, or back
    "missing_now",      # recorded, but the role does not exist in this environment
    "not_recorded",     # exists now, but the run did not record it
    "unknown_now",      # cannot be fingerprinted in this environment
    "not_exact",        # recorded and current agree on paper, but that is not proof (dirty tree)
    "unrecognised",     # a recorded key this code does not know how to compare
]
_CHANGES = {"changed", "source_changed", "missing_now"}


@dataclass(frozen=True)
class Difference:
    role: str
    kind: DifferenceKind
    recorded: str | None
    current: str | None
    affects: Affects
    reason: str

    def line(self) -> str:
        return f"{self.role} ({self.affects}): {self.reason}"


@dataclass
class Comparison:
    status: Status
    differences: list[Difference] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    legacy: bool = False

    @property
    def changed_roles(self) -> list[str]:
        return [d.role for d in self.differences if d.kind in _CHANGES]


def _short(value: str | None) -> str | None:
    return value[:12] if isinstance(value, str) and len(value) == 64 else value


def _role_of_recorded(key: str) -> tuple[str | None, str | None]:
    """(role, source) for a key in a legacy `Run.input_hashes`, or (None, None)."""
    if key.startswith("defaults/") and key[len("defaults/"):] in CONFIG_FILES:
        return f"config/{key[len('defaults/'):]}", "package"
    if key.startswith("knowledge/") and key.endswith(".yaml") and "/" not in key[len("knowledge/"):]:
        return key, "package"
    # An override was recorded by its absolute path. The path is never
    # opened: only its file name says which role it played.
    name = key.replace("\\", "/").rsplit("/", 1)[-1]
    if key.startswith(("/", "~")) or (len(key) > 2 and key[1] == ":"):
        if name in CONFIG_FILES:
            return f"config/{name}", "override"
    return None, None


def _compare_inputs(
    recorded: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]], out: list[Difference]
) -> None:
    for role in sorted(set(recorded) | set(current)):
        rec, cur = recorded.get(role), current.get(role)
        kind_of = affects(role)
        if rec is None:
            out.append(Difference(role, "not_recorded", None, _short(cur and cur.get("sha256")), kind_of,
                                  "in effect now, but this run did not record it"))
            continue
        if cur is None:
            out.append(Difference(role, "missing_now", _short(rec.get("sha256")), None, kind_of,
                                  "the run recorded it; it does not exist in this environment"))
            continue
        rsha, csha = rec.get("sha256"), cur.get("sha256")
        if not csha:
            why = ("the knowledge table in use was built in memory, so its content has no fingerprint"
                   if cur.get("source") == "memory" else "cannot be fingerprinted here")
            out.append(Difference(role, "unknown_now", _short(rsha), None, kind_of, why))
            continue
        if not rsha:
            out.append(Difference(role, "not_recorded", None, _short(csha), kind_of,
                                  "the run recorded the role but not its content"))
            continue
        if rec.get("source") != cur.get("source") and {rec.get("source"), cur.get("source")} <= {"package", "override"}:
            if rsha != csha:
                out.append(Difference(
                    role, "source_changed", _short(rsha), _short(csha), kind_of,
                    f"the run read the {rec.get('source')} file; the {cur.get('source')} file is in effect now "
                    "and differs (the file the run read is not substituted or reopened)",
                ))
            continue
        if rsha != csha:
            out.append(Difference(role, "changed", _short(rsha), _short(csha), kind_of,
                                  "content differs from what the run recorded"))


def _compare_code(recorded: dict[str, Any], current: CodeIdentity, out: list[Difference]) -> None:
    rv, cv = recorded.get("version"), current.version
    if not rv:
        out.append(Difference("code", "not_recorded", None, cv, "generation+verification",
                              "the run did not record a code version"))
    elif not cv:
        out.append(Difference("code", "unknown_now", rv, None, "generation+verification",
                              "the code version cannot be determined in this environment"))
    elif rv != cv:
        out.append(Difference("code", "changed", rv, cv, "generation+verification",
                              f"code version was {rv}, is {cv} (a different version proves a different "
                              "environment, not a different verdict)"))
    elif recorded.get("dirty") or current.dirty or str(rv).endswith("-dirty"):
        out.append(Difference("code", "not_exact", rv, cv, "generation+verification",
                              "same version label, but the working tree had uncommitted changes, so equal "
                              "labels do not prove equal code"))


def _status(differences: list[Difference], notes: list[str], incomplete: bool) -> Status:
    if any(d.kind in _CHANGES for d in differences):
        return "changed"
    if incomplete or differences:
        return "incomplete"
    return "unchanged"


def compare_contexts(
    contexts: list[dict[str, Any]],
    *,
    current_in: dict[str, dict[str, Any]],
    current_pr: dict[str, dict[str, str]],
    current_code: CodeIdentity,
    current_verifier: str,
    current_schemas: dict[str, str],
) -> Comparison:
    """A run with `verification_context.v1`: compare the context of the
    invocation that wrote the report (the last one) with now."""
    last = contexts[-1]
    diffs: list[Difference] = []
    notes: list[str] = []
    incomplete = False
    if last.get("schema_version") != SCHEMA_VERSION:
        notes.append(f"the recorded context has schema version {last.get('schema_version')!r}; "
                     f"this code compares version {SCHEMA_VERSION}")
        incomplete = True
    inputs = last.get("inputs")
    if isinstance(inputs, dict):
        _compare_inputs(inputs, current_in, diffs)
    else:
        notes.append("the recorded context has no readable inputs")
        incomplete = True
    prompts = last.get("prompts") if isinstance(last.get("prompts"), dict) else {}
    for name in sorted(set(prompts) | set(current_pr)):
        rec, cur = prompts.get(name) or {}, current_pr.get(name) or {}
        role = f"prompt/{name}"
        if not rec.get("sha256"):
            diffs.append(Difference(role, "not_recorded", None, _short(cur.get("sha256")), "generation",
                                    "prompt content was not recorded"))
        elif not cur:
            diffs.append(Difference(role, "missing_now", _short(rec["sha256"]), None, "generation",
                                    "the prompt no longer exists"))
        elif rec["sha256"] != cur.get("sha256"):
            same_label = rec.get("version") == cur.get("version")
            diffs.append(Difference(
                role, "changed", f"v{rec.get('version')} {_short(rec['sha256'])}",
                f"v{cur.get('version')} {_short(cur.get('sha256'))}", "generation",
                "content changed without a version bump" if same_label else "prompt version and content changed",
            ))
    _compare_code(last.get("code") or {}, current_code, diffs)
    rv = last.get("verifier_sha256")
    if not rv:
        diffs.append(Difference("verifier_source", "not_recorded", None, _short(current_verifier), "verification",
                                "the verifier fingerprint was not recorded"))
    elif rv != current_verifier:
        diffs.append(Difference("verifier_source", "changed", _short(rv), _short(current_verifier), "verification",
                                "the source of the checks differs from the one the run was built with"))
    schemas = last.get("schemas") if isinstance(last.get("schemas"), dict) else {}
    for role in sorted(set(schemas) | set(current_schemas)):
        rec, cur = schemas.get(role), current_schemas.get(role)
        if rec is None:
            diffs.append(Difference(role, "not_recorded", None, _short(cur), affects(role),
                                    "schema not recorded"))
        elif cur is None:
            diffs.append(Difference(role, "missing_now", _short(rec), None, affects(role), "schema no longer exists"))
        elif rec != cur:
            diffs.append(Difference(role, "changed", _short(rec), _short(cur), affects(role),
                                    "the stored data model's schema differs"))
    if len(contexts) > 1:
        distinct = {json.dumps({k: c.get(k) for k in ("inputs", "prompts", "code", "verifier_sha256", "schemas")},
                               sort_keys=True) for c in contexts}
        if len(distinct) > 1:
            notes.append(f"the run was built over {len(contexts)} invocations under {len(distinct)} different "
                         "environments; the comparison uses the last, which wrote the report")
    return Comparison(_status(diffs, notes, incomplete), diffs, notes)


def compare_legacy(
    run,
    *,
    current_in: dict[str, dict[str, Any]],
    current_pr: dict[str, dict[str, str]],
    current_code: CodeIdentity,
) -> Comparison:
    """A run from before `verification_context.v1`: compare what it did record."""
    diffs: list[Difference] = []
    notes = [
        "legacy run: prompt content, the verifier fingerprint and data-model schemas were not recorded",
    ]
    incomplete = True
    raw = run.input_hashes if isinstance(run.input_hashes, dict) else {}
    recorded: dict[str, dict[str, Any]] = {}
    resumed: dict[str, str] = {}
    for key, value in sorted(raw.items()):
        if not isinstance(key, str) or not isinstance(value, str):
            notes.append(f"input hash entry {key!r} is malformed and was not compared")
            continue
        if key.startswith("corpus:"):
            # The corpus file a run read is pinned by the paper manifest,
            # which the artifact check covers; it is not verifier policy.
            continue
        if key.startswith("resumed:"):
            role, _ = _role_of_recorded(key[len("resumed:"):])
            if role:
                resumed[role] = value
            else:
                diffs.append(Difference(key, "unrecognised", _short(value), None, "unknown",
                                        "a recorded input this code does not know how to compare"))
            continue
        role, source = _role_of_recorded(key)
        if role is None:
            diffs.append(Difference(key, "unrecognised", _short(value), None, "unknown",
                                    "a recorded input this code does not know how to compare"))
            continue
        recorded[role] = {"source": source, "sha256": value}
    if not raw:
        notes.append("the run recorded no input hashes")
    for role, value in resumed.items():
        # The report was written by the last invocation, so the latest hash
        # recorded for a role is the one to compare with. The legacy record
        # keeps one `resumed:` value per role, so more than two environments
        # cannot be told apart.
        if role in recorded and recorded[role]["sha256"] != value:
            notes.append(f"{role} changed between the run's invocations; compared with the later version")
            recorded[role] = {**recorded[role], "sha256": value}
    _compare_inputs(recorded, current_in, diffs)
    labels = run.prompt_versions if isinstance(run.prompt_versions, dict) else {}
    for name in sorted(set(labels) | set(current_pr)):
        rec, cur = labels.get(name), (current_pr.get(name) or {}).get("version")
        role = f"prompt/{name}"
        if rec is None:
            diffs.append(Difference(role, "not_recorded", None, cur and f"v{cur}", "generation",
                                    "prompt version not recorded"))
        elif cur is None:
            diffs.append(Difference(role, "missing_now", f"v{rec}", None, "generation", "the prompt no longer exists"))
        elif str(rec) != str(cur):
            diffs.append(Difference(role, "changed", f"v{rec}", f"v{cur}", "generation", "prompt version changed"))
    _compare_code({"version": run.code_version}, current_code, diffs)
    return Comparison(_status(diffs, notes, incomplete), diffs, notes, legacy=True)


def compare_run(db, run, kb=None, env=None) -> Comparison:
    """The run's recorded environment beside this process's. Reads only."""
    current_in = current_inputs(kb, env)
    current_pr = current_prompts()
    current_code = code_identity()
    contexts, problem = load_contexts(db, run.id)
    if contexts:
        return compare_contexts(
            contexts, current_in=current_in, current_pr=current_pr, current_code=current_code,
            current_verifier=verifier_fingerprint(), current_schemas=schema_fingerprints(),
        )
    comparison = compare_legacy(run, current_in=current_in, current_pr=current_pr, current_code=current_code)
    if problem:
        comparison.notes.insert(0, problem + "; compared from the run's legacy fields instead")
        comparison.legacy = False
    return comparison
