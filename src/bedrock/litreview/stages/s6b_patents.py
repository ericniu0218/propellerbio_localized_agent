"""Stage 6b - the patent landscape's raw material. Competitor landscape only,
never evidence (`sources/patents_base.py`).

It runs straight after Stage 6 because that is when the question's *targets* are
known, and a landscape is about targets. The first version searched the disease
alone, right after the question was read, and a measured run showed what that
buys: "psoriasis" has 10,787 patent hits and only 5 of the first 100 mention
IL-23, whereas "IL-23 AND psoriasis" has 26 and all 26 do. So the search asks
about each leading target together with the question's other entities.

Which targets is decided in code, from the evidence the run has already read, in
the order Stage 6 ranked them, and only by names the knowledge table lists (or a
name that plainly is a target). A model never adds a search term - the rule that
protects the paper search protects this one.

What a run found is saved as the `patents.v1` artifact so that a resumed run
reuses it (rather than spending EPO's quota again, and getting a different list)
and a stored run can be re-checked later. It is an artifact and not a field of
`Report`, because a new defaulted field there changes the bytes of every stored
report (`report/invariants.py` `check_pinned`).
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from bedrock.litreview.knowledge import Knowledge, looks_like_target
from bedrock.litreview.models import EvidenceSummary, PatentRecord, QuestionSpec
from bedrock.litreview.sources.http import SourceError
from bedrock.litreview.sources.patents_base import PatentSearcher, PatentSearchRequest

log = logging.getLogger(__name__)

ARTIFACT = "patents.v1"
DEFAULT_MAX_TARGETS = 5


def candidate_targets(summary: EvidenceSummary, kb: Knowledge, limit: int = DEFAULT_MAX_TARGETS
                      ) -> dict[str, list[str]]:
    """Canonical target name -> its exact terms, for the leading targets.

    Stage 6 already put the targets in order of the evidence behind them, so the
    first `limit` are the ones worth a patent search. A name the knowledge table
    lists brings its listed terms; one it does not brings only itself, and only
    if it reads as a target name at all (not "weight loss" or "HaCaT cells").
    """
    out: dict[str, list[str]] = {}
    for target in summary.targets:
        if len(out) >= limit:
            break
        name = target.target
        entity = kb.get(name)
        if entity is not None:
            out[name] = list(entity.terms)
        elif looks_like_target(name):
            out[name] = [name]
    return out


def landscape_request(spec: QuestionSpec, targets: dict[str, list[str]], *, max_results: int,
                      per_target: int) -> PatentSearchRequest:
    return PatentSearchRequest(
        question=spec.text, spec=spec,
        exact_terms={e.canonical: e.exact_terms for e in spec.entities},
        max_results=max_results, targets=targets, per_target=per_target,
    )


def load_patents(db, run_id: str) -> list[PatentRecord] | None:
    """The patents a run saved, or None when it saved none (it never searched,
    or its search failed - which is retried, not remembered)."""
    body = db.get_artifact(run_id, ARTIFACT)
    if body is None:
        return None
    return [PatentRecord.model_validate(r) for r in json.loads(body)["records"]]


def save_patents(db, run_id: str, records: list[PatentRecord], *, searched: list[str]) -> None:
    db.put_artifact(run_id, ARTIFACT, json.dumps(
        {"version": 1, "searched_targets": searched,
         "records": [r.model_dump(mode="json", exclude_none=True) for r in records]},
        sort_keys=True, ensure_ascii=False))


def gather_patents(
    *, db, run_id: str, spec: QuestionSpec, summary: EvidenceSummary, kb: Knowledge,
    given: list[PatentRecord] | None, searcher: PatentSearcher | None,
    knob: Callable[[str, object], object], say: Callable[[str], None],
) -> tuple[list[PatentRecord], bool]:
    """-> (patents, whether a search was asked for and failed).

    In order of preference: records handed in (a `--patents` file: the air-gapped
    way), records this run saved before it stopped, then a search. A failed
    search costs the landscape and nothing else - failing the run would throw
    away the evidence work, which is the part that answers the question, the
    same call `sources/integrity.py` makes about a retraction check that could
    not be run.
    """
    if given is not None:
        save_patents(db, run_id, given, searched=[])
        return given, False
    saved = load_patents(db, run_id)
    if saved is not None:
        say(f"patents: reusing the {len(saved)} this run already found")
        return saved, False
    if searcher is None:
        return [], False

    targets = candidate_targets(summary, kb, int(knob("patents.max_targets", DEFAULT_MAX_TARGETS)))
    if targets:
        say("patents: searching for " + ", ".join(targets) + f" with {spec.subject.canonical if spec.subject else 'the question'}")
    else:
        say("patents: no target to search for yet, so searching the question's own terms (this is broad)")
    try:
        records = searcher.search(landscape_request(
            spec, targets, max_results=int(knob("patents.max_results", 100)),
            per_target=int(knob("patents.per_target", 40))))
    except (SourceError, ValueError) as exc:
        log.warning("run %s: the patent search failed: %s", run_id, exc)
        say(f"patents: WARNING - the patent search failed ({exc}); this report has no competitor landscape")
        return [], True
    save_patents(db, run_id, records, searched=list(targets))
    return records, False
