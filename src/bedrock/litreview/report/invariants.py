"""The integrity check — what code can prove about a finished report.

Every other check in this repository runs while a report is being built, one
finding or one sentence at a time. This one runs over the finished report and
the exact papers it was built from, and asks whether the guarantees hold for
the whole thing:

  quote       every quote shown is the paper's text at its recorded position,
              and so is every context sentence, every qualifying next
              sentence and every sentence shown for naming a target or agent
  fields      every field the model wrote that the report shows was found in
              the paper by `extract/provenance.py`, and is still there; and
              no finding whose conclusion names people its paper's Methods
              never studied is labelled human
  sentences   every kept sentence passes the citation guard, re-run here
              against the paper text it cites. Pure code: a sentence the
              optional entailment check (`report/entailment.py`) dropped is
              stored as not kept, so it is not re-checked and no model is
              needed to re-prove a report
  flags       no paper lost a review or retraction flag in a merge, and no
              finding comes from a paper flagged either way
  pinned      (for a stored run) the paper bodies still hash to the run's
              manifest, and the stored report re-serialises to its own bytes

`assess_run` returns the same checks split into separate answers, and adds
a comparison of the environment the run recorded with the one checking it now
(`report/verification_context.py`) - so "passes today's rules" is never read
as "passes the rules it was written under". `verify_run` keeps its list.

None of these can fail if the code is right. So a violation is not a finding
about the literature, it is a bug, and it is loud: the runner writes it on the
report (`Report.integrity`), the rendered report opens with a warning banner,
and `litreview ask` and `litreview verify-run` exit non-zero.

What this does not prove: that a reworded sentence means what its quote
means. No string check can. It proves that everything a reader is shown as
the paper's words is the paper's words, and that nothing a sentence states as
a number, a name or a subject is absent from the text it cites.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bedrock.litreview.extract.provenance import n_in, name_in, species_in, text_supported
from bedrock.litreview.extract.verify import normalize
from bedrock.litreview.knowledge import Knowledge
from bedrock.litreview.models import EvidenceUnit, Paper, QuestionSpec, Report
from bedrock.litreview.report.guard import CitationGuard

if TYPE_CHECKING:
    from bedrock.litreview.report.verification_context import Comparison

# Model-written fields a report can show, and what each is checked against.
_NAME_FIELDS = (("intervention", "intervention"), ("combo_with", "combo_with"), ("disease_raw", "disease_raw"))


def _anchored(text: str | None, paper: Paper, section: str | None, start: int | None, end: int | None) -> str | None:
    """Why a stored span is not the paper's text at its offsets, or None."""
    if not text:
        return None
    if start is None or end is None or section is None:
        return "has no recorded position"
    source = paper.sections.get(section)
    if source is None:
        return f"cites section {section!r}, which the paper does not have"
    if source[start:end] != text:
        return f"is not the paper's text at {section}[{start}:{end}]"
    return None


def check_units(
    units: list[EvidenceUnit], papers: dict[str, Paper], kb: Knowledge, added_back: frozenset[str] = frozenset()
) -> list[str]:
    out: list[str] = []
    for u in units:
        paper = papers.get(u.paper_id)
        where = f"finding {u.id} (paper {u.paper_id})"
        if paper is None:
            out.append(f"quote: {where}: its paper is not among the run's papers")
            continue
        if (why := _anchored(u.quote, paper, u.section, u.quote_start, u.quote_end)):
            out.append(f"quote: {where}: quote {why}")
        if (why := _anchored(u.context_quote, paper, u.context_section, u.context_start, u.context_end)):
            out.append(f"quote: {where}: context sentence {why}")
        if (why := _anchored(u.qualifier_quote, paper, u.section, u.qualifier_start, u.qualifier_end)):
            out.append(f"quote: {where}: qualifying sentence {why}")
        if (why := _anchored(u.naming_quote, paper, u.naming_section, u.naming_start, u.naming_end)):
            out.append(f"quote: {where}: naming sentence {why}")
        # A paper a scientist added back after a recorded reinstatement is the
        # one retracted paper that may carry findings (`added_back`).
        if paper.is_review or (paper.signals.is_retracted and paper.id not in added_back):
            out.append(f"flags: {where}: comes from a paper flagged "
                       f"{'retracted' if paper.signals.is_retracted else 'review'}")

        if not u.provenance:
            out.append(f"fields: {where}: was never traced")
            continue
        seen = f"{u.quote} {u.context_quote or ''}"
        # Where provenance says each name is. 'chunk' and 'methods' are what
        # units stored before names had to be shown say; newer units name
        # the displayed sentence instead, which must then exist.
        homes = {"quote": u.quote, "chunk": paper.sections.get(u.section, ""),
                 "methods": paper.sections.get("methods", ""), "context": u.context_quote or "",
                 "qualifier": u.qualifier_quote or "", "naming": u.naming_quote or ""}
        if u.provenance.get("target", "none") == "none":
            out.append(f"fields: {where}: target {u.target_raw!r} was not found in the paper")
        elif not name_in(u.target_raw, homes.get(u.provenance["target"], ""), kb, "target"):
            out.append(f"fields: {where}: target {u.target_raw!r} is not in its {u.provenance['target']}")
        for attr, key in _NAME_FIELDS:
            value = getattr(u, attr)
            if not value:
                continue
            place = u.provenance.get(key, "none")
            if place == "none" or not name_in(value, homes.get(place, ""), kb):
                out.append(f"fields: {where}: {attr} {value!r} is shown but not in the paper's {place}")
        if u.species_raw and (u.provenance.get("species", "none") == "none" or not species_in(u.species_raw, seen)):
            out.append(f"fields: {where}: subjects {u.species_raw!r} are shown but not in the quote or context")
        if u.n is not None and (u.provenance.get("n", "none") == "none" or not n_in(u.n, seen)):
            out.append(f"fields: {where}: sample size {u.n} is shown but not in the quote or context")
        if u.effect_detail and u.provenance.get("effect_detail", "none") == "none":
            out.append(f"fields: {where}: a result summary is shown that the paper does not support")
        # A conclusion's people the Methods do not back never make a finding
        # human evidence (`extract/provenance.py` `methods_subjects`).
        if u.methods_subjects and (u.species == "human" or u.evidence_type.startswith("human")):
            out.append(f"fields: {where}: is labelled human, but its paper's methods studied only "
                       f"{u.methods_subjects}")
    return out


def check_sentences(report: Report, spec: QuestionSpec, kb: Knowledge, papers: dict[str, Paper]) -> list[str]:
    from bedrock.litreview.report.layout import load_layout
    from bedrock.litreview.stages.s7_write import paper_label  # the labels the writer's guard used

    by_id = {u.id: u for u in report.evidence}
    by_short = {short: by_id[uid] for uid, short in report.short_ids.items() if uid in by_id}
    labels = {pid: paper_label(p) for pid, p in papers.items()}
    vocab = sorted({v for u in report.evidence for v in (u.target_raw, u.intervention or "", u.target) if v})
    # A per-target subsection's id is "spec_id:target_name" (see
    # `write_report`); `findings` is a property of the section in
    # report.yaml, so it is looked up by the part before the colon. Two guards
    # - one per `findings` value - so this re-check applies exactly the rule
    # the writer applied to that same section, never a looser or stricter one.
    findings_by_id = {s.id: s.findings for s in load_layout(spec.question_type)}
    guards = {
        findings: CitationGuard(spec, kb, by_short, paper_labels=labels, vocabulary=vocab, findings=findings)
        for findings in (True, False)
    }
    out = []
    for section_id, s in report.kept_sentences():
        findings = findings_by_id.get(section_id.split(":", 1)[0], True)
        verdict = guards[findings].check(s.text, s.citations)
        if not verdict.kept:
            out.append(f"sentences: [{section_id}] {s.text[:80]!r}: {verdict.reason}")
    return out


def check_flags(papers: list[Paper]) -> list[str]:
    out = []
    for p in papers:
        types = set(p.signals.publication_types)
        for copy in p.merged_from:
            lost = set(copy.get("publication_types") or []) - types  # type: ignore[arg-type]
            if lost:
                out.append(f"flags: paper {p.id}: lost publication types {sorted(lost)} from copy {copy.get('id')}")
            if copy.get("is_review") and not p.is_review:
                out.append(f"flags: paper {p.id}: copy {copy.get('id')} was a review; the merged paper is not")
            if copy.get("is_retracted") and not p.signals.is_retracted:
                out.append(f"flags: paper {p.id}: copy {copy.get('id')} was retracted; the merged paper is not")
    return out


def check_best_source(report: Report, papers: dict[str, Paper], kb: Knowledge) -> list[str]:
    """The "most relevant source" block quotes the paper's title and abstract
    and gives the ranking model's reason: the quote must be the paper's words,
    and the reason must restate them."""
    paper = papers.get(report.best_source_id or "")
    if paper is None:
        return []
    head = paper.head_text()
    out = []

    def walk(sections):
        for section in sections:
            yield section
            yield from walk(section.subsections)

    for section in walk(report.sections):
        for line in (section.body_markdown or "").splitlines():
            # The background section quotes reviews, not the best source;
            # `check_background` holds those quotes to their own papers.
            if section.id != "background" and line.startswith("> ") and normalize(line[2:]) not in normalize(head):
                out.append(f"quote: [{section.id}] the most relevant source's quote is not in its title or abstract")
            if line.startswith("Why it ranks first: ") and not text_supported(line[20:], head, kb):
                out.append(f"fields: [{section.id}] the ranking reason shown is not supported by the abstract")
    return out


def check_background(report: Report, papers: dict[str, Paper]) -> list[str]:
    """Every quote in the background-from-reviews section is a review's own
    words, and no finding in the report comes from a review."""
    heads = [normalize(p.head_text()) for p in papers.values()]
    out = []
    for section in report.sections:
        if section.id != "background":
            continue
        for line in (section.body_markdown or "").splitlines():
            if line.startswith("> ") and not any(normalize(line[2:]) in h for h in heads):
                out.append(f"quote: [background] {line[2:][:80]!r} is not in any collected paper's title or abstract")
    return out


def check_report(
    report: Report, papers: list[Paper], spec: QuestionSpec, kb: Knowledge, added_back: frozenset[str] = frozenset()
) -> list[str]:
    """Every violation in a report built from `papers`. Empty on a sound report.
    `added_back`: the run's reinstated papers a person added back (from its
    publication-integrity snapshot)."""
    by_id = {p.id: p for p in papers}
    return [*check_units(report.evidence, by_id, kb, added_back), *check_sentences(report, spec, kb, by_id),
            *check_best_source(report, by_id, kb), *check_background(report, by_id), *check_flags(papers)]


def check_pinned(db, run_id: str) -> list[str]:
    """The stored run still matches what it pinned: manifest hash, paper
    bodies, and the report's own bytes."""
    from bedrock.litreview.stages.s8_render import to_json

    out = []
    run = db.get_run(run_id)
    manifest_text = db.get_artifact(run_id, "papers_manifest")
    if manifest_text is None:
        out.append("pinned: the run has no paper manifest (written before runs were pinned)")
    else:
        if run.papers_sha256 and hashlib.sha256(manifest_text.encode("utf-8")).hexdigest() != run.papers_sha256:
            out.append("pinned: the paper manifest does not match the hash recorded on the run")
        for paper_id, sha in json.loads(manifest_text):
            body = db.paper_version_body(paper_id, sha)
            if body is None:
                out.append(f"pinned: paper {paper_id} version {sha[:12]} is missing")
            elif hashlib.sha256(body.encode("utf-8")).hexdigest() != sha:
                out.append(f"pinned: paper {paper_id} body no longer hashes to {sha[:12]}")
    body = db.get_artifact(run_id, "report")
    if body is not None:
        if to_json(Report.model_validate_json(body)) != body:
            out.append("pinned: the stored report does not re-serialise to its own bytes")
        if run.report_sha256 and hashlib.sha256(body.encode("utf-8")).hexdigest() != run.report_sha256:
            out.append("pinned: the stored report does not match the hash recorded on the run")
    return out


def verify_run(db, run_id: str, kb: Knowledge | None = None) -> list[str]:
    """Re-check a stored run from scratch, on its pinned papers."""
    from bedrock.litreview.knowledge import load_knowledge
    from bedrock.litreview.runner import load_result

    kb = kb or load_knowledge()
    result = load_result(db, run_id)
    report = result.report.model_copy(update={"integrity": []})
    return [*check_pinned(db, run_id), *check_report(report, result.papers, result.spec, kb, _added_back(db, run_id))]


def _added_back(db, run_id: str) -> frozenset[str]:
    from bedrock.litreview.publication_integrity import SNAPSHOT_ARTIFACT, added_back_in

    body = db.get_artifact(run_id, SNAPSHOT_ARTIFACT)
    return frozenset(added_back_in(json.loads(body))) if body else frozenset()


# --- the richer answer: what passed, under which rules -------------------------------

_LEGACY_MANIFEST = "pinned: the run has no paper manifest"


@dataclass
class RunAssessment:
    """`verify_run`, split into the three questions it mixes together.

      artifact_integrity      are the stored bytes the bytes the run pinned?
                              pass | fail | incomplete (nothing was pinned)
      current_rule_check      does the report pass today's checks?
                              pass | fail | unavailable (could not be run)
      environment_comparison  are today's checks and inputs the ones the run
                              recorded? unchanged | changed | incomplete

    The first two are what `verify_run` has always returned, as violations;
    `violations` is still exactly that list. The third is a disclosure, not a
    violation: a report that passes today's rules under a changed environment
    is intact, and saying it is corrupt would be false. Publication-integrity
    advisories, when there are any, are local lookups only and never a
    network refresh.
    """

    run_id: str
    artifact_integrity: str
    artifact_errors: list[str]
    current_rule_check: str
    rule_violations: list[str]
    environment: "Comparison"
    advisories: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[str]:
        return [*self.artifact_errors, *self.rule_violations]

    def exit_code(self, *, require_same_rules: bool = False) -> int:
        if self.violations or self.current_rule_check == "unavailable":
            return 1
        if require_same_rules and self.environment.status != "unchanged":
            return 1
        return 0

    def summary(self) -> str:
        """One plain sentence per axis, e.g. "Pinned artifacts intact; current-rule
        checks pass; environment changed in knowledge/meaning_words.yaml; ..."."""
        parts = [{
            "pass": "pinned artifacts intact",
            "fail": "pinned artifacts FAILED",
            "incomplete": "nothing pinned to check (legacy run)",
        }[self.artifact_integrity]]
        parts.append({
            "pass": "current-rule checks pass",
            "fail": "current-rule checks FAILED",
            "unavailable": "current-rule checks could not run",
        }[self.current_rule_check])
        env = self.environment
        changed = env.changed_roles
        if env.status == "unchanged":
            parts.append("environment unchanged since the run")
        elif changed:
            parts.append("environment changed in " + ", ".join(changed))
        else:
            parts.append("environment comparison incomplete")
        if env.legacy:
            parts.append("prompt content not recorded in this legacy run")
        parts.append("historical-rule replay not performed")
        text = "; ".join(parts)
        return text[0].upper() + text[1:] + "."


def assess_run(db, run_id: str, kb: Knowledge | None = None, *, env=None, now=None) -> RunAssessment:
    """Re-check a stored run and say, separately, whether its bytes are intact,
    whether it passes today's rules, and whether today's rules are the ones it
    recorded. Local reads only: no model, no network, no writes."""
    from bedrock.litreview.knowledge import load_knowledge
    from bedrock.litreview.report.verification_context import compare_run
    from bedrock.litreview.runner import load_result

    run = db.get_run(run_id)
    if run is None:
        raise ValueError(f"there is no run {run_id!r}")
    kb_used = kb or load_knowledge()
    try:
        pinned = check_pinned(db, run_id)
    except Exception as exc:  # noqa: BLE001 - an unreadable manifest is a failed check, not a crash
        pinned = [f"pinned: cannot be checked: {type(exc).__name__}: {exc}"]
    if not pinned:
        integrity = "pass"
    elif all(v.startswith(_LEGACY_MANIFEST) for v in pinned):
        integrity = "incomplete"
    else:
        integrity = "fail"
    result = None
    try:
        result = load_result(db, run_id)
        report = result.report.model_copy(update={"integrity": []})
        rules = check_report(report, result.papers, result.spec, kb_used, _added_back(db, run_id))
        rule_state = "fail" if rules else "pass"
    except Exception as exc:  # noqa: BLE001 - a missing pinned paper is exactly what this reports
        rules = [f"cannot be verified: {type(exc).__name__}: {exc}"]
        rule_state = "unavailable"
    # The comparison is about the knowledge actually used for the checks
    # above: a caller's own table, or the loaded default.
    environment = compare_run(db, run, kb=kb_used, env=env)
    advisories: list[str] = []
    if result is not None:
        advisories = _publication_advisories(db, run_id, result, now=now)
    return RunAssessment(run_id, integrity, pinned, rule_state, rules, environment, advisories)


def _publication_advisories(db, run_id: str, result, *, now=None) -> list[str]:
    """Notices recorded on this machine about the papers the report cites, beside
    what the run knew. Local reads only; never a refresh."""
    from datetime import datetime, timedelta, timezone

    from bedrock.litreview.config import load_settings
    from bedrock.litreview.publication_integrity import SNAPSHOT_ARTIFACT, advisories_for_run

    body = db.get_artifact(run_id, SNAPSHOT_ARTIFACT)
    snapshot = json.loads(body) if body else None
    cited = {ref.paper_id for ref in result.report.references}
    papers = [p for p in result.papers if p.id in cited]
    try:
        days = int(load_settings(check_endpoint=False, require_models=False).knob("integrity.stale_after_days", 7))
    except Exception:  # noqa: BLE001 - a broken config must not stop a verification
        days = 7
    return advisories_for_run(db, papers, snapshot, now=now or datetime.now(timezone.utc), ttl=timedelta(days=days))
