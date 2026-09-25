"""The integrity check — what code can prove about a finished report.

Every other check in this repository runs while a report is being built, one
finding or one sentence at a time. This one runs over the finished report and
the exact papers it was built from, and asks whether the guarantees hold for
the whole thing:

  quote       every quote shown is the paper's text at its recorded position,
              and so is every context sentence
  fields      every field the model wrote that the report shows was found in
              the paper by `extract/provenance.py`, and is still there
  sentences   every kept sentence passes the citation guard, re-run here
              against the paper text it cites
  flags       no paper lost a review or retraction flag in a merge, and no
              finding comes from a paper flagged either way
  pinned      (for a stored run) the paper bodies still hash to the run's
              manifest, and the stored report re-serialises to its own bytes

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

from bedrock.litreview.extract.provenance import n_in, name_in, species_in, text_supported
from bedrock.litreview.extract.verify import normalize
from bedrock.litreview.knowledge import Knowledge
from bedrock.litreview.models import EvidenceUnit, Paper, QuestionSpec, Report
from bedrock.litreview.report.guard import CitationGuard

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


def check_units(units: list[EvidenceUnit], papers: dict[str, Paper], kb: Knowledge) -> list[str]:
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
        if paper.is_review or paper.signals.is_retracted:
            out.append(f"flags: {where}: comes from a paper flagged "
                       f"{'retracted' if paper.signals.is_retracted else 'review'}")

        if not u.provenance:
            out.append(f"fields: {where}: was never traced")
            continue
        seen = f"{u.quote} {u.context_quote or ''}"
        homes = {"quote": u.quote, "chunk": paper.sections.get(u.section, ""),
                 "methods": paper.sections.get("methods", "")}
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
    return out


def check_sentences(report: Report, spec: QuestionSpec, kb: Knowledge, papers: dict[str, Paper]) -> list[str]:
    from bedrock.litreview.stages.s7_write import paper_label  # the labels the writer's guard used

    by_id = {u.id: u for u in report.evidence}
    by_short = {short: by_id[uid] for uid, short in report.short_ids.items() if uid in by_id}
    labels = {pid: paper_label(p) for pid, p in papers.items()}
    vocab = sorted({v for u in report.evidence for v in (u.target_raw, u.intervention or "", u.target) if v})
    guard = CitationGuard(spec, kb, by_short, paper_labels=labels, vocabulary=vocab)
    out = []
    for section_id, s in report.kept_sentences():
        verdict = guard.check(s.text, s.citations)
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
            if line.startswith("> ") and normalize(line[2:]) not in normalize(head):
                out.append(f"quote: [{section.id}] the most relevant source's quote is not in its title or abstract")
            if line.startswith("Why it ranks first: ") and not text_supported(line[20:], head, kb):
                out.append(f"fields: [{section.id}] the ranking reason shown is not supported by the abstract")
    return out


def check_report(report: Report, papers: list[Paper], spec: QuestionSpec, kb: Knowledge) -> list[str]:
    """Every violation in a report built from `papers`. Empty on a sound report."""
    by_id = {p.id: p for p in papers}
    return [*check_units(report.evidence, by_id, kb), *check_sentences(report, spec, kb, by_id),
            *check_best_source(report, by_id, kb), *check_flags(papers)]


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
    return [*check_pinned(db, run_id), *check_report(report, result.papers, result.spec, kb)]
