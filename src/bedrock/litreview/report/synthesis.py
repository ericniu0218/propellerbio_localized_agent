"""The code-written half of a report a scientist can read: who was studied,
how the studies line up, what the numbers were, and what the letters mean.

The model writes narrow sections from quotes, under the citation guard. That
leaves the parts of a review a scientist needs most - "how strong is this?",
"do the studies agree?", "what was the effect, in whom, compared with what?" -
to either the model's prose or nothing. The first real report (tirzepatide and
liver fat, 28 September 2026) got nothing: p-values without the changes they
tested, abbreviations never expanded, no word on design or sample size, and a
list of 118 excluded papers where a synthesis should have been.

Everything here is computed, never generated:

* **What each study was** comes from the paper's own publication type and
  design wording (`study_design.py`), refined by its title ("post hoc",
  "case report", "retrospective"), and never from a model's label.
* **Who and how many** are the finding's `species_raw` and `n`, which
  `extract/provenance.py` has already found in the paper's text or blanked.
* **Results** are the paper's own sentences, verbatim, with the section they
  came from. No model-written field (`effect_detail`) is shown.
* **Independent studies.** Publications that name the same trial - a trial
  acronym in the title, or the same registration number in the abstract -
  are one study, so a trial and its three post hoc analyses are not four
  confirmations.
* **Evidence level** is a written rule on design and replication, stated in
  the report beside every use of it. It is this system's reading of the
  evidence, labelled as such, and never a judgement of any paper's quality -
  that stays with the scientist (CLAUDE.md: no model scores quality, and
  code's rules are shown).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from bedrock.litreview.knowledge import TermMatcher
from bedrock.litreview.knowledge.abbreviations import defined_as, paper_abbreviations
from bedrock.litreview.models import EvidenceUnit, Paper
from bedrock.litreview.report import claims

if TYPE_CHECKING:  # pragma: no cover
    from bedrock.litreview.stages.s7_write import WriteContext

GLOSSARY = Path(__file__).resolve().parents[1] / "knowledge" / "glossary.yaml"

# --- what each study was -------------------------------------------------------------

HUMAN_TIERS = ("rct", "trial", "observational", "case", "genetic", "human")
TIER_TEXT = {
    "rct": "randomised controlled trial",
    "trial": "clinical trial (randomisation not stated)",
    "observational": "observational study",
    "case": "case report",
    "genetic": "human genetic study",
    "human": "human study (design not stated)",
    "animal": "animal study",
    "cell": "cell or tissue study",
    "unknown": "design not stated",
}
_POST_HOC = re.compile(r"\bpost[- ]hoc\b|\bsecondary analys[ie]s\b|\bsubgroup analys[ie]s\b|\bexploratory analys[ie]s\b",
                       re.I)
_SUBSTUDY = re.compile(r"\bsub-?stud(?:y|ies)\b", re.I)
_CASE = re.compile(r"\bcase reports?\b|\ba (?:rare )?case of\b|\bcase series\b", re.I)
_RETRO = re.compile(r"\bretrospective\b", re.I)
_PROSP = re.compile(r"\bprospective\b", re.I)
_REAL_WORLD = re.compile(r"\breal[- ]world\b", re.I)
_CROSS = re.compile(r"\bcross[- ]sectional\b", re.I)
_RANDOM_TITLE = re.compile(r"\brandomi[sz]ed\b", re.I)


@dataclass
class Study:
    """One paper as a reader needs to see it."""

    paper_id: str
    tier: str
    design: str
    """Human wording: "post hoc analysis of a randomised controlled trial"."""
    trial: str | None = None
    """The trial it belongs to, when the paper names one ("SURPASS-3")."""
    key: str = ""
    """Which underlying study it counts as: the trial, or the paper itself."""
    full_text: bool = False
    subjects: str = ""
    """Who was studied and how many, from its verified findings."""
    access: str = "abstract only"
    """What of the paper this review actually read, from the plan and the
    paper record together: "full text read", "abstract read (full text was
    acquired but not read)", or "abstract only" when nothing more was
    available. The same words everywhere the report names it."""


def _tier(paper: Paper, record: dict[str, Any] | None, units: list[EvidenceUnit]) -> str:
    from bedrock.litreview.study_design import grounded_labels

    head = f"{paper.title}\n{paper.abstract or ''}"
    types = " ".join(paper.signals.publication_types).lower()
    if "case report" in types or _CASE.search(paper.title):
        return "case"
    labels = grounded_labels(record)
    if "human_rct" in labels or ("randomized controlled trial" in types):
        return "rct"
    if "human_interventional_other" in labels:
        return "trial"
    if "human_genetic" in labels:
        return "genetic"
    if "human_observational" in labels:
        return "observational"
    # Who the findings were about decides before the findings' own labels: a
    # mouse study whose extraction filed one finding as "human_clinical" (a
    # conclusion naming patients) is still a mouse study.
    species = {u.species for u in units}
    if species & {"mouse", "rat", "cyno"} and "human" not in species:
        return "animal"
    if "cell_line" in species and "human" not in species:
        return "cell"
    kinds = {u.evidence_type for u in units}
    if "human_clinical" in kinds or "human_genetic" in kinds:
        if _RANDOM_TITLE.search(paper.title):
            return "rct"
        if _REAL_WORLD.search(head) or _RETRO.search(head) or _PROSP.search(head) or _CROSS.search(head):
            return "observational"
        return "genetic" if kinds == {"human_genetic"} else "human"
    if "animal_interventional" in labels or "in_vivo" in kinds:
        return "animal"
    if "in_vitro" in labels or "in_vitro" in kinds:
        return "cell"
    return "unknown"


def _design_text(tier: str, paper: Paper) -> str:
    base = TIER_TEXT[tier]
    head = f"{paper.title}\n{paper.abstract or ''}"
    if tier in ("rct", "trial") and _POST_HOC.search(paper.title):
        return f"post hoc or secondary analysis of a {base}"
    if tier in ("rct", "trial") and _SUBSTUDY.search(paper.title):
        return f"substudy of a {base}"
    if tier == "observational":
        words = [w for w, rx in (("retrospective", _RETRO), ("prospective", _PROSP), ("cross-sectional", _CROSS))
                 if rx.search(head)]
        lead = " ".join(words[:1])
        rw = "real-world " if _REAL_WORLD.search(head) else ""
        return f"{lead + ' ' if lead else ''}{rw}observational study".strip()
    return base


# --- which publications are one study ---------------------------------------------------

_ACRONYM = re.compile(r"\b([A-Z]{4,}(?:-[A-Z0-9]+)+)\b")
_NAMED_TRIAL = re.compile(r"\b(?:the\s+)?([A-Z]{4,}(?:-[A-Z0-9]+)*)\s+(?:trial|substudy|sub-study|study)\b")
_NCT = re.compile(r"\bNCT\d{8}\b")
_NOT_TRIALS = frozenset({"HOMA-IR", "COVID-19", "SARS-COV-2", "MRI-PDFF", "GLP-1", "GLP-1RA", "LDL-C", "HDL-C",
                         "FIB-4", "FGF-21", "SGLT-2", "MASLD", "MASH", "NAFLD", "NASH", "MAFLD", "HFPEF", "HFREF"})


def trial_keys(paper: Paper) -> set[str]:
    """Names of the trial a paper reports, from its title, and trial
    registration numbers from its abstract. Background trials an abstract
    mentions in passing are not read: only the title names the paper's own."""
    keys = set(_NCT.findall(paper.abstract or ""))
    for rx in (_ACRONYM, _NAMED_TRIAL):
        for name in rx.findall(paper.title):
            if name.upper() in _NOT_TRIALS or not re.search(r"[A-Z]{4}", name):
                continue
            keys.add(name.upper())
    return keys


def group_studies(paper_ids: list[str], papers: dict[str, Paper]) -> dict[str, tuple[str, str | None]]:
    """Paper id -> (study key, trial name or None). Papers sharing any trial
    name or registration number are one study (union-find over shared keys)."""
    parent = {pid: pid for pid in paper_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    names: dict[str, set[str]] = defaultdict(set)
    for pid in paper_ids:
        for key in sorted(trial_keys(papers[pid])):
            names[pid].add(key)
            if key in owner:
                a, b = find(owner[key]), find(pid)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[key] = pid
    out = {}
    for pid in paper_ids:
        root = find(pid)
        members = [p for p in paper_ids if find(p) == root]
        trial_names = sorted({n for p in members for n in names[p] if not n.startswith("NCT")})
        registrations = sorted({n for p in members for n in names[p] if n.startswith("NCT")})
        # Linked by a registration number alone, the number is the name shown.
        shared = trial_names[0] if trial_names else (registrations[0] if len(members) > 1 and registrations else None)
        out[pid] = (f"study:{root}", shared)
    return out


def _subjects(units: list[EvidenceUnit]) -> str:
    """Who was studied, in the paper's words, with the largest stated n - and
    "per group" when the paper's sentence says that is what it counts."""
    words = sorted({u.species_raw.strip() for u in units
                    if u.species_raw.strip() and u.provenance.get("species", "none") != "none"}, key=len)
    with_n = [u for u in units if u.n and u.provenance.get("n", "none") != "none"]
    who = words[0] if words else ""
    if with_n:
        top = max(with_n, key=lambda u: u.n)
        scope = claims.n_scope(top)
        n = f"n={top.n}" + (f" {scope}" if scope else "")
        return f"{who} ({n})".strip() if who else n
    return who


def profiles(ctx: "WriteContext") -> dict[str, Study]:
    """Every paper that gave at least one verified finding, profiled."""
    by_paper: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for u in ctx.summary.units:
        by_paper[u.paper_id].append(u)
    pids = sorted(by_paper, key=lambda p: (ctx.rank_of.get(p, 10**6), p))
    groups = group_studies(pids, ctx.papers)
    abstract_only = {j.paper_id for j in ctx.plan.jobs if j.abstract_only}
    out = {}
    for pid in pids:
        paper = ctx.papers[pid]
        tier = _tier(paper, ctx.designs.get(pid), by_paper[pid])
        key, trial = groups[pid]
        read_full = paper.has_full_text and pid not in abstract_only
        access = ("full text read" if read_full else
                  "abstract read (full text was acquired but not read)" if paper.has_full_text else "abstract only")
        out[pid] = Study(paper_id=pid, tier=tier, design=_design_text(tier, paper), trial=trial, key=key,
                         full_text=read_full, subjects=_subjects(by_paper[pid]), access=access)
    return out


def _refs(pids, refs: dict[str, int]) -> str:
    nums = sorted({refs[p] for p in pids if p in refs})
    return "".join(f"[{n}]" for n in nums)


def study_label(study: Study, refs: dict[str, int]) -> str:
    """"[3] randomised controlled trial (SURPASS-3), people with T2D (n=296), full text"."""
    bits = [study.design + (f" ({study.trial})" if study.trial else "")]
    if study.subjects:
        bits.append(study.subjects)
    bits.append(study.access)
    return f"{_refs([study.paper_id], refs)} " + ", ".join(bits)


# --- 1. evidence at a glance ------------------------------------------------------------


def glance(ctx: "WriteContext", refs: dict[str, int], studies: dict[str, Study]) -> str:
    if not studies:
        if ctx.client is None:
            return ("No findings were extracted: this report was assembled without a model, so the papers "
                    "below were ranked and planned but not read. Run the review with a model to get findings.")
        return "No paper that was read gave a verified finding for this question."
    independent = {s.key for s in studies.values()}
    tiers = defaultdict(set)
    for s in studies.values():
        tiers[s.tier].add(s.key)
    mix = ", ".join(f"{len(keys)} {_plural(TIER_TEXT[t], len(keys))}"
                    for t, keys in sorted(tiers.items(), key=lambda kv: HUMAN_TIERS.index(kv[0])
                                          if kv[0] in HUMAN_TIERS else 99))
    full = sum(1 for s in studies.values() if s.full_text)
    lines = [
        f"This report rests on **{len(studies)} publication{'' if len(studies) == 1 else 's'}** that gave "
        f"verified findings, from **{len(independent)} independent stud{'y' if len(independent) == 1 else 'ies'}** "
        f"({mix}). {full} read in full text; {len(studies) - full} from the abstract alone"
        + (", so their methods, tables and full results were not seen." if len(studies) - full else "."),
        "",
        "| Ref | What it found (counted for the treatment) | What the study was | Who was studied | Read from |",
        "|---|---|---|---|---|",
    ]
    status = getattr(ctx, "status", {})
    found: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    set_aside: dict[str, int] = defaultdict(int)
    for u in ctx.summary.units:
        a = status.get(u.id)
        if a is not None and a.status != claims.COUNTED:
            set_aside[u.paper_id] += 1
            continue
        cls = claims.endpoint_of(u)
        measure = cls.label.split(" (")[0] if cls else (u.target or "other")
        found[u.paper_id][f"{measure}: {DIRECTION.get(u.outcome, u.outcome)}"].append(ctx.summary.short_ids[u.id])
    for pid, s in studies.items():
        title = ctx.papers[pid].title.rstrip(".")
        design = s.design + (f" — {s.trial}" if s.trial else "")
        results = [f"{what} ({', '.join(ids[:3])}{'…' if len(ids) > 3 else ''})"
                   for what, ids in list(found[pid].items())[:3]]
        if len(found[pid]) > 3:
            results.append(f"{len(found[pid]) - 3} more")
        if set_aside[pid]:
            results.append(f"{set_aside[pid]} quote(s) not counted - see the evidence table")
        lines.append(f"| {_refs([pid], refs)} | {_cell('; '.join(results) or 'nothing counted')} | "
                     f"{_cell(design)}: *{_cell(title[:90])}* | {_cell(s.subjects) or 'not stated'} | {s.access} |")
    shared = defaultdict(list)
    for s in studies.values():
        if s.trial:
            shared[s.key].append(s)
    repeats = [group for group in shared.values() if len(group) > 1]
    if repeats:
        lines.append("")
        for group in repeats:
            lines.append(f"- {_refs([s.paper_id for s in group], refs)} report the same trial "
                         f"(**{group[0].trial}**): they are counted as one study below, not as independent "
                         "confirmations.")
    return "\n".join(lines)


# --- 2. how the studies line up ------------------------------------------------------------

DIRECTION = {
    "improved": "improved", "no_effect": "no significant effect", "worsened": "worsened",
    "mixed": "mixed results", "association": "an association only (not a tested effect)",
}


def _measure(u: EvidenceUnit, ctx: "WriteContext", subject: TermMatcher | None, abbreviations) -> str:
    """What a finding measured, as one label: the question's subject when the
    finding's own target names it (directly or through the paper's own
    definition of an abbreviation), else the normalised target."""
    raw = (u.target_raw or u.target or "").strip()
    long_form = defined_as(raw, abbreviations(u.paper_id)) or ""
    if subject is not None and (subject.exact_count(raw) or subject.exact_count(long_form)
                                or subject.exact_count(u.target or "")):
        return "__subject__"
    drugs = [e for e in ctx.spec.entities if e.kind == "drug"]
    if any(TermMatcher(e.exact_terms).exact_count(raw) for e in drugs):
        return "__several__"
    if long_form:
        # Grouped under the letters, which every paper spells the same, and
        # labelled with the first long form seen ("HOMA-IR" is written out
        # three different ways in three papers).
        return f"{long_form} ({raw})"
    return (u.target or raw or "other").strip()


_BASELINE = re.compile(r"\bbaseline\b", re.I)
_OVER_TIME = re.compile(
    r"\b(?:from|since|versus|vs\.?|compared\s+(?:with|to)|relative\s+to)\s+baseline\b|\bbaseline\s+to\b|"
    r"\bchange[sd]?\b|\bafter\b|\bweeks?\b|\bmonths?\b|\bat\s+week\b|\bfollow[- ]up\b|\bdecreas|\bincreas|"
    r"\breduc|\bimprov|\bfell\b|\brose\b", re.I)
_DEMOGRAPHIC = re.compile(
    r"\bmean\s+age\b|\bmedian\s+age\b|\b(?:were|was)\s+(?:women|men|female|male)\b|\bstudy\s+population\b|"
    r"\b(?:was|were)\s+taken\s+by\b|\bmost\s+common\s+[\w-]+\s+(?:agent|medication|drug|therapy)\b|"
    r"\b(?:were|was)\s+(?:enrolled|included|randomi[sz]ed\s+to|recruited)\b", re.I)
_AIM = re.compile(r"^\s*(?:this|the\s+present|the\s+current|our)\s+(?:study|analysis|trial|work|review)\s+"
                  r"(?:investigated|examined|assessed|evaluated|explored|aimed|compared|describes|reports)\b", re.I)


def substantive(u: EvidenceUnit) -> bool:
    """Does this finding's quote state a result a scientist can use?

    Extraction keeps some verified quotes that are true and the paper's own,
    but are not results: "A total of 276 participants were included", a
    table of baseline characteristics, "This study investigated ...". The
    citation guard already refuses to let a findings sentence rest on one
    (report/guard.py check 6); here they are also kept out of the synthesis,
    the key results and the front of every section's evidence, so they do not
    crowd out the result beside them in the same paper. A quote cut off
    mid-bracket is left out of display for the same reason."""
    from bedrock.litreview.extract.verify import reports_result

    quote = u.quote
    if not reports_result(quote) or _AIM.search(quote) or _DEMOGRAPHIC.search(quote):
        return False
    if _BASELINE.search(quote) and not _OVER_TIME.search(quote):
        return False  # a description of the groups at the start, not what happened to them
    return quote.count("(") <= quote.count(")") and quote.count("[") <= quote.count("]")


def subject_test(ctx: "WriteContext"):
    """A test: does this finding measure the question's own subject? (The same
    judgement `outcome_groups` files it under; the paper's abbreviations are
    read once per paper.)"""
    entity = ctx.spec.subject
    if entity is None:
        return lambda u: False
    matcher = TermMatcher(entity.exact_terms)
    cache: dict[str, dict[str, str]] = {}

    def abbreviations(pid: str) -> dict[str, str]:
        if pid not in cache:
            cache[pid] = paper_abbreviations(ctx.papers[pid]) if pid in ctx.papers else {}
        return cache[pid]

    def about(u):
        cls = claims.endpoint_of(u)
        if cls is not None:
            return cls.subject == entity.canonical
        return _measure(u, ctx, matcher, abbreviations) == "__subject__"

    return about


def section_preference(ctx: "WriteContext"):
    """The order a section's evidence is offered in: a result the requested
    intervention is responsible for, about the question's subject, then any
    other result, then the rest."""
    about = subject_test(ctx)
    return lambda u: (not substantive(u),
                      not claims.attributable(u, ctx.spec, ctx.papers.get(u.paper_id)), not about(u))


def _tested_studies(directions: dict[str, set[str]]) -> set[str]:
    studies = set().union(*directions.values()) if directions else set()
    return studies - (directions.get("association", set()) - (directions.get("improved", set())
                      | directions.get("no_effect", set()) | directions.get("worsened", set())))


def _level(directions: dict[str, set[str]], tiers: dict[str, str], abstract_only: bool) -> tuple[str, str]:
    """(basis, why) by the written rule in `LEVEL_RULE`: what kind of evidence
    there is, stated as a fact - never a verdict on how good it is.

    Disagreement is judged within human studies and within animal or cell
    studies, never across them: a mouse model that differs from a human trial
    is a difference between systems, not a conflict in the human evidence. A
    study counts against only when it reports no benefit at all for this
    measure (two subgroups of one trial going different ways is that trial's
    mixed result)."""
    tested = _tested_studies(directions)
    human = {k for k in tested if tiers.get(k) in HUMAN_TIERS}
    pre = {k for k in tested if tiers.get(k) in ("animal", "cell")}
    rct = {k for k in human if tiers.get(k) == "rct"}
    case = {k for k in human if tiers.get(k) == "case"}
    better = directions.get("improved", set())
    other = (directions.get("no_effect", set()) | directions.get("worsened", set())) - better
    if (better & human and other & human) or (not human and better & pre and other & pre):
        basis, why = "Studies disagree", "independent studies of the same kind report different directions"
    elif rct:
        basis = f"Randomised comparison{'s' if len(rct) > 1 else ''}: {len(rct)} trial{'s' if len(rct) > 1 else ''}"
        why = (f"plus {len(human) - len(rct)} non-randomised human stud{'y' if len(human) - len(rct) == 1 else 'ies'}"
               if len(human) > len(rct) else "no other human study")
    elif human and human == case:
        basis, why = "Case reports only", f"{len(case)} case report{'s' if len(case) > 1 else ''}"
    elif human:
        basis = "Non-randomised human studies only"
        why = f"{len(human)} stud{'y' if len(human) == 1 else 'ies'}, no randomised comparison"
    elif pre:
        linked = {k for k in directions.get("association", set()) if tiers.get(k) in HUMAN_TIERS}
        basis, why = "Animal or cell studies only", (
            f"no tested effect in humans; {len(linked)} human stud{'y' if len(linked) == 1 else 'ies'} "
            f"report{'s' if len(linked) == 1 else ''} associations only" if linked
            else "no eligible human finding was extracted from the papers read")
    else:
        basis, why = "Associations only", "no tested effect of an intervention"
    if pre and human:
        why += f"; {len(pre)} animal or cell stud{'y' if len(pre) == 1 else 'ies'} as supporting context"
    if abstract_only:
        why += "; every study behind it was read from the abstract alone"
    return basis, why


LEVEL_RULE = (
    "**Basis** says what kind of evidence stands behind each line, by this report's written rule - not the papers' "
    "own claim, not a judgement of any study's quality, and not a formal certainty rating (such as GRADE): "
    "*Randomised comparisons* - at least one randomised trial compared the treatment with something else; "
    "*Non-randomised human studies only*; *Case reports only*; *Animal or cell studies only*; *Studies disagree* - "
    "independent studies of the same kind (human, or animal and cell) report different directions; *Associations "
    "only*. Publications from the same trial count once. A finding the requested treatment is not responsible for "
    "(the disease model it was tested in, another drug), or that compares subgroups or arms that all received it, is "
    "listed below its line and not counted."
)


@dataclass
class Group:
    label: str
    units: list[EvidenceUnit] = field(default_factory=list)


def outcome_groups(ctx: "WriteContext", studies: dict[str, Study]) -> list[Group]:
    """Findings grouped by what they measured, the question's subject first,
    then by how many independent studies measured it."""
    subject_entity = ctx.spec.subject
    subject = TermMatcher(subject_entity.exact_terms) if subject_entity else None
    cache: dict[str, dict[str, str]] = {}

    def abbreviations(pid: str) -> dict[str, str]:
        if pid not in cache:
            cache[pid] = paper_abbreviations(ctx.papers[pid])
        return cache[pid]

    groups: dict[str, Group] = {}
    subject_classes: set[str] = set()
    for u in ctx.summary.units:
        if not substantive(u):
            continue
        cls = None if ctx.spec.question_type == "target_identification" and not u.outcome_measure \
            else claims.endpoint_of(u)
        if cls is not None:
            label, key = cls.label, "endpoint:" + cls.id
            if subject_entity is not None and cls.subject == subject_entity.canonical:
                subject_classes.add(label)
        elif ctx.spec.question_type == "target_identification" and not u.outcome_measure:
            label = u.target
            key = label.lower()
        else:
            label = _measure(u, ctx, subject, abbreviations)
            key = label.rsplit(" (", 1)[1].rstrip(")").lower() if label.endswith(")") and " (" in label else label.lower()
        groups.setdefault(key, Group(label=label)).units.append(u)
    subject_name = _subject_name(ctx)
    for g in groups.values():
        if g.label == "__subject__":
            g.label = subject_name
            subject_classes.add(subject_name)
        elif g.label == "__several__":
            g.label = "Several measures reported together"

    def order(g: Group):
        keys = {studies[u.paper_id].key for u in g.units if u.paper_id in studies}
        return (g.label not in subject_classes, g.label.startswith("Several"), -len(keys), g.label.lower())

    return sorted(groups.values(), key=order)


def _subject_name(ctx: "WriteContext") -> str:
    e = ctx.spec.subject
    if e is None:
        return "the subject"
    return e.text if e.text.lower() != e.canonical.lower() else e.canonical


@dataclass
class SynthesisRow:
    """One measure's comparison, shared by synthesis, disagreements and gaps.

    This is a transient view, never added to stored report or evidence models.
    `units` contains only counted findings; `aside` explains the rest.
    Compute every measure before a renderer applies its display limit.
    """

    label: str
    units: list[EvidenceUnit]
    directions: dict[str, set[str]]
    tiers: dict[str, str]
    papers: dict[str, set[str]]
    aside: dict[str, set[str]]
    within: set[str]
    basis: str | None
    why: str

    endpoint: str | None = None
    """The endpoint class id (`knowledge/endpoints.yaml`) the row measures, if any."""

    @property
    def tested_human(self) -> bool:
        """Human tested effects by the same rule that computes the basis.

        An animal-only disagreement still has no tested human effect, even
        though its basis says 'Studies disagree' rather than 'Animal only'.
        """
        return any(self.tiers.get(k) in HUMAN_TIERS for k in _tested_studies(self.directions))


def synthesis_rows(ctx: "WriteContext", studies: dict[str, Study]) -> list[SynthesisRow]:
    rows = []
    for g in outcome_groups(ctx, studies):
        directions: dict[str, set[str]] = defaultdict(set)
        tiers: dict[str, str] = {}
        papers: dict[str, set[str]] = defaultdict(set)
        aside: dict[str, set[str]] = defaultdict(set)
        within: set[str] = set()
        counted = []
        for u in g.units:
            s = studies.get(u.paper_id)
            if s is None:
                continue
            paper = ctx.papers.get(u.paper_id)
            contrast = claims.contrast_of(u, ctx.spec, paper)
            assessed = getattr(ctx, "status", {}).get(u.id)
            if assessed is not None:
                # The one decision every view reads (claims.assess): a finding
                # set aside here is set aside everywhere, for the same reason.
                if assessed.status != claims.COUNTED:
                    aside[claims.REASONS[assessed.reasons[0]]].add(u.paper_id)
                    continue
                why_not = None
            else:
                why_not = None if contrast in ("subgroup", "co_intervention") else claims.attribution(u, ctx.spec, paper)
            if why_not:
                aside[claims.NOT_ATTRIBUTED[why_not]].add(u.paper_id)
                continue
            if contrast == "subgroup":
                aside["compares subgroups of treated people, not treatment with no treatment"].add(u.paper_id)
                continue
            if contrast == "co_intervention":
                aside["every arm received the requested treatment; the arms differed by something else"].add(u.paper_id)
                continue
            if contrast == "within_arm" and u.outcome in ("improved", "worsened"):
                within.add(u.paper_id)
            directions[u.outcome].add(s.key)
            papers[u.outcome].add(u.paper_id)
            counted.append(u)
            # A trial's strongest design stands for it (the RCT, not its post hoc).
            if tiers.get(s.key) is None or _rank(s.tier) < _rank(tiers[s.key]):
                tiers[s.key] = s.tier
        keys = set().union(*directions.values()) if directions else set()
        all_abstract = all(not studies[u.paper_id].full_text for u in g.units if u.paper_id in studies)
        basis, why = _level(directions, tiers, all_abstract) if keys else (None, "")
        rows.append(SynthesisRow(g.label, counted, dict(directions), tiers, dict(papers), dict(aside), within,
                                 basis, why, endpoint=_endpoint_id(g.label)))
    return rows


def _endpoint_id(label: str) -> str | None:
    return next((c.id for c in claims.endpoint_classes() if c.label == label), None)


_COMPARATIVE = re.compile(r"\b(?:more|less)\s+(?:frequent|common|often)|\b(?:higher|lower)\s+(?:incidence|rate|"
                          r"risk|frequency|proportion)|\bthan\b|\bcompared\s+(?:with|to)\b|\bversus\b|\bvs\.?\b|"
                          r"\b(?:HR|OR|RR)\b", re.I)
_SEVERE = re.compile(r"\bserious\b|\bsevere\b|\bdiscontinu\w*|\bhospitali[sz]\w*|\bdeath|\bgrade\s*[3-5]\b", re.I)


def _adverse_parts(row: SynthesisRow, refs: dict[str, int]) -> list[str]:
    """Adverse events as what the papers report about them - that they
    occurred, any comparison of rates, any serious events - never as a
    treatment "effect". "Gastrointestinal events were the most frequently
    reported" is not "no significant effect"."""
    occurred = {u.paper_id for u in row.units}
    compared = {u.paper_id for u in row.units if _COMPARATIVE.search(u.quote)}
    severe = {u.paper_id for u in row.units if _SEVERE.search(u.quote)}
    parts = [f"reported in {len(occurred)} stud{'y' if len(occurred) == 1 else 'ies'} {_refs(occurred, refs)}"]
    parts.append(f"rates compared between groups in {_refs(compared, refs)}" if compared
                 else "no comparison of rates between groups in the quotes")
    if severe:
        parts.append(f"serious events or stopping treatment described in {_refs(severe, refs)}")
    return parts


def render_synthesis(rows: list[SynthesisRow], refs: dict[str, int], limit: int = 10) -> str:
    """One table row per measure - what was measured, which studies, what they
    found, and what kind of evidence that is - then, per measure, what was not
    counted and why. A reader compares measures down the table; the reasons a
    finding was set aside are there to check, not to read first."""
    if not rows:
        return "No verified findings to compare."
    table = ["| Measure | Studies | What they found | Basis |", "|---|---|---|---|"]
    not_counted = []
    for row in rows[:limit]:
        directions, tiers, papers = row.directions, row.tiers, row.papers
        keys = set().union(*directions.values()) if directions else set()
        if row.aside:
            not_counted.append(f"- **{_cell(row.label)}** — "
                               + "; ".join(f"{why} {_refs(pids, refs)}" for why, pids in row.aside.items()) + ".")
        if not keys:
            table.append(f"| **{_cell(row.label)}** | – | nothing here can be counted for the requested treatment "
                         "(see below) | – |")
            continue
        if row.endpoint == "adverse_events":
            parts = _adverse_parts(row, refs)
        else:
            parts = []
            for outcome in ("improved", "no_effect", "worsened", "mixed", "association"):
                if directions.get(outcome):
                    n = len(directions[outcome])
                    parts.append(f"{DIRECTION[outcome]} in {n} stud{'y' if n == 1 else 'ies'} "
                                 f"{_refs(papers[outcome], refs)}")
        if row.within:
            parts.append(f"change from baseline, with no comparison group, in {_refs(row.within, refs)}")
        designs = defaultdict(int)
        for k in keys:
            designs[TIER_TEXT[tiers[k]]] += 1
        mix = ", ".join(f"{v} {_plural(d, v)}" for d, v in sorted(designs.items()))
        table.append(f"| **{_cell(row.label)}** | {len(keys)} independent: {_cell(mix)} | {_cell('; '.join(parts))} | "
                     f"**{_cell(row.basis)}** — {_cell(row.why)} |")
    lines = table
    if len(rows) > limit:
        lines += ["", f"{len(rows) - limit} further measures are in the evidence table (Appendix)."]
    if not_counted:
        lines += ["", "**Not counted, and why** (each finding stays in the evidence table with its status):", ""]
        lines += not_counted
    lines += ["", LEVEL_RULE]
    return "\n".join(lines)


def synthesis(ctx: "WriteContext", refs: dict[str, int], studies: dict[str, Study], limit: int = 10) -> str:
    return render_synthesis(synthesis_rows(ctx, studies), refs, limit)


def _rank(tier: str) -> int:
    order = ("rct", "trial", "genetic", "observational", "human", "case", "animal", "cell", "unknown")
    return order.index(tier) if tier in order else len(order)


# --- where the studies disagree -------------------------------------------------------

DISAGREEMENT_RULE = (
    "The study records below show design, subjects, access and reported directions for each disagreement. "
    "The scientist decides which differences, if any, explain it. Publications from the same trial are shown "
    "together as one independent study."
)


@dataclass
class Disagreement:
    """One set of disagreeing papers and the report lines it explains."""

    explanations: list[str] = field(default_factory=list)
    units: list[EvidenceUnit] = field(default_factory=list)


def disagreement_entries(ctx: "WriteContext", studies: dict[str, Study],
                         rows: list[SynthesisRow]) -> list[Disagreement]:
    """Both disagreement rules, with identical paper sets displayed once.

    Synthesis disagrees within systems; Stage 6 target conflicts also span
    systems. Keep each rule's wording and each supporting finding when the
    same papers explain both. Nothing here infers a cause of disagreement.
    """
    entries: dict[frozenset[str], Disagreement] = {}

    def add(explanation: str, units: list[EvidenceUnit]) -> None:
        units = [u for u in units if u.paper_id in studies]
        if not units:
            return
        key = frozenset(u.paper_id for u in units)
        entry = entries.setdefault(key, Disagreement())
        if explanation not in entry.explanations:
            entry.explanations.append(explanation)
        seen = {u.id for u in entry.units}
        entry.units.extend(u for u in units if u.id not in seen)

    for row in rows:
        if row.basis != "Studies disagree":
            continue
        better = row.directions.get("improved", set())
        other = (row.directions.get("no_effect", set()) | row.directions.get("worsened", set())) - better
        human = {k for k in better | other if row.tiers.get(k) in HUMAN_TIERS}
        kind = HUMAN_TIERS if human else ("animal", "cell")
        keys = {k for k in better | other if row.tiers.get(k) in kind}
        add(f"How the studies line up: **{_cell(row.label)}** (Studies disagree)",
            [u for u in row.units if studies[u.paper_id].key in keys
             and u.outcome in ("improved", "no_effect", "worsened")])
    for conflict in ctx.summary.conflicts:
        ids = set(conflict.unit_ids)
        pids = set(conflict.supporting_papers) | set(conflict.opposing_papers)
        label = "Candidate targets" if ctx.spec.question_type == "target_identification" else "Target conflict"
        add(f"{label}: **{_cell(conflict.target)}** (Papers disagree on this target; {conflict.direction})",
            [u for u in ctx.summary.units if u.id in ids and u.paper_id in pids
             and u.outcome in ("improved", "no_effect", "worsened")])
    return list(entries.values())


def disagreements(ctx: "WriteContext", refs: dict[str, int], studies: dict[str, Study],
                  rows: list[SynthesisRow]) -> str:
    entries = disagreement_entries(ctx, studies, rows)
    if not entries:
        return ""
    lines = [DISAGREEMENT_RULE]
    order = {outcome: i for i, outcome in enumerate(("improved", "no_effect", "worsened"))}
    for entry in entries:
        lines += ["", "; ".join(entry.explanations) + "."]
        independent: dict[str, list[EvidenceUnit]] = defaultdict(list)
        for u in entry.units:
            independent[studies[u.paper_id].key].append(u)
        groups = sorted(independent.values(),
                        key=lambda units: (min(order[u.outcome] for u in units),
                                           min(refs[u.paper_id] for u in units)))
        for units in groups:
            lines += ["", f"- **Independent study {_refs([u.paper_id for u in units], refs)}**"]
            seen: set[tuple[str, str, str, str | None]] = set()
            for u in sorted(units, key=lambda u: (order[u.outcome], refs[u.paper_id], u.id)):
                # Abstract and Results can repeat the same finding. Keep a
                # distinct qualifier or direction, and every cited paper.
                key = (u.paper_id, u.outcome, u.quote, u.qualifier_quote)
                if key in seen:
                    continue
                seen.add(key)
                lines += ["", f"{study_label(studies[u.paper_id], refs)}; direction: **{DIRECTION[u.outcome]}**.",
                          "", "> " + " ".join(u.quote.split())]
                if u.qualifier_quote:
                    lines += ["", "> " + " ".join(u.qualifier_quote.split())]
    return "\n".join(lines)


# --- 3. key results, with their context -------------------------------------------------------

_QUANT = re.compile(r"\d(?:[\d.,]*)\s*(?:%|kg|mg|g/|mmol|µ|mcg|U/L|IU|cm|mm|points?|years?|weeks?|months?|days?|"
                    r"fold|ms|mL|L/|/L|/dL|/mL)|±|\b95%\s*CI\b|\bCI\b|\bOR\b|\bHR\b|\bRR\b|\bp\s*[<=>]", re.I)
_SECTION_ORDER = {"results": 0, "tables": 0, "abstract": 1, "discussion": 2}


def _other_estimates(ctx: "WriteContext", u: EvidenceUnit, refs: dict[str, int]) -> list[str]:
    """Where the same paper gives a different estimate for an outcome this
    quote reports: said beside the result, not only in the Limitations."""
    mine = claims.estimates(u.quote)
    if not mine:
        return []
    out = []
    for other in ctx.summary.units:
        if other.paper_id != u.paper_id or other.id == u.id:
            continue
        for label, value in claims.estimates(other.quote).items():
            if label in mine and mine[label] != value:
                out.append(f"The same paper gives a different estimate for {label} elsewhere ({value}, "
                           f"{ctx.summary.short_ids[other.id]}) - perhaps a different analysis or follow-up; "
                           "not reconciled here.")
    return list(dict.fromkeys(out))


def key_results(ctx: "WriteContext", refs: dict[str, int], studies: dict[str, Study],
                per_group: int = 3, limit: int = 14) -> str:
    """The quantitative findings, as the papers state them, with the study they
    come from beside each. A number without its comparison, time point and
    population is not shown alone: the quote is the paper's whole sentence."""
    rows: list[str] = []
    for g in outcome_groups(ctx, studies):
        chosen: list[EvidenceUnit] = []
        seen_papers: dict[str, int] = defaultdict(int)
        units = sorted(g.units, key=lambda u: (_SECTION_ORDER.get(u.section, 3), -len(_QUANT.findall(u.quote)),
                                               ctx.rank_of.get(u.paper_id, 10**6), u.id))
        for u in units:
            assessed = getattr(ctx, "status", {}).get(u.id)
            if assessed is not None and assessed.status != claims.COUNTED:
                continue
            if not _QUANT.search(u.quote) or not substantive(u) or u.paper_id not in studies \
                    or seen_papers[u.paper_id] >= 2 or claims.statistical_anomalies(u.quote) \
                    or not claims.attributable(u, ctx.spec, ctx.papers.get(u.paper_id)):
                continue
            chosen.append(u)
            seen_papers[u.paper_id] += 1
            if len(chosen) >= per_group:
                break
        for u in chosen:
            s = studies[u.paper_id]
            where = f"{u.section}" + (", from the Discussion" if u.from_discussion else "")
            rows.append(f"- **{_cell(g.label)}** — “{" ".join(u.quote.split())}” — {study_label(s, refs)}; "
                        f"{ctx.summary.short_ids[u.id]}, {where}.")
            if u.qualifier_quote:
                rows.append(f"  - The paper's next sentence qualifies it: “{" ".join(u.qualifier_quote.split())}”")
            for note in _other_estimates(ctx, u, refs):
                rows.append(f"  - {note}")
        if len(rows) >= limit:
            break
    if not rows:
        return "No verified finding states a number."
    return ("Each result is the paper's own sentence, quoted exactly, with the study it comes from. Numbers are "
            "never taken out of the sentence that gives their comparison and time point.\n\n" + "\n".join(rows[:limit]))


# --- 4. abbreviations -------------------------------------------------------------------------


_NOT_ABBREVIATIONS = frozenset({"figure", "fig", "figs", "table", "tables", "supplementary", "appendix", "panel",
                                "results", "methods", "discussion", "conclusion", "conclusions", "background", "data"})
"""Words the abbreviation finder can mistake for a defined short form
("... gamma-glutamyl transpeptidase (Figure 2)")."""

STATISTICAL = frozenset({"CI", "OR", "HR", "RR", "SD", "SEM", "IQR", "ETD", "LS"})
"""Statistical abbreviations: in a report's numbers they always mean the
statistic, so the standard expansion is shown even where a paper defined the
letters as something else."""


def _load_glossary() -> dict[str, str]:
    data = yaml.safe_load(GLOSSARY.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in data.items()}


def glossary(ctx: "WriteContext", text: str, refs: dict[str, int]) -> str:
    """Every abbreviation the report's main text uses, expanded: the paper's
    own definition when a cited paper gives one, else the standard expansion
    from `knowledge/glossary.yaml`."""
    standard = _load_glossary()
    own: dict[str, list[tuple[str, int]]] = defaultdict(list)

    def same_meaning(a: str, b: str) -> bool:
        # Capitalisation, stray punctuation, a plural and a near-identical
        # wording ("homeostatic" / "homeostasis model assessment") are not a
        # different meaning; a different expansion (heart failure / hazard
        # ratio) is.
        from rapidfuzz import fuzz

        norm = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip().rstrip("s")  # noqa: E731
        return norm(a) == norm(b) or fuzz.ratio(norm(a), norm(b)) >= 85

    for pid in sorted(refs, key=refs.get):
        for short, long_form in paper_abbreviations(ctx.papers[pid]).items():
            if short.lower() in _NOT_ABBREVIATIONS:
                continue
            long_form = long_form.strip(" /-:;,")
            if not any(same_meaning(long_form, seen) for seen, _ in own[short]):
                own[short].append((long_form, refs[pid]))
    found = []
    tokens = set(re.findall(r"(?<![\w-])([A-Za-zΔ]*[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)(?![\w-])", text))
    # "HRs" and "MALOs" are the plurals of HR and MALO: one entry each.
    plural_of: dict[str, str] = {}
    for token in list(tokens):
        bare = token.lstrip("Δ")
        if len(bare) > 2 and bare.endswith("s") and bare[:-1].upper() == bare[:-1] and any(c.isalpha() for c in bare[:-1]):
            plural_of[bare[:-1]] = bare
            tokens.discard(token)
            tokens.add(bare[:-1])
    for token in sorted(tokens, key=str.lower):
        bare = token.lstrip("Δ")
        also = f" (also {plural_of[bare]})" if bare in plural_of else ""
        if bare not in own and bare not in standard and plural_of.get(bare) in own:
            own[bare] = own[plural_of[bare]]
        if bare.lower() in _NOT_ABBREVIATIONS:
            continue
        if len(bare) < 2 or not any(c.isupper() for c in bare[1:] + bare[:1]) or bare.isdigit():
            continue
        if bare in STATISTICAL and bare in standard:
            # "HR, 0.66; 95% CI" is a hazard ratio whatever a cited paper
            # abbreviated as HR elsewhere (one defined it as heart failure).
            found.append(f"- **{bare}**{also} — {standard[bare]}")
        elif bare in own:
            meanings = own[bare]
            if len(meanings) == 1:
                long_form, ref = meanings[0]
                found.append(f"- **{bare}**{also} — {long_form} (as defined in [{ref}])")
            else:
                found.append(f"- **{bare}**{also} — the cited papers use it differently: "
                             + "; ".join(f"{lf} in [{ref}]" for lf, ref in meanings))
        elif bare in standard:
            found.append(f"- **{bare}**{also} — {standard[bare]}")
    found = list(dict.fromkeys(found))
    if not found:
        return "No abbreviations need expanding."
    return "\n".join(found) + "\n\nΔ before a measure means the change in it from baseline."


# --- 5. notes for the gaps --------------------------------------------------------------------


def population_note(ctx: "WriteContext", studies: dict[str, Study]) -> str | None:
    """How many studies are about the population the question names."""
    population = ctx.spec.population
    if not population or not studies:
        return None
    entities = [e for e in ctx.spec.entities if e.text and e.text.lower() in population.lower()]
    if not entities:
        return None
    matchers = [TermMatcher(e.exact_terms) for e in entities]
    keys = {s.key for s in studies.values()}
    about = {s.key for s in studies.values()
             if any(m.exact_count(f"{ctx.papers[s.paper_id].title}\n{ctx.papers[s.paper_id].abstract or ''}")
                    for m in matchers)}
    names = ", ".join(e.canonical for e in entities)
    if about == keys:
        return None
    return (f"The question is about {population}. {len(about)} of the {len(keys)} independent studies name "
            f"{names} in their title or abstract; the others studied a related population (see who was studied, "
            "above), so their results may not carry over.")


def cited_population_note(ctx: "WriteContext", paper_ids: set[str], refs: dict[str, int]) -> str | None:
    """For the papers behind one section: which name the question's
    population in their title or abstract, said beside that section. None
    when the question names no population or every cited paper names it."""
    population = ctx.spec.population
    entities = [e for e in ctx.spec.entities if population and e.text and e.text.lower() in population.lower()]
    pids = [p for p in paper_ids if p in ctx.papers]
    if not entities or not pids:
        return None
    matchers = [TermMatcher(e.exact_terms) for e in entities]
    names = {p for p in pids if any(m.exact_count(f"{ctx.papers[p].title}\n{ctx.papers[p].abstract or ''}")
                                    for m in matchers)}
    if len(names) == len(pids):
        return None
    word = ", ".join(e.canonical for e in entities)
    others = [p for p in pids if p not in names]
    return (f"Who was studied: the question asks about {population}; "
            + (f"{_refs(names, refs)} name{'s' if len(names) == 1 else ''} {word} in the title or abstract, "
               if names else "")
            + f"{_refs(others, refs)} do{'es' if len(others) == 1 else ''} not, so "
            + ("its" if len(others) == 1 else "their") + " results may not carry over.")


def basis_note(units: list[EvidenceUnit], studies: dict[str, Study], refs: dict[str, int]) -> str | None:
    """"Evidence basis: [3] randomised controlled trial, ..." for one section."""
    seen, parts = set(), []
    for u in units:
        s = studies.get(u.paper_id)
        if s is None or u.paper_id in seen:
            continue
        seen.add(u.paper_id)
        parts.append(study_label(s, refs))
    return ("Evidence basis: " + "; ".join(parts) + ".") if parts else None


def _plural(noun: str, n: int) -> str:
    """"observational study" -> "observational studies"; the head noun is the
    one before any bracket ("clinical trial (randomisation not stated)")."""
    if n == 1:
        return noun
    head, sep, rest = noun.partition(" (")
    head = head[:-1] + "ies" if head.endswith("y") else head + "s"
    return head + (sep + rest if sep else "")


def _cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ").strip()


# --- 6. what the numbers themselves say is wrong -------------------------------------------------


def statistics_notes(ctx: "WriteContext", refs: dict[str, int]) -> list[str]:
    """Statistics printed in a way that cannot be right, and a paper that gives
    two different estimates for what reads as the same outcome. Shown as the
    papers print them; never corrected, and kept out of Key results."""
    notes = []
    for u in ctx.summary.units:
        for problem in claims.statistical_anomalies(u.quote):
            notes.append(f"{ctx.summary.short_ids[u.id]} {_refs([u.paper_id], refs)}: {problem}. Shown as printed; "
                         "not used in Key results. Check the paper before relying on it.")
    by_paper: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for u in ctx.summary.units:
        for label, value in claims.estimates(u.quote).items():
            by_paper[u.paper_id][label].add(value)
    for pid, labels in by_paper.items():
        for label, values in sorted(labels.items()):
            if len(values) > 1:
                notes.append(f"{_refs([pid], refs)} gives different estimates for {label} "
                             f"({', '.join(sorted(values))}) in different sentences - perhaps different analyses or "
                             "follow-up; not reconciled here.")
    return notes
