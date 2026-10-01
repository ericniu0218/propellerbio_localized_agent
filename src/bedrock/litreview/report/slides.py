"""The target-proposal PowerPoint, made from a finished Report. Code only - no model.

The deck follows Propeller's target-proposal template (one target, a
bispecific of two, or - for a "what should I target?" question - a disease with
its candidate targets) and is drawn in ArcDeck's slide format. Like the other
renderers of the report (stages/s8_render.py), it is not a second piece of
research: every statement on a slide is

  - a sentence the report kept under the citation guard, word for word, with
    its [n] reference numbers;
  - a line the report's own code generated (the key limitations, the review
    articles, a disagreement);
  - a count code makes from the report's verified findings (papers read,
    findings by outcome, targets by number of papers); or
  - a paper's own words (its verbatim quotes in the appendix).

Nothing is summarised, reworded or added, so the deck cannot say more than
the report does. Where the template asks for something no paper can supply -
who owns a drug, what the unmet need is - the slide says "fill in" rather than
guess, and slides assembled by a written rule rather than by the report's own
sentences (indications, competitors, SWOT) are marked `by_rule` so the
renderer tells the presenter to review them.

Two rules from the report carry over unchanged. A field a paper does not
support (provenance "none": `extract/provenance.py`) is never shown - the
agent, disease and combination partner on these slides each come from a field
that was located in the paper, and nothing shows `effect_detail`. And a gap is
stated as a gap: a slot with no verified finding still appears, saying so, so
the scientist sees the template slot and knows nothing was found.

Which report section feeds which slide is `defaults/deck.yaml` (loaded the way
report.yaml is), so changing the report's layout means editing that file. A slide
shows only its first few sentences; the rest follow the SWOT as "further
findings" slides, so a long section never makes a long slide. This
module only decides what each slide holds; `report/deck_render.py` draws it and
splits what does not fit over "(cont.)" slides, so nothing here shrinks text.

Needs python-pptx only to draw (`pip install -e ".[server]"`); `outline` builds
the slides as plain data without it, and the core install never imports it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

from bedrock.litreview import study_design
from bedrock.litreview.config import config_file
from bedrock.litreview.models import EvidenceUnit, Paper, Reference, Report, ReportSection
from bedrock.litreview.report.deck_model import (
    DECK_VERSION, FigureAsset, Para, RefMark, Slide, Visual, VisualItem)
from bedrock.litreview.report.guard import _located
from bedrock.litreview.stages.s8_render import _numbers, _ref_map, lede

ABSENCE = ("No reported harm among these papers is not evidence of safety: the papers read may not have looked for, "
           "or been large or long enough to see, a given harm.")
NO_FINDING = "No verified finding in the papers read."
NO_COMBINATION = "No verified finding in the papers read tested both targets together."
NO_COMPETITOR = "No verified human finding that names an agent in the papers read."
FILL_IN = "- fill in"
_INTEGRITY = ("This report failed its integrity check: a quote or field did not match its paper. Do not rely on it "
              "until it is fixed.")

HUMAN_TYPES = ("human_clinical", "human_genetic")
_NUM = re.compile(r"\[(\d+)\]")
_LIST_MAX = 6
"""Most targets, or diseases, a stacked list shows; the rest are counted."""

SLIDE_IDS = ("executive_summary", "introduction", "candidate_target", "moa", "human", "animal", "bispecific", "indications",
             "competitive", "swot", "references", "appendix")


def plain(md: str) -> str:
    """Markdown the report's code wrote, as slide text: emphasis and link
    brackets dropped, whitespace collapsed. Words are never changed."""
    md = re.sub(r"<sub>|</sub>", "", md)
    md = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", md)
    md = re.sub(r"<(https?://[^>\s]+)>", r"\1", md)
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    md = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"\1", md)
    return " ".join(md.split())


def _cited_numbers(text: str) -> list[int]:
    """The [n] reference numbers a line of text carries, once each, in order."""
    return list(dict.fromkeys(int(n) for n in _NUM.findall(text)))


def _bullets(md: str | None) -> list[str]:
    """The top-level "- " items of a generated section's Markdown."""
    return [plain(line[2:]) for line in (md or "").splitlines() if line.startswith("- ")]


# The limitations that most change how the Answer should be read, in order:
# whether the studies are the population asked about, which measures have no
# tested human effect, whether their numbers
# were printed wrongly, what was only read as an abstract, notices and
# preprints. The report's Limitations section keeps all of them in its order.
_KEY_LIMITS = (r"^The question is about", r"^No tested effect in humans", r"interval printed|p-value printed|different estimates",
               r"abstract only", r"publication notice", r"preprint")


def key_limitations(bullets: list[str], limit: int = 3) -> list[str]:
    chosen = []
    for pattern in _KEY_LIMITS:
        chosen += [b for b in bullets if re.search(pattern, b) and b not in chosen]
    return chosen[:limit]


# --- which section feeds which slide: defaults/deck.yaml -------------------------------------------------------


def load_deck_layout(path: Path | None = None) -> dict:
    """`defaults/deck.yaml`, or a whole-file replacement in the config
    directory (`config_file`), read on every call like report.yaml so an
    override dropped in later is picked up."""
    raw = yaml.safe_load(Path(path or config_file("deck.yaml")).read_text(encoding="utf-8")) or {}
    if raw.get("version") != DECK_VERSION:
        raise ValueError(f"deck.yaml: version must be {DECK_VERSION!r}, not {raw.get('version')!r}")
    missing = [i for i in SLIDE_IDS if i not in (raw.get("slides") or {})]
    if missing:
        raise ValueError(f"deck.yaml: no entry for slide(s) {', '.join(missing)}")
    return raw


def _pick(mapping: dict | None, question_type: str) -> list[str]:
    """The list for this question type, else the `default` one."""
    if not mapping:
        return []
    return list(mapping[question_type] if question_type in mapping else mapping.get("default") or [])


def _title(entry: dict, mode: str) -> str:
    title = entry["title"]
    return str(title[mode] if isinstance(title, dict) else title)


# --- what the question is about ------------------------------------------------------------------------------------


def _entities(report: Report) -> list[dict]:
    return [e for e in (report.interpretation.get("entities") or []) if isinstance(e, dict)]


def _targets(report: Report) -> list[str]:
    names = [str(e["canonical"]) for e in _entities(report) if e.get("kind") == "target" and e.get("canonical")]
    return list(dict.fromkeys(names))


def _mode(report: Report, targets: list[str]) -> str:
    """`bispecific` when the question names two targets, `disease` when a
    target-identification question names none (the deck then lists candidate
    targets for the disease), else `single`."""
    if len(targets) >= 2:
        return "bispecific"
    if not targets and report.interpretation.get("question_type") == "target_identification":
        return "disease"
    return "single"


def _deck_title(report: Report, targets: list[str], mode: str) -> str:
    subject = str(report.interpretation.get("subject") or "").strip()
    if mode == "bispecific":
        return " x ".join(targets)
    if mode == "disease":
        name = subject or report.question
        return f"{name[:1].upper()}{name[1:]} - candidate targets"  # only the first letter; the subject's other casing stays
    drug = next((str(e["canonical"]) for e in _entities(report) if e.get("kind") == "drug" and e.get("canonical")), "")
    return targets[0] if targets else (drug or subject or report.question)


_INITIALS = re.compile(r"^[A-Z]{1,3}$")
_YEAR_AFTER_JOURNAL = re.compile(r"\*\s+((?:19|20)\d{2})\b")


def short_name(ref: Reference) -> str:
    """"Smith 2020", from the paper record the reference was made from - the
    first author's surname, as the citation lists it, and the year the paper
    record carries. Never model text."""
    head = ref.citation.split(",")[0].split(" et al")[0]
    surname: list[str] = []
    for word in head.split():
        if surname and _INITIALS.match(word):
            break
        surname.append(word)
    else:
        surname = surname[:1]  # no initials to stop at: a lone author runs straight into the title
    year = ref.signals.year
    if not year and (m := _YEAR_AFTER_JOURNAL.search(ref.citation)):
        year = int(m.group(1))
    return " ".join(surname + ([str(year)] if year else [])) or f"[{ref.number}]"


# --- the report's kept sentences, with what they rest on -----------------------------------------------------


@dataclass
class _Kept:
    """A kept sentence of a written section, as the slides show it."""

    section: str
    text: str
    """The sentence word for word, then its [n] reference numbers."""
    numbers: list[int]
    cites: list[str]
    """Short ids ("E3") of the findings it cites that have a reference."""
    units: list[EvidenceUnit]
    sub_id: str = ""
    sub_title: str = ""
    """The report subsection the sentence is in ("targets:SIK3", "SIK3"), else empty. Its section is the
    parent's: a subsection is part of its section, so deck.yaml lists "targets", never "targets:SIK3"."""

    def para(self) -> Para:
        return Para(self.text, refs=self.numbers)


def _kept_sentences(report: Report, refs: dict[str, int], by_short: dict[str, EvidenceUnit]) -> list[_Kept]:
    out: list[_Kept] = []

    def walk(section, top: str, sub: ReportSection | None) -> None:
        for s in section.sentences:
            if not s.kept:
                continue
            nums = _numbers(s, refs)
            marker = "".join(f"[{n}]" for n in nums)
            cites = [c for c in s.citations if c in refs]
            out.append(_Kept(top, f"{s.text.rstrip()} {marker}".rstrip(), nums, cites,
                             [by_short[c] for c in cites if c in by_short], sub.id if sub else "",
                             sub.title if sub else ""))
        for inner in section.subsections:
            walk(inner, top, sub or inner)  # the first-level subsection names the group

    for section in report.sections:
        if section.kind == "written":
            walk(section, section.id, None)
    return out


def _human(u: EvidenceUnit) -> bool:
    return u.evidence_type in HUMAN_TYPES


def _combo(u: EvidenceUnit) -> bool:
    """A finding that names a combination partner the paper supports."""
    return bool(u.combo_with) and _located(u, "combo_with")


def _kind(units: list[EvidenceUnit]) -> str | None:
    """What a sentence's cited findings all are, if they are all one kind:
    human, animal (live animal), or cell/association work (`moa`)."""
    if not units:
        return None
    if all(_human(u) for u in units):
        return "human"
    if all(u.evidence_type == "in_vivo" for u in units):
        return "animal"
    if all(u.evidence_type == "in_vitro" or u.outcome == "association" for u in units):
        return "moa"
    return None


def _notes(kept: list[_Kept], refs: dict[str, int], after: int = 0) -> list[str]:
    """One speaker-note line per sentence naming the findings it rests on;
    `after` continues the numbering when the sentences are the rest of a slide's."""
    return [f"Sentence {i}: " + ", ".join(f"{c} [{refs[c]}]" for c in k.cites) for i, k in enumerate(kept, after + 1)]


# --- counts made by code -------------------------------------------------------------------------------------------


def _papers(units: list[EvidenceUnit]) -> str:
    n = len({u.paper_id for u in units})
    human = len({u.paper_id for u in units if _human(u)})
    animal = len({u.paper_id for u in units if u.evidence_type == "in_vivo"})
    return f"{n} paper{'s' if n != 1 else ''}: {human} human, {animal} animal"


def _cited_units(kept: list[_Kept]) -> list[EvidenceUnit]:
    """The findings a group of sentences cites, once each."""
    return list({u.id: u for k in kept for u in k.units}.values())


def _finding_counts(units: list[EvidenceUnit]) -> str:
    papers = len({u.paper_id for u in units})
    human = sum(1 for u in units if _human(u))
    animal = sum(1 for u in units if u.evidence_type == "in_vivo")
    return (f"{len(units)} finding{'s' if len(units) != 1 else ''} from {papers} paper{'s' if papers != 1 else ''}: "
            f"{human} human, {animal} animal")


def _tally(units: list[EvidenceUnit], name_of) -> list[VisualItem]:
    """One item per name (a target, a disease), most papers first, then by
    name, each with "N papers: x human, y animal"."""
    groups: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for u in units:
        name = name_of(u)
        if name:
            groups[name].append(u)
    ranked = sorted(groups.items(), key=lambda kv: (-len({u.paper_id for u in kv[1]}), kv[0].lower()))
    return [VisualItem(label=name, sublabel=_papers(us), value=float(len({u.paper_id for u in us}))) for name, us in ranked]


def _listed(items: list[VisualItem], what: str) -> tuple[list[VisualItem], list[Para]]:
    shown = items[:_LIST_MAX]
    more = len(items) - len(shown)
    return shown, ([Para(f"{more} further {what} are in the report's evidence table.", muted=True)] if more > 0 else [])


_EVIDENCE_TYPE_LABELS = (("in_vitro", "Cell or in vitro"), ("in_vivo", "Animal (in vivo)"),
                         ("human_clinical", "Human clinical"), ("human_genetic", "Human genetic"))
_OUTCOME_LABELS = (("improved", "Improved", "positive"), ("no_effect", "No effect", "neutral"),
                   ("worsened", "Worsened", "negative"), ("mixed", "Mixed", "neutral"),
                   ("association", "Association", "neutral"))
_SPECIES_LABELS = (("mouse", "Mouse"), ("rat", "Rat"), ("cyno", "Cynomolgus"))


def _evidence_types(units: list[EvidenceUnit]) -> Visual | None:
    items = [VisualItem(label, value=float(n)) for key, label in _EVIDENCE_TYPE_LABELS
             if (n := sum(1 for u in units if u.evidence_type == key))]
    return Visual("doughnut_chart", items, "Verified findings, by type of evidence") if items else None


def _by_outcome(units: list[EvidenceUnit]) -> Visual | None:
    human = [u for u in units if _human(u)]
    # sublabel carries the outcome's own key so a renderer can colour by meaning, not by wording
    items = [VisualItem(label, sublabel=key, value=float(n), tone=tone) for key, label, tone in _OUTCOME_LABELS
             if (n := sum(1 for u in human if u.outcome == key))]
    return Visual("bar_chart", items, "Verified findings in people, by outcome") if items else None


def _species(units: list[EvidenceUnit]) -> Visual | None:
    animal = [u for u in units if u.evidence_type == "in_vivo"]
    named = {key for key, _ in _SPECIES_LABELS}
    counts = [(label, sum(1 for u in animal if u.species == key)) for key, label in _SPECIES_LABELS]
    counts.append(("Other", sum(1 for u in animal if u.species not in named)))
    items = [VisualItem(label, value=float(n)) for label, n in counts if n]
    return Visual("doughnut_chart", items, "Verified findings in live animals, by species") if items else None


# --- competitors: verified human findings that name an agent ------------------------------------------------------


@dataclass
class _Agent:
    agent: str
    target: str
    modality: str
    disease: str
    design: str
    number: int
    short: str


_DESIGN_WORDS = (("human_rct", "randomised controlled trial"), ("human_interventional_other", "clinical trial"),
                 ("human_genetic", "human genetic study"), ("human_observational", "observational study"))
_PHASE = re.compile(r"clinical trial, phase (i{1,3}v?|[1-4])$", re.I)


def _design(ref: Reference) -> str:
    """The stage or design of a paper's study, from the paper record's own
    publication types (`study_design.py`, metadata only: a reference carries no
    abstract), else "human study". A phase is named when the type says one."""
    for t in ref.signals.publication_types:
        if m := _PHASE.match(t.strip()):
            return f"phase {m.group(1).upper()} trial"
    record = study_design.design_signals(Paper(id=ref.paper_id, title="", signals=ref.signals))
    labels = study_design.grounded_labels(record)
    return next((words for label, words in _DESIGN_WORDS if label in labels), "human study")


# A competitor is something a company could own: a molecule of one of these kinds. The extractor also records
# "inhibition", "estrogen" or "hormone replacement therapy" as an intervention; none of those is a drug to compete with.
_DRUG_MODALITIES = ("antibody", "peptide", "siRNA", "small_molecule", "vaccine")
# Words that only say what was done to a target, so as a whole name they name no agent.
_GENERIC_AGENTS = {"inhibition", "blockade", "treatment", "therapy", "knockdown", "knockout", "deletion",
                   "administration", "supplementation", "overexpression", "neutralization", "inhibitor", "antagonist",
                   "agonist"}


def _is_agent(u: EvidenceUnit) -> bool:
    name = (u.intervention or "").strip().lower()
    return (bool(name) and _located(u, "intervention") and u.modality in _DRUG_MODALITIES
            and name not in _GENERIC_AGENTS and name not in {u.target.lower(), u.target_raw.strip().lower()})


def _agents(report: Report, number_of: dict[str, int], ref_of: dict[str, Reference]) -> list[_Agent]:
    """Rows for the Competitive landscape: verified findings from human
    clinical evidence whose agent is a drug-type molecule the paper supports
    (a provenance of "none" blanks the agent: never shown; a bare "inhibition"
    or the target's own name is no agent). One row per agent, target and
    disease, in the order of the paper's reference number."""
    units = sorted((u for u in report.evidence if u.evidence_type == "human_clinical" and _is_agent(u)
                    and u.paper_id in number_of),
                   key=lambda u: (number_of[u.paper_id], u.id))
    seen: set[tuple[str, str, str]] = set()
    rows: list[_Agent] = []
    designs: dict[str, str] = {}
    for u in units:
        target = "" if u.outcome_measure else u.target  # a measured outcome is not a target
        disease = u.disease if u.disease and _located(u, "disease_raw") else ""
        key = (u.intervention.strip().lower(), target.lower(), disease.lower())
        if key in seen:
            continue
        seen.add(key)
        designs.setdefault(u.paper_id, _design(ref_of[u.paper_id]))
        rows.append(_Agent(u.intervention.strip(), target, u.modality.replace("_", " "), disease, designs[u.paper_id], number_of[u.paper_id], report.short_ids.get(u.id, "")))
    return rows


# --- the SWOT -----------------------------------------------------------------------------------------------------


@dataclass
class _Item:
    text: str
    refs: list[int] = field(default_factory=list)
    cites: list[str] = field(default_factory=list)
    muted: bool = False

    def para(self) -> Para:
        return Para(self.text, level=1, muted=self.muted, refs=self.refs)


# A card shows three items; the rest are counted, not listed (the report has them all).
_CELL_ITEMS = 3
_EMPTY = {"Strengths": NO_FINDING, "Weaknesses": NO_FINDING, "Opportunities": "Not in the papers read - fill in.",
          "Threats": "No agent tested in people and no disagreement was recorded in the report."}
_GAP_PHRASE = "No tested effect in humans"


def _item_of(k: _Kept) -> _Item:
    return _Item(k.text, k.numbers, k.cites)


def _disagreement_items(md: str | None) -> list[_Item]:
    """One item per disagreement the report lists: its explanation line, with
    the reference numbers of the independent studies listed under it."""
    explained: list[tuple[str, list[int]]] = []
    for block in (md or "").split("\n\n"):
        block = block.strip()
        if "(Studies disagree)" in block or "(Papers disagree on this target" in block:
            explained.append((plain(block), []))
        elif block.startswith("- **Independent study") and explained:
            explained[-1][1].extend(int(n) for n in _NUM.findall(block))
    out = []
    for text, nums in explained:
        nums = sorted(set(nums))
        out.append(_Item(f"{text.rstrip()} {''.join(f'[{n}]' for n in nums)}".rstrip(), nums))
    return out


# --- finishing a slide --------------------------------------------------------------------------------------------


def _numbers_on(slide: Slide) -> list[int]:
    nums = {n for p in slide.paras for n in p.refs}
    nums |= {int(n) for row in (slide.table or [])[1:] for cell in row for n in _NUM.findall(cell)}
    if slide.figure_ref:
        nums.add(slide.figure_ref)
    return sorted(nums)


def _gap_if_empty(slide: Slide) -> Slide:
    """A slide with no sentence and no visual still appears, saying so."""
    if not slide.paras and slide.visual is None and slide.table is None and slide.figure is None:
        slide.paras = [Para(NO_FINDING)]
    return slide


# --- the deck ------------------------------------------------------------------------------------------------------


def _background_paras(md: str | None) -> list[Para]:
    """The review articles the report quotes for context: each review's line
    (with its link), then its own sentences, exactly as the report has them."""
    link = re.compile(r"<(https?://[^>\s]+)>")
    paras: list[Para] = []
    for block in (md or "").split("\n\n"):
        block = block.strip()
        lines = block.splitlines()
        if block.startswith("**"):
            m = link.search(lines[0])
            paras.append(Para(plain(link.sub("", lines[0])).rstrip(" ·"), link=m.group(1) if m else None))
            paras += [Para(f"“{' '.join(x[2:].split())}”", level=1, muted=True) for x in lines[1:] if x.startswith("> ")]
        elif block.startswith("Also collected:"):
            for x in lines[1:]:
                if x.startswith("- "):
                    m = link.search(x)
                    paras.append(Para(plain(link.sub("", x[2:])).rstrip(" ·"), link=m.group(1) if m else None))
        elif block.startswith("…"):
            paras.append(Para(plain(block), muted=True))
    if not paras and (md or "").strip():
        paras = [Para(plain(md or ""))]  # the report's own "no review article was collected"
    return paras


def outline(report: Report, source: str = "", *, figures: Sequence[FigureAsset] = (), author: str = "",
            date: str = "") -> list[Slide]:
    """The slides, as data: what the deck says, independent of python-pptx.
    Deterministic - the same report, figures, author and date give the same
    slides. `figures` are the paper figures chosen for the run; only those
    that were downloaded (have a path) are placed, each on one slide."""
    layout = load_deck_layout()
    entries = layout["slides"]
    qtype = str(report.interpretation.get("question_type") or "")
    targets = _targets(report)
    mode = _mode(report, targets)
    refs, quotes = _ref_map(report)
    units = list(report.evidence)
    by_uid = {u.id: u for u in units}
    by_short = {short: by_uid[uid] for uid, short in report.short_ids.items() if uid in by_uid}
    number_of = {r.paper_id: r.number for r in report.references}
    ref_of = {r.paper_id: r for r in report.references}
    marks = {r.number: RefMark(r.number, short_name(r), r.url) for r in report.references}
    warning = _INTEGRITY if report.integrity else ""
    cited: list[str] = []  # short ids shown anywhere, in deck order: the appendix quotes them

    def finish(slide: Slide) -> Slide:
        slide.refs = [marks[n] for n in _numbers_on(slide) if n in marks]
        return slide

    # Which slide each kept sentence belongs on: the one whose deck.yaml `sections` lists its section. Only a
    # sentence from a section no slide lists is placed by what it cites.
    home: dict[str, str] = {}
    for sid, entry in entries.items():
        for section_id in _pick(entry.get("sections"), qtype):
            home.setdefault(section_id, sid)
    summary_sections = set(_pick(entries["executive_summary"].get("sections"), qtype))
    sentence_slides = {"introduction", "moa", "human", "animal", "indications"} | (
        {"bispecific"} if mode == "bispecific" else set())
    fallback = str(layout.get("unmapped") or "moa")
    kept = _kept_sentences(report, refs, by_short)
    summary = [k for k in kept if k.section in summary_sections]
    others = [k for k in kept if k.section not in summary_sections]
    # A target-identification report writes one subsection per candidate target. They are shown on a slide of
    # their own each (below), not routed: only where the introduction lists the section and the deck does not
    # open with review articles.
    intro_entry = entries["introduction"]
    candidate_sections = set(_pick(intro_entry.get("sections"), qtype)) if not _pick(intro_entry.get("background"), qtype) \
        else set()
    groups: dict[str, tuple[str, list[_Kept]]] = {}
    for k in others:
        if k.sub_id and k.section in candidate_sections:
            groups.setdefault(k.sub_id, (k.sub_title, []))[1].append(k)
    routed: dict[str, list[_Kept]] = {sid: [] for sid in sentence_slides}
    for k in others:
        if k.sub_id in groups:
            continue
        if mode == "bispecific" and any(_combo(u) for u in k.units):
            target = "bispecific"
        else:
            target = home.get(k.section) or _kind(k.units) or fallback
        if target == "swot":
            continue  # no slide of its own: the SWOT's weaknesses draw on it
        if target not in sentence_slides:
            target = fallback if fallback in sentence_slides else "moa"
        routed[target].append(k)
    # No section is mapped to the animal slide for this question type (target identification has no preclinical
    # section): it takes the live-animal sentences of the slides named in deck.yaml.
    animal = entries["animal"]
    if not _pick(animal.get("sections"), qtype):
        position = {id(k): i for i, k in enumerate(others)}
        for sid in animal.get("in_vivo_from") or []:
            if sid in routed and sid != "animal":
                routed["animal"] += [k for k in routed[sid] if _kind(k.units) == "animal"]
                routed[sid] = [k for k in routed[sid] if _kind(k.units) != "animal"]
        routed["animal"].sort(key=lambda k: position[id(k)])

    # A slide shows its first few sentences, in report order: `max_sentences`, or fewer when it also carries a
    # chart. A paper's own figure takes the chart's place and may come from any sentence routed to the slide. The
    # rest go to "Appendix: further findings" slides after the SWOT, so a long section never makes a long slide
    # and nothing is dropped.
    cap_plain = int(layout.get("max_sentences") or 6)
    cap_chart = int(layout.get("max_sentences_with_visual") or 4)
    usable = [f for f in figures if f.path]
    used_figures: set[tuple[str, str]] = set()
    further: list[tuple[str, str, list[_Kept], int]] = []

    def lay_out(entry: dict, title: str, kicker: str, here: list[_Kept], visual: Visual | None):
        found = _pick_figure(here, usable, used_figures, number_of) if entry.get("figures") else None
        if found:
            visual = None
        cap = int(entry.get("max_sentences") or cap_plain) if visual is None else int(
            entry.get("max_sentences_with_visual") or cap_chart)
        if len(here) > cap:
            further.append((title, kicker, here[cap:], cap))
        return here[:cap], visual, found

    slides: list[Slide] = []

    # 1. Title: the target, or the disease whose candidate targets follow.
    when = " | ".join(x for x in (author.strip(), date.strip()) if x)
    notes = [f"Deck {DECK_VERSION}, made by code from the stored report; no model was used.",
             "Models behind the report: " + ", ".join(f"{k} {v}" for k, v in sorted(report.model_versions.items())),
             lede(report)]
    if source:
        notes.insert(1, f"Source report: {source}")
    slides.append(Slide(title=_deck_title(report, targets, mode), layout="title",
                        subtitle="\n".join(x for x in (when, report.question) if x),
                        kicker="Target proposal - from a literature review", notes=notes, warning=warning))

    # 2. Executive summary: the Answer, the limitations that change how to read it, and what was counted.
    paras = [k.para() for k in summary]
    notes = _notes(summary, refs)
    cited += [c for k in summary for c in k.cites]
    answer = next((s for s in report.sections if s.id in summary_sections and s.kind == "written"), None)
    note = plain(answer.note) if answer is not None and answer.note else ""
    if not summary:
        paras = [Para(note or NO_FINDING)]  # the report's own reason, else the gap stated as a gap
    elif note:
        paras.append(Para(note, muted=True))
    gap_sections = _pick(entries["executive_summary"].get("limitations"), qtype)
    limits = key_limitations([b for s in report.sections if s.id in gap_sections for b in _bullets(s.body_markdown)])
    if limits:
        paras.append(Para("Key limitations", bold=True))
        paras += [Para(x, level=1, refs=_cited_numbers(x)) for x in limits]
    st = report.stats
    count = lambda pred: float(len({u.paper_id for u in units if pred(u)}))  # noqa: E731
    stats = Visual("stats_callout", [
        VisualItem("Papers read", f"of {st.get('papers_collected', 0)} collected" if st.get("papers_collected") else "",
                   float(st.get("papers_read", 0))),
        VisualItem("Papers with human findings", value=count(_human)),
        VisualItem("Papers with animal findings", value=count(lambda u: u.evidence_type == "in_vivo")),
        VisualItem("Verified findings", "each with the paper's own quote", float(len(units)))],
        "Counted by code from the report's verified findings")
    slides.append(finish(Slide(title=_title(entries["executive_summary"], mode), paras=paras, visual=stats, notes=notes,
                               warning=warning)))

    # 3. Introduction: the candidate targets (target identification), else the review articles and what was asked.
    entry = entries["introduction"]
    background = _pick(entry.get("background"), qtype)
    kicker = str(entry.get("kicker") or "") if background else ""
    paras: list[Para] = []
    if background:
        paras = _background_paras("\n\n".join(s.body_markdown or "" for s in report.sections if s.id in background))
        cards, seen = [], set()
        for e in _entities(report):
            name = str(e.get("canonical") or "")
            if name and name.lower() not in seen:
                seen.add(name.lower())
                cards.append(VisualItem(name, str(e.get("kind") or "").capitalize()))
        if report.interpretation.get("population"):
            cards.append(VisualItem(str(report.interpretation["population"]), "Population"))
        visual = Visual("callout_cards", cards, "What the question names") if cards else None
        tail: list[Para] = []
    elif groups:
        # The report's own candidates, in its order, each with what its sentences rest on.
        kicker = str(entry.get("candidates_kicker") or "")
        items = [VisualItem(title, _finding_counts(_cited_units(ks)), float(len(_cited_units(ks))))
                 for title, ks in groups.values()]
        shown, more = items[:_LIST_MAX], len(items) - _LIST_MAX
        tail = []
        visual = Visual("stacked_list", shown, "Candidate targets, in the report's order"
                        + (f" (+{more} more in the report)" if more > 0 else ""))
    else:
        # No subsections: count the targets the findings name, leaving out what the question itself names.
        own = {str(n).lower() for e in _entities(report) for n in (e.get("canonical"), *(e.get("exact_terms") or [])) if n}
        shown, tail = _listed(_tally([u for u in units if not u.outcome_measure],
                                     lambda u: "" if u.target.lower() in own else u.target), "targets")
        visual = Visual("stacked_list", shown, "Candidate targets, by number of papers") if shown else None
    kept_here, visual, _ = lay_out(entry, _title(entry, mode), kicker, routed["introduction"], visual)
    cited += [c for k in kept_here for c in k.cites]
    slides.append(finish(_gap_if_empty(Slide(title=_title(entry, mode), paras=[k.para() for k in kept_here] + paras + tail,
                                             visual=visual, kicker=kicker, notes=_notes(kept_here, refs)))))

    # 3b. One slide per candidate target, up to `max_target_slides`; the rest are in an appendix slide.
    entry = entries["candidate_target"]
    extra_groups: list[tuple[str, list[_Kept]]] = []
    for n, (title, ks) in enumerate(groups.values()):
        if n >= int(layout.get("max_target_slides") or 6):
            extra_groups.append((title, ks))
            continue
        full = _title(entry, mode).replace("{target}", title)
        cites = _cited_units(ks)
        count = lambda pred: float(len({u.paper_id for u in cites if pred(u)}))  # noqa: E731
        stats = Visual("stats_callout", [
            VisualItem("Papers", value=count(lambda u: True)),
            VisualItem("Human findings", value=float(sum(1 for u in cites if _human(u)))),
            VisualItem("Animal findings", value=float(sum(1 for u in cites if u.evidence_type == "in_vivo")))],
            "Counted by code from the findings this target's sentences cite")
        kept_here, visual, found = lay_out(entry, full, "", ks, stats)
        slide = Slide(title=full, paras=[k.para() for k in kept_here], visual=visual, notes=_notes(kept_here, refs))
        if found:
            slide.figure, slide.figure_ref = found
        cited += [c for k in kept_here for c in k.cites]
        slides.append(finish(slide))

    # 4-6. Mechanism, then human and animal proof of concept. A paper's own figure replaces the chart.
    chart = {"moa": _evidence_types(units), "human": _by_outcome(units), "animal": _species(units)}
    for sid in ("moa", "human", "animal"):
        entry = entries[sid]
        kicker = str(entry.get("kicker") or "")
        kept_here, visual, found = lay_out(entry, _title(entry, mode), kicker, routed[sid], chart[sid])
        # The chart counts every finding of its kind, not this slide's, so a
        # slide with a chart but no sentence still says it has none.
        paras = [k.para() for k in kept_here] or [Para(NO_FINDING)]
        slide = Slide(title=_title(entry, mode), paras=paras, kicker=kicker, visual=visual,
                      notes=_notes(kept_here, refs))
        if found:
            slide.figure, slide.figure_ref = found
        cited += [c for k in kept_here for c in k.cites]
        slides.append(finish(slide))

    # 7. Bispecific only: why the two targets belong together.
    if mode == "bispecific":
        entry = entries["bispecific"]
        a, b = targets[0], targets[1]
        pair = {a.lower(), b.lower()}
        together = [u for u in units if _combo(u) and {u.target.lower(), u.combo_with.lower()} == pair]
        alone = lambda name: [u for u in units if u.target.lower() == name.lower() and not _combo(u)]  # noqa: E731
        better = lambda us: float(sum(1 for u in us if u.outcome == "improved"))  # noqa: E731
        visual = Visual("comparison", [
            VisualItem(f"{a} alone", "improved findings", better(alone(a))),
            VisualItem(f"{b} alone", "improved findings", better(alone(b))),
            VisualItem(f"{a} and {b} together", "improved findings", better(together))],
            "Verified findings that improved, by what was tested") if routed["bispecific"] or together else None
        kicker = str(entry.get("kicker") or "")
        kept_here, visual, _ = lay_out(entry, _title(entry, mode), kicker, routed["bispecific"], visual)
        slide = Slide(title=_title(entry, mode), paras=[k.para() for k in kept_here], kicker=kicker, visual=visual,
                      notes=_notes(kept_here, refs))
        if visual is None:
            slide.paras = [Para(NO_COMBINATION)]
        cited += [c for k in kept_here for c in k.cites]
        slides.append(finish(slide))

    # 8. Disease indications: the diseases the findings are about, what the report says about prevention, and a blank for the rest.
    entry = entries["indications"]
    named = [u for u in units if u.disease and _located(u, "disease_raw")]
    shown, tail = _listed(_tally(named, lambda u: u.disease), "diseases")
    visual = Visual("stacked_list", shown, "Diseases the verified findings are about") if shown else None
    kept_here, visual, _ = lay_out(entry, _title(entry, mode), "", routed["indications"], visual)
    paras = [k.para() for k in kept_here] + tail
    if not shown and not kept_here:
        paras.insert(0, Para(NO_FINDING))
    paras.append(Para("Unmet need and opportunities: not in the papers read - fill in.", muted=True))
    cited += [c for k in kept_here for c in k.cites]
    slides.append(finish(Slide(
        title=_title(entry, mode), paras=paras, by_rule=True, notes=_notes(kept_here, refs) + [
            "Diseases and their counts are made by code from the verified findings; a disease the paper does not "
            "support is not listed. Unmet need is not in any paper: it is for the presenter to fill in."],
        visual=visual)))

    # 9. Competitive landscape: agents tested in people, from the verified findings; owners are not in the papers.
    entry = entries["competitive"]
    agents = _agents(report, number_of, ref_of)
    note = ("One row per agent, target and disease, from verified findings in human clinical evidence whose agent the "
            "paper supports. Stage/design is the paper's own publication type; the company is not in the papers.")
    if agents:
        table = [["Drug/agent", "Target", "Modality", "Company", "Indication", "Stage/design", "Ref."]]
        table += [[a.agent, a.target or "-", a.modality, FILL_IN, a.disease or "-", a.design, f"[{a.number}]"]
                  for a in agents]
        cited += [a.short for a in agents if a.short]
        slides.append(finish(Slide(title=_title(entry, mode), layout="table", table=table, by_rule=True, notes=[note],
                                   kicker="Company: not in the papers read - fill in")))
    else:
        slides.append(Slide(title=_title(entry, mode), by_rule=True, notes=[note], paras=[
            Para(NO_COMPETITOR), Para("Company, stage and unmet need: not in the papers read - fill in.", muted=True)]))

    # 10. SWOT: strengths and weaknesses from the sentences; opportunities and threats from the report's gaps,
    # disagreements and the agents above. A card shows three items and counts the rest.
    entry = entries["swot"]
    weak_sections = set(_pick(entry.get("sections"), qtype))
    improved = lambda k: any(u.outcome == "improved" for u in k.units)  # noqa: E731
    # Strengths: sentences citing only human findings that improved; with none, any whose findings all improved.
    strong = [k for k in others if k.section not in weak_sections and k.units
              and all(_human(u) and u.outcome == "improved" for u in k.units)]
    strong = strong or [k for k in others if k.section not in weak_sections and k.units
                        and all(u.outcome == "improved" for u in k.units)]
    # Weaknesses: the negative and safety sentences, then any whose findings all had no effect or were worse. A
    # sentence citing a finding that improved is never one: "reduced exacerbations" is a strength, whatever section.
    weak = [k for k in others if k.section in weak_sections and not improved(k)]
    weak += [k for k in others if k.section not in weak_sections and k.units
             and all(u.outcome in ("no_effect", "worsened") for u in k.units)]
    gaps = [b for s in report.sections if s.id in _pick(entry.get("opportunities"), qtype)
            for b in _bullets(s.body_markdown) if b.startswith(_GAP_PHRASE)]
    by_agent: dict[tuple[str, str], list[_Agent]] = defaultdict(list)
    for a in agents:
        by_agent[(a.agent, a.target)].append(a)
    threats = []
    for (agent, target), rows in by_agent.items():
        nums = sorted({r.number for r in rows})
        name = f"{agent} ({target})" if target else agent
        threats.append(_Item(f"{name} tested in people {''.join(f'[{n}]' for n in nums)}", nums,
                             [r.short for r in rows if r.short]))
    for s in report.sections:
        if s.id in _pick(entry.get("threats"), qtype):
            threats += _disagreement_items(s.body_markdown)
    quadrants = [("Strengths", [_item_of(k) for k in strong]), ("Weaknesses", [_item_of(k) for k in weak]),
                 ("Opportunities", [_Item(g, _cited_numbers(g)) for g in gaps]), ("Threats", threats)]
    paras = []
    for label, items in quadrants:
        paras.append(Para(label, bold=True))
        paras += [i.para() for i in items[:_CELL_ITEMS]] or [Para(_EMPTY[label], level=1, muted=True)]
        if len(items) > _CELL_ITEMS:
            paras.append(Para(f"+{len(items) - _CELL_ITEMS} more in the report", level=1, muted=True))
        cited += [c for i in items[:_CELL_ITEMS] for c in i.cites]
    swot_notes = ["Strengths: kept sentences citing only human findings that improved (with none, any whose findings all "
                  "improved). Weaknesses: the report's negative and safety sentences that cite no improved finding, "
                  "then sentences whose findings all had no effect or were worse. Opportunities: measures with no "
                  "tested effect in humans. Threats: agents tested in people, and the disagreements the report lists."]
    if "safety" in weak_sections and any(s.id == "safety" and s.kind == "written" for s in report.sections) \
            and not any(k.section == "safety" for k in kept):
        swot_notes.append(f"No verified safety finding in the papers read. {ABSENCE}")
    slides.append(finish(Slide(title=_title(entry, mode), layout="matrix", paras=paras, by_rule=True, notes=swot_notes,
                               kicker="From the report's sentences, gaps and disagreements")))

    # Candidate targets past `max_target_slides`, and the sentences a slide had no room for: still in the deck, after the SWOT.
    if extra_groups:
        paras, notes = [], []
        for title, ks in extra_groups:
            paras += [Para(title, bold=True), *[k.para() for k in ks]]
            notes += [f"{title} - {line}" for line in _notes(ks, refs)]
            cited += [c for k in ks for c in k.cites]
        slides.append(finish(Slide(title="Appendix: further candidate targets", paras=paras, notes=notes)))
    for title, kicker, rest, shown_count in further:
        cited += [c for k in rest for c in k.cites]
        slides.append(finish(Slide(title=f"Appendix: further findings - {title}", paras=[k.para() for k in rest],
                                   kicker=kicker, notes=_notes(rest, refs, shown_count))))

    # 11. References: every paper the report cites, linked to its record.
    ref_paras = [Para(f"[{r.number}] {plain(r.citation)}", link=r.url, refs=[r.number]) for r in report.references]
    if ref_paras:
        slides.append(Slide(title=_title(entries["references"], mode), layout="references", paras=ref_paras, size=10,
                            notes=["Each reference links to its record (Europe PMC or the publisher)."]))

    # 12. Appendix: each paper's own words, exactly as verified, for every finding cited above.
    shown_ids = list(dict.fromkeys(c for c in cited if c in quotes))
    quote_paras = [Para(f"{c} [{refs[c]}] “{' '.join(quotes[c].split())}”", refs=[refs[c]]) for c in shown_ids]
    if quote_paras:
        slides.append(finish(Slide(title=_title(entries["appendix"], mode), paras=quote_paras, size=12,
                                   kicker="Each paper's own words, exactly as verified; cited above by these ids")))
    return slides


def _pick_figure(kept: list[_Kept], usable: list[FigureAsset], used: set[tuple[str, str]],
                 number_of: dict[str, int]) -> tuple[FigureAsset, int] | None:
    """The first unused downloaded figure that one of these sentences' findings
    points to (the figure the paper cites beside that finding's quote), in the
    order the sentences cite them, and its paper's reference number. Every
    sentence the slide holds counts, including those that overflow to a
    "further findings" slide. A figure is used once, and one whose paper has no
    reference number is not used: its credit line would have nothing to cite."""
    for k in kept:
        for short in k.cites:
            for fig in usable:
                key = (fig.paper_id, fig.fig_id)
                if key in used or short not in fig.evidence_ids or fig.paper_id not in number_of:
                    continue
                used.add(key)
                return fig, number_of[fig.paper_id]
    return None


def to_pptx(report: Report, source: str = "", *, figures: Sequence[FigureAsset] = (), author: str = "",
            date: str = "") -> bytes:
    """The deck as a .pptx file. python-pptx is imported by the renderer, so a
    missing install surfaces here as an ImportError the caller words."""
    from bedrock.litreview.report import deck_render

    return deck_render.render(outline(report, source, figures=figures, author=author, date=date))
