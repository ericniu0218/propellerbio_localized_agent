"""What one verified finding can and cannot be used to claim. Code only.

A quote that is really in the paper can still be counted for the wrong thing.
The tirzepatide/liver-fat report of 29 September showed five ways:

* "HFHFr feeding induced ... marked hepatic steatosis" counted as tirzepatide
  *worsening* liver fat - the diet caused it; the drug was the treatment;
* "no statistically significant differences ... between any of the genotype
  subgroups" counted as tirzepatide having *no effect* - it compares
  subgroups, not treatment with no treatment;
* "Between-group differences were statistically significant for both CAP and
  LSM" credited to tirzepatide - both arms took tirzepatide, and the arms
  differed by diet;
* mortality and major adverse liver outcomes filed under "liver fat" - the
  finding's target was the disease, so it matched the subject;
* "HR, 0.66; 95% CI, 0.58-0.42" used as a result - the interval is reversed.

Each is decided here from the quote, the finding's own fields and the
knowledge tables, so the synthesis counts a finding only for what it
supports, and says what it left out and why. Nothing is rewritten: a finding
set aside here stays in the evidence table and the quotes, with its reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from bedrock.litreview.knowledge import TermMatcher
from bedrock.litreview.knowledge.abbreviations import paper_abbreviations
from bedrock.litreview.models import EvidenceUnit, Paper, QuestionSpec
from bedrock.litreview.report.guard import numeric_view

ENDPOINTS = Path(__file__).resolve().parents[1] / "knowledge" / "endpoints.yaml"
CLAIMS_VERSION = "claims.v2"


# --- endpoint classes ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EndpointClass:
    id: str
    label: str
    kind: str
    subject: str | None
    patterns: tuple[re.Pattern[str], ...]


@lru_cache(maxsize=1)
def endpoint_classes() -> tuple[EndpointClass, ...]:
    data = yaml.safe_load(ENDPOINTS.read_text(encoding="utf-8")) or []
    return tuple(EndpointClass(id=e["id"], label=e["label"], kind=e["kind"], subject=e.get("subject"),
                               patterns=tuple(re.compile(p, re.I) for p in e["patterns"])) for e in data)


# "After adjustment for sex, age, baseline body mass index, and hepatic
# steatosis using ANCOVA, weight loss ..." measured weight loss: the list of
# what was adjusted for names covariates, not the endpoint.
_COVARIATES = re.compile(r"\badjust\w*\s+for\s+(?:[\w\s-]+,\s*)*(?:and\s+)?[\w\s-]+?(?=\s+using\b|[,;).])", re.I)


def endpoint_of(u: EvidenceUnit) -> EndpointClass | None:
    """The kind of endpoint a finding's quote reports, or None when no class fits."""
    full = f"{u.quote}\n{u.target_raw or ''}"
    for text in (_COVARIATES.sub(" ", full), full):
        for cls in endpoint_classes():
            if any(p.search(text) for p in cls.patterns):
                return cls
    return None


# --- what kind of comparison ----------------------------------------------------------------------

_SUBGROUP = re.compile(
    r"\b(?:between|across|among)\s+(?:any\s+of\s+)?(?:the\s+)?(?:\w+[\s-]+){0,2}(?:subgroups?|genotypes?|alleles?|"
    r"sexes|age\s+groups?|strata)\b|\bgenotype\s+subgroups?\b|\b(?:CC|CG|GG|AA|AG)\s+(?:vs\.?|versus)\b|"
    r"\bconsistent\s+across\s+subgroups\b", re.I)
# "regardless of the G allele" is not a subgroup comparison: it says the
# treated people's result held in every subgroup, which is the drug's result.
_REGARDLESS = re.compile(r"\b(?:regardless|irrespective)\s+of\s+(?:the\s+)?(?:[\w-]+\s+){0,4}?"
                         r"(?:subgroups?|genotypes?|alleles?|carriers?|status|presence)\b", re.I)
_WITHIN = re.compile(r"\bwithin[- ]group\b|\bfrom\s+baseline\b|\bcompared\s+with\s+baseline\b|\bvs\.?\s+baseline\b|"
                     r"\bbaseline\s+values\b", re.I)
_BETWEEN = re.compile(r"\bbetween[- ]group\b|\bintergroup\b|\bversus\b|\bvs\.?\b|\bcompared\s+(?:with|to)\b|"
                      r"\brelative\s+to\b|\bthan\b|\bplacebo\b", re.I)
_BOTH_ARMS = re.compile(r"\bboth\s+(?:groups|arms)\b|\ball\s+(?:three|two|four|treatment)\s+(?:treatment\s+)?groups\b|"
                        r"\bacross\s+(?:all\s+)?treatment\s+groups\b", re.I)
_EXPOSURE = re.compile(r"\b(?:feeding|diet|fed|induced|induction|model|injury|challenge|exposure|HFD|HFHFr|"
                       r"high[- ]fat|diet-induced)\b", re.I)


def _drug_names(spec: QuestionSpec, paper: Paper | None) -> list[str]:
    """The requested drug's names, plus what this paper abbreviates them to ("TZP")."""
    names = [t for e in spec.entities if e.kind == "drug" for t in e.exact_terms]
    if paper is not None and names:
        matcher = TermMatcher(names)
        names += [short for short, long in paper_abbreviations(paper).items() if matcher.exact_count(long)]
    return names


def other_agents(spec: QuestionSpec, paper: Paper | None, text: str = "") -> set[str]:
    """Other drugs named in `text` or the paper's title and abstract: the
    question drug's known near-misses ("semaglutide" for tirzepatide) and any
    other drug in the knowledge tables. Arms that got them are not arms that
    all got the requested drug."""
    from bedrock.litreview.knowledge import load_knowledge

    asked = {e.canonical.lower() for e in spec.entities if e.kind == "drug"}
    near = {n for e in spec.entities if e.kind == "drug" for n in e.near_miss_terms}
    where = " ".join(x for x in (text, paper.title if paper else "", (paper.abstract or "") if paper else "") if x)
    found = {n.lower() for n in near if TermMatcher([n]).exact_count(where)}
    found |= {ent.canonical.lower() for ent, _m in load_knowledge().scan(where, kinds=("drug",))}
    return found - asked


_BOTH_CHANGED = re.compile(r"\bboth\s+(?:groups|arms)\s+(?:showed|had|achieved|experienced)\s+significant\s+"
                           r"(?:reductions?|improvements?|decreases?|increases?|weight\s+loss)\b|"
                           r"\bsignificant\s+within[- ]group\s+(?:reductions?|improvements?|decreases?|changes?)\b",
                           re.I)


def contrast_of(u: EvidenceUnit, spec: QuestionSpec, paper: Paper | None) -> str:
    """subgroup | co_intervention | within_arm | between_arm | association | unclear.

    `co_intervention`: the requested drug is in every arm compared ("TZP +
    LEKT versus TZP + LCD", "both groups ... tirzepatide"), so a difference
    between arms belongs to what differed, not to the drug. Decided from what
    the arms were: a paper whose other arms got another drug (liraglutide,
    semaglutide) is an active comparison, not a co-intervention, however its
    sentence words "all treatment groups". And what every arm did from its own
    baseline ("Both groups showed significant reductions in body weight") is
    still the drug's within-arm change, counted as such."""
    quote = u.quote
    # "... regardless of the genotype subgroup" says the treated people's
    # result held in every subgroup: the drug's result, not a comparison.
    if _SUBGROUP.search(_REGARDLESS.sub(" ", quote)):
        return "subgroup"
    # "compared with baseline values" compares with the start, not with a
    # control: read before the between-arm words, which it would otherwise match.
    beyond_baseline = _WITHIN.sub(" ", quote)
    names = _drug_names(spec, paper)
    if names:
        arms = re.findall(r"(" + "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))
                          + r")\s*\+\s*[\w-]+", quote, re.I)
        shared = len(arms) >= 2 or (_BOTH_ARMS.search(quote) and re.search(r"\bbetween[- ]group|\bintergroup", quote, re.I))
        if shared and len(arms) < 2 and other_agents(spec, paper, quote):
            shared = False  # the other arms got other drugs: an active comparison
        if shared:
            return "within_arm" if _BOTH_CHANGED.search(quote) and u.outcome in ("improved", "worsened") \
                else "co_intervention"
    if u.outcome == "association":
        return "association"
    if _BETWEEN.search(beyond_baseline):
        return "between_arm"
    if _WITHIN.search(quote) or _BOTH_ARMS.search(quote):
        return "within_arm"
    return "unclear"


NOT_ATTRIBUTED = {
    "exposure": "what the disease model did (diet, induction), not the treatment",
    "other_agent": "another agent's result, not the requested treatment's",
    "unnamed": "does not say the requested treatment caused it (groups described or compared, or an association)",
}


def attribution(u: EvidenceUnit, spec: QuestionSpec, paper: Paper | None) -> str | None:
    """None when this finding is the requested drug's effect as far as code can
    tell; otherwise which kind of finding it is instead (a key of NOT_ATTRIBUTED).

    Only intervention questions have a drug to attribute to. A finding whose
    agent is an exposure ("HFHFr feeding") or another drug, and whose quote
    does not name the requested drug as what was given, is context: the model
    of disease the drug was tested in, not the drug's result."""
    names = _drug_names(spec, paper)
    if not names:
        return None
    matcher = TermMatcher(names)
    agent = (u.intervention or "").strip()
    if agent and matcher.exact_count(agent):
        return None
    if agent and _EXPOSURE.search(agent):
        return "exposure"
    if agent and not matcher.exact_count(u.quote):
        return "other_agent"
    if matcher.exact_count(u.quote):
        return None
    # No agent recorded and the quote does not name the drug: a result that
    # plainly concerns the drug ("LFC fell at week 52") still counts, but not
    # an association between groups of people ("those with T2D were older")
    # or what a disease model did to the animals ("HFD rats vs control rats").
    if _EXPOSURE.search(u.quote):
        return "exposure"
    return "unnamed" if u.outcome == "association" else None


def attributable(u: EvidenceUnit, spec: QuestionSpec, paper: Paper | None) -> bool:
    return attribution(u, spec, paper) is None


# --- one status per finding ---------------------------------------------------------------------------
#
# Every view of the evidence - the synthesis, Key results, the evidence table,
# the written sections, the chat's context and the deck - reads this one
# decision, so a finding cannot be set aside in the summary and reused as a
# "verified finding" somewhere else. The finding itself is never changed: its
# record stays in the evidence table with its status and reasons.

COUNTED, CONTEXT, EXCLUDED = "counted", "context", "excluded"

_LIMITED = re.compile(r"\b(?:remains?|are|is)\s+(?:still\s+)?(?:limited|scarce|lacking|unclear|unknown|"
                      r"poorly\s+understood|not\s+(?:well\s+)?(?:established|understood))\b|"
                      r"\blittle\s+is\s+known\b|\bfurther\s+(?:studies|research|trials)\s+(?:are|is)\s+"
                      r"(?:needed|required|warranted)\b|\bremains?\s+to\s+be\s+(?:determined|established|elucidated)\b",
                      re.I)
_SETTING = re.compile(r"\b(?:study|trial|analysis|cohort)\s+(?:was|were)\s+(?:conducted|performed|carried\s+out)\b|"
                      r"\bdata\s+(?:were|was)\s+(?:retrieved|collected|obtained|extracted)\b|"
                      r"\b(?:were|was)\s+(?:included|enrolled|randomi[sz]ed|recruited|assigned)\b(?![^.]*\b(?:reduc|improv|decreas|increas))",
                      re.I)
_GLUED_HEADING = re.compile(r"[a-z0-9)%](?:Interpretation|Conclusions?|Results|Methods|Background|Findings|"
                            r"Objectives?|Purpose|Significance|Introduction)[A-Z]|\bp\s?(?:Interpretation|Conclusions?)")
_BAD_CHAR = re.compile("[\ufffd\u2370]")
_DANGLING = re.compile(r"\b(?:seen|shown|presented|listed|given|summari[sz]ed|depicted|illustrated)\s+in\s*"
                       r"(?:\(\s*\))?\s*\.?\s*$", re.I)
_UNRESOLVED = re.compile(r"^\s*(?:the\s+\w+(?:\s+\w+)?\s+of\s+)?(?:these|those|such)\s+(?:markers?|changes?|effects?|"
                         r"parameters?|genes?|proteins?|analytes?|abnormalities|lesions|values?|levels?|findings?|"
                         r"variables?|indices|outcomes?)\b", re.I)
_ANIMAL_WORDS = re.compile(r"\b(?:mice|mouse|rats?|murine|rodents?|animals?|zebrafish|pigs?|primates?|monkeys?|"
                           r"cynomolgus|dogs?|rabbits?|hamsters?)\b", re.I)
_CELL_WORDS = re.compile(r"\b(?:cells?|cell\s+lines?|organoids?|hepatocytes?|in\s+vitro|cultured)\b", re.I)
_PEOPLE_WORDS = re.compile(r"\b(?:patients?|participants?|people|adults?|subjects|volunteers|individuals|children|"
                           r"women|men)\b", re.I)

REASONS = {
    "not_a_result": "describes the study (who, where, when, how many), not a result",
    "background": "background or the authors' view (\"data remain limited\"), not this study's result",
    "damaged": "the text was damaged when the paper was read (a heading or character run into the sentence); "
               "re-read the paper, or refresh the papers to fetch it again",
    "dangling": "points at a table or figure that was not read, and states no result itself",
    "unresolved": "refers to something named earlier (\"these markers\") that the quote does not say",
    "anomaly": "a statistic is printed impossibly (see Limitations); not used in any summary",
    "subgroup": "compares subgroups of treated people, not treatment with no treatment",
    "co_intervention": "every arm received the requested treatment; the arms differed by something else",
    **{f"not_attributed:{k}": v for k, v in NOT_ATTRIBUTED.items()},
    "design_corrected": "labelled with a study type its paper contradicts; counted as the paper's own type",
}


@dataclass(frozen=True)
class Assessment:
    status: str
    """counted | context | excluded."""
    reasons: tuple[str, ...] = ()
    """Keys of `REASONS`."""
    evidence_type: str | None = None
    """The evidence type to use instead of the extracted one, when the paper's
    own design contradicts it (a mouse study's finding labelled human)."""

    def text(self) -> str:
        return "; ".join(REASONS[r] for r in self.reasons)


def _type_from_design(u: EvidenceUnit, tier: str | None) -> str | None:
    """The evidence type the paper's design and the finding's own subjects
    say, when the extracted label contradicts them; None when it does not."""
    from bedrock.litreview.report.synthesis import HUMAN_TIERS

    subjects = f"{u.species_raw or ''} {u.quote}"
    if u.evidence_type in ("human_clinical", "human_genetic"):
        if tier in ("animal", "cell") or (_ANIMAL_WORDS.search(u.species_raw or "") and not
                                          _PEOPLE_WORDS.search(u.species_raw or "")):
            return "in_vivo" if (tier == "animal" or _ANIMAL_WORDS.search(subjects)) else "in_vitro"
    if u.evidence_type == "in_vitro" and _ANIMAL_WORDS.search(u.species_raw or "") \
            and not _CELL_WORDS.search(u.species_raw or ""):
        return "in_vivo"
    if u.evidence_type in ("in_vivo", "in_vitro") and tier in HUMAN_TIERS and \
            _PEOPLE_WORDS.search(u.species_raw or "") and not _ANIMAL_WORDS.search(subjects):
        return "human_clinical"
    return None


def assess(u: EvidenceUnit, spec: QuestionSpec, paper: Paper | None, tier: str | None = None) -> Assessment:
    """The one decision on how this finding may be used. `tier` is its
    paper's study tier (`synthesis.Study.tier`) when known."""
    from bedrock.litreview.extract.verify import other_work_reason, reports_result
    from bedrock.litreview.report.synthesis import substantive

    quote = u.quote
    excluded = []
    if _GLUED_HEADING.search(quote) or _BAD_CHAR.search(quote):
        excluded.append("damaged")
    if _DANGLING.search(quote) and not re.search(r"\d", _DANGLING.sub("", quote)):
        excluded.append("dangling")
    if _LIMITED.search(quote) or other_work_reason(quote, u.section or "results"):
        excluded.append("background")
    if _UNRESOLVED.search(quote):
        excluded.append("unresolved")
    # A setting sentence ("was conducted at ...", "276 were included") is not
    # a result - unless it also reports one ("Of 190 randomised, MASH resolved
    # in 62% versus 21%"), judged by the same deliberately wide test the
    # extraction uses, because a false "not a result" withholds real evidence.
    if not substantive(u) or (_SETTING.search(quote) and not reports_result(quote)):
        excluded.append("not_a_result")
    if statistical_anomalies(quote):
        excluded.append("anomaly")
    corrected = _type_from_design(u, tier)
    if excluded:
        return Assessment(EXCLUDED, tuple(dict.fromkeys(excluded)), corrected)
    notes = ("design_corrected",) if corrected else ()
    contrast = contrast_of(u, spec, paper)
    if contrast in ("subgroup", "co_intervention"):
        return Assessment(CONTEXT, (contrast, *notes), corrected)
    why_not = attribution(u, spec, paper)
    if why_not:
        return Assessment(CONTEXT, (f"not_attributed:{why_not}", *notes), corrected)
    return Assessment(COUNTED, notes, corrected)


# --- statistics printed wrongly --------------------------------------------------------------------

_INTERVAL = re.compile(r"\b(?:95\s*%\s*)?(?:CI|UR|CrI|confidence interval)[,:]?\s*\(?\s*(-?\d+(?:\.\d+)?)\s*"
                       r"(?:-|to|,)\s*(-?\d+(?:\.\d+)?)", re.I)
_P = re.compile(r"\bp\s*[=<>≤≥]\s*(\d+(?:\.\d+)?)", re.I)


def statistical_anomalies(quote: str) -> list[str]:
    """Problems in how a quote prints its statistics, shown as printed and
    never repaired. An interval whose lower bound exceeds its upper bound, a
    p-value above 1."""
    view = numeric_view(quote)
    out = []
    for m in _INTERVAL.finditer(view):
        low, high = float(m.group(1)), float(m.group(2))
        if low > high:
            out.append(f"interval printed as {m.group(1)} to {m.group(2)}: the lower bound is above the upper")
    for m in _P.finditer(view):
        if float(m.group(1)) > 1:
            out.append(f"p-value printed as {m.group(1)}, above 1")
    return out


_LABELLED_ESTIMATE = re.compile(r"\b([A-Z][A-Za-z]{2,})\s*\(\s*(?:a?HR|OR|RR)[,:]?\s*(\d+(?:\.\d+)?)")


def estimates(quote: str) -> dict[str, str]:
    """Outcome label -> the ratio a quote gives for it ("MALO" -> "0.50")."""
    out = {}
    for m in _LABELLED_ESTIMATE.finditer(numeric_view(quote)):
        label = m.group(1)
        if label.endswith("s") and label[:-1].isupper():
            label = label[:-1]  # "MALOs" is MALO
        out[label.upper()] = m.group(2)
    return out


# --- sample size, and what it counts -------------------------------------------------------------

_PER_GROUP = re.compile(r"\b(?:in\s+)?each\s+(?:group|arm|cohort)\b|\bper\s+(?:group|arm)\b|\beach\s+treatment\s+arm\b",
                        re.I)


def n_scope(u: EvidenceUnit) -> str:
    """"per group" when the sentence that gives n says it counts each group."""
    for text in (u.context_quote or "", u.quote):
        if u.n is not None and str(u.n) in text.replace(",", "") and _PER_GROUP.search(text):
            return "per group"
    return ""
