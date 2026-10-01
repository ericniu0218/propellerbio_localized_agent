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
CLAIMS_VERSION = "claims.v1"


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
    r"\bregardless\s+of\s+(?:the\s+)?(?:presence|genotype|sex|age)\b|\bconsistent\s+across\s+subgroups\b", re.I)
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


def contrast_of(u: EvidenceUnit, spec: QuestionSpec, paper: Paper | None) -> str:
    """subgroup | co_intervention | within_arm | between_arm | association | unclear.

    `co_intervention`: the requested drug is in every arm compared ("TZP +
    LEKT versus TZP + LCD", "both groups ... tirzepatide"), so a difference
    between arms belongs to what differed, not to the drug."""
    quote = u.quote
    if _SUBGROUP.search(quote):
        return "subgroup"
    # "compared with baseline values" compares with the start, not with a
    # control: read before the between-arm words, which it would otherwise match.
    beyond_baseline = _WITHIN.sub(" ", quote)
    names = _drug_names(spec, paper)
    if names:
        arms = re.findall(r"(" + "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))
                          + r")\s*\+\s*[\w-]+", quote, re.I)
        if len(arms) >= 2 or (_BOTH_ARMS.search(quote) and re.search(r"\bbetween[- ]group|\bintergroup", quote, re.I)):
            return "co_intervention"
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
