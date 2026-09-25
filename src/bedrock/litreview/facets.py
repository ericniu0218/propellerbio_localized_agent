"""What information each kind of question needs. Code, not model.

A "facet" is one piece of the answer: for "what should I target to prevent X?"
the answer needs the disease biology, targets with causal evidence, human
clinical evidence, prevention-specific evidence, negative results and safety.
These templates decide that — the same way for every question of a type — and
the planner (Stage 4) decides which papers can supply each facet.

This is the part an agent loop would have the model improvise. It is written
down instead, because the scientists' process is the same every time and an
improvised plan is a plan that differs between runs.
"""

from __future__ import annotations

import re

from bedrock.litreview.models import Facet, Paper, QuestionSpec


def subject_label(spec: QuestionSpec) -> str:
    entity = spec.subject
    if entity is None:
        return "the condition asked about"
    label = entity.canonical
    if entity.text and entity.text.lower() != entity.canonical.lower():
        label = f"{entity.canonical} ({entity.text})"
    if entity.kind == "pathogen":
        label += " infection"
    return label


def agent_label(spec: QuestionSpec) -> str:
    entity = spec.first("drug", "target")
    return entity.canonical if entity else "the intervention"


_TEMPLATES: dict[str, list[dict]] = {
    "target_identification": [
        dict(id="mechanism", title="Disease biology", min_sources=2, max_sources=4,
             need="which pathways, cell types and molecules drive {subject}",
             fields=["target", "outcome", "evidence_type"]),
        dict(id="causal", title="Targets with causal evidence", min_sources=3, max_sources=6,
             need="experiments in which blocking, activating or deleting a specific molecule "
                  "changed {subject} — animal models, genetics, human tissue",
             fields=["target", "direction", "species", "outcome", "effect_detail", "n"]),
        dict(id="clinical", title="Human clinical evidence", min_sources=2, max_sources=5,
             need="trials or human studies in which modulating a target changed {subject} "
                  "outcomes, with effect sizes and sample sizes",
             fields=["target", "intervention", "outcome", "effect_detail", "n"]),
        dict(id="prevention", title="Prevention-specific evidence", min_sources=1, max_sources=4,
             need="evidence that acting on a target before onset, early, or during remission "
                  "prevents {subject}, its progression or its recurrence",
             fields=["target", "setting", "outcome", "effect_detail"]),
        dict(id="negative", title="Negative and conflicting results", min_sources=0, max_sources=3,
             need="targets whose modulation did not help, or made {subject} worse",
             fields=["target", "outcome", "effect_detail"]),
        dict(id="safety", title="Safety liabilities", min_sources=1, max_sources=3,
             need="adverse effects or risks of modulating the candidate targets",
             fields=["target", "intervention", "effect_detail"]),
    ],
    "intervention_efficacy": [
        dict(id="clinical", title="Clinical efficacy", min_sources=2, max_sources=5,
             need="human trials or cohorts testing {agent} for {subject}: outcomes, effect sizes, sample sizes",
             fields=["intervention", "outcome", "effect_detail", "n"]),
        dict(id="preclinical", title="Preclinical efficacy", min_sources=1, max_sources=4,
             need="animal or cell studies testing {agent} against {subject}",
             fields=["intervention", "species", "outcome", "effect_detail"]),
        dict(id="mechanism", title="Mechanism of action", min_sources=1, max_sources=3,
             need="how {agent} acts on {subject}", fields=["target", "effect_detail"]),
        dict(id="prevention", title="Prevention-specific evidence", min_sources=1, max_sources=4,
             need="evidence that {agent} given before onset or exposure prevents {subject}",
             fields=["setting", "outcome", "effect_detail"]),
        dict(id="negative", title="Negative and conflicting results", min_sources=0, max_sources=3,
             need="studies in which {agent} did not help, or made {subject} worse",
             fields=["outcome", "effect_detail"]),
        dict(id="safety", title="Safety", min_sources=1, max_sources=3,
             need="adverse events and risks of {agent}", fields=["intervention", "effect_detail", "n"]),
    ],
    "mechanism": [
        dict(id="mechanism", title="Pathways and cell types", min_sources=2, max_sources=5,
             need="which pathways and cell types drive {subject}", fields=["target", "outcome"]),
        dict(id="causal", title="Key molecules with causal evidence", min_sources=2, max_sources=5,
             need="experiments showing that a specific molecule causes or controls {subject}",
             fields=["target", "direction", "outcome", "effect_detail"]),
        dict(id="negative", title="Conflicting results", min_sources=0, max_sources=3,
             need="results that contradict the proposed mechanism of {subject}", fields=["outcome"]),
    ],
    "safety": [
        dict(id="safety", title="Adverse effects", min_sources=2, max_sources=5,
             need="adverse events and risks associated with {agent} in {subject}",
             fields=["intervention", "effect_detail", "n"]),
        dict(id="mechanism", title="Mechanism of toxicity", min_sources=1, max_sources=3,
             need="why {agent} causes the adverse effects seen", fields=["target", "effect_detail"]),
        dict(id="clinical", title="Human data", min_sources=1, max_sources=4,
             need="human trials or cohorts reporting safety outcomes of {agent}",
             fields=["intervention", "outcome", "n"]),
    ],
    "other": [
        dict(id="direct", title="Direct evidence", min_sources=2, max_sources=6,
             need="findings that directly answer the question about {subject}",
             fields=["target", "outcome", "effect_detail"]),
        dict(id="mechanism", title="Supporting biology", min_sources=1, max_sources=3,
             need="the biology of {subject} relevant to the question", fields=["target"]),
        dict(id="negative", title="Negative and conflicting results", min_sources=0, max_sources=3,
             need="findings that contradict the others about {subject}", fields=["outcome"]),
    ],
}


def facets_for(spec: QuestionSpec) -> list[Facet]:
    """The facets this question needs. Prevention only when the goal is prevention."""
    subject, agent = subject_label(spec), agent_label(spec)
    out = []
    for raw in _TEMPLATES.get(spec.question_type, _TEMPLATES["other"]):
        if raw["id"] == "prevention" and spec.goal != "prevent":
            continue
        body = dict(raw)
        body["need"] = body["need"].format(subject=subject, agent=agent)
        out.append(Facet(**body))
    return out


# Which information needs a paper can supply, read from its title and abstract
# by code. A model asked this ticks every need in the order they are listed;
# study-design words are cruder but give the same answer every time, and a
# wrong tag here costs little — the extraction stage decides, finding by
# finding, what a paper actually supplied.
_FACET_WORDS: dict[str, re.Pattern[str]] = {
    "clinical": re.compile(
        r"\b(randomi[sz]ed|placebo|phase\s*(?:i{1,3}|[1-3])[ab]?\b|clinical trials?|double-blind|open-label|"
        r"enrolled|participants|cohort|real-world|patients (?:were|received|treated|who received))", re.I),
    "causal": re.compile(
        r"\b(knock\w*|deficien\w+|deleti\w+|blockade|blocking|inhibit\w*|neutrali[sz]\w+|antagonis\w+|"
        r"agonis\w+|silenc\w+|overexpress\w+|genetic variants?|mendelian|loss-of-function)", re.I),
    "preclinical": re.compile(r"\b(mice|mouse|murine|rats?|in vivo|in vitro|cell lines?|animal models?|cotton rats?)", re.I),
    "prevention": re.compile(
        r"\b(prevent\w*|prophyla\w*|onset|recurren\w+|relapse|maintenance|protect\w*|incidence|immuni[sz]\w+|vaccin\w+)",
        re.I),
    "negative": re.compile(
        r"\b(did not|does not|no significant|not significantly|failed to|not associated|no effect|"
        r"no difference|worsen\w*|paradoxical)", re.I),
    "safety": re.compile(r"\b(adverse|safety|tolerab\w*|side effects?|toxicit\w*|serious infections?)", re.I),
    "mechanism": re.compile(r"\b(pathway|signal\w+|express\w+|cytokines?|mechanis\w+|pathogen\w+|drives?|mediat\w+)", re.I),
    "direct": re.compile(r".", re.S),
}

# Needs that are about an experiment changing something: when a model has
# said what the paper changed, a paper that changed nothing cannot supply them.
_NEEDS_A_TEST = {"causal", "preclinical"}


def paper_facets(paper: Paper, facets: list[Facet], tested: bool | None = None) -> list[str]:
    """The needs this paper's title and abstract can supply.

    `tested` is whether the relevance model named a molecule, pathway or drug
    the paper's own experiments changed (and code found that name in the
    abstract). None means no model graded the paper: words alone decide.
    """
    text = paper.head_text()
    out = []
    for f in facets:
        pattern = _FACET_WORDS.get(f.id)
        if pattern is None or not pattern.search(text):
            continue
        if f.id in _NEEDS_A_TEST and tested is False:
            continue
        out.append(f.id)
    return out
