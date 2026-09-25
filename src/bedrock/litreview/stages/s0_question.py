"""Stage 0 — read the question. Frozen once a run starts.

"If I want to prevent psoriasis, what should I target?" becomes a
`QuestionSpec`: the question type, the goal, and each entity with the exact
names that count as it and the near-misses that must not.

Division of labour, as everywhere in this pipeline:

- The model reads the question and **copies** entity names out of it. Each one
  is verified against the question text; a name that is not in the question
  is discarded, so the model cannot quietly widen the search.
- Code **resolves** each name against the knowledge table: "RSV" becomes the
  respiratory syncytial virus entry, with its synonyms (hRSV, human
  orthopneumovirus) and near-misses (hMPV, bovine RSV). An entity the table
  does not know is matched verbatim and nothing else.
- Code also scans the question against the table itself and classifies the
  question type from its wording. Where the rules are sure, they win; the
  model fills in what they cannot tell.

With no model at all, the rules alone produce a usable spec for any question
that names a known entity.
"""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from bedrock.litreview.knowledge import Knowledge, clean, family_near_miss_patterns, fold, load_knowledge
from bedrock.litreview.llm import LLMError, ask_json, load_prompt, strict_schema, version_of
from bedrock.litreview.models import ParsedQuestion, QuestionEntity, QuestionSpec, stable_id

log = logging.getLogger(__name__)

PROMPT = "question"


class QuestionError(ValueError):
    """The question does not name anything the pipeline can gate on."""


_TARGET_Q = re.compile(
    r"\b(what|which)\b[^?]*\b(target\w*|pathways?|genes?|proteins?|molecules?|receptors?|mechanisms? to)\b"
    r"|\bshould (i|we) (target|focus|go after|inhibit|block)\b|\bdruggable\b|\btarget (areas?|candidates?)\b",
    re.I,
)
_YES_NO = re.compile(r"^\s*(does|do|did|is|are|can|could|will|would|has|have)\b", re.I)
_SAFETY_Q = re.compile(r"\b(safe|safety|side[\s-]effects?|adverse|toxicit\w*|tolerab\w*|risks? of)\b", re.I)
_MECH_Q = re.compile(r"\b(mechanism\w*|how (does|do|is|are)|why (does|do)|role of)\b", re.I)
_PREVENT = re.compile(
    r"\b(prevent\w*|prophyla\w*|avoid\w*|protect\w* against|reduce the risk|risk of developing|onset)\b", re.I
)
_TREAT = re.compile(r"\b(treat\w*|therap\w*|cure|manag\w*|remission|reverse)\b", re.I)
_SUBJECT_AFTER_VERB = re.compile(
    r"\b(?:prevent|treat|cure|stop|avoid|against|for)\s+(?:the\s+|an?\s+)?([^,?;.]+?)(?=\s*(?:,|\?|;|\.|$|\bwhat\b|\bwhich\b|\bin\b|\bwith\b))",
    re.I,
)


def classify(text: str, entity_kinds: set[str]) -> tuple[str | None, str | None]:
    """Question type and goal from wording alone. None where unsure."""
    goal = "prevent" if _PREVENT.search(text) else "treat" if _TREAT.search(text) else None
    has_agent = bool(entity_kinds & {"drug", "target"})
    if has_agent and _YES_NO.match(text):
        qtype = "safety" if _SAFETY_Q.search(text) else "intervention_efficacy"
    elif _TARGET_Q.search(text):
        qtype = "target_identification"
    elif _SAFETY_Q.search(text):
        qtype = "safety"
    elif _MECH_Q.search(text):
        qtype = "mechanism"
    elif has_agent:
        qtype = "intervention_efficacy"
    else:
        qtype = None
    return qtype, goal


def _from_table(text: str, known, kb: Knowledge) -> QuestionEntity:
    return QuestionEntity(
        text=text,
        kind=known.kind,  # type: ignore[arg-type]
        canonical=known.canonical,
        exact_terms=list(dict.fromkeys([*known.terms])),
        near_miss_terms=list(known.near_miss),
        near_miss_patterns=list(known.near_miss_patterns),
        resolved_from="knowledge_table",
    )


def resolve_entity(text: str, kind: str, kb: Knowledge) -> QuestionEntity:
    """A name copied from the question -> the entity, with its exact terms."""
    known = kb.lookup(text)
    if known is None:
        inside = kb.scan(text, ("disease", "pathogen", "drug", "target"))
        if len({e.canonical for e, _ in inside}) == 1:
            known = inside[0][0]
    if known is not None:
        return _from_table(text, known, kb)
    return QuestionEntity(
        text=text,
        kind=kind,  # type: ignore[arg-type]
        canonical=clean(text).strip(),
        exact_terms=[clean(text).strip()],
        near_miss_patterns=family_near_miss_patterns(text),
        resolved_from="verbatim",
    )


def _in_question(span: str, question: str) -> bool:
    return bool(span) and fold(span) in fold(question) and len(fold(span)) >= 2


def parse_question(
    text: str,
    client=None,
    *,
    model: str | None = None,
    knowledge: Knowledge | None = None,
    approved_terms: dict[str, list[str]] | None = None,
) -> QuestionSpec:
    kb = knowledge or load_knowledge()
    question = re.sub(r"\s+", " ", clean(text)).strip()
    if not question:
        raise QuestionError("empty question")

    entities: dict[str, QuestionEntity] = {}

    def add(entity: QuestionEntity) -> None:
        key = entity.canonical.lower()
        if key not in entities:
            entities[key] = entity

    # 1. Code: every table entity named in the question.
    for known, match in kb.scan(question, ("disease", "pathogen", "drug", "target")):
        add(_from_table(match.text, known, kb))

    # 2. Model: names the table does not know, plus type and goal.
    parsed: ParsedQuestion | None = None
    prompt = load_prompt(PROMPT)
    parsed_by = "rules"
    if client is not None and model:
        schema = strict_schema(ParsedQuestion.model_json_schema())
        try:
            data = ask_json(client, prompt.render(question=question), model=model, schema=schema, max_tokens=512)
            parsed = ParsedQuestion.model_validate(data)
            parsed_by = version_of(client, model)
        except (LLMError, ValidationError) as exc:
            log.warning("question model failed, using rules only: %s", exc)

    if parsed is not None:
        for pe in parsed.entities:
            if not _in_question(pe.text, question):
                # The fabrication guard, applied to the question: a name the
                # scientist did not write does not get to steer the search.
                log.info("discarding entity not present in the question: %r", pe.text)
                continue
            add(resolve_entity(pe.text, pe.kind, kb))

    # 3. Rules fallback for an unknown subject with no model to find it.
    if not any(e.kind in ("disease", "pathogen", "drug", "target") for e in entities.values()):
        m = _SUBJECT_AFTER_VERB.search(question)
        if m and parsed is None:
            add(resolve_entity(m.group(1).strip(), "disease", kb))

    kinds = {e.kind for e in entities.values()}
    rule_type, rule_goal = classify(question, kinds)
    qtype = rule_type or (parsed.question_type if parsed else None) or "other"
    goal = rule_goal or (parsed.goal if parsed else None) or "understand"

    for entity in entities.values():
        if entity.kind in ("disease", "pathogen", "drug"):
            entity.required = True
        elif entity.kind == "target":
            # In "what should I target?" the targets are the answer, not a filter.
            entity.required = qtype != "target_identification"
        else:
            entity.required = False
        extra = (approved_terms or {}).get(entity.canonical) or []
        if extra:
            entity.exact_terms = list(dict.fromkeys([*entity.exact_terms, *extra]))
            entity.resolved_from = "approved"

    if not any(e.required for e in entities.values()):
        raise QuestionError(
            "Could not find a disease, pathogen, drug or target in the question. "
            "Name it explicitly, e.g. 'what should I target to prevent psoriasis?'"
        )

    population = None
    if parsed and parsed.population and _in_question(parsed.population, question):
        population = parsed.population

    ordered = sorted(
        entities.values(),
        key=lambda e: (["disease", "pathogen", "drug", "target", "species", "other"].index(e.kind), e.canonical),
    )
    return QuestionSpec(
        id=stable_id("question", question.lower()),
        text=question,
        question_type=qtype,  # type: ignore[arg-type]
        goal=goal,  # type: ignore[arg-type]
        entities=ordered,
        population=population,
        parsed_by=parsed_by,
        prompt_version=prompt.version if parsed is not None else None,
    )
