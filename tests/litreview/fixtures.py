"""Small synthetic corpus and a scripted fake model, shared by the tests.

The papers are invented, and say so. They exist to exercise each path: a
primary psoriasis paper with a positive and a negative result, a paper about
a near-miss disease, a review, an off-topic paper, and one that mentions
psoriasis only in passing.
"""

from __future__ import annotations

import json
import re

from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import Paper, PaperSignals

QUESTION = "If I want to prevent psoriasis, what areas should I target?"


def corpus() -> list[Paper]:
    return [
        Paper(
            id="P1",
            doi="10.1000/p1",
            title="IL-23 blockade prevents psoriasis-like skin inflammation in mice",
            abstract=(
                "IL-23 drives psoriasis through the Th17 pathway in skin. Anti-IL-23 antibody given before imiquimod challenge reduced "
                "ear thickness by 60% in C57BL/6 mice. Blocking IL-17A had no significant effect on "
                "epidermal thickness (p=0.41)."
            ),
            authors=["Smith J", "Lee K"],
            signals=PaperSignals(journal="J Invest Dermatol", year=2020, citation_count=40),
            sections={
                "methods": "Female C57BL/6 mice (n=12 per group) received imiquimod cream daily.",
                "results": (
                    "Anti-IL-23 antibody given before imiquimod challenge reduced ear thickness by 60% "
                    "in C57BL/6 mice.\n\nBlocking IL-17A had no significant effect on epidermal "
                    "thickness (p=0.41)."
                ),
            },
            text_source="europepmc_xml",
        ),
        Paper(
            id="P2",
            doi="10.1000/p2",
            title="Guselkumab in plaque psoriasis: a randomized placebo-controlled trial",
            abstract=(
                "At week 16, PASI-90 was achieved by 73% of patients receiving guselkumab versus "
                "3% receiving placebo (n=837). Guselkumab maintained remission and prevented relapse "
                "in 81% of patients through week 48."
            ),
            authors=["Garcia M"],
            signals=PaperSignals(journal="Lancet", year=2017, citation_count=900, publication_types=["Randomized Controlled Trial"]),
        ),
        Paper(
            id="P3",
            doi="10.1000/p3",
            title="IL-4 receptor blockade in atopic dermatitis",
            abstract="Dupilumab improved eczema scores in atopic dermatitis patients (n=671).",
            signals=PaperSignals(journal="NEJM", year=2016),
        ),
        Paper(
            id="P4",
            doi="10.1000/p4",
            title="Therapeutic targets in psoriasis: a review",
            abstract="We review IL-23, IL-17 and TNF as targets in psoriasis.",
            signals=PaperSignals(journal="Nat Rev Drug Discov", year=2021, publication_types=["Review"]),
        ),
        Paper(
            id="P5",
            doi="10.1000/p5",
            title="Gut microbiota in type 2 diabetes",
            abstract="Microbial diversity was reduced in type 2 diabetes (n=120).",
            signals=PaperSignals(journal="Cell", year=2019),
        ),
        Paper(
            id="P6",
            doi="10.1000/p6",
            title="Keratinocyte metabolism in wound healing",
            abstract="Glycolysis increased in wound-edge keratinocytes.",
            sections={"results": "Unlike psoriasis, wound healing resolved within days."},
            signals=PaperSignals(journal="J Cell Biol", year=2018),
        ),
    ]


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i == -1:
        return ""
    i += len(start)
    j = text.find(end, i)
    return text[i:j if j != -1 else None].strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def responder(prompt: str, schema: dict | None) -> str:
    title = (schema or {}).get("title")
    if title == "ParsedQuestion":
        return json.dumps({
            "question_type": "target_identification", "goal": "prevent",
            # "eczema" is not in the question: the guard must discard it.
            "entities": [{"text": "psoriasis", "kind": "disease"}, {"text": "eczema", "kind": "disease"}],
            "population": None,
        })
    if title == "RelevanceJudgement":
        paper_title = _between(prompt, "Title:", "\n")
        abstract = _between(prompt, "Abstract:", "## Answer each field")
        first = _sentences(abstract)[0] if abstract else paper_title
        on_topic = "IL-23" in paper_title or "Guselkumab" in paper_title
        return json.dumps({
            "quote": first, "reason": "fake", "about_exact_entity": on_topic,
            "subject_role": "studied" if on_topic else "passing", "study_kind": "primary_research",
            "tested": "IL-23" if on_topic else "",
        })
    if title == "ExtractionResult":
        text = _between(prompt, "Section:", "Now extract the findings")
        text = text.split("\n", 1)[1] if "\n" in text else text
        findings = []
        for s in _sentences(text):
            if "reduced ear thickness" in s:
                findings.append(dict(quote=s, target="IL-23", direction="block", modality="antibody",
                                     intervention="anti-IL-23 antibody", disease="psoriasis-like inflammation",
                                     evidence_type="in_vivo", species="C57BL/6 mice", outcome="improved",
                                     setting="prevention", effect_detail="ear thickness reduced by 60%", n=None,
                                     combo_with=None, target_is_primary=True, facet="prevention"))
            if "no significant effect" in s:
                findings.append(dict(quote=s, target="IL-17A", direction="block", modality="antibody",
                                     intervention=None, disease="psoriasis-like inflammation",
                                     evidence_type="in_vivo", species="mice", outcome="no_effect",
                                     setting="unclear", effect_detail="no significant effect, p=0.41", n=None,
                                     combo_with=None, target_is_primary=True, facet="negative"))
            if "PASI-90" in s:
                findings.append(dict(quote=s, target="IL-23", direction="block", modality="antibody",
                                     intervention="guselkumab", disease="plaque psoriasis",
                                     evidence_type="human_clinical", species="patients", outcome="improved",
                                     setting="treatment", effect_detail="PASI-90 73% vs 3% placebo at week 16",
                                     n=837, combo_with=None, target_is_primary=True, facet="clinical"))
        # A fabricated quote, and a finding about a near-miss disease: both
        # must be discarded by code.
        findings.append(dict(quote="IL-23 blockade cured every patient in the study within one week.",
                             target="IL-23", direction="block", modality="antibody", intervention=None,
                             disease="psoriasis", evidence_type="human_clinical", species="patients",
                             outcome="improved", setting="treatment", effect_detail="cured all", n=None,
                             combo_with=None, target_is_primary=True, facet="clinical"))
        if findings and _sentences(text):
            findings.append(dict(quote=_sentences(text)[0], target="IL-4R", direction="block", modality="antibody",
                                 intervention="dupilumab", disease="atopic dermatitis", evidence_type="human_clinical",
                                 species="patients", outcome="improved", setting="treatment",
                                 effect_detail="n/a", n=None, combo_with=None, target_is_primary=False,
                                 facet="clinical"))
        return json.dumps({"findings": findings, "is_review": False})
    if title == "DraftSection":
        ids = schema["properties"]["sentences"]["items"]["properties"]["citations"]["items"]["enum"]
        evidence = _between(prompt, "## Evidence you may use", "## Rules")
        first_quote = _between(evidence, 'Quote: "', '"\n')
        return json.dumps({"sentences": [
            {"citations": [ids[0]], "text": f"In the cited study, {first_quote[:1].lower()}{first_quote[1:]}"},
            {"citations": [ids[0]], "text": "This reduced disease activity by 99% in every model tested."},
            {"citations": [ids[0]], "text": "Secukinumab showed the same benefit."},
        ]})
    return "{}"


def fake_client() -> FakeLLMClient:
    return FakeLLMClient(responder=responder, version="fake-1")
