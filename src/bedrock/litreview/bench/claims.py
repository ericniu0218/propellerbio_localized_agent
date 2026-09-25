"""Turn an answer into claims — one per sentence — with what each one cites."""

from __future__ import annotations

import re

from bedrock.litreview.bench.models import Claim
from bedrock.litreview.models import Report

_ABBREV = re.compile(r"\b(e\.g|i\.e|et al|vs|approx|Fig|Figs|ref|refs|No|Dr|cf)\.", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# "[P3]", "[P1, P4]" and, as the MedGemma baseline also wrote, "(P3)".
_CITE = re.compile(r"[\[(]\s*(P\d+(?:\s*(?:[,;]|and)\s*P\d+)*)\s*[\])]")
_MARKUP = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+|#+\s+)")
MIN_CLAIM_CHARS = 25


def split_sentences(text: str) -> list[str]:
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
    out = []
    for block in re.split(r"\n+", protected):
        block = _MARKUP.sub("", block).replace("**", "").strip()
        if not block:
            continue
        for sentence in _SPLIT.split(block):
            sentence = sentence.replace("<DOT>", ".").strip()
            if sentence:
                out.append(sentence)
    return out


def pipeline_claims(report: Report) -> list[Claim]:
    return [
        Claim(id=f"pipeline-{i}", system="pipeline", section=section, text=s.text, citations=s.citations)
        for i, (section, s) in enumerate(report.kept_sentences(), 1)
    ]


def text_claims(text: str, system: str, id_map: dict[str, str] | None = None) -> list[Claim]:
    """Claims from free text. `[P3]` or `(P3)` markers become citations via `id_map`."""
    claims = []
    for i, sentence in enumerate(split_sentences(text), 1):
        cited = []
        for group in _CITE.findall(sentence):
            for token in re.split(r"\s*(?:[,;]|and)\s*", group):
                if id_map and token in id_map:
                    cited.append(id_map[token])
        clean = _CITE.sub("", sentence).replace(" .", ".").strip()
        if len(clean) < MIN_CLAIM_CHARS or clean.endswith(":"):
            continue
        claims.append(Claim(id=f"{system}-{i}", system=system, text=clean, citations=list(dict.fromkeys(cited))))
    return claims
