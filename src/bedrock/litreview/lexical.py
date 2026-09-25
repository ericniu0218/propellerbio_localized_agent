"""BM25 — lexical relevance, in plain Python, deterministic.

Used for two things: choosing which papers get a model relevance grade when
there are more than the budget allows, and breaking ties between papers the
model graded the same. It is also what ranks papers when no model is
available at all. It never overrides the model's grade.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from bedrock.litreview.knowledge import clean

STOPWORDS = frozenset(
    """a an and are as at be by can could do does for from had has have how i if in into is it
    its of on or our should so than that the their them then there these they this to was we
    were what when where which who why will with would want area areas best way ways help
    about any some me my you your us""".split()
)

_JOIN = re.compile(r"(?<=[a-z])[-\s](?=\d)")
_TOKEN = re.compile(r"[a-z0-9α-ω]+")


def tokenize(text: str) -> list[str]:
    text = _JOIN.sub("", clean(text).lower())
    out = []
    for tok in _TOKEN.findall(text):
        if tok in STOPWORDS or len(tok) < 2 and not tok.isdigit():
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = {doc_id: Counter(tokenize(text)) for doc_id, text in docs.items()}
        self.len = {doc_id: sum(c.values()) for doc_id, c in self.tf.items()}
        self.avg = (sum(self.len.values()) / len(self.len)) if self.len else 0.0
        df: Counter[str] = Counter()
        for counts in self.tf.values():
            df.update(counts.keys())
        n = len(self.tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str, doc_id: str) -> float:
        counts = self.tf.get(doc_id)
        if not counts or not self.avg:
            return 0.0
        norm = self.k1 * (1 - self.b + self.b * self.len[doc_id] / self.avg)
        total = 0.0
        for term in set(tokenize(query)):
            f = counts.get(term, 0)
            if f:
                total += self.idf.get(term, 0.0) * f * (self.k1 + 1) / (f + norm)
        return round(total, 6)

    def rank(self, query: str) -> list[tuple[str, float]]:
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self.tf]
        return sorted(scored, key=lambda x: (-x[1], x[0]))


def top_passages(query: str, passages: dict[str, str], k: int = 5) -> list[tuple[str, float]]:
    """Best-matching passages for a query. The benchmark uses this to find
    where in the corpus a claim could have come from."""
    if not passages:
        return []
    return [p for p in BM25(passages).rank(query)[:k] if p[1] > 0]
