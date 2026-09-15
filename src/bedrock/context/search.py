"""Keyword ranking (BM25) over stored chunks.

Why BM25 rather than embeddings: it needs no model, no GPU and no extra
service, it is a few dozen lines you can read in one sitting, and for the kind
of lookup this store exists for - "which chunk mentions pH?" - it is genuinely
good. BM25 is the standard bag-of-words ranking function: a chunk scores
higher when it uses a rare query word often, and the score is damped so that a
very long chunk does not win just by being long.

The trade-off is honest and worth knowing: it matches *words*, not *meaning*.
A search for "acidity" will not find a chunk that only ever says "pH". If that
becomes the limiting factor, this module is the one to replace, and nothing
else has to change - the store keeps the word counts it needs in each
manifest and calls into these functions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

K1 = 1.5
B = 0.75

_WORD = re.compile(r"[a-z0-9_]+")

# Words too common to help ranking. Kept short on purpose: an over-eager
# stopword list quietly makes real queries unanswerable.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how in into is it its of on
    or that the their there they this to was were what when where which who
    will with you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words, minus stopwords and single characters."""
    return [
        word
        for word in _WORD.findall(text.lower())
        if len(word) > 1 and word not in _STOPWORDS
    ]


def term_frequencies(text: str) -> dict[str, int]:
    """How often each word appears - this is what gets stored in the manifest."""
    counts: dict[str, int] = {}
    for word in tokenize(text):
        counts[word] = counts.get(word, 0) + 1
    return counts


@dataclass(frozen=True)
class ScoredDoc:
    """A document in the ranking sense: one chunk, reduced to its word counts."""

    key: tuple
    terms: dict[str, int]
    length: int


def bm25(query: str, docs: list[ScoredDoc], *, limit: int = 5) -> list[tuple[tuple, float]]:
    """Rank ``docs`` against ``query``; returns ``(key, score)``, best first.

    Documents that match none of the query words are dropped rather than
    returned with a zero score, so an unmatched search says "nothing found"
    instead of handing back arbitrary chunks.
    """
    query_terms = tokenize(query)
    if not query_terms or not docs:
        return []

    total = len(docs)
    avg_length = sum(doc.length for doc in docs) / total or 1.0

    # How rare each query word is across the corpus. Rare words carry the
    # ranking; a word in every chunk contributes almost nothing.
    idf: dict[str, float] = {}
    for term in set(query_terms):
        containing = sum(1 for doc in docs if term in doc.terms)
        if containing:
            idf[term] = math.log(1 + (total - containing + 0.5) / (containing + 0.5))

    scored: list[tuple[tuple, float]] = []
    for doc in docs:
        score = 0.0
        for term in query_terms:
            frequency = doc.terms.get(term)
            if not frequency or term not in idf:
                continue
            norm = 1 - B + B * (doc.length / avg_length)
            score += idf[term] * (frequency * (K1 + 1)) / (frequency + K1 * norm)
        if score > 0:
            scored.append((doc.key, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]
