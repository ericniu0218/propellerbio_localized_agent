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
manifest and calls into these functions. See NEXT_STEPS.md section 6 for the
planned replacement.

A measured note on the other half of that trade-off, because it is tempting to
try to fix it here: a query whose distinctive word is absent ("how acidic are
the samples") scores *identically* to a genuine one whose distinctive word is
also absent ("what temperature were samples held at") - same score, same
matched term, same idf, because lexically they are the same event. No
threshold on score, matched-term count or rarity separates them. Precision
here needs meaning, not tuning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

K1 = 1.5
B = 0.75

# Bumped whenever tokenisation changes, because term counts are written into
# each artifact's manifest at ingest time. ``bedrock.context.store`` folds
# older manifests forward on read - see ``fold_terms``.
INDEX_VERSION = 2

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


def stem(word: str) -> str:
    """Fold plurals, so "samples" and "sample" rank as one term.

    Deliberately plural-only. A fuller stemmer is tempting but has to fold
    *both* sides of a comparison to the same string, and the usual suffix
    lists do not: strip "es" before "s" and "samples" becomes "sampl" while
    "sample" stays "sample", so the two never meet. (``bedrock.recall`` can
    use exactly that list because it matches stems as prefixes, where the
    mismatch is harmless - here the comparison is equality.) Verb forms like
    "stored"/"storage" are left alone rather than folded half-correctly.

    Words ending in "ss", "us" or "is" keep their "s": glass, apparatus,
    analysis.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase words, minus stopwords and single characters, plurals folded.

    Stopwords are removed before stemming: the list holds real words ("was",
    "is"), not stems.
    """
    return [
        stem(word)
        for word in _WORD.findall(text.lower())
        if len(word) > 1 and word not in _STOPWORDS
    ]


def fold_terms(terms: dict[str, int]) -> dict[str, int]:
    """Bring a manifest written before ``INDEX_VERSION`` 2 up to date.

    Stemming is a pure function of the word, so folding the stored counts
    gives exactly what re-reading the chunk would give - {"samples": 3,
    "sample": 2} becomes {"sample": 5} - without decompressing anything. The
    total count is unchanged, so the stored ``term_count`` stays correct.
    """
    folded: dict[str, int] = {}
    for word, count in terms.items():
        key = stem(word)
        folded[key] = folded.get(key, 0) + count
    return folded


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
