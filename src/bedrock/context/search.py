"""Fold stored words and queries for keyword ranking over stored chunks.

The store indexes these folded terms in SQLite FTS5, where BM25 ranks matching
chunks. The Python BM25 implementation remains as a fallback if FTS5 is not
available. Moving ranking to disk fixes search speed and memory, not meaning:
"acidity" still will not find "pH". See NEXT_STEPS.md section 6 for embeddings.

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


# Singular words that end in "s" and must never be shortened. Words ending in
# "ss", "us" or "is" are already safe via the suffix guard below (glass,
# apparatus, analysis); these are the ones that would otherwise be stripped.
# It matters more than it looks: auto_inject searches on every outgoing
# message, so "news" matching text that only says "new" would quietly put the
# wrong passage in front of the model.
_NEVER_FOLD = frozenset("news lens bias series species atlas alias canvas gas".split())


def stem(word: str) -> str:
    """Fold regular plurals, so "samples" and "sample" rank as one term.

    Regular plurals only. The limits are worth stating plainly, because the
    alternative is a full stemmer that nobody here wants to maintain:

    * "-s" folds: samples -> sample. Not after "ss", "us" or "is" (glass,
      apparatus, analysis), nor for the words in ``_NEVER_FOLD``.
    * "-es" after a sibilant folds: batches -> batch, boxes -> box,
      dishes -> dish, glasses -> glass.
    * "-ies" becomes "-y": studies -> study.
    * Irregular plurals are *not* handled. "analyses" never reaches
      "analysis", and "calories" folds to "calory" rather than "calorie".
      Both are self-consistent - text and query fold the same way, so a word
      still finds itself - they simply never meet their singular form.
    * Verb forms ("stored"/"storage") are left alone rather than folded
      half-correctly.

    Deliberately not ``bedrock.recall``'s stemmer: that one strips "es"
    before "s", so "samples" becomes "sampl" while "sample" stays "sample"
    and the two never meet. recall matches stems as *prefixes*, where that
    mismatch never shows; here the comparison is equality.
    """
    if word in _NEVER_FOLD:
        return word
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
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
