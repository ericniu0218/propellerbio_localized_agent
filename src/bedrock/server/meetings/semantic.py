"""Finding a meeting by what it meant, not only by the words it used.

The gap this closes was documented in this repo long before it was fixed:
`meeting_transcripts` ranked by keyword overlap, so "what did we decide about
recruitment?" scored **zero** against a meeting that only ever said
"enrollment" — and a clinical team says both, interchangeably, in the same
week. Every meeting was findable only by the exact words somebody happened to
use, which for a product whose whole purpose is institutional memory is close
to the worst failure available.

Measured on the real sentences, with the model this ships:

    recruitment?  vs  "enrollment is 148 of 200 patients"   +0.219
    recruitment?  vs  "the assay plate needs another run"    +0.067
    interim look? vs  "we push the interim analysis to..."   +0.443

**Hybrid, not pure semantic.** Keyword overlap is kept and the two scores are
added. Static embeddings are good at "recruitment means enrollment" and bad at
"G12C is not G12D" — in a biotech company, an exact match on a gene variant
matters more than any amount of topical similarity, and a ranker that blurred
those would be worse than the keyword one it replaced. Keyword finds the
variant; embeddings find the synonym; the sum finds both.

**Static embeddings, no PyTorch.** model2vec distils a sentence transformer
into a fixed table of token vectors, so inference is numpy over a lookup — no
GPU, no torch, about 30 MB, and fast enough to embed a month of meetings
inside a request. That matters more here than the last few points of quality:
this has to run on the same air-gapped box as everything else, and a dependency
that drags in a 2 GB deep-learning stack would not be deployed.

The model is downloaded once and then read from disk. `MEETING_TEXT_MODEL` may
be a local directory, which is how an air-gapped deployment gets one: fetch it
on a machine with a network, copy the directory in, point the variable at it.
Without a model, ranking silently falls back to keyword overlap — the old
behaviour, which was never wrong, only narrow.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    import numpy
    from model2vec import StaticModel

    SEMANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - deployments without the extra
    SEMANTIC_AVAILABLE = False

SEMANTIC_MISSING = (
    "Semantic meeting search is unavailable: model2vec is not installed. "
    'Install it with pip install "bedrock-private-ai[recorder]"; ranking falls back to keyword overlap.'
)

DEFAULT_MODEL = "minishlab/potion-base-8M"

# How much the semantic score counts against keyword overlap. Keyword scores
# are whole numbers of matched terms, so a weight near a single match keeps an
# exact term worth roughly one topical match rather than drowning in it.
SEMANTIC_WEIGHT = 2.0

# Below this, a passage does not count as an answer at all.
#
# This threshold is load-bearing, and not for ranking. Cosine similarity
# between any two English sentences is almost never zero, so without a floor
# *everything* scores above nothing, and the tool loses its ability to say
# "none of this actually matched your question" — the property that stops an
# agent presenting an unrelated meeting as an answer.
#
# Measured against one real meeting transcript:
#
#     "who is chasing site 4?"                 0.291   should match
#     "what did we decide about recruitment?"  0.179   should match (synonym)
#     "quarterly travel budget"                0.124   should NOT match
#     "what did we do about it"                0.062   should NOT match
#
# and, from a second passage, "who is updating the protocol?" against "Priya
# will revise the study document" scored 0.141 — a true match that lands
# *below* an unrelated question on another passage. Static embeddings do not
# separate these cleanly, so the threshold is a choice about which error to
# make. It is set to reject: a miss falls through to "nothing matched
# directly, here is how each meeting opened", which is honest and useful,
# while a false positive hands somebody a meeting about travel budgets and
# calls it an answer. Raising the ceiling here needs a better model, not a
# lower number.
MIN_SEMANTIC_MATCH = 0.15


def model_name() -> str:
    return os.environ.get("MEETING_TEXT_MODEL", DEFAULT_MODEL)


def enabled() -> bool:
    """Off with one variable, for a deployment that wants the old ranking."""
    return SEMANTIC_AVAILABLE and os.environ.get("MEETING_SEMANTIC", "true").lower() != "false"


@lru_cache(maxsize=2)
def _model(name: str):
    """Loaded once per process. Costs a couple of seconds, so never per request."""
    return StaticModel.from_pretrained(name)


def load() -> object | None:
    """The embedding model, or None if this deployment has no usable one.

    Never raises. A meeting search that fails because a model file moved is a
    worse outcome than one that quietly ranks by keyword, which is what this
    did for its whole life until now.
    """
    if not enabled():
        return None
    try:
        return _model(model_name())
    except Exception as exc:
        logger.warning("Semantic search unavailable (%s); falling back to keywords", exc)
        return None


def embed(model, texts: list[str]):
    """Unit-normalised vectors, so a dot product is a cosine similarity."""
    vectors = numpy.asarray(model.encode(texts), dtype=numpy.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = numpy.linalg.norm(vectors, axis=1, keepdims=True)
    # A text that embeds to nothing — punctuation, a cue of pure filler —
    # would divide by zero and poison every comparison with nan.
    norms[norms == 0] = 1.0
    return vectors / norms


def similarities(model, query: str, documents: list[str]):
    """Cosine similarity of one question against many passages."""
    if not documents:
        return []
    query_vector = embed(model, [query])[0]
    return (embed(model, documents) @ query_vector).tolist()
