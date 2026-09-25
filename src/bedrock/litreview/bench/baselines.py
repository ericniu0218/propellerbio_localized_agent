"""The two chatbot-style answers the pipeline is compared against.

Same local model, same temperature, same papers available — only the method
differs, so the difference in scores is the pipeline's contribution:

  closed_book  the question alone, as a scientist would ask a chatbot
  naive_rag    the most query-similar abstracts pasted in with the question,
               as a scientist would paste papers into a chatbot: no exact-
               entity gate, no relevance grading, no extraction, no quote
               checks, no citation guard
"""

from __future__ import annotations

from bedrock.litreview.lexical import BM25
from bedrock.litreview.llm import load_prompt
from bedrock.litreview.models import Paper

ABSTRACT_CHARS = 1500
# Room for a thinking trace before the answer (MedGemma writes one, ~1,000
# tokens, and Ollama cannot turn it off); the client strips it.
MAX_TOKENS = 3000


def closed_book(client, model: str, question: str) -> str:
    prompt = load_prompt("baseline_closed_book").render(question=question)
    return client.complete(prompt, model=model, schema=None, temperature=0.0, max_tokens=MAX_TOKENS)


def naive_rag(client, model: str, question: str, papers: list[Paper], k: int = 10) -> tuple[str, dict[str, str]]:
    """Returns the answer and the map from its [P#] labels to paper ids."""
    bm25 = BM25({p.id: f"{p.title} {p.abstract or ''}" for p in papers if p.abstract})
    top = [pid for pid, score in bm25.rank(question)[:k] if score > 0]
    by_id = {p.id: p for p in papers}
    id_map, blocks = {}, []
    for i, pid in enumerate(top, 1):
        label = f"P{i}"
        id_map[label] = pid
        p = by_id[pid]
        blocks.append(f"[{label}] {p.title}\n{(p.abstract or '')[:ABSTRACT_CHARS]}")
    prompt = load_prompt("baseline_rag").render(question=question, papers="\n\n".join(blocks))
    return client.complete(prompt, model=model, schema=None, temperature=0.0, max_tokens=MAX_TOKENS), id_map
