import json

import numpy as np

from rag_financeiro import config
from rag_financeiro.embeddings.local_embedder import embed_documents, embed_query
from rag_financeiro.observability import tracer

_entries: list[dict] | None = None
_embeddings: np.ndarray | None = None


def _load() -> tuple[list[dict], np.ndarray]:
    global _entries, _embeddings
    if _entries is not None:
        return _entries, _embeddings

    if not config.CACHE_PATH.exists():
        _entries, _embeddings = [], np.empty((0, 0))
        return _entries, _embeddings

    payload = json.loads(config.CACHE_PATH.read_text(encoding="utf-8"))
    _entries = payload.get("entries", [])
    _embeddings = (
        np.array(embed_documents([e["question"] for e in _entries]))
        if _entries
        else np.empty((0, 0))
    )
    return _entries, _embeddings


def warmup() -> None:
    _load()


def lookup(question: str) -> dict | None:
    with tracer.start_as_current_span("cache_lookup") as span:
        entries, embeddings = _load()
        if not entries:
            span.set_attribute("cache.hit", False)
            return None

        query_emb = np.array(embed_query(question))
        scores = embeddings @ query_emb
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        span.set_attribute("cache.best_score", best_score)

        if best_score < config.CACHE_SIMILARITY_THRESHOLD:
            span.set_attribute("cache.hit", False)
            return None

        span.set_attribute("cache.hit", True)
        return {
            **entries[best_idx],
            "match_score": best_score,
            "related": _related(entries, scores, best_idx),
        }


RELATED_COUNT = 3


def _related(entries: list[dict], scores: np.ndarray, best_idx: int) -> list[str]:
    """Perguntas vizinhas no cache, para um hit também render sugestões de continuação."""
    source = entries[best_idx].get("source")
    ranked = sorted(range(len(entries)), key=lambda i: scores[i], reverse=True)
    return [
        entries[i]["question"]
        for i in ranked
        if i != best_idx and entries[i].get("source") == source
    ][:RELATED_COUNT]
