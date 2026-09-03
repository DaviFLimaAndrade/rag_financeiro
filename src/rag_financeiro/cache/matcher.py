import json

import numpy as np

from rag_financeiro import config
from rag_financeiro.embeddings.local_embedder import embed_documents, embed_query

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
    entries, embeddings = _load()
    if not entries:
        return None

    query_emb = np.array(embed_query(question))
    scores = embeddings @ query_emb
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    if best_score < config.CACHE_SIMILARITY_THRESHOLD:
        return None

    return {**entries[best_idx], "match_score": best_score}
