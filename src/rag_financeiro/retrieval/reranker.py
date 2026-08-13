from sentence_transformers import CrossEncoder

from rag_financeiro import config

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANKER_MODEL)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]
