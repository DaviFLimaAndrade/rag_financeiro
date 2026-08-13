from rag_financeiro import config
from rag_financeiro.embeddings.local_embedder import embed_query
from rag_financeiro.retrieval import bm25_index, fusion, reranker
from rag_financeiro.vector_store import chroma_store


def _dense_search(query: str, k: int) -> tuple[list[str], dict[str, dict]]:
    query_emb = embed_query(query)
    results = chroma_store.query(query_emb, k)
    ids = results["ids"][0]
    candidates = {
        doc_id: {"text": doc, "metadata": meta}
        for doc_id, doc, meta in zip(ids, results["documents"][0], results["metadatas"][0])
    }
    return ids, candidates


def retrieve(query: str, k: int | None = None) -> list[dict]:
    k = k or config.TOP_K

    dense_ids, dense_candidates = _dense_search(query, config.DENSE_TOP_K)
    bm25_hits = bm25_index.search(query, config.BM25_TOP_K)
    bm25_ids = [hit["id"] for hit in bm25_hits]

    candidates_by_id = dict(dense_candidates)
    for hit in bm25_hits:
        candidates_by_id.setdefault(hit["id"], {"text": hit["text"], "metadata": hit["metadata"]})

    fused_scores = fusion.reciprocal_rank_fusion([dense_ids, bm25_ids])
    pool_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[: config.RERANK_POOL_SIZE]

    pool = [{"id": doc_id, **candidates_by_id[doc_id]} for doc_id in pool_ids]
    reranked = reranker.rerank(query, pool, top_k=k)

    return [
        {
            "text": c["text"],
            "page_no": c["metadata"].get("page_no"),
            "section": c["metadata"].get("section"),
            "source": c["metadata"].get("source"),
            "rerank_score": c["rerank_score"],
        }
        for c in reranked
    ]
