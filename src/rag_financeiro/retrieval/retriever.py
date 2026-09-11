from rag_financeiro import config
from rag_financeiro.embeddings.local_embedder import embed_query
from rag_financeiro.observability import tracer
from rag_financeiro.retrieval import bm25_index, fusion, reranker, synonyms
from rag_financeiro.vector_store import chroma_store


def _dense_search(query: str, k: int) -> tuple[list[str], dict[str, dict]]:
    with tracer.start_as_current_span("dense_search") as span:
        query_emb = embed_query(query)
        results = chroma_store.query(query_emb, k)
        ids = results["ids"][0]
        candidates = {
            doc_id: {"text": doc, "metadata": meta}
            for doc_id, doc, meta in zip(ids, results["documents"][0], results["metadatas"][0])
        }
        span.set_attribute("retrieval.dense.k", k)
        span.set_attribute("retrieval.dense.num_hits", len(ids))
        return ids, candidates


def retrieve(query: str, k: int | None = None) -> list[dict]:
    k = k or config.TOP_K

    with tracer.start_as_current_span("retrieve") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.k", k)

        query = synonyms.expand(query)
        span.set_attribute("retrieval.expanded_query", query)

        dense_ids, dense_candidates = _dense_search(query, config.DENSE_TOP_K)

        with tracer.start_as_current_span("bm25_search") as bm25_span:
            bm25_hits = bm25_index.search(query, config.BM25_TOP_K)
            bm25_span.set_attribute("retrieval.bm25.num_hits", len(bm25_hits))
        bm25_ids = [hit["id"] for hit in bm25_hits]

        candidates_by_id = dict(dense_candidates)
        for hit in bm25_hits:
            candidates_by_id.setdefault(
                hit["id"], {"text": hit["text"], "metadata": hit["metadata"]}
            )

        fused_scores = fusion.reciprocal_rank_fusion([dense_ids, bm25_ids])
        pool_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[
            : config.RERANK_POOL_SIZE
        ]

        pool = [{"id": doc_id, **candidates_by_id[doc_id]} for doc_id in pool_ids]

        with tracer.start_as_current_span("rerank") as rerank_span:
            reranked = reranker.rerank(query, pool, top_k=k)
            rerank_span.set_attribute("retrieval.rerank.pool_size", len(pool))
            rerank_span.set_attribute(
                "retrieval.rerank.top_score", reranked[0]["rerank_score"] if reranked else 0.0
            )

        span.set_attribute("retrieval.num_results", len(reranked))

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
