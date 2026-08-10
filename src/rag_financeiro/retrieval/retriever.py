from rag_financeiro import config
from rag_financeiro.embeddings.local_embedder import embed_query
from rag_financeiro.vector_store import chroma_store


def retrieve(query: str, k: int | None = None) -> list[dict]:
    k = k or config.TOP_K
    query_emb = embed_query(query)
    results = chroma_store.query(query_emb, k)
    return [
        {
            "text": doc,
            "page_no": meta.get("page_no"),
            "section": meta.get("section"),
            "source": meta.get("source"),
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
