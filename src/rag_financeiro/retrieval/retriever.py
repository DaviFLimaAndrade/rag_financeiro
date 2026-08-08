import chromadb
from rag_financeiro.embeddings.gemini_embedder import embed_text

client = chromadb.PersistentClient(path="data/processed/chroma_db")
collection = client.get_or_create_collection("res_bcb")

def index_chunks(chunks: list[dict]):
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=[embed_text(c["text"]) for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"page_no": c["page_no"], "section": c["section"] or ""} for c in chunks],
    )

def retrieve(query: str, k: int = 5) -> list[dict]:
    query_emb = embed_text(query, task_type="RETRIEVAL_QUERY")
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    return [
        {"text": doc, "page_no": meta["page_no"], "section": meta["section"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]