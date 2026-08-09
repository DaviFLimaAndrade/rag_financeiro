import hashlib
import chromadb
from rag_financeiro import config

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = client.get_or_create_collection(
            config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def _chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    collection = get_collection()
    ids = [_chunk_id(c["text"]) for c in chunks]
    metadatas = [
        {
            "page_no": c.get("page_no") or 0,
            "section": c.get("section") or "",
            "source": c.get("source") or "",
        }
        for c in chunks
    ]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=metadatas,
    )
    return len(ids)



def query(query_embedding: list[float], k: int) -> dict:
    return get_collection().query(query_embeddings=[query_embedding], n_results=k)

def count() -> int:
    return get_collection().count()