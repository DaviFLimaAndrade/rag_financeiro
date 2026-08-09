from langchain_huggingface import HuggingFaceEmbeddings
from rag_financeiro import config

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed_documents(texts)

def embed_query(text: str) -> list[float]:
    return get_embedder().embed_query(text)