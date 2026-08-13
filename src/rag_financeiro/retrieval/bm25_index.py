import re
import unicodedata

from rank_bm25 import BM25Okapi

from rag_financeiro.vector_store import chroma_store

_index = None
_ids: list[str] = []
_documents: list[str] = []
_metadatas: list[dict] = []


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.lower()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", _normalize(text))


def _get_index() -> BM25Okapi:
    global _index, _ids, _documents, _metadatas
    if _index is None:
        data = chroma_store.get_all()
        _ids = data["ids"]
        _documents = data["documents"]
        _metadatas = data["metadatas"]
        tokenized_corpus = [_tokenize(doc) for doc in _documents]
        _index = BM25Okapi(tokenized_corpus)
    return _index


def search(query: str, k: int) -> list[dict]:
    index = _get_index()
    scores = index.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "id": _ids[i],
            "text": _documents[i],
            "metadata": _metadatas[i],
            "score": float(scores[i]),
        }
        for i in ranked
    ]
