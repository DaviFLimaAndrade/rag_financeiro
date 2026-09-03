from rag_financeiro import config
from rag_financeiro.cache.builder import build_cache
from rag_financeiro.vector_store import chroma_store

if __name__ == "__main__":
    n_chunks = chroma_store.count()
    if n_chunks == 0:
        print("Coleção vazia — rode scripts/ingest.py antes de construir o cache.")
    else:
        print(f"Construindo cache a partir de {n_chunks} chunks indexados...")
        entries = build_cache()
        print(f"{len(entries)} entradas salvas em {config.CACHE_PATH}")
