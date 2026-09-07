from pathlib import Path

from rag_financeiro import config
from rag_financeiro.embeddings.local_embedder import embed_documents
from rag_financeiro.ingestion.chunking import chunk_document
from rag_financeiro.ingestion.pdf_loader import load_document
from rag_financeiro.vector_store import chroma_store


def ingest_pdf(pdf_path: Path) -> int:
    doc = load_document(str(pdf_path))
    chunks = chunk_document(doc)
    for c in chunks:
        c["source"] = pdf_path.name
    embeddings = embed_documents([c["text"] for c in chunks])
    return chroma_store.add_chunks(chunks, embeddings)


def ingest_all() -> int:
    total = 0
    for pdf_path in sorted(config.DATA_DIR.glob("*.pdf")):
        total += ingest_pdf(pdf_path)
    return total
