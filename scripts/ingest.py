from rag_financeiro import config
from rag_financeiro.ingestion.pdf_loader import load_document
from rag_financeiro.ingestion.chunking import chunk_document
from rag_financeiro.embeddings.local_embedder import embed_documents
from rag_financeiro.vector_store import chroma_store

pdf_files = sorted(config.DATA_DIR.glob("*.pdf"))

if __name__ == "__main__":
    for pdf_path in pdf_files:
        print(f"Processando {pdf_path.name}...")
        doc = load_document(str(pdf_path))
        chunks = chunk_document(doc)
        for c in chunks:
            c["source"] = pdf_path.name

        embeddings = embed_documents([c["text"] for c in chunks])
        n = chroma_store.add_chunks(chunks, embeddings)
        print(f"  {len(chunks)} chunks, {n} salvos.")
