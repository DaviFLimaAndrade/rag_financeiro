from docling.chunking import HybridChunker

def chunk_document(doc, tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    chunker = HybridChunker(tokenizer=tokenizer_model, max_tokens=512)
    chunks = []
    for chunk in chunker.chunk(doc):
        chunks.append({
            "text": chunker.contextualize(chunk),
            "page_no": chunk.meta.doc_items[0].prov[0].page_no,
            "headings": chunk.meta.headings,
            "section": chunk.meta.headings[-1] if chunk.meta.headings else None,
        })
    return chunks