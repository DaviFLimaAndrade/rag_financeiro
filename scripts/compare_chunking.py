"""
Compara duas estratégias de chunking do PDF do relatório de estabilidade financeira:
- naive: extração de texto crua via pypdf + split por caracteres com overlap (como um
  RecursiveCharacterTextSplitter simplificado).
- docling: HybridChunker table-aware, o mesmo usado em produção (scripts/ingest.py).

Isola o chunking como única variável: mesmo embedder local (bge-m3), mesma métrica
(key_fact_recall). Não faz nenhuma chamada de LLM -- roda inteiro local/offline, sem
gastar cota de API. Não popula o ChromaDB de produção; a busca aqui é feita em memória
(cosine similarity via numpy) sobre embeddings descartáveis.

A ideia aqui é explorar mesmo orçamento de caracteres para o LLM, em vez de top-k fixo
"""

import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from rag_financeiro import config
from rag_financeiro.ingestion.pdf_loader import load_document
from rag_financeiro.ingestion.chunking import chunk_document
from rag_financeiro.embeddings.local_embedder import embed_documents, embed_query
from rag_financeiro.evaluation.golden_dataset import load_golden_dataset
from rag_financeiro.evaluation.metrics import key_fact_recall

CHARS_PER_TOKEN_APPROX = 4


def naive_chunk_pdf(pdf_path: str, chunk_size_tokens: int, chunk_overlap_chars: int) -> list[dict]:
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    chunk_size_chars = chunk_size_tokens * CHARS_PER_TOKEN_APPROX
    step = max(chunk_size_chars - chunk_overlap_chars, 1)

    chunks = []
    start = 0
    while start < len(full_text):
        text = full_text[start : start + chunk_size_chars]
        if text.strip():
            chunks.append({"text": text})
        start += step
    return chunks


def docling_chunk_pdf(pdf_path: str) -> list[dict]:
    doc = load_document(pdf_path)
    return chunk_document(doc)


def retrieve_by_budget(
    query_embedding, chunk_embeddings, chunks: list[dict], char_budget: int
) -> list[dict]:
    """Acumula chunks em ordem de similaridade até bater o orçamento de caracteres.

    Em vez de top-k fixo: se um dos dois usa chunks 4x maiores, top-k fixo dá 4x
    mais texto bruto pra ele, o que infla key_fact_recall por volume, não por
    precisão de retrieval. Fixar o orçamento de caracteres iguala o que as duas
    estratégias entregam pro prompt do LLM em produção a base de comparação que
    de fato importa (custo/latência/contexto), não a contagem de chunks em si.
    """
    sims = np.array(chunk_embeddings) @ np.array(query_embedding)
    ranked_idx = np.argsort(-sims)

    selected, total_chars = [], 0
    for i in ranked_idx:
        chunk = chunks[i]
        if selected and total_chars + len(chunk["text"]) > char_budget:
            break
        selected.append(chunk)
        total_chars += len(chunk["text"])
    return selected


def run_strategy(name: str, chunks: list[dict], cases: list[dict], char_budget: int) -> dict:
    print(f"\n{name}: {len(chunks)} chunks gerados, embedando...")
    embeddings = embed_documents([c["text"] for c in chunks])

    recalls = []
    chunks_retrieved_counts = []
    for case in cases:
        query_embedding = embed_query(case["question"])
        retrieved = retrieve_by_budget(query_embedding, embeddings, chunks, char_budget)
        chunks_retrieved_counts.append(len(retrieved))
        kfr = key_fact_recall(case["ground_truth"], retrieved)
        if kfr is not None:
            recalls.append(kfr)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_chunk_chars = sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
    avg_chunks_retrieved = sum(chunks_retrieved_counts) / len(chunks_retrieved_counts)

    return {
        "strategy": name,
        "num_chunks": len(chunks),
        "avg_chunk_chars": round(avg_chunk_chars),
        "avg_chunks_retrieved_per_query": round(avg_chunks_retrieved, 1),
        "avg_key_fact_recall": round(avg_recall, 3),
        "cases_scored": len(recalls),
    }


def main():
    pdf_files = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Nenhum PDF encontrado em {config.DATA_DIR}")
        return

    cases = load_golden_dataset()

    naive_chunks, docling_chunks = [], []
    for pdf_path in pdf_files:
        print(f"Processando {pdf_path.name}...")
        naive_chunks += naive_chunk_pdf(str(pdf_path), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        docling_chunks += docling_chunk_pdf(str(pdf_path))

    avg_docling_chunk_chars = sum(len(c["text"]) for c in docling_chunks) / len(docling_chunks)
    char_budget = round(config.TOP_K * avg_docling_chunk_chars)
    print(f"\nOrçamento de contexto (TOP_K={config.TOP_K} x ~{round(avg_docling_chunk_chars)} chars/chunk do Docling): {char_budget} chars")

    results = [
        run_strategy("naive (pypdf + split por caracteres)", naive_chunks, cases, char_budget),
        run_strategy("docling (table-aware, produção)", docling_chunks, cases, char_budget),
    ]

    print("\n" + "=" * 60)
    print("COMPARAÇÃO DE CHUNKING — key-fact recall (local, sem LLM)")
    print("=" * 60)
    for r in results:
        print(f"\n{r['strategy']}")
        print(f"  Chunks gerados: {r['num_chunks']} (média ~{r['avg_chunk_chars']} chars/chunk)")
        print(f"  Chunks usados por pergunta (mesmo orçamento): {r['avg_chunks_retrieved_per_query']}")
        print(
            f"  Key-fact recall médio: {r['avg_key_fact_recall']*100:.0f}% "
            f"({r['cases_scored']}/{len(cases)} perguntas com fato-chave em negrito)"
        )
    print("=" * 60 + "\n")

    Path("chunking_comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Resultado salvo em chunking_comparison.json")


if __name__ == "__main__":
    main()
