import os

# Desabilita o torch.compile (Dynamo/Inductor) -- ver run_comparison.py para detalhes.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

DATA_DIR = "./data/raw"
os.makedirs(DATA_DIR, exist_ok=True)


def load_and_chunk_pdfs(directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Lê todos os PDFs de um diretório usando o Docling (IBM Research), que entende
    layout, ordem de leitura e ESTRUTURA DE TABELAS -- em vez de tratar o PDF como
    texto corrido, como fazia o PyPDFLoader + RecursiveCharacterTextSplitter.

    Isso evita que tabelas (ex: indicadores como ROE, Índice de Basileia, Selic)
    sejam quebradas no meio durante o chunking, separando o número do rótulo
    que dá contexto a ele -- a causa raiz de vários erros observados no eval.

    Mantém a mesma assinatura da função anterior para não quebrar embeddings.py.
    chunk_overlap não é usado diretamente aqui (o HybridChunker do Docling gerencia
    isso internamente de forma estrutural), mas fica no parâmetro por compatibilidade.
    """
    converter = DocumentConverter()
    chunker = HybridChunker(max_tokens=chunk_size)

    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"Nenhum documento encontrado. Adicione PDFs na pasta '{directory_path}'.")
        return []

    all_chunks = []

    for file in pdf_files:
        file_path = os.path.join(directory_path, file)
        print(f"Carregando (Docling): {file}")

        result = converter.convert(file_path)
        docling_doc = result.document

        for chunk in chunker.chunk(docling_doc):
            # contextualize() inclui cabeçalhos/seção ao redor do chunk, além do
            # próprio conteúdo (com tabelas preservadas em formato estruturado)
            text = chunker.contextualize(chunk)

            page_numbers = set()
            for item in chunk.meta.doc_items:
                for prov in item.prov:
                    page_numbers.add(prov.page_no)
            page = min(page_numbers) if page_numbers else None

            all_chunks.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page},
                )
            )

    print("-" * 30)
    print(f"Total de PDFs processados: {len(pdf_files)}")
    print(f"Total de chunks gerados (table-aware): {len(all_chunks)}")
    print("-" * 30)

    return all_chunks


if __name__ == "__main__":
    chunks_gerados = load_and_chunk_pdfs(DATA_DIR)

    if chunks_gerados:
        print(f"\nExemplo do primeiro chunk gerado:\n\n{chunks_gerados[0].page_content[:300]}...")