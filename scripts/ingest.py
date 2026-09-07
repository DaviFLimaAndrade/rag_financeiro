from rag_financeiro import config
from rag_financeiro.ingestion.pipeline import ingest_pdf

if __name__ == "__main__":
    for pdf_path in sorted(config.DATA_DIR.glob("*.pdf")):
        print(f"Processando {pdf_path.name}...")
        n = ingest_pdf(pdf_path)
        print(f"  {n} chunks salvos.")
