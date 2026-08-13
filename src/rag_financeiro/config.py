import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "data" / "processed" / "chroma_db"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financeiro_rag")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" ou "gemini"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

TOP_K = int(os.getenv("TOP_K", 8))

# Retrieval híbrido: fusão RRF entre busca densa e BM25, reordenada por cross-encoder antes do TOP_K.
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", 20))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 20))
RERANK_POOL_SIZE = int(os.getenv("RERANK_POOL_SIZE", 20))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# Score mínimo do reranker pro melhor candidato; abaixo disso, reescreve a query e tenta de novo.
RETRIEVAL_CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", 0.1))
