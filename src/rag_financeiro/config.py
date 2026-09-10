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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL", "openai/gpt-oss-20b")

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "openrouter")

TOP_K = int(os.getenv("TOP_K", 8))

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 600))

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 4))

DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", 20))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 20))
RERANK_POOL_SIZE = int(os.getenv("RERANK_POOL_SIZE", 10))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

RETRIEVAL_CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", 0.1))

CACHE_PATH = BASE_DIR / "data" / "processed" / "cache.json"
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.85))
CACHE_BUILD_BATCH_SIZE = int(os.getenv("CACHE_BUILD_BATCH_SIZE", 8))

ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID", "")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY", "")
ARIZE_PROJECT_NAME = os.getenv("ARIZE_PROJECT_NAME", "rag-financeiro")
