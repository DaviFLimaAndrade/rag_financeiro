"""
Configuração centralizada do projeto RAG Financeiro.
Mantém todos os parâmetros ajustáveis em um único lugar,
facilitando manutenção e demonstrando boas práticas de engenharia.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Caminhos ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DIR.mkdir(exist_ok=True)

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# --- Embeddings ---
# Modelo multilingue com bom desempenho em português.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- Vector Store ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financeiro_rag")

# --- LLM (para a etapa de geração, usada depois no retriever) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" ou "gemini"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 8))