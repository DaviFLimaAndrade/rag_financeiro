# RAG Financeiro — Relatório de Estabilidade Financeira (BCB)

Sistema de RAG (Retrieval-Augmented Generation) em português para consultar o Relatório de
Estabilidade Financeira do Banco Central do Brasil (`data/raw/relatorio_estabilidade_bcb.pdf`).

## Arquitetura

```
PDF -> Docling (parsing/chunking table-aware) -> embeddings locais (bge-m3)
    -> ChromaDB -> geração via Groq ou Gemini -> Streamlit
```

- **Docling** faz o parsing do PDF preservando layout e tabelas. O relatório do BCB tem indicadores
  (ROE, índice de Basileia, Selic etc.) organizados em tabelas — um chunking ingênuo por caracteres
  quebraria essas tabelas no meio, misturando linhas/colunas de indicadores diferentes. O
  `HybridChunker` do Docling é table-aware e mantém cada tabela coesa dentro de um chunk.
- **Embeddings locais** (`sentence-transformers`, modelo `BAAI/bge-m3`, multilingue) rodam a
  indexação inteira sem chamar nenhuma API externa — evita esbarrar em limites de free-tier ao
  reprocessar o PDF.
- **Geração via Groq ou Gemini** (`LLM_PROVIDER` no `.env`, com override por requisição) — só essa
  etapa (e o judge da avaliação) chama API externa.
- **ChromaDB** local (`data/processed/chroma_db`), com `upsert()` por hash do conteúdo do chunk —
  rodar a ingestão de novo não duplica dados.
- **Streamlit** como interface de chat, com seletor de provider na sidebar.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Configure o `.env` (veja `config.py` para todas as variáveis): no mínimo `GEMINI_API_KEY` e/ou
`GROQ_API_KEY`, conforme o `LLM_PROVIDER` escolhido.

## Como rodar

```bash
# 1. Ingestão: parseia o PDF, gera chunks e popula o ChromaDB
python scripts/ingest.py

# 2. Interface de chat
streamlit run app/streamlit_app.py

# 3. Avaliação (LLM-as-judge contra o golden dataset)
python scripts/evaluate.py
```

## Avaliação

`scripts/evaluate.py` roda cada pergunta de `data/golden_dataset_reduzido.jsonl` pelo pipeline
completo e usa um LLM como juiz para comparar a resposta gerada com o `ground_truth`, numa escala
de 1 a 5, com regras de calibração explícitas (ex.: uma recusa honesta quando a informação existe
no documento é sempre nota 2 — falha de retrieval — nunca é "perdoada" por ser honesta). O
resultado é salvo em `eval_results.json` com nota média, taxa de aprovação e acurácia de fonte.
