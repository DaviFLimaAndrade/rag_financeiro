# Lastro — publicações do Banco Central (BCB)

[![RAG Retrieval](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DaviFLimaAndrade/rag_financeiro/main/retrieval_badge.json)](.github/workflows/eval.yml)

Sistema de RAG (Retrieval-Augmented Generation) em português para consultar publicações do Banco
Central do Brasil. Toda resposta sai com a página e a seção do documento que a sustentam — daí o
nome.

O projeto é menos sobre "montar um RAG" e mais sobre as decisões que aparecem quando ele precisa
funcionar com documentos reais: PDFs de centenas de páginas cheios de tabela, seis publicações de
datas diferentes disputando a mesma pergunta, e um free tier de API que não aguenta o pipeline de
avaliação. Cada seção abaixo tenta registrar não só o que foi feito, mas por quê — e o que foi
medido para sustentar a escolha.

---

## Sumário

- [Corpus](#corpus)
- [Arquitetura](#arquitetura)
  - [1. Ingestão](#1-ingestão-pdf--chunks-indexados)
  - [2. Retrieval híbrido](#2-retrieval-híbrido)
  - [3. Roteador de intenção](#3-roteador-de-intenção)
  - [4. O grafo (LangGraph)](#4-o-grafo-langgraph)
  - [5. Geração](#5-geração)
  - [6. Cache (CAG)](#6-cache-cag--cache-augmented-generation)
  - [7. Observabilidade](#7-observabilidade)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Setup](#setup)
- [Como rodar](#como-rodar)
- [Avaliação](#avaliação)
- [Experimentos e decisões](#experimentos-e-decisões)
- [Limitações conhecidas](#limitações-conhecidas)

---

## Corpus

Seis publicações do BCB em `data/raw/`, 902 chunks no índice:

| Documento | Cobre | Chunks |
| --- | --- | ---: |
| `relatorio_estabilidade_bcb.pdf` (REF, nov/2024) | Riscos e resiliência do SFN | 223 |
| `bcb_relatorio_estabilidade_2026_05.pdf` (REF, mai/2026) | Idem, edição atual | 291 |
| `bcb_relatorio_politica_monetaria_2026_06.pdf` (RPM, jun/2026) | Projeções de inflação, PIB e Selic | 153 |
| `bcb_ata_copom_279.pdf` (jun/2026) | Decisão de juros | 11 |
| `bcb_ata_copom_280.pdf` (ago/2026) | Decisão de juros | 10 |
| `bcb_caderno_educacao_financeira.pdf` | Orçamento, juros, dívidas — para quem sabe pouco de economia | 214 |

**Por que mais de um documento.** Os relatórios do BCB são escritos para o mercado: um único REF
não sustenta um Q&A aberto, porque a pergunta óbvia do leigo ("quanto está a Selic?") simplesmente
não tem resposta no texto dele. O corpus mistura de propósito três registros — conjuntura (REF/RPM),
decisão de juros (atas) e material didático (Caderno) — para que perguntas de níveis diferentes
caiam em algum documento.

**Por que isso é difícil.** Documentos de datas diferentes convivem no mesmo índice, e dois deles
são a *mesma publicação* em edições distintas: o REF de 2024 e o de 2026 têm seções com a mesma
numeração e até gráficos homônimos (ambos têm um "Gráfico 1.3.6 — PEF - Ciclo econômico", com dados
completamente diferentes). Duas defesas contra isso:

1. O nome do arquivo carrega a data, vai nos metadados de cada chunk e entra no contexto do LLM,
   que é instruído a responder pelo documento mais recente e a dizer de quando é o dado.
2. No golden dataset, as perguntas do REF antigo são ancoradas no enunciado ("No REF de novembro de
   2024, ..."). Sem âncora a pergunta é genuinamente ambígua e não dá pra dizer se o retrieval
   errou — a avaliação estaria medindo ruído.

---

## Arquitetura

```
                    ┌──────────────────────── ingestão (offline) ────────────────────────┐
                    │                                                                    │
   PDF ─► Docling ─►│ HybridChunker (table-aware) ─► bge-m3 (local) ─► ChromaDB + BM25   │
                    │                                                                    │
                    └────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────── consulta (LangGraph) ──────────────────────┐
                    │                                                                    │
 pergunta ─────────►│ condense_query ─► route_intent ─┬─ saudação ─────► resposta fixa ──┼─► resposta
                    │                                 ├─ fora de escopo ► recusa ────────┼─  + página
                    │                                 │                                  │   + seção
                    │                                 └─ documentos ──► cache_lookup     │   + 3 follow-ups
                    │                                                        │           │
                    │                                            ┌───hit─────┴──miss──┐  │
                    │                                            ▼                    ▼  │
                    │                                  generate_from_cache      retrieve ─► generate
                    │                                                              │  ▲   │
                    │                                              score baixo ────┘  │   │
                    │                                                       rewrite_query  │
                    └────────────────────────────────────────────────────────────────────┘
```

### 1. Ingestão (PDF → chunks indexados)

**Docling** (`ingestion/pdf_loader.py`) faz o parsing preservando layout e tabelas. Os relatórios do
BCB organizam os indicadores que mais importam (ROE, índice de Basileia, RWA, Selic) em tabelas — um
chunking ingênuo por caracteres quebraria essas tabelas no meio, misturando linhas e colunas de
indicadores diferentes. O `HybridChunker` (`ingestion/chunking.py`) é table-aware e mantém cada
tabela coesa dentro de um chunk, cortando em fronteira estrutural em vez de a cada N caracteres.

Cada chunk carrega `page_no`, `section` (o último heading da hierarquia) e `source` (o nome do
arquivo). É daí que sai a citação de página e seção em toda resposta.

**Embeddings locais** (`embeddings/local_embedder.py`, `sentence-transformers` com `BAAI/bge-m3`,
multilingue) rodam a indexação inteira sem chamar nenhuma API externa. Reprocessar os 6 PDFs não
consome cota de lugar nenhum — o que importa num projeto que vive de free tier.

**ChromaDB** local (`data/processed/chroma_db`) com `upsert()` por hash SHA-256 do conteúdo do
chunk: rodar `scripts/ingest.py` de novo não duplica dados e é idempotente por construção.

### 2. Retrieval híbrido

`retrieval/retriever.py`, nesta ordem:

| Etapa | O que faz |
| --- | --- |
| `synonyms.expand` | Acrescenta à pergunta o vocabulário que o BCB realmente usa |
| busca densa | bge-m3 contra o ChromaDB, `DENSE_TOP_K=20` |
| BM25 | `retrieval/bm25_index.py`, `BM25_TOP_K=20` |
| fusão | Reciprocal Rank Fusion (`k=60`) sobre as duas listas |
| rerank | Cross-encoder `BAAI/bge-reranker-base` no pool de `RERANK_POOL_SIZE=10`, devolve `TOP_K=8` |

**Por que híbrido.** Busca densa acha paráfrase e conceito; BM25 acha número, sigla e nome próprio
exato — que é metade das perguntas em documento financeiro ("qual foi o RWA Operacional?"). RRF
combina as duas sem precisar calibrar peso entre scores de escalas diferentes: só posição no ranking
importa.

**Por que reranker.** A fusão devolve candidatos plausíveis, mas o cross-encoder lê pergunta e chunk
*juntos* e reordena com muito mais precisão do que similaridade de vetores independentes. O custo é
rodar 10 pares por pergunta, local, sem API.

**Expansão de vocabulário** (`retrieval/synonyms.py`) é o tipo de detalhe que só aparece quando se
mede. O usuário pergunta pela "Selic", mas a ata do Copom escreve *"reduzir a taxa básica de juros
para 14,00% a.a."* e nunca usa a palavra "Selic" na frase da decisão. Sem expandir, o BM25 não casa
e o chunk certo não entra nem no pool do reranker: **score 0,07, fora do top-8**. Com a pergunta
expandida com o vocabulário do BCB, o mesmo trecho sobe para **0,89** — sem reindexar nada.

### 3. Roteador de intenção

Antes de qualquer busca, `routing/` classifica a mensagem em três rotas — `documentos`, `saudacao`
e `fora_de_escopo`. Só a primeira atravessa o pipeline completo; as outras duas respondem com texto
fixo, sem embedding, sem BM25, sem rerank e sem nenhuma chamada de LLM.

**Por que isso existe.** Até então o único filtro de escopo era o system prompt: "me conta uma
piada" custava busca densa + BM25 + cross-encoder + uma chamada de LLM para terminar numa recusa
educada. O roteador move a decisão para antes do gasto — e, de quebra, tira as tentativas de prompt
injection do caminho do gerador, porque elas nunca chegam a virar um prompt.

**Rota segura.** Abaixo de `ROUTER_CONFIDENCE_THRESHOLD` (0.7) a decisão é rebaixada para
`documentos`. Os dois erros não custam a mesma coisa: mandar uma saudação para o pipeline completo
gasta alguns segundos, enquanto recusar uma pergunta legítima sobre o REF entrega ao usuário uma
resposta errada. O limiar existe para que o erro caia sempre no lado barato.

**Plugável.** `ROUTER_PROVIDER` escolhe a implementação, e o default é `atual` — o roteador vem
desligado, com o grafo se comportando exatamente como antes:

| `ROUTER_PROVIDER` | O que faz | Custo por decisão |
| --- | --- | --- |
| `atual` | Baseline: manda tudo para o pipeline completo | zero |
| `heuristica` | Lista de saudações + marcadores de fora-de-escopo | zero |
| `embeddings` | Similaridade contra protótipos de cada rota, com o bge-m3 que já roda no RAG | zero (local) |
| `jev` | [Jev](https://docs.typesafe.ai/models), System One model da TypeSafe: devolve a rota e uma confiança calibrada | US$ 0,042/milhão de tokens de entrada |
| `llm` | LLM generativo classificando a intenção | 1 chamada de API |

Qualquer erro do roteador (timeout, cota, provedor fora do ar) cai na rota segura em vez de
derrubar a resposta. O comparativo entre as implementações está em
[Experimentos](#qual-roteador-de-intenção).

### 4. O grafo (LangGraph)

`generation/rag_chain.py` monta um `StateGraph` com nove nós:

| Nó | Função | Chama LLM? |
| --- | --- | --- |
| `condense_query` | Transforma pergunta de acompanhamento ("e sobre isso?") numa pergunta autocontida usando o histórico | sim, só se houver histórico |
| `route_intent` | Classifica a intenção da mensagem | depende do `ROUTER_PROVIDER` |
| `answer_saudacao` | Responde saudação com texto fixo e sugestões do cache | não |
| `answer_fora_de_escopo` | Recusa pedido fora do escopo | não |
| `cache_lookup` | Busca semântica no cache pré-construído | não |
| `generate_from_cache` | Responde direto do cache | não |
| `retrieve` | Pipeline de busca híbrida | não |
| `rewrite_query` | Reescreve a query quando o retrieval veio fraco | sim |
| `generate` | Monta o contexto e gera a resposta | sim |

O roteamento condicional é onde está a lógica interessante: depois do `retrieve`, se o melhor score
do reranker fica abaixo de `RETRIEVAL_CONFIDENCE_THRESHOLD` (0.1) **e** ainda não houve retry, o
grafo desvia para `rewrite_query` e tenta buscar de novo — uma vez só, para não entrar em loop.

### 5. Geração

**Groq** (`openai/gpt-oss-120b`) como gerador padrão; `openai/gpt-oss-20b`, menor e mais barato, para
a tarefa mecânica de reescrever query. Chamadas ao LLM rodam num `ThreadPoolExecutor` com timeout
explícito (`REQUEST_TIMEOUT_SECONDS=15`), porque um free tier travado é indistinguível de um app
quebrado para o usuário.

**Fallback de provider.** Quando o Groq devolve 429, `llm_provider.py` classifica o erro por
marcadores no texto (`rate limit`, `quota`, `429`, ...) e tenta uma vez no OpenRouter, com timeout
maior (30s), já que modelo free compartilhado é mais lento.

O fallback usa modelos free **pinados** (`OPENROUTER_MODEL` + `OPENROUTER_MODEL_FALLBACKS`,
repassados como a lista `models` do OpenRouter, que aceita no máximo 3). Antes o fallback era o
alias de roteamento `openrouter/free`, e isso causou um bug memorável: o alias às vezes roteava para
`nvidia/nemotron-3.5-content-safety:free` — um classificador de moderação, não um modelo de chat. A
resposta ao usuário virava literalmente `User Safety: safe`. Na avaliação isso aparecia como nota 2
do juiz **com key-fact recall de 100% e fonte correta**: o retrieval acertava e o gerador é que
estragava a resposta. Foi a métrica separada que denunciou — nota baixa com recuperação perfeita não
é um problema de busca.

**Sugestões de continuação.** Cada resposta termina com três perguntas de follow-up, geradas no mesmo
request da resposta (nenhuma chamada extra ao LLM) e ancoradas nas seções que já estão no contexto
recuperado — não em temas inventados. Em caso de hit no cache, que não chama o LLM, as sugestões vêm
das perguntas vizinhas do próprio cache.

**Escopo.** O system prompt restringe o assistente às publicações do BCB: ele recusa pedidos de
código, tradução ou conhecimento geral, ignora instruções embutidas na pergunta que tentem mudar
suas regras, e responde saudações sem fingir que consultou documento. Também é instruído a explicar
siglas (Selic, Copom, IPCA, provisões) em uma frase — simplificar a linguagem em volta dos números,
nunca os números.

### 6. Cache (CAG — Cache-Augmented Generation)

`cache/` guarda um `cache.json` pré-construído com **71 pares pergunta/resposta** estáveis, cada um
com página e fonte. No `cache_lookup`, a pergunta é embedada e comparada por similaridade de cosseno
contra as perguntas do cache; acima de `CACHE_SIMILARITY_THRESHOLD` (0.85) é hit.

Em caso de hit, o pipeline pula embedding denso, BM25 e rerank e responde direto — a latência cai de
segundos para milissegundos nas perguntas mais comuns, que são justamente as que mais se repetem
numa demo.

### 7. Observabilidade

Tracing opcional via **Arize AX** (`observability.py`), com OpenTelemetry/OpenInference: cada etapa
do pipeline vira um span (grafo LangGraph, chamadas de LLM, busca densa, BM25, rerank, cache lookup),
com atributos como query expandida, número de hits e score do topo do rerank. Sem `ARIZE_SPACE_ID` /
`ARIZE_API_KEY` no `.env`, o app roda normalmente, sem tracing.

---

## Estrutura do repositório

```
app/streamlit_app.py              interface de chat (múltiplas conversas, tratamento de erro)
scripts/
  ingest.py                       parseia os PDFs e popula o ChromaDB
  build_cache.py                  gera o cache.json de perguntas estáveis
  evaluate_retrieval.py           avaliação de retrieval — offline, sem API
  build_routing_dataset.py        monta o dataset rotulado de roteamento
  evaluate_routing.py             compara os roteadores de intenção
  evaluate.py                     avaliação de geração — LLM-as-judge
  validate_golden.py              checagem offline do golden dataset
  compare_chunking.py             experimento naive vs. table-aware
  build_retrieval_badge.py        badge a partir de retrieval_results.json
  build_eval_badge.py             badge a partir de eval_results.json
src/rag_financeiro/
  config.py                       todas as variáveis de ambiente com default
  ingestion/                      pdf_loader, chunking, pipeline
  embeddings/local_embedder.py    bge-m3 local
  vector_store/chroma_store.py    ChromaDB com upsert por hash
  retrieval/                      retriever, bm25_index, fusion, reranker, synonyms
  routing/                        roteador de intenção: null, heuristic, embedding, jev, llm
  generation/                     rag_chain (LangGraph), llm_provider (fallback/timeout)
  cache/                          builder, importer, matcher
  evaluation/                     golden_dataset, judge, metrics
  observability.py                tracing OpenTelemetry/Arize
data/
  raw/                            os 6 PDFs
  processed/chroma_db/            índice vetorial (versionado)
  processed/cache.json            71 pares do CAG
  golden_dataset_reduzido.jsonl   26 perguntas de avaliação
  routing_dataset.jsonl           76 mensagens rotuladas por rota
```

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Configure o `.env`. O mínimo para o chat funcionar é `GROQ_API_KEY`.

| Variável | Default | Para quê |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` | Provider de geração (`groq`, `gemini`, `openrouter`) |
| `GROQ_API_KEY` | — | **Obrigatória** para o chat |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Modelo de geração |
| `GROQ_REWRITE_MODEL` | `openai/gpt-oss-20b` | Modelo menor para reescrita de query |
| `FALLBACK_PROVIDER` | `openrouter` | Provider acionado em erro de cota |
| `OPENROUTER_API_KEY` | — | Necessária para o fallback e para o juiz |
| `OPENROUTER_MODEL` | `inclusionai/ling-3.0-flash-fin:free` | Modelo do fallback de geração |
| `OPENROUTER_MODEL_FALLBACKS` | 2 modelos free | Lista `models` do OpenRouter (máx. 3 no total) |
| `JUDGE_PROVIDER` | `openrouter` | Provider do LLM-as-judge |
| `OPENROUTER_JUDGE_MODEL` | `nex-agi/nex-n2.5-pro:free` | Modelo do juiz, pinado à parte do gerador |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embeddings (local) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder do rerank (local) |
| `CHUNK_SIZE` | `1000` | `max_tokens` do HybridChunker |
| `CHUNK_OVERLAP` | `200` | Só usado pelo baseline naive do `compare_chunking.py` |
| `TOP_K` | `8` | Chunks entregues ao LLM |
| `DENSE_TOP_K` / `BM25_TOP_K` | `20` / `20` | Candidatos de cada perna da busca |
| `RERANK_POOL_SIZE` | `10` | Pool que vai para o cross-encoder |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.1` | Abaixo disso, reescreve a query e tenta de novo |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | Limiar de hit no cache |
| `ROUTER_PROVIDER` | `atual` | Roteador de intenção (`atual`, `heuristica`, `embeddings`, `jev`, `llm`) |
| `ROUTER_CONFIDENCE_THRESHOLD` | `0.7` | Abaixo disso, a rota vira `documentos` |
| `ROUTER_TIMEOUT_SECONDS` | `5` | Timeout do roteador que usa API |
| `TYPESAFE_API_KEY` | — | Necessária para `ROUTER_PROVIDER=jev` |
| `JEV_MODEL` | `jev-latest` | Modelo do System One |
| `LLM_MAX_TOKENS` | `700` | Teto da resposta |
| `MAX_HISTORY_TURNS` | `4` | Turnos de histórico no condense |
| `ARIZE_SPACE_ID` / `ARIZE_API_KEY` | — | Tracing opcional |

---

## Como rodar

```bash
# 1. Ingestão: parseia os PDFs, gera chunks e popula o ChromaDB
python scripts/ingest.py

# 2. Interface de chat
streamlit run app/streamlit_app.py

# 3. Avaliação offline (sem nenhuma chamada de API)
python scripts/validate_golden.py
python scripts/evaluate_retrieval.py
python scripts/evaluate_routing.py

# 3b. Roteamento com os provedores que consomem cota (opt-in)
python scripts/evaluate_routing.py --routers atual,embeddings,jev

# 4. Avaliação da geração (LLM-as-judge; consome cota de API)
python scripts/evaluate.py --limit 5
```

---

## Avaliação

O sistema é avaliado contra `data/golden_dataset_reduzido.jsonl`: **26 perguntas** cobrindo os 6
documentos, em quatro categorias — busca léxica (extrair número exato), raciocínio analítico (ler
tabela e comparar), busca semântica (explicar conceito) e multidocumento (comparar duas publicações).

Cada caso tem `question`, `ground_truth`, `categoria` e `expected_source`. No caso multidocumento o
`expected_source` é uma **lista**, e o retrieval só conta como acerto se trouxer todos os documentos
citados — trazer metade da comparação não é meio acerto.

### Duas avaliações separadas

O LLM-as-judge mede a geração, mas custa ~2 chamadas de API por pergunta e não cabe no free tier:
com 26 perguntas, um único run estoura o limite diário de 50 requisições dos modelos `:free` do
OpenRouter — a mesma chave usada pelo CI e pelo desenvolvimento local. Daí a divisão:

| | `evaluate_retrieval.py` | `evaluate.py` |
| --- | --- | --- |
| mede | retrieval (busca + rerank) | geração (LLM-as-judge) |
| chamadas de API | **nenhuma** | ~2 por pergunta |
| roda | todo push no `main` | só no disparo manual, com `run_judge` marcado |
| determinístico | sim | não |

Separar as duas não é só economia de cota: é o que permite atribuir culpa. Nota baixa **com**
key-fact recall alto é problema de geração; nota baixa **com** recall baixo é problema de busca. Foi
exatamente assim que o bug do modelo de moderação apareceu.

### Resultado atual do retrieval

26 perguntas, sem nenhuma chamada de API:

```
Acurácia de fonte (documento certo recuperado):  100%
Key-fact recall médio:                            94%
Abaixo do limiar de confiança (0.1):             0/26
Latência média do retrieval:                    4,24s
```

Por categoria:

| Categoria | Casos | Fonte | Key-fact recall |
| --- | ---: | ---: | ---: |
| Busca Semântica | 8 | 100% | 100% |
| Busca Léxica | 10 | 100% | 98% |
| Raciocínio Analítico | 7 | 100% | 80% |
| Multidocumento | 1 | 100% | 75% |

Com seis documentos no índice — dois deles edições diferentes da mesma publicação — o retrieval
**nunca trouxe o documento errado**. O que ainda falha é leitura de tabela e gráfico, e a pergunta
multidocumento, onde `TOP_K=8` precisa acomodar trechos de dois PDFs ao mesmo tempo.

### Como o key-fact recall funciona

O golden dataset não tem página ou chunk anotado, só o documento esperado. Então a precisão do
retrieval é medida por **key-fact recall** (`evaluation/metrics.py`): os trechos em `**negrito**` do
`ground_truth` — os valores que a resposta precisa conter — são extraídos e procurados no texto dos
chunks recuperados, após normalizar acento, caixa e espaço.

É aproximado, e a aproximação tem nome: o docling extrai tabela como texto corrido do tipo
`Riscos fiscais, ... Fev 2024 = 31`, sem o sinal de `%`, então o matcher tem um fallback para
percentual solto.

### `validate_golden.py` — o teto da métrica

Roda offline e checa duas coisas: que todo `expected_source` existe no índice, e que todo fato em
negrito aparece **literalmente no texto extraído daquele PDF**. Se o fato não está nem no documento
inteiro, nenhum retrieval consegue trazê-lo — a nota baixa seria culpa do dataset, não do sistema.

Isso não é teórico: rodar essa checagem pela primeira vez reprovou **5 dos 12 casos originais** e
mostrou que boa parte do key-fact recall medido até então era artefato de dataset. Havia três
problemas distintos:

- fatos em negrito que eram **paráfrase** do PDF, não citação;
- números **lidos a olho de gráficos** que o docling extrai como imagem — impossíveis de recuperar
  por texto (perderam o negrito e ficaram a cargo só do juiz);
- percentuais que a extração de tabela devolve **sem o `%`**.

Depois da correção, os 26 casos passam. O key-fact recall saltou de 48% para 94% — parte disso é o
corpus e o retrieval terem melhorado, parte é a métrica ter parado de medir o próprio dataset. Ela
roda no CI antes da avaliação, então um golden dataset quebrado falha o build.

### CI

`.github/workflows/eval.yml`, a cada push no `main` que toque no pipeline: valida o dataset, roda a
avaliação de retrieval, publica o badge e sobe os resultados como artifact — **sem consumir cota de
API**. A avaliação com LLM-as-judge fica no disparo manual (Actions → Run workflow → `run_judge`),
desmarcada por padrão.

A avaliação de geração tolera falha de provider: um caso que estoura cota vira "sem nota" em vez de
derrubar o run inteiro, a execução para após 3 erros seguidos, e o badge **não** é publicado a
partir de execução parcial — um badge que mente é pior que um badge desatualizado.

### Sobre o juiz

O juiz (`JUDGE_PROVIDER`) é sempre um provider diferente do gerador, de propósito: um LLM avaliando
a própria resposta tende a ser leniente consigo mesmo, o que inflaria a nota. Como o OpenRouter
também é o fallback da geração, o modelo do juiz é pinado à parte em `OPENROUTER_JUDGE_MODEL` em vez
de herdar `OPENROUTER_MODEL`, garantindo que juiz e gerador nunca sejam o mesmo modelo.

A escala é 1 a 5 com regras de calibração explícitas — por exemplo, uma recusa honesta quando a
informação **existe** no documento é sempre nota 2 (falha de retrieval), nunca "perdoada" por ser
honesta.

---

## Experimentos e decisões

### Chunking naive vs. table-aware

`scripts/compare_chunking.py` isola o chunking como única variável, usando `key_fact_recall` e sem
chamar nenhum LLM:

- **naive**: extração crua via `pypdf` + split por caracteres com overlap.
- **docling**: `HybridChunker` table-aware, o que a produção usa.

A primeira tentativa comparou top-8 chunks de cada estratégia e deu vitória fácil para o naive
(73% vs. 59%) — resultado enganoso. Os chunks do Docling saem bem menores (~954 chars, corte em
fronteira estrutural) que os do naive (~3.948 chars, corte por tamanho fixo), então top-8 entregava
~4x mais texto bruto para o naive. Mais texto recuperado facilita achar um número solto no meio,
independentemente de o retrieval ter sido preciso.

Corrigindo para **orçamento de caracteres igual** (7.629 chars — o que `TOP_K=8` do Docling entrega
em produção), o resultado inverte:

| Estratégia | Chunks/pergunta (mesmo orçamento) | Key-fact recall |
| --- | ---: | ---: |
| naive | 1,0 | 44% |
| docling (produção) | 3,1 | 46% |

Com o mesmo espaço de contexto, o naive aposta tudo em 1 chunk gigante; o Docling encaixa ~3 chunks
menores e mais focados. Duas ressalvas honestas: a margem (2 p.p.) é pequena, e os números são de
quando o corpus tinha 1 documento e o dataset 12 perguntas — reexecutar hoje mede os 6. É sinal
direcional, não prova estatística. E `key_fact_recall` só checa se o número aparece em algum lugar
do texto recuperado, não se ele está coeso com o rótulo da tabela que lhe dá sentido — a vantagem
real do chunking table-aware é provavelmente maior do que a métrica captura.

Reproduzir: `python scripts/compare_chunking.py` (grava `chunking_comparison.json`).

### Qual roteador de intenção

`scripts/evaluate_routing.py` compara as implementações no mesmo dataset rotulado
(`data/routing_dataset.jsonl`, 76 casos: 40 `documentos` — as 26 perguntas do golden mais 14
sorteadas do cache —, 14 `saudacao` e 22 `fora_de_escopo` escritos à mão, incluindo seis tentativas
de prompt injection). Reproduzir: `python scripts/build_routing_dataset.py` e
`python scripts/evaluate_routing.py`.

| Roteador | Acurácia | p50 | p95 | Pipeline evitado | Recusas indevidas |
| --- | ---: | ---: | ---: | ---: | ---: |
| `atual` (baseline) | 52,6% | 0 ms | 0 ms | 0% | 0 |
| `heuristica` | 80,3% | 0 ms | 0,1 ms | 30,3% | **2** |
| `embeddings` | 86,8% | 212 ms | 417 ms | 34,2% | 0 |

"Pipeline evitado" é a fração de perguntas que o roteador tirou do caminho **corretamente**. Com a
latência de retrieval medida nesta máquina (8,27s por pergunta), os 34,2% do roteador de embeddings
equivalem a 215s poupados só neste dataset — além das chamadas de LLM que deixam de acontecer.

**A acurácia de 52,6% do baseline não é um bug**: é exatamente a proporção de perguntas in-scope no
dataset. O sistema atual acerta 100% do recall em `documentos` porque manda tudo para lá, e 0% nas
outras duas rotas. É o piso contra o qual os outros se comparam.

**As duas recusas indevidas da heurística têm a mesma causa e valem mais que a média.** O marcador
`clima` — que existe para pegar "previsão do tempo" — casa dentro de "risco **climá**tico de
transição", um assunto que ocupa uma seção inteira do REF. A regra de palavra-chave passa de filtro
a censor em cima do próprio corpus, e o erro é invisível na acurácia agregada (80,3% parece
razoável) porque ele acontece em 2 de 76 casos. Foi a métrica separada de recusa indevida que
mostrou.

**O roteador de embeddings erra só para o lado barato.** Todos os seus 10 erros são
`fora_de_escopo` classificado como `documentos` — nunca o contrário. Na prática o usuário recebe a
recusa que já recebia antes, pelo caminho caro. Recall de `fora_de_escopo` fica em 0,55: perguntas
como "vale a pena investir em bitcoin agora?" e "qual o melhor notebook pra comprar em 2026?" ficam
perto demais do vocabulário financeiro do corpus para a similaridade separar.

**A calibração é o que o limiar de confiança usa.** Nas faixas do roteador de embeddings, confiança
declarada e acurácia real andam juntas em cima (0,98 declarado → 1,00 real; 0,83 → 0,94) e
desabam embaixo (0,63 → 0,12). É essa separação que faz o corte em 0,7 funcionar: ele pegou 8 casos
de baixa confiança e os devolveu para o pipeline completo. A heurística não tem nada disso — ela
declara 1,0 nas duas recusas indevidas, porque "confiança" de regra é só a regra ter casado.

### Por que o badge do LLM-as-judge saiu do README

Ele media a geração, mas só pode rodar manualmente por causa da cota — um badge que só atualiza de
vez em quando passa a informar a data do último run, não a qualidade atual do sistema. O badge que
ficou é o de retrieval: determinístico, sem custo, atualizado em todo commit. A avaliação de geração
continua no repositório e continua sendo rodada; ela só não vira mais um selo no topo da página.

---

## Limitações conhecidas

- **Gráficos são imagens.** Números que só existem em gráfico não são extraídos pelo docling e
  portanto não são recuperáveis por texto. É a maior causa de erro na categoria "raciocínio
  analítico" (80% de key-fact recall).
- **Perguntas multidocumento competem por espaço.** Com `TOP_K=8`, uma comparação entre duas
  publicações precisa acomodar trechos dos dois PDFs no mesmo orçamento de contexto.
- **A avaliação offline mede a primeira tentativa.** Em produção, uma pergunta de baixa confiança
  dispara reescrita de query (que usa LLM) e um segundo retrieval. O script determinístico mede só o
  primeiro passe — caminho de 100% das perguntas, e o único reproduzível sem API.
- **O roteador de intenção não sabe dizer "não sei".** Ele escolhe entre três rotas e usa o limiar
  de confiança como válvula de escape, mas uma mensagem ambígua ("e o bitcoin?") vai para
  `documentos` por construção. É a escolha certa para o custo assimétrico do erro, e continua sendo
  um limite.
- **`fora_de_escopo` é a rota difícil.** Recall de 0,55 no melhor roteador offline: pedidos que
  usam vocabulário financeiro sem estar nas publicações do BCB ("vale a pena investir em bitcoin?")
  ficam perto demais das perguntas legítimas.
- **O free tier é o gargalo real do projeto.** Groq e OpenRouter têm limites diários que uma
  avaliação completa estoura sozinha. Boa parte das decisões de arquitetura aqui (embeddings locais,
  reranker local, cache, avaliação offline) existe para manter o custo marginal em zero.
