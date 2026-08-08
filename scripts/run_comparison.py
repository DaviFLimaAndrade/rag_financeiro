"""
run_comparison.py

Roda o experimento controlado completo em um único comando:

1. Ingestão + indexação com chunking NAIVE (PyPDFLoader + RecursiveCharacterTextSplitter)
   -> persiste em ./chroma_db_naive
2. Ingestão + indexação com chunking DOCLING (table-aware)
   -> persiste em ./chroma_db_docling
3. Avalia as duas variantes com o MESMO golden dataset e o MESMO LLM/judge
   (definido em LLM_PROVIDER no .env), garantindo que a única variável
   testada seja a estratégia de chunking.
4. Imprime um relatório comparativo lado a lado e salva em
   comparison_results.json

Uso:
    python run_comparison.py
"""

import os

# Desabilita o torch.compile (Dynamo/Inductor) ANTES de qualquer import que
# carregue o PyTorch (via docling). No Windows, sem o compilador C++ (cl.exe)
# instalado, o torch.compile falha com "InvalidCxxCompiler". Isso não afeta
# a qualidade dos resultados, só desliga uma otimização de velocidade que
# não faz diferença relevante rodando em CPU.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from config import DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from rag_llm import get_llm, current_provider_label, extract_text
from ingestion_naive import load_and_chunk_pdfs_naive
from scripts.ingest import load_and_chunk_pdfs as load_and_chunk_pdfs_docling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path("golden_dataset_reduzido.jsonl")

PROMPT_TEMPLATE = """Você é um assistente especializado em análise financeira.
Responda à pergunta do usuário utilizando APENAS as informações do contexto abaixo.
Se a resposta não estiver no contexto, diga claramente que não encontrou a informação.
Sempre cite de qual documento/página veio a informação, quando disponível.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

JUDGE_PROMPT = """Você é um avaliador rigoroso de sistemas de RAG (Retrieval-Augmented Generation).

Compare a RESPOSTA GERADA com a RESPOSTA ESPERADA (ground truth) para a pergunta abaixo.
Avalie a RESPOSTA GERADA numa escala de 1 a 5, considerando:
- Correção factual em relação ao ground truth
- Se cobre os pontos principais do ground truth (não precisa ser idêntica em texto)
- Se não inventa informação que não está no ground truth nem contradiz ele
- Se NÃO inclui informação extra, não solicitada, que não está no ground truth (isso deve reduzir a nota, mesmo que a informação extra seja verdadeira)

REGRAS DE CALIBRAÇÃO (aplique de forma consistente, sempre da mesma forma para o mesmo padrão):

A) Se a RESPOSTA GERADA afirma que não encontrou a informação (ex: "não encontrei", "não há dados"),
   e o ground truth contém um valor/fato específico real -> isso é uma FALHA DE RETRIEVAL.
   Nota-se SEMPRE 2, independentemente de quão "educada" ou "bem escrita" seja a recusa.
   Não dê nota 3 ou 4 para esse padrão só porque a resposta é honesta sobre não ter encontrado.

B) Se a RESPOSTA GERADA afirma que não encontrou a informação, e o ground truth TAMBÉM afirma
   que a informação não está disponível/fora de escopo (caso de guardrail/anti-alucinação)
   -> isso é um ACERTO. Nota 5.

C) Se a RESPOSTA GERADA contém o valor/fato correto, mas com texto adicional não solicitado
   -> nota 4 (não 5), pela falta de concisão.

D) Reserve nota 1 apenas para alucinação real (a resposta inventa ou contradiz um fato do
   ground truth) -- não para "não encontrei".

Escala:
5 = Correta e completa, equivalente em conteúdo ao ground truth (ou acerto de guardrail, regra B)
4 = Correta, mas incompleta, com detalhes a menos, ou com informação extra não solicitada (regra C)
3 = Parcialmente correta, com alguma imprecisão factual relevante (não é caso de "não encontrei")
2 = Falha de retrieval: não encontrou informação que existia no documento (regra A)
1 = Alucinação: inventa ou contradiz o ground truth (regra D)

Pergunta: {question}
Resposta esperada (ground truth): {ground_truth}
Resposta gerada pelo sistema: {generated_answer}

Responda APENAS com um JSON no formato:
{{"score": <número de 1 a 5>, "failure_type": "<'retrieval', 'generation', 'guardrail_ok' ou 'none'>", "justificativa": "<uma frase curta>"}}
"""


@dataclass
class EvalCase:
    question: str
    ground_truth: str
    expected_source: str | None
    categoria: str = ""


@dataclass
class EvalResult:
    question: str
    categoria: str
    score: int
    failure_type: str
    justificativa: str
    source_ok: bool
    latency_seconds: float
    generated_answer: str


def load_dataset(path: Path) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {path.resolve()}")
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cases.append(
                EvalCase(
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                    expected_source=item.get("expected_source"),
                    categoria=item.get("categoria", ""),
                )
            )
    return cases


def _chunk_id(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_index(chunks, persist_dir: str, collection_name: str, embeddings) -> Chroma:
    Path(persist_dir).mkdir(exist_ok=True)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    ids = [_chunk_id(c.page_content) for c in chunks]
    existing = set(store.get()["ids"]) if store.get()["ids"] else set()
    new_docs, new_ids = [], []
    for c, cid in zip(chunks, ids):
        if cid not in existing:
            new_docs.append(c)
            new_ids.append(cid)
    if new_docs:
        store.add_documents(documents=new_docs, ids=new_ids)
    logger.info(f"[{collection_name}] indexados {len(new_docs)} novos chunks (total pedido: {len(chunks)})")
    return store


def answer_with_store(question: str, store: Chroma, llm) -> dict:
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    if not docs:
        return {"answer": "Não encontrei informações relevantes no índice.", "sources": []}

    context = "\n\n".join(
        f"[Fonte: {d.metadata.get('source', 'desconhecida')}, página {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    answer_text = extract_text(response.content)

    sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]
    return {"answer": answer_text, "sources": sources}


def judge_answer(judge_llm, question: str, ground_truth: str, generated_answer: str) -> tuple[int, str, str]:
    import re

    prompt = JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, generated_answer=generated_answer)
    response = judge_llm.invoke(prompt)
    text = extract_text(response.content).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return 1, "generation", "Falha ao interpretar avaliação do judge."
    try:
        parsed = json.loads(match.group())
        return (
            int(parsed["score"]),
            parsed.get("failure_type", "none"),
            parsed.get("justificativa", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return 1, "generation", "Falha ao interpretar avaliação do judge."


def run_eval_for_variant(label: str, store: Chroma, llm, judge_llm, cases: list[EvalCase]) -> list[EvalResult]:
    results = []
    checkpoint_file = f"checkpoint_{label}.jsonl"
    
    # 1. Carrega progresso anterior (Checkpoints)
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                results.append(EvalResult(**data))
        start_index = len(results)
        if start_index > 0:
            logger.info(f"[{label}] Retomando do checkpoint. {start_index} casos já avaliados.")

    # 2. Continua a avaliação de onde parou
    for i in range(start_index, len(cases)):
        case = cases[i]
        logger.info(f"[{label}] [{i+1}/{len(cases)}] {case.question}")
        
        start = time.perf_counter()
        response = answer_with_store(case.question, store, llm)
        elapsed = time.perf_counter() - start

        if case.expected_source is None:
            source_ok = True
        else:
            source_ok = any(
                case.expected_source.lower() in str(s.get("source", "")).lower()
                for s in response["sources"]
            )

        # 3. Chama o Judge com a nova rubrica (retornando as 3 variáveis)
        score, failure_type, justificativa = judge_answer(
            judge_llm, case.question, case.ground_truth, response["answer"]
        )

        eval_result = EvalResult(
            question=case.question,
            categoria=case.categoria,
            score=score,
            failure_type=failure_type, # Adicionado o novo campo aqui
            justificativa=justificativa,
            source_ok=source_ok,
            latency_seconds=round(elapsed, 2),
            generated_answer=response["answer"][:200],
        )
        
        results.append(eval_result)

        # 4. Salva o checkpoint imediatamente após a avaliação individual
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(eval_result), ensure_ascii=False) + "\n")

        # 5. Delay (Throttle) para não estourar o Rate Limit da API
        if i < len(cases) - 1:
            logger.info(f"[{label}] Aguardando 15 segundos para evitar Rate Limit da API...")
            time.sleep(15)

    return results

def summarize(results: list[EvalResult]) -> dict:
    total = len(results)
    avg_score = sum(r.score for r in results) / total if total else 0
    approved = sum(1 for r in results if r.score >= 4)
    source_accuracy = sum(1 for r in results if r.source_ok) / total if total else 0
    avg_latency = sum(r.latency_seconds for r in results) / total if total else 0
    retrieval_failures = sum(1 for r in results if r.failure_type == "retrieval")
    return {
        "avg_score": round(avg_score, 2),
        "approval_rate": round(approved / total, 2) if total else 0,
        "source_accuracy": round(source_accuracy, 2),
        "avg_latency_seconds": round(avg_latency, 2),
        "retrieval_failures": retrieval_failures,
    }


def print_comparison(cases, naive_results, docling_results, naive_summary, docling_summary):
    print("\n" + "=" * 70)
    print("COMPARAÇÃO: chunking NAIVE vs DOCLING (mesmo dataset, mesmo LLM)")
    print("=" * 70)

    for case, r_naive, r_docling in zip(cases, naive_results, docling_results):
        print(f"\n[{case.categoria}] {case.question}")
        print(f"  NAIVE   -> nota {r_naive.score}/5 ({r_naive.failure_type}) | fonte ok: {r_naive.source_ok} | {r_naive.latency_seconds}s")
        print(f"  DOCLING -> nota {r_docling.score}/5 ({r_docling.failure_type}) | fonte ok: {r_docling.source_ok} | {r_docling.latency_seconds}s")

    print("\n" + "-" * 70)
    print(f"{'Métrica':<28} {'Naive':>15} {'Docling':>15}")
    print("-" * 70)
    for key in naive_summary:
        print(f"{key:<28} {naive_summary[key]:>15} {docling_summary[key]:>15}")
    print("=" * 70 + "\n")


def main():
    logger.info(f"Provider/modelo em uso (respostas + judge): {current_provider_label()}")

    cases = load_dataset(DATASET_PATH)
    logger.info(f"{len(cases)} casos carregados de {DATASET_PATH}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Gerando chunks NAIVE...")
    naive_chunks = load_and_chunk_pdfs_naive(str(DATA_DIR), CHUNK_SIZE, CHUNK_OVERLAP)

    logger.info("Gerando chunks DOCLING...")
    docling_chunks = load_and_chunk_pdfs_docling(str(DATA_DIR), CHUNK_SIZE, CHUNK_OVERLAP)

    naive_store = build_index(naive_chunks, "./chroma_db_naive", "financeiro_naive", embeddings)
    docling_store = build_index(docling_chunks, "./chroma_db_docling", "financeiro_docling", embeddings)

    llm = get_llm(temperature=0)
    judge_llm = get_llm(temperature=0)

    logger.info("Avaliando variante NAIVE...")
    naive_results = run_eval_for_variant("naive", naive_store, llm, judge_llm, cases)

    logger.info("Avaliando variante DOCLING...")
    docling_results = run_eval_for_variant("docling", docling_store, llm, judge_llm, cases)

    naive_summary = summarize(naive_results)
    docling_summary = summarize(docling_results)

    print_comparison(cases, naive_results, docling_results, naive_summary, docling_summary)

    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "provider": current_provider_label(),
                "naive": {"summary": naive_summary, "results": [asdict(r) for r in naive_results]},
                "docling": {"summary": docling_summary, "results": [asdict(r) for r in docling_results]},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Resultados salvos em comparison_results.json")


if __name__ == "__main__":
    main()