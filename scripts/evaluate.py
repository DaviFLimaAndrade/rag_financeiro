"""
evaluate.py

Avaliação do sistema RAG usando um golden dataset (data/golden_dataset_reduzido.jsonl)
e a técnica de LLM-as-judge: um LLM compara a resposta gerada com o
ground_truth e atribui uma nota de correção/fidelidade, capturando
nuance semântica que keyword matching não consegue.

Métricas produzidas:
- Nota média do judge (1-5)
- Taxa de aprovação (score >= 4)
- Acurácia de retrieval (fonte esperada foi recuperada?)
- Latência média
"""

import json
import logging
import time
from dataclasses import dataclass, asdict

from rag_financeiro.evaluation.golden_dataset import load_golden_dataset
from rag_financeiro.evaluation.judge import judge_answer
from rag_financeiro.generation.rag_chain import answer_question
from rag_financeiro.generation.llm_provider import get_llm, current_provider_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def run_evaluation() -> list[EvalResult]:
    cases = load_golden_dataset()
    logger.info(f"{len(cases)} casos carregados")
    logger.info(f"Provider/modelo em uso: {current_provider_label()}")

    judge_llm = get_llm(temperature=0)
    results = []

    for i, case in enumerate(cases, start=1):
        logger.info(f"[{i}/{len(cases)}] Avaliando: {case['question']}")

        start = time.perf_counter()
        response = answer_question(case["question"])
        elapsed = time.perf_counter() - start

        expected_source = case.get("expected_source")
        if expected_source is None:
            source_ok = True
        else:
            source_ok = any(
                expected_source.lower() in str(s.get("source", "")).lower()
                for s in response["sources"]
            )

        score, failure_type, justificativa = judge_answer(
            judge_llm, case["question"], case["ground_truth"], response["answer"]
        )

        results.append(
            EvalResult(
                question=case["question"],
                categoria=case.get("categoria", ""),
                score=score,
                failure_type=failure_type,
                justificativa=justificativa,
                source_ok=source_ok,
                latency_seconds=round(elapsed, 2),
                generated_answer=response["answer"][:200],
            )
        )

    return results


def print_report(results: list[EvalResult]):
    total = len(results)
    avg_score = sum(r.score for r in results) / total if total else 0
    approved = sum(1 for r in results if r.score >= 4)
    source_accuracy = sum(1 for r in results if r.source_ok) / total if total else 0
    avg_latency = sum(r.latency_seconds for r in results) / total if total else 0
    retrieval_failures = sum(1 for r in results if r.failure_type == "retrieval")

    print("\n" + "=" * 60)
    print("RELATÓRIO DE AVALIAÇÃO — RAG FINANCEIRO (LLM-as-judge)")
    print("=" * 60)
    for r in results:
        status = "✅" if r.score >= 4 else "⚠️" if r.score == 3 else "❌"
        print(f"\n[{status} nota {r.score}/5] ({r.failure_type}) {r.question}")
        print(f"  Justificativa do judge: {r.justificativa}")
        print(f"  Fonte correta: {r.source_ok} | Latência: {r.latency_seconds}s")
        print(f"  Resposta (preview): {r.generated_answer}...")

    print("\n" + "-" * 60)
    print(f"Nota média (judge): {avg_score:.2f}/5")
    print(f"Taxa de aprovação (nota >= 4): {approved}/{total} ({approved/total*100:.0f}%)")
    print(f"Acurácia de retrieval (fonte correta): {source_accuracy*100:.0f}%")
    print(f"Falhas de retrieval (dado existia mas não foi encontrado): {retrieval_failures}/{total}")
    print(f"Latência média: {avg_latency:.2f}s")
    print("=" * 60 + "\n")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": [asdict(r) for r in results],
                "summary": {
                    "provider": current_provider_label(),
                    "avg_score": round(avg_score, 2),
                    "approval_rate": round(approved / total, 2) if total else 0,
                    "source_accuracy": round(source_accuracy, 2),
                    "retrieval_failures": retrieval_failures,
                    "avg_latency_seconds": round(avg_latency, 2),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Resultados salvos em eval_results.json")


if __name__ == "__main__":
    results = run_evaluation()
    print_report(results)
