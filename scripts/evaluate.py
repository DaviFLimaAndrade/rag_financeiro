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

O corpus tem 6 publicações do BCB, então o relatório também quebra os números por
documento esperado: com vários PDFs no mesmo índice, uma nota baixa pode ser o
retrieval trazendo o trecho certo do documento errado (ex.: REF de 2024 vs REF de
2026), e a média global esconde isso.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict

from rag_financeiro.observability import setup_tracing

setup_tracing()

from rag_financeiro import config
from rag_financeiro.evaluation.golden_dataset import load_golden_dataset
from rag_financeiro.evaluation.judge import judge_answer
from rag_financeiro.evaluation.metrics import key_fact_recall
from rag_financeiro.generation.rag_chain import answer_question
from rag_financeiro.generation.llm_provider import get_llm, current_provider_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    question: str
    categoria: str
    expected_source: list[str]
    retrieved_sources: list[str]
    score: int
    failure_type: str
    justificativa: str
    source_ok: bool
    key_fact_recall: float | None
    latency_seconds: float
    generated_answer: str


def run_evaluation(limit: int | None = None) -> list[EvalResult]:
    cases = load_golden_dataset()
    if limit:
        cases = cases[:limit]
    logger.info(f"{len(cases)} casos carregados")
    logger.info(f"Gerador: {current_provider_label()}")
    logger.info(f"Judge: {current_provider_label(config.JUDGE_PROVIDER, purpose='judge')}")

    judge_llm = get_llm(provider=config.JUDGE_PROVIDER, temperature=0, purpose="judge")
    results = []

    for i, case in enumerate(cases, start=1):
        logger.info(f"[{i}/{len(cases)}] Avaliando: {case['question']}")

        start = time.perf_counter()
        response = answer_question(case["question"])
        elapsed = time.perf_counter() - start

        expected = case.get("expected_source") or []
        expected = [expected] if isinstance(expected, str) else list(expected)
        retrieved = sorted({str(s.get("source", "")) for s in response["sources"]})
        # Numa pergunta multidocumento o retrieval precisa trazer todos os documentos
        # citados no ground_truth, não só um deles.
        source_ok = all(
            any(exp.lower() in src.lower() for src in retrieved) for exp in expected
        )

        score, failure_type, justificativa = judge_answer(
            judge_llm, case["question"], case["ground_truth"], response["answer"]
        )

        kfr = key_fact_recall(case["ground_truth"], response["sources"])

        results.append(
            EvalResult(
                question=case["question"],
                categoria=case.get("categoria", ""),
                expected_source=expected,
                retrieved_sources=retrieved,
                score=score,
                failure_type=failure_type,
                justificativa=justificativa,
                source_ok=source_ok,
                key_fact_recall=round(kfr, 2) if kfr is not None else None,
                latency_seconds=round(elapsed, 2),
                generated_answer=response["answer"][:200],
            )
        )

    return results


def print_report(results: list[EvalResult], save: bool = True):
    total = len(results)
    avg_score = sum(r.score for r in results) / total if total else 0
    approved = sum(1 for r in results if r.score >= 4)
    source_accuracy = sum(1 for r in results if r.source_ok) / total if total else 0
    avg_latency = sum(r.latency_seconds for r in results) / total if total else 0
    retrieval_failures = sum(1 for r in results if r.failure_type == "retrieval")

    kfr_values = [r.key_fact_recall for r in results if r.key_fact_recall is not None]
    avg_key_fact_recall = sum(kfr_values) / len(kfr_values) if kfr_values else None

    by_document = defaultdict(list)
    for r in results:
        for doc in r.expected_source or ["(sem fonte esperada)"]:
            by_document[doc].append(r)

    by_category = defaultdict(list)
    for r in results:
        by_category[r.categoria or "(sem categoria)"].append(r)

    print("\n" + "=" * 60)
    print("RELATÓRIO DE AVALIAÇÃO — RAG FINANCEIRO (LLM-as-judge)")
    print("=" * 60)
    for r in results:
        status = "✅" if r.score >= 4 else "⚠️" if r.score == 3 else "❌"
        print(f"\n[{status} nota {r.score}/5] ({r.failure_type}) {r.question}")
        print(f"  Justificativa do judge: {r.justificativa}")
        kfr_display = f"{r.key_fact_recall*100:.0f}%" if r.key_fact_recall is not None else "n/a"
        print(f"  Fonte correta: {r.source_ok} | Key-fact recall: {kfr_display} | Latência: {r.latency_seconds}s")
        if not r.source_ok:
            print(f"  Esperava {r.expected_source}, recuperou {r.retrieved_sources}")
        print(f"  Resposta (preview): {r.generated_answer}...")

    print("\n" + "-" * 60)
    print(f"Nota média (judge): {avg_score:.2f}/5")
    print(f"Taxa de aprovação (nota >= 4): {approved}/{total} ({approved/total*100:.0f}%)")
    print(f"Acurácia de retrieval (fonte correta): {source_accuracy*100:.0f}%")
    if avg_key_fact_recall is not None:
        print(f"Key-fact recall médio (fatos-chave recuperados): {avg_key_fact_recall*100:.0f}%")
    print(f"Falhas de retrieval (dado existia mas não foi encontrado): {retrieval_failures}/{total}")
    print(f"Latência média: {avg_latency:.2f}s")

    print("\nPor documento esperado:")
    for doc, rows in sorted(by_document.items()):
        n = len(rows)
        doc_score = sum(r.score for r in rows) / n
        doc_source = sum(1 for r in rows if r.source_ok) / n
        print(f"  {doc}: {n} casos | nota {doc_score:.2f}/5 | fonte correta {doc_source*100:.0f}%")

    print("\nPor categoria:")
    for cat, rows in sorted(by_category.items()):
        n = len(rows)
        print(f"  {cat}: {n} casos | nota {sum(r.score for r in rows)/n:.2f}/5")
    print("=" * 60 + "\n")

    if not save:
        logger.info("Execução parcial (--limit): eval_results.json não foi sobrescrito.")
        return

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": [asdict(r) for r in results],
                "summary": {
                    "provider": current_provider_label(),
                    "judge_provider": current_provider_label(config.JUDGE_PROVIDER, purpose="judge"),
                    "avg_score": round(avg_score, 2),
                    "approval_rate": round(approved / total, 2) if total else 0,
                    "source_accuracy": round(source_accuracy, 2),
                    "avg_key_fact_recall": round(avg_key_fact_recall, 2) if avg_key_fact_recall is not None else None,
                    "retrieval_failures": retrieval_failures,
                    "avg_latency_seconds": round(avg_latency, 2),
                    "by_document": {
                        doc: {
                            "cases": len(rows),
                            "avg_score": round(sum(r.score for r in rows) / len(rows), 2),
                            "source_accuracy": round(
                                sum(1 for r in rows if r.source_ok) / len(rows), 2
                            ),
                        }
                        for doc, rows in sorted(by_document.items())
                    },
                    "by_category": {
                        cat: {
                            "cases": len(rows),
                            "avg_score": round(sum(r.score for r in rows) / len(rows), 2),
                        }
                        for cat, rows in sorted(by_category.items())
                    },
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Resultados salvos em eval_results.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="avalia só os N primeiros casos")
    args = parser.parse_args()

    results = run_evaluation(args.limit)
    print_report(results, save=args.limit is None)
