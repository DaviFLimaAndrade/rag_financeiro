"""Avaliação do retrieval sem nenhuma chamada de LLM."""

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict

from rag_financeiro import config
from rag_financeiro.evaluation.golden_dataset import load_golden_dataset
from rag_financeiro.evaluation.metrics import key_fact_recall
from rag_financeiro.retrieval.retriever import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = "retrieval_results.json"


@dataclass
class RetrievalResult:
    question: str
    categoria: str
    expected_source: list[str]
    retrieved_sources: list[str]
    source_ok: bool
    key_fact_recall: float | None
    missing_facts: list[str]
    top_rerank_score: float | None
    low_confidence: bool
    latency_seconds: float


def _expected_list(case: dict) -> list[str]:
    expected = case.get("expected_source") or []
    return [expected] if isinstance(expected, str) else list(expected)


def run_evaluation() -> list[RetrievalResult]:
    from rag_financeiro.evaluation.metrics import BOLD_SPAN, _found, _normalize

    cases = load_golden_dataset()
    logger.info(f"{len(cases)} casos carregados | top_k={config.TOP_K}")
    results = []

    for i, case in enumerate(cases, start=1):
        logger.info(f"[{i}/{len(cases)}] {case['question'][:80]}")

        start = time.perf_counter()
        chunks = retrieve(case["question"])
        elapsed = time.perf_counter() - start

        expected = _expected_list(case)
        retrieved = sorted({str(c.get("source", "")) for c in chunks})
        source_ok = all(
            any(exp.lower() in src.lower() for src in retrieved) for exp in expected
        )

        kfr = key_fact_recall(case["ground_truth"], chunks)
        context = _normalize(" ".join(c["text"] for c in chunks))
        missing = [f for f in BOLD_SPAN.findall(case["ground_truth"]) if not _found(f, context)]

        top_score = max((c["rerank_score"] for c in chunks), default=None)

        results.append(
            RetrievalResult(
                question=case["question"],
                categoria=case.get("categoria", ""),
                expected_source=expected,
                retrieved_sources=retrieved,
                source_ok=source_ok,
                key_fact_recall=round(kfr, 2) if kfr is not None else None,
                missing_facts=missing,
                top_rerank_score=round(top_score, 4) if top_score is not None else None,
                low_confidence=(
                    top_score is not None
                    and top_score < config.RETRIEVAL_CONFIDENCE_THRESHOLD
                ),
                latency_seconds=round(elapsed, 2),
            )
        )

    return results


def _rate(rows: list[RetrievalResult], attr: str) -> float:
    return sum(1 for r in rows if getattr(r, attr)) / len(rows) if rows else 0.0


def _avg_kfr(rows: list[RetrievalResult]) -> float | None:
    values = [r.key_fact_recall for r in rows if r.key_fact_recall is not None]
    return sum(values) / len(values) if values else None


def print_report(results: list[RetrievalResult]):
    total = len(results)
    source_accuracy = _rate(results, "source_ok")
    avg_kfr = _avg_kfr(results)
    low_conf = sum(1 for r in results if r.low_confidence)
    avg_latency = sum(r.latency_seconds for r in results) / total if total else 0

    by_document = defaultdict(list)
    for r in results:
        for doc in r.expected_source or ["(sem fonte esperada)"]:
            by_document[doc].append(r)

    by_category = defaultdict(list)
    for r in results:
        by_category[r.categoria or "(sem categoria)"].append(r)

    print("\n" + "=" * 60)
    print("RELATÓRIO DE RETRIEVAL — RAG FINANCEIRO (offline, sem LLM)")
    print("=" * 60)
    for r in results:
        kfr_display = f"{r.key_fact_recall*100:.0f}%" if r.key_fact_recall is not None else "n/a"
        flags = []
        if not r.source_ok:
            flags.append("FONTE ERRADA")
        if r.low_confidence:
            flags.append("BAIXA CONFIANÇA")
        status = "✅" if r.source_ok and not r.missing_facts else "⚠️"
        print(f"\n[{status}] {r.question[:90]}")
        print(f"  Fonte ok: {r.source_ok} | Key-fact recall: {kfr_display} | "
              f"score: {r.top_rerank_score} | {r.latency_seconds}s")
        if flags:
            print(f"  {' | '.join(flags)}")
        if not r.source_ok:
            print(f"  Esperava {r.expected_source}, recuperou {r.retrieved_sources}")
        if r.missing_facts:
            print(f"  Fatos não recuperados: {r.missing_facts}")

    print("\n" + "-" * 60)
    print(f"Perguntas: {total}")
    print(f"Acurácia de fonte (documento certo recuperado): {source_accuracy*100:.0f}%")
    if avg_kfr is not None:
        print(f"Key-fact recall médio: {avg_kfr*100:.0f}%")
    print(f"Abaixo do limiar de confiança ({config.RETRIEVAL_CONFIDENCE_THRESHOLD}): {low_conf}/{total}")
    print(f"Latência média do retrieval: {avg_latency:.2f}s")

    print("\nPor documento esperado:")
    for doc, rows in sorted(by_document.items()):
        kfr = _avg_kfr(rows)
        kfr_display = f"{kfr*100:.0f}%" if kfr is not None else "n/a"
        print(f"  {doc}: {len(rows)} casos | fonte {_rate(rows, 'source_ok')*100:.0f}% | "
              f"key-fact recall {kfr_display}")

    print("\nPor categoria:")
    for cat, rows in sorted(by_category.items()):
        kfr = _avg_kfr(rows)
        kfr_display = f"{kfr*100:.0f}%" if kfr is not None else "n/a"
        print(f"  {cat}: {len(rows)} casos | fonte {_rate(rows, 'source_ok')*100:.0f}% | "
              f"key-fact recall {kfr_display}")
    print("=" * 60 + "\n")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": [asdict(r) for r in results],
                "summary": {
                    "cases": total,
                    "top_k": config.TOP_K,
                    "embedding_model": config.EMBEDDING_MODEL,
                    "reranker_model": config.RERANKER_MODEL,
                    "source_accuracy": round(source_accuracy, 2),
                    "avg_key_fact_recall": round(avg_kfr, 2) if avg_kfr is not None else None,
                    "low_confidence": low_conf,
                    "avg_latency_seconds": round(avg_latency, 2),
                    "by_document": {
                        doc: {
                            "cases": len(rows),
                            "source_accuracy": round(_rate(rows, "source_ok"), 2),
                            "avg_key_fact_recall": (
                                round(_avg_kfr(rows), 2) if _avg_kfr(rows) is not None else None
                            ),
                        }
                        for doc, rows in sorted(by_document.items())
                    },
                    "by_category": {
                        cat: {
                            "cases": len(rows),
                            "source_accuracy": round(_rate(rows, "source_ok"), 2),
                            "avg_key_fact_recall": (
                                round(_avg_kfr(rows), 2) if _avg_kfr(rows) is not None else None
                            ),
                        }
                        for cat, rows in sorted(by_category.items())
                    },
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Resultados salvos em {RESULTS_PATH}")

    return source_accuracy


if __name__ == "__main__":
    results = run_evaluation()
    print_report(results)
    sys.exit(0)
