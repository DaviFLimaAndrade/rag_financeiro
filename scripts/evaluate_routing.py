"""Compara roteadores de intenção no mesmo dataset rotulado.

Os roteadores `atual`, `heuristica` e `embeddings` rodam offline, sem chamar nenhuma API.
`jev` e `llm` são opt-in porque consomem cota.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass

from rag_financeiro import config
from rag_financeiro.routing import ROUTES, get_router
from rag_financeiro.routing.base import DOCUMENTOS, FORA_DE_ESCOPO, apply_confidence_floor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = "routing_results.json"
RETRIEVAL_RESULTS_PATH = "retrieval_results.json"
OFFLINE_ROUTERS = ("atual", "heuristica", "embeddings")
CALIBRATION_BUCKETS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))


@dataclass
class RoutingResult:
    question: str
    expected: str
    predicted: str
    confidence: float
    correct: bool
    fallback_applied: bool
    latency_ms: float
    input_tokens: int | None
    error: str | None


def load_dataset() -> list[dict]:
    with open(config.ROUTING_DATASET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def retrieval_latency_seconds() -> float | None:
    try:
        with open(RETRIEVAL_RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)["summary"]["avg_latency_seconds"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def run_router(name: str, cases: list[dict], threshold: float) -> list[RoutingResult]:
    router = get_router(name)
    results = []

    for i, case in enumerate(cases, start=1):
        decision = apply_confidence_floor(router.decide(case["question"]), threshold)
        if i % 20 == 0 or i == len(cases):
            logger.info(f"  {name}: {i}/{len(cases)}")
        results.append(
            RoutingResult(
                question=case["question"],
                expected=case["route"],
                predicted=decision.route,
                confidence=round(decision.confidence, 4),
                correct=decision.route == case["route"],
                fallback_applied=decision.fallback_applied,
                latency_ms=round(decision.latency_seconds * 1000, 1),
                input_tokens=decision.input_tokens,
                error=decision.error,
            )
        )

    return results


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(pct / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def _per_class(results: list[RoutingResult]) -> dict:
    per_class = {}
    for route in ROUTES:
        tp = sum(1 for r in results if r.expected == route and r.predicted == route)
        fp = sum(1 for r in results if r.expected != route and r.predicted == route)
        fn = sum(1 for r in results if r.expected == route and r.predicted != route)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[route] = {
            "support": tp + fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    return per_class


def _calibration(results: list[RoutingResult]) -> list[dict]:
    buckets = []
    for low, high in CALIBRATION_BUCKETS:
        rows = [r for r in results if low <= r.confidence < high]
        if not rows:
            continue
        buckets.append(
            {
                "faixa": f"{low:.1f}–{min(high, 1.0):.1f}",
                "casos": len(rows),
                "confianca_media": round(sum(r.confidence for r in rows) / len(rows), 3),
                "acuracia": round(sum(1 for r in rows if r.correct) / len(rows), 3),
            }
        )
    return buckets


def summarize(name: str, results: list[RoutingResult], retrieval_seconds: float | None) -> dict:
    total = len(results)
    validos = [r for r in results if not r.error]
    accuracy = sum(1 for r in validos if r.correct) / len(validos) if validos else None
    latencies = [r.latency_ms for r in results]

    desviadas = [r for r in results if r.predicted != DOCUMENTOS]
    desviadas_corretas = [r for r in desviadas if r.correct]
    recusas_indevidas = [
        r for r in results if r.expected == DOCUMENTOS and r.predicted == FORA_DE_ESCOPO
    ]

    tokens = [r.input_tokens for r in results if r.input_tokens is not None]
    custo_por_mil = None
    if tokens:
        media_tokens = sum(tokens) / len(tokens)
        custo_por_mil = round(
            media_tokens * 1000 / 1_000_000 * config.JEV_INPUT_PRICE_PER_MILLION, 6
        )

    return {
        "router": name,
        "casos": total,
        "casos_validos": len(validos),
        "acuracia": round(accuracy, 3) if accuracy is not None else None,
        "por_classe": _per_class(validos),
        "confusao": {
            expected: {
                predicted: sum(
                    1 for r in validos if r.expected == expected and r.predicted == predicted
                )
                for predicted in ROUTES
            }
            for expected in ROUTES
        },
        "calibracao": _calibration(validos),
        "latencia_ms": {
            "media": round(sum(latencies) / total, 1) if total else 0.0,
            "p50": round(_percentile(latencies, 50), 1),
            "p95": round(_percentile(latencies, 95), 1),
        },
        "desvios": {
            "perguntas_desviadas": len(desviadas),
            "desvios_corretos": len(desviadas_corretas),
            "pct_pipeline_evitado": round(len(desviadas_corretas) / total, 3) if total else 0.0,
            "segundos_de_retrieval_poupados": (
                round(len(desviadas_corretas) * retrieval_seconds, 1)
                if retrieval_seconds is not None
                else None
            ),
        },
        "recusas_indevidas": len(recusas_indevidas),
        "fallbacks_por_baixa_confianca": sum(1 for r in results if r.fallback_applied),
        "erros_de_provedor": sum(1 for r in results if r.error),
        "custo_usd_por_mil_decisoes": custo_por_mil,
    }


def print_report(summaries: list[dict], retrieval_seconds: float | None):
    print("\n" + "=" * 78)
    print("RELATÓRIO DE ROTEAMENTO DE INTENÇÃO — RAG FINANCEIRO")
    print("=" * 78)
    if retrieval_seconds is not None:
        print(f"Latência de retrieval usada como referência: {retrieval_seconds:.2f}s por pergunta")

    print(f"\n{'router':<12} {'acurácia':>9} {'p50 ms':>8} {'p95 ms':>8} "
          f"{'pipeline evitado':>17} {'recusas indevidas':>18}")
    print("-" * 78)
    for s in summaries:
        acuracia = f"{s['acuracia']*100:.1f}%" if s["acuracia"] is not None else "sem nota"
        print(f"{s['router']:<12} {acuracia:>9} {s['latencia_ms']['p50']:>8.1f} "
              f"{s['latencia_ms']['p95']:>8.1f} {s['desvios']['pct_pipeline_evitado']*100:>16.1f}% "
              f"{s['recusas_indevidas']:>18}")

    for s in summaries:
        print(f"\n--- {s['router']} " + "-" * (74 - len(s['router'])))
        for route, m in s["por_classe"].items():
            print(f"  {route:<16} n={m['support']:<4} precisão {m['precision']:.2f} | "
                  f"recall {m['recall']:.2f} | f1 {m['f1']:.2f}")
        if s["calibracao"]:
            print("  calibração (confiança declarada vs. acurácia real):")
            for b in s["calibracao"]:
                print(f"    {b['faixa']}: {b['casos']:>3} casos | "
                      f"confiança {b['confianca_media']:.2f} | acurácia {b['acuracia']:.2f}")
        if s["fallbacks_por_baixa_confianca"]:
            print(f"  fallback por baixa confiança: {s['fallbacks_por_baixa_confianca']}")
        if s["erros_de_provedor"]:
            print(f"  erros de provedor: {s['erros_de_provedor']}/{s['casos']} "
                  f"(casos com nota: {s['casos_validos']})")
            print("  ATENÇÃO: decisão com erro cai na rota segura e não conta para a acurácia — "
                  "run parcial não vira número publicável.")
        if s["custo_usd_por_mil_decisoes"] is not None:
            print(f"  custo: US$ {s['custo_usd_por_mil_decisoes']:.6f} por mil decisões")
        if s["desvios"]["segundos_de_retrieval_poupados"] is not None:
            print(f"  retrieval poupado no dataset: "
                  f"{s['desvios']['segundos_de_retrieval_poupados']}s")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routers", default=",".join(OFFLINE_ROUTERS))
    parser.add_argument("--threshold", type=float, default=config.ROUTER_CONFIDENCE_THRESHOLD)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cases = load_dataset()
    if args.limit:
        cases = cases[: args.limit]

    by_route = defaultdict(int)
    for case in cases:
        by_route[case["route"]] += 1
    logger.info(f"{len(cases)} casos | {dict(by_route)} | limiar={args.threshold}")

    retrieval_seconds = retrieval_latency_seconds()
    summaries, detailed = [], {}
    for name in [n.strip() for n in args.routers.split(",") if n.strip()]:
        logger.info(f"rodando router={name}")
        results = run_router(name, cases, args.threshold)
        summaries.append(summarize(name, results, retrieval_seconds))
        detailed[name] = [asdict(r) for r in results]

    print_report(summaries, retrieval_seconds)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "casos": len(cases),
                    "por_rota": dict(by_route),
                    "limiar_de_confianca": args.threshold,
                    "latencia_retrieval_referencia": retrieval_seconds,
                    "routers": summaries,
                },
                "results": detailed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Resultados salvos em {RESULTS_PATH}")


if __name__ == "__main__":
    main()
    sys.exit(0)
