import time

from rag_financeiro.routing.base import DOCUMENTOS, RouteDecision


class NullRouter:
    """Baseline: o comportamento atual, em que toda pergunta atravessa o pipeline completo."""

    name = "atual"

    def decide(self, question: str) -> RouteDecision:
        start = time.perf_counter()
        return RouteDecision(
            route=DOCUMENTOS,
            confidence=1.0,
            router=self.name,
            latency_seconds=time.perf_counter() - start,
            probabilities={DOCUMENTOS: 1.0},
        )
