import time

from langchain_core.messages import HumanMessage, SystemMessage

from rag_financeiro.generation.llm_provider import extract_text, invoke_with_timeout
from rag_financeiro.routing.base import (
    ROUTE_CRITERIA,
    ROUTES,
    SAFE_ROUTE,
    RouteDecision,
)

SYSTEM_PROMPT = (
    "Classifique a mensagem do usuário em uma destas rotas, considerando que ela foi enviada a "
    "um assistente que só responde sobre publicações do Banco Central do Brasil:\n\n"
    + "\n".join(f"- {route}: {criteria}" for route, criteria in ROUTE_CRITERIA.items())
    + "\n\nResponda APENAS com o nome da rota, sem explicação e sem pontuação."
)


class LLMRouter:
    """Baseline caro: um LLM generativo classificando a intenção. Consome cota."""

    name = "llm"

    def __init__(self, provider: str | None = None):
        self.provider = provider

    def decide(self, question: str) -> RouteDecision:
        start = time.perf_counter()
        try:
            response, _ = invoke_with_timeout(
                self.provider,
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
                purpose="rewrite",
            )
            raw = extract_text(response.content).strip().lower()
        except Exception as error:
            return RouteDecision(
                route=SAFE_ROUTE,
                confidence=0.0,
                router=self.name,
                latency_seconds=time.perf_counter() - start,
                fallback_applied=True,
                error=str(error),
            )

        route = next((r for r in ROUTES if r in raw), SAFE_ROUTE)
        return RouteDecision(
            route=route,
            confidence=1.0,
            router=self.name,
            latency_seconds=time.perf_counter() - start,
            probabilities={route: 1.0},
        )
