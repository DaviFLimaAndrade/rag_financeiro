import time

import requests

from rag_financeiro import config
from rag_financeiro.routing.base import (
    ROUTE_CRITERIA,
    SAFE_ROUTE,
    RouteDecision,
)

QUESTION_KEY = "rota"

INSTRUCTIONS = (
    "A mensagem é de um usuário conversando com um assistente que só responde sobre as "
    "publicações do Banco Central do Brasil. Para onde essa mensagem deve ser roteada?"
)


class JevRouter:
    """Roteador via System One da TypeSafe: devolve a rota escolhida e a confiança calibrada."""

    name = "jev"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.TYPESAFE_API_KEY
        self.model = model or config.JEV_MODEL
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY não configurada no .env")

    def decide(self, question: str) -> RouteDecision:
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{config.TYPESAFE_BASE_URL}/v1/systemone",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "state": question,
                    "model": self.model,
                    "questions": {
                        QUESTION_KEY: {
                            "type": "choice",
                            "instructions": INSTRUCTIONS,
                            "criteria": ROUTE_CRITERIA,
                        }
                    },
                },
                timeout=config.ROUTER_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            return RouteDecision(
                route=SAFE_ROUTE,
                confidence=0.0,
                router=self.name,
                latency_seconds=time.perf_counter() - start,
                fallback_applied=True,
                error=str(error),
            )

        answer = payload["answers"][QUESTION_KEY]
        usage = payload.get("usage") or {}
        return RouteDecision(
            route=answer["choice"],
            confidence=float(answer.get("confidence", 0.0)),
            router=self.name,
            latency_seconds=time.perf_counter() - start,
            probabilities={k: float(v) for k, v in (answer.get("probabilities") or {}).items()},
            input_tokens=usage.get("input_tokens") or usage.get("input"),
        )
