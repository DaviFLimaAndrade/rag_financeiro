import re
import time
import unicodedata

from rag_financeiro.routing.base import (
    DOCUMENTOS,
    FORA_DE_ESCOPO,
    SAUDACAO,
    RouteDecision,
)

_SAUDACOES = {
    "oi",
    "ola",
    "opa",
    "eai",
    "e ai",
    "bom dia",
    "boa tarde",
    "boa noite",
    "tudo bem",
    "tudo bom",
    "como vai",
    "obrigado",
    "obrigada",
    "valeu",
    "vlw",
    "tchau",
    "ate mais",
    "ate logo",
    "falou",
    "bye",
    "hello",
    "hi",
}

_FORA_DE_ESCOPO_MARCADORES = (
    "codigo",
    "script",
    "python",
    "javascript",
    "sql",
    "funcao em",
    "traduz",
    "traduza",
    "traduzir",
    "receita",
    "piada",
    "poema",
    "musica",
    "filme",
    "futebol",
    "capital d",
    "quem foi",
    "quem e o presidente",
    "clima",
    "previsao do tempo",
    "ignore as instrucoes",
    "ignore tudo",
    "esqueca as instrucoes",
    "aja como",
    "finja que",
    "voce agora e",
    "system prompt",
    "prompt do sistema",
)

_PONTUACAO = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _PONTUACAO.sub(" ", text).strip()


MAX_SAUDACAO_WORDS = 4


class HeuristicRouter:
    """Baseline de regra: casa saudação por lista e fora-de-escopo por marcador de texto."""

    name = "heuristica"

    def decide(self, question: str) -> RouteDecision:
        start = time.perf_counter()
        normalized = _normalize(question)
        collapsed = " ".join(normalized.split())

        route, confidence = DOCUMENTOS, 0.5
        if collapsed in _SAUDACOES:
            route, confidence = SAUDACAO, 1.0
        elif len(collapsed.split()) <= MAX_SAUDACAO_WORDS and any(
            collapsed.startswith(s) for s in _SAUDACOES
        ):
            route, confidence = SAUDACAO, 1.0
        elif any(m in collapsed for m in _FORA_DE_ESCOPO_MARCADORES):
            route, confidence = FORA_DE_ESCOPO, 1.0

        return RouteDecision(
            route=route,
            confidence=confidence,
            router=self.name,
            latency_seconds=time.perf_counter() - start,
            probabilities={route: confidence},
        )
