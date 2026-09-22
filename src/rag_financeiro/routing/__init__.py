from rag_financeiro.routing.base import (
    DOCUMENTOS,
    FORA_DE_ESCOPO,
    ROUTES,
    SAFE_ROUTE,
    SAUDACAO,
    RouteDecision,
    apply_confidence_floor,
)

ROUTERS = ("atual", "heuristica", "embeddings", "jev", "llm")

_cache: dict[str, object] = {}


def get_router(name: str):
    if name in _cache:
        return _cache[name]

    if name in ("atual", "none", ""):
        from rag_financeiro.routing.null import NullRouter

        router = NullRouter()
    elif name == "heuristica":
        from rag_financeiro.routing.heuristic import HeuristicRouter

        router = HeuristicRouter()
    elif name == "embeddings":
        from rag_financeiro.routing.embedding import EmbeddingRouter

        router = EmbeddingRouter()
    elif name == "jev":
        from rag_financeiro.routing.jev import JevRouter

        router = JevRouter()
    elif name == "llm":
        from rag_financeiro.routing.llm import LLMRouter

        router = LLMRouter()
    else:
        raise RuntimeError(f"router desconhecido: {name!r} (use um de {ROUTERS})")

    _cache[name] = router
    return router


__all__ = [
    "DOCUMENTOS",
    "FORA_DE_ESCOPO",
    "ROUTERS",
    "ROUTES",
    "SAFE_ROUTE",
    "SAUDACAO",
    "RouteDecision",
    "apply_confidence_floor",
    "get_router",
]
