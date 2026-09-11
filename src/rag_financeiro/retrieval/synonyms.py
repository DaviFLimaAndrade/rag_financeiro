import re
import unicodedata

_EXPANSIONS = {
    "selic": "taxa básica de juros",
    "juro": "taxa de juros",
    "juros": "taxa de juros",
    "copom": "Comitê de Política Monetária decisão de política monetária",
    "calote": "inadimplência ativos problemáticos",
    "inadimplencia": "ativos problemáticos",
    "banco": "instituição financeira sistema bancário",
    "bancos": "instituições financeiras sistema bancário",
    "divida": "endividamento",
    "dividas": "endividamento",
    "emprestimo": "crédito operações de crédito",
    "emprestimos": "crédito operações de crédito",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.lower()


def expand(query: str) -> str:
    """Acrescenta à pergunta os termos equivalentes usados nos documentos do BCB."""
    normalized = _normalize(query)
    extras = []
    for term, expansion in _EXPANSIONS.items():
        if not re.search(rf"\b{re.escape(term)}\b", normalized):
            continue
        if _normalize(expansion) in normalized:
            continue
        extras.append(expansion)
    if not extras:
        return query
    return f"{query} ({'; '.join(extras)})"
