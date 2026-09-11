import re
import unicodedata

# O usuário pergunta "qual a Selic?"; a ata do Copom escreve "reduzir a taxa básica de juros para
# 14,00% a.a." e nunca usa a palavra "Selic" na frase da decisão. O BM25 não casa e o embedding
# denso fica fraco o suficiente pro chunk certo não entrar nem no pool do reranker.
#
# Expandir a *pergunta* (não o corpus) com o vocabulário do BCB resolve sem reindexar nada: a chave
# é o termo que o leigo usa, o valor é como o documento escreve. Só entram pares verificados contra
# o corpus — sinônimo especulativo aqui vira ruído no reranker.
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
    """Acrescenta à pergunta os termos equivalentes usados nos documentos do BCB.

    A pergunta original é preservada no início — a expansão só adiciona recall, nunca substitui o
    que o usuário escreveu. O texto expandido vale para busca densa, BM25 e rerank; o que vai pro
    LLM continua sendo a pergunta original.
    """
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
