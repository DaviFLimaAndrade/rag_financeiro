import re
import unicodedata

BOLD_SPAN = re.compile(r"\*\*(.+?)\*\*")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def key_fact_recall(ground_truth: str, retrieved_chunks: list[dict]) -> float | None:
    """Fração dos trechos em **negrito** do ground_truth que aparecem literalmente
    (após normalização de acento/caixa/espaço) no texto dos chunks recuperados.

    Proxy heurístico de retrieval accuracy: o golden dataset não tem página/chunk
    esperado, só um `expected_source` (sempre o mesmo, único PDF do projeto), então
    checar presença dos fatos-chave é o sinal mais direto disponível hoje sobre se
    o retrieval trouxe o trecho certo. Falsos negativos são possíveis quando o PDF
    formata o número de um jeito diferente do texto do ground_truth (ex.: espaço
    diferente em "R$ 37,8" vs "R$37,8") -- é aproximado, não exato.
    """
    facts = BOLD_SPAN.findall(ground_truth)
    if not facts:
        return None

    context = _normalize(" ".join(c["text"] for c in retrieved_chunks))
    hits = sum(1 for fact in facts if _normalize(fact) in context)
    return hits / len(facts)


# depois dá pra adicionar faithfulness, answer_relevancy etc via RAGAS
