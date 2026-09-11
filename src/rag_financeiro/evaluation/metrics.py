import re
import unicodedata

BOLD_SPAN = re.compile(r"\*\*(.+?)\*\*")
BARE_PERCENT = re.compile(r"^[\d.,]+%$")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def _found(fact: str, context: str) -> bool:
    norm = _normalize(fact)
    if norm in context:
        return True
    # Tabelas e gráficos extraídos pelo docling viram texto do tipo
    # "Riscos fiscais, ... Frequência (%).Fev 2024 = 31", sem o sinal de %.
    if BARE_PERCENT.match(norm):
        return f"= {norm[:-1]}" in context
    return False


def key_fact_recall(ground_truth: str, retrieved_chunks: list[dict]) -> float | None:
    """Fração dos trechos em **negrito** do ground_truth que aparecem literalmente
    (após normalização de acento/caixa/espaço) no texto dos chunks recuperados.

    Proxy heurístico de retrieval accuracy: o golden dataset não tem página/chunk
    esperado, só o `expected_source`, então checar presença dos fatos-chave é o
    sinal mais direto sobre se o retrieval trouxe o trecho certo — e, com 6 PDFs
    no índice, se trouxe o trecho do documento certo. Falsos negativos são
    possíveis quando o PDF formata o número de um jeito diferente do ground_truth
    (ex.: "R$ 37,8" vs "R$37,8") -- é aproximado, não exato. Rode
    `python scripts/validate_golden.py` para separar o que o retrieval não achou
    do que nem existe no texto extraído do PDF.
    """
    facts = BOLD_SPAN.findall(ground_truth)
    if not facts:
        return None

    context = _normalize(" ".join(c["text"] for c in retrieved_chunks))
    hits = sum(1 for fact in facts if _found(fact, context))
    return hits / len(facts)
