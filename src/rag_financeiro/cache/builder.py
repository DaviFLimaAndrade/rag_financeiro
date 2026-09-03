import json
import re
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from rag_financeiro import config
from rag_financeiro.generation.llm_provider import extract_text, invoke_with_timeout
from rag_financeiro.vector_store import chroma_store

BUILD_SYSTEM_PROMPT = (
    "Você extrai fatos estáveis e perguntas frequentes de trechos de um relatório de estabilidade "
    "financeira do Banco Central do Brasil, para alimentar um cache de respostas rápidas.\n\n"
    "Para cada trecho, gere de 0 a 3 pares pergunta/resposta APENAS para fatos objetivos e estáveis "
    "(indicadores, definições, números, datas) — ignore trechos sem fato extraível (ex: só título de "
    "seção ou texto introdutório). A resposta deve ser autocontida e citar o dado exato do trecho.\n\n"
    "Responda APENAS com uma lista JSON no formato:\n"
    '[{"question": "...", "answer": "...", "page_no": <número da página do trecho de origem>, '
    '"source": "<nome do arquivo fonte do trecho>"}]\n'
    "Se nenhum trecho do lote render fato extraível, responda []."
)


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _parse_entries(text: str, fallback_page_no, fallback_source) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    entries = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("question") or not item.get("answer"):
            continue
        entries.append(
            {
                "question": str(item["question"]).strip(),
                "answer": str(item["answer"]).strip(),
                "page_no": item.get("page_no") or fallback_page_no,
                "source": item.get("source") or fallback_source,
            }
        )
    return entries


def build_cache(provider: str | None = None) -> list[dict]:
    data = chroma_store.get_all()
    chunks = [{"text": doc, **meta} for doc, meta in zip(data["documents"], data["metadatas"])]

    entries: list[dict] = []
    for batch in _batches(chunks, config.CACHE_BUILD_BATCH_SIZE):
        context = "\n\n".join(
            f"[p.{c.get('page_no')}, fonte: {c.get('source')}] {c['text']}" for c in batch
        )
        response = invoke_with_timeout(
            provider,
            [SystemMessage(content=BUILD_SYSTEM_PROMPT), HumanMessage(content=context)],
        )
        text = extract_text(response.content).strip()
        entries.extend(_parse_entries(text, batch[0].get("page_no"), batch[0].get("source")))

    save_cache(entries, provider)
    return entries


def save_cache(entries: list[dict], provider: str | None) -> None:
    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider or config.LLM_PROVIDER,
        "entries": entries,
    }
    config.CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
