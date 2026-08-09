from langchain_core.messages import HumanMessage, SystemMessage

from rag_financeiro.retrieval.retriever import retrieve
from rag_financeiro.generation.llm_provider import get_llm, extract_text

NO_CONTEXT_ANSWER = "Não encontrei essa informação no relatório de estabilidade financeira."

SYSTEM_PROMPT = (
    "Você é um assistente que ajuda a consultar o Relatório de Estabilidade Financeira do "
    "Banco Central do Brasil.\n\n"
    "Primeiro verifique: a pergunta é uma saudação, agradecimento, despedida ou conversa casual "
    "(ex: 'olá', 'oi', 'tudo bem?', 'obrigado')? Se SIM, responda de forma breve e natural e "
    "pare por aí — não use o contexto fornecido na mensagem seguinte, não mencione o relatório e "
    "não diga que não encontrou informação.\n\n"
    "Se NÃO — a pergunta é sobre o conteúdo do relatório —, responda em português com base APENAS "
    "no contexto fornecido, citando a página. Se a resposta não estiver no contexto, diga "
    "claramente que não encontrou a informação no relatório."
)


def answer_question(question: str, k: int | None = None, provider: str | None = None) -> dict:
    chunks = retrieve(question, k=k)
    if not chunks:
        return {"answer": NO_CONTEXT_ANSWER, "sources": []}

    context = "\n\n".join(
        f"[Fonte: {c['source']}, p.{c['page_no']}, {c['section'] or 'sem seção'}] {c['text']}"
        for c in chunks
    )

    llm = get_llm(provider=provider)
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Contexto:\n{context}\n\nPergunta: {question}"),
        ]
    )
    return {"answer": extract_text(response.content), "sources": chunks}
