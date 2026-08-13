from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from rag_financeiro import config
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

REWRITE_SYSTEM_PROMPT = (
    "Reformule a pergunta a seguir para maximizar a chance de encontrar trechos relevantes numa "
    "busca por similaridade semântica em um relatório de estabilidade financeira do Banco Central "
    "do Brasil. Retorne APENAS a pergunta reformulada, sem explicações nem aspas."
)


class RAGState(TypedDict):
    question: str  #
    original_question: str
    provider: str | None
    k: int | None
    chunks: list[dict]
    top_rerank_score: float | None
    retried: bool
    answer: str


def _retrieve_node(state: RAGState) -> dict:
    chunks = retrieve(state["question"], k=state["k"])
    top_rerank_score = max((c["rerank_score"] for c in chunks), default=None)
    return {"chunks": chunks, "top_rerank_score": top_rerank_score}


def _route_after_retrieve(state: RAGState) -> str:
    if not state["chunks"]:
        return "generate"
    if (
        state["top_rerank_score"] is not None
        and state["top_rerank_score"] < config.RETRIEVAL_CONFIDENCE_THRESHOLD
        and not state["retried"]
    ):
        return "rewrite"
    return "generate"


def _rewrite_query_node(state: RAGState) -> dict:
    llm = get_llm(provider=state["provider"])
    response = llm.invoke(
        [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=state["original_question"]),
        ]
    )
    rewritten = extract_text(response.content).strip()
    return {"question": rewritten or state["question"], "retried": True}


def _generate_node(state: RAGState) -> dict:
    chunks = state["chunks"]
    if not chunks:
        return {"answer": NO_CONTEXT_ANSWER}

    context = "\n\n".join(
        f"[Fonte: {c['source']}, p.{c['page_no']}, {c['section'] or 'sem seção'}] {c['text']}"
        for c in chunks
    )
    llm = get_llm(provider=state["provider"])
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Contexto:\n{context}\n\nPergunta: {state['original_question']}"),
        ]
    )
    return {"answer": extract_text(response.content)}


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", _retrieve_node)
        builder.add_node("rewrite_query", _rewrite_query_node)
        builder.add_node("generate", _generate_node)

        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            _route_after_retrieve,
            {"rewrite": "rewrite_query", "generate": "generate"},
        )
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("generate", END)

        _graph = builder.compile()
    return _graph


def answer_question(question: str, k: int | None = None, provider: str | None = None) -> dict:
    graph = _get_graph()
    initial_state: RAGState = {
        "question": question,
        "original_question": question,
        "provider": provider,
        "k": k,
        "chunks": [],
        "top_rerank_score": None,
        "retried": False,
        "answer": "",
    }
    final_state = graph.invoke(initial_state)
    return {"answer": final_state["answer"], "sources": final_state["chunks"]}
