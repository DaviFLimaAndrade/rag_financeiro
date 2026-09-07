import time
import uuid

import streamlit as st

from rag_financeiro.observability import setup_tracing

setup_tracing()

from rag_financeiro import config
from rag_financeiro.vector_store import chroma_store
from rag_financeiro.embeddings import local_embedder
from rag_financeiro.retrieval import reranker
from rag_financeiro.cache import matcher as cache_matcher
from rag_financeiro.generation.rag_chain import answer_question
from rag_financeiro.generation.llm_provider import (
    classify_error,
    current_provider_label,
    get_llm,
    LLMTimeoutError,
)

st.set_page_config(page_title="RAG Financeiro — BCB", page_icon=":material/monitoring:")

ERROR_MESSAGES = {
    "quota": "O provedor {provider} está sem cota disponível agora. Tente de novo em instantes.",
    "context": "A conversa ficou grande demais pro modelo processar de uma vez. Inicie uma nova conversa ou faça uma pergunta mais direta.",
    "unknown": "Erro ao consultar {provider}: {detail}",
}
ERROR_ICONS = {
    "quota": ":material/speed:",
    "context": ":material/straighten:",
    "unknown": ":material/error:",
}

TITLE_MAX_LEN = 40
SUGESTOES = [
    "Qual o risco de crédito atual?",
    "Resumo do último relatório",
    "Principais riscos sistêmicos",
]


def _create_conversation(title: str) -> dict:
    conv_id = str(uuid.uuid4())
    conv = {"title": title, "messages": [], "last_error": None}
    st.session_state.conversations[conv_id] = conv
    st.session_state.conv_order.insert(0, conv_id)
    st.session_state.current_conv_id = conv_id
    return conv


def _delete_conversation(conv_id: str) -> None:
    del st.session_state.conversations[conv_id]
    st.session_state.conv_order.remove(conv_id)
    if st.session_state.current_conv_id == conv_id:
        st.session_state.current_conv_id = (
            st.session_state.conv_order[0] if st.session_state.conv_order else None
        )


@st.cache_resource(show_spinner="Carregando modelos locais...")
def _warmup_models():
    local_embedder.warmup()
    reranker.warmup()
    cache_matcher.warmup()
    if config.GROQ_API_KEY:
        get_llm(provider="groq")
    return True


_warmup_models()

st.session_state.setdefault("conversations", {})
st.session_state.setdefault("conv_order", [])
st.session_state.setdefault("current_conv_id", None)

provider = "groq"

with st.sidebar:
    st.title(":material/monitoring: RAG Financeiro")
    st.caption("Relatório de Estabilidade Financeira — BCB")
    st.caption(f"Modelo: {current_provider_label(provider)}")
    st.metric("Documentos indexados", chroma_store.count())

    st.subheader("Conversas")
    if st.button("Nova conversa", icon=":material/add:", width="stretch", type="primary"):
        st.session_state.current_conv_id = None
        st.rerun()

    for conv_id in st.session_state.conv_order:
        conv = st.session_state.conversations[conv_id]
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            if st.button(
                conv["title"],
                key=f"conv_{conv_id}",
                width="stretch",
                type="primary" if conv_id == st.session_state.current_conv_id else "tertiary",
            ):
                st.session_state.current_conv_id = conv_id
                st.rerun()
            if st.button(
                "",
                icon=":material/delete:",
                key=f"del_{conv_id}",
                width="content",
                help="Excluir conversa",
            ):
                _delete_conversation(conv_id)
                st.rerun()

conversation = st.session_state.conversations.get(st.session_state.current_conv_id)
messages = conversation["messages"] if conversation else []

if chroma_store.count() == 0:
    st.warning(
        "O índice vetorial está vazio. Rode `python scripts/ingest.py` antes de fazer perguntas.",
        icon=":material/warning:",
    )
    st.stop()

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Fontes", icon=":material/menu_book:"):
                for s in message["sources"]:
                    st.markdown(
                        f"- **{s['source']}**, p.{s['page_no']} — {s['section'] or 'sem seção'}"
                    )

question = None
is_retry = False

if conversation and conversation.get("last_error"):
    err = conversation["last_error"]
    with st.chat_message("assistant"):
        st.error(err["message"], icon=err.get("icon", ":material/error:"))
        if st.button("Tentar novamente", icon=":material/refresh:"):
            question = err["question"]
            is_retry = True
            conversation["last_error"] = None

if not messages and not question:
    selected = st.pills(
        "Sugestões",
        SUGESTOES,
        label_visibility="collapsed",
        key="suggestion_pills",
    )
    if selected:
        question = selected

chat_input = st.chat_input("Pergunte algo sobre o relatório de estabilidade financeira...")
if chat_input:
    question = chat_input
    is_retry = False

if question:
    if not is_retry:
        if conversation is None:
            title = (
                question if len(question) <= TITLE_MAX_LEN else question[: TITLE_MAX_LEN - 1] + "…"
            )
            conversation = _create_conversation(title)
        conversation["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

    with st.chat_message("assistant"):
        start = time.time()
        max_history_messages = config.MAX_HISTORY_TURNS * 2
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in conversation["messages"][:-1][-max_history_messages:]
        ]
        try:
            with st.spinner("Consultando o relatório..."):
                response = answer_question(question, provider=provider, history=history)
        except LLMTimeoutError:
            msg = f"O provedor {current_provider_label(provider)} não respondeu a tempo. Tente de novo."
            st.error(msg, icon=":material/schedule:")
            conversation["last_error"] = {
                "question": question,
                "message": msg,
                "icon": ":material/schedule:",
            }
            st.stop()
        except Exception as e:
            kind = classify_error(e)
            msg = ERROR_MESSAGES[kind].format(provider=current_provider_label(provider), detail=e)
            icon = ERROR_ICONS[kind]
            st.error(msg, icon=icon)
            conversation["last_error"] = {"question": question, "message": msg, "icon": icon}
            st.stop()
        conversation["last_error"] = None
        elapsed = time.time() - start

        def stream_answer(text):
            for word in text.split(" "):
                yield word + " "

        st.write_stream(stream_answer(response["answer"]))
        st.caption(f":material/schedule: {elapsed:.1f}s • modelo: {current_provider_label(provider)}")

        if response["sources"]:
            with st.expander("Fontes", icon=":material/menu_book:"):
                for s in response["sources"]:
                    st.markdown(
                        f"- **{s['source']}**, p.{s['page_no']} — {s['section'] or 'sem seção'}"
                    )

    conversation["messages"].append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        }
    )
