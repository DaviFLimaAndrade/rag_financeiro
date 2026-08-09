import time

import streamlit as st

from rag_financeiro.vector_store import chroma_store
from rag_financeiro.generation.rag_chain import answer_question
from rag_financeiro.generation.llm_provider import current_provider_label

st.set_page_config(page_title="RAG Financeiro — BCB", page_icon="📊")
st.title("📊 RAG Financeiro — Relatório de Estabilidade Financeira (BCB)")

with st.sidebar:
    st.header("Configurações")
    provider = st.radio("Provedor de LLM", options=["groq", "gemini"])
    st.caption(f"Modelo atual: {current_provider_label(provider)}")
    st.metric("Documentos indexados", chroma_store.count())
    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

if chroma_store.count() == 0:
    st.warning(
        "O índice vetorial está vazio. Rode `python scripts/ingest.py` antes de fazer perguntas."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Histórico da conversa
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Fontes"):
                for s in message["sources"]:
                    st.markdown(
                        f"- **{s['source']}**, p.{s['page_no']} — {s['section'] or 'sem seção'}"
                    )

question = None

# Perguntas sugeridas quando ainda não há conversa
if not st.session_state.messages:
    st.markdown("**Experimente perguntar:**")
    cols = st.columns(3)
    sugestoes = [
        "Qual o risco de crédito atual?",
        "Resumo do último relatório",
        "Principais riscos sistêmicos",
    ]
    for col, sugestao in zip(cols, sugestoes):
        if col.button(sugestao):
            question = sugestao

chat_input = st.chat_input("Pergunte algo sobre o relatório de estabilidade financeira...")
if chat_input:
    question = chat_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.time()
        try:
            with st.spinner("Consultando o relatório..."):
                response = answer_question(question, provider=provider)
        except Exception as e:
            st.error(f"Erro ao consultar: {e}")
            st.stop()
        elapsed = time.time() - start

        def stream_answer(text):
            for word in text.split(" "):
                yield word + " "

        st.write_stream(stream_answer(response["answer"]))
        st.caption(f"⏱️ {elapsed:.1f}s • modelo: {current_provider_label(provider)}")

        if response["sources"]:
            with st.expander("Fontes"):
                for s in response["sources"]:
                    st.markdown(
                        f"- **{s['source']}**, p.{s['page_no']} — {s['section'] or 'sem seção'}"
                    )

        col1, col2, _ = st.columns([1, 1, 10])
        with col1:
            st.button("👍", key=f"up_{len(st.session_state.messages)}")
        with col2:
            st.button("👎", key=f"down_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        }
    )