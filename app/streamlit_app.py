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

APP_NAME = "Lastro"
APP_TAGLINE = "Relatório de Estabilidade Financeira — Banco Central do Brasil"
APP_ICON = ":material/account_balance:"

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON)

# Único CSS do app. Sem ele, um título de conversa longo empurra o botão de
# excluir pra fora da linha e quebra o layout da sidebar — Streamlit não tem
# truncamento nativo de label de botão. Cores e fontes ficam em config.toml.
st.html(
    """
    <style>
      [class*="st-key-convrow_"],
      [class*="st-key-convrow_"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; }
      [class*="st-key-convbtn_"] { min-width: 0; flex: 1 1 auto; }
      [class*="st-key-convbtn_"] button { width: 100%; min-width: 0; }
      [class*="st-key-convbtn_"] button p {
        display: block;
        width: 100%;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      [class*="st-key-convdel_"] { flex: 0 0 auto; }
    </style>
    """
)

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

# Quanto do título cabe no botão da sidebar, e quanto guardamos pro tooltip.
TITLE_MAX_LEN = 32
TITLE_STORE_LEN = 120

SUGESTOES = [
    (":material/credit_score:", "Qual o risco de crédito atual?"),
    (":material/summarize:", "Resumo do último relatório"),
    (":material/warning:", "Principais riscos sistêmicos"),
]


def _make_title(question: str) -> str:
    """Título da conversa: uma linha só, sem quebras que estourem o botão."""
    return " ".join(question.split())[:TITLE_STORE_LEN]


def _short_title(title: str) -> str:
    if len(title) <= TITLE_MAX_LEN:
        return title
    return title[: TITLE_MAX_LEN - 1].rstrip() + "…"


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


def _assistant_message():
    return st.chat_message("assistant", avatar=APP_ICON)


def _render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Fontes ({len(sources)})", icon=":material/menu_book:"):
        for s in sources:
            secao = s["section"] or "sem seção"
            st.markdown(f"- **{s['source']}**, p. {s['page_no']} — :gray[{secao}]")


@st.cache_resource(show_spinner="Carregando modelos locais...")
def _warmup_models():
    local_embedder.warmup()
    reranker.warmup()
    cache_matcher.warmup()
    try:
        get_llm(provider=config.LLM_PROVIDER)
    except RuntimeError:
        pass
    return True


_warmup_models()

st.session_state.setdefault("conversations", {})
st.session_state.setdefault("conv_order", [])
st.session_state.setdefault("current_conv_id", None)

provider = config.LLM_PROVIDER
doc_count = chroma_store.count()

with st.sidebar:
    st.title(APP_NAME)
    st.caption(APP_TAGLINE)

    with st.container(horizontal=True, gap="small"):
        st.badge(
            f"{doc_count} trechos",
            icon=":material/library_books:",
            color="gray",
            help="Trechos do relatório indexados no banco vetorial",
        )
        st.badge(
            current_provider_label(provider),
            icon=":material/neurology:",
            color="green",
            help="Modelo de linguagem em uso",
        )

    st.subheader("Conversas")
    if st.button("Nova conversa", icon=":material/add:", width="stretch", type="primary"):
        st.session_state.current_conv_id = None
        st.rerun()

    if not st.session_state.conv_order:
        st.caption("Suas conversas aparecem aqui.")

    for conv_id in st.session_state.conv_order:
        conv = st.session_state.conversations[conv_id]
        label = _short_title(conv["title"])
        with st.container(
            horizontal=True,
            vertical_alignment="center",
            gap="small",
            key=f"convrow_{conv_id}",
        ):
            if st.button(
                label,
                key=f"convbtn_{conv_id}",
                width="stretch",
                type="secondary" if conv_id == st.session_state.current_conv_id else "tertiary",
                help=conv["title"] if label != conv["title"] else None,
            ):
                st.session_state.current_conv_id = conv_id
                st.rerun()
            if st.button(
                "",
                icon=":material/delete:",
                key=f"convdel_{conv_id}",
                width="content",
                type="tertiary",
                help="Excluir conversa",
            ):
                _delete_conversation(conv_id)
                st.rerun()

conversation = st.session_state.conversations.get(st.session_state.current_conv_id)
messages = conversation["messages"] if conversation else []

if doc_count == 0:
    st.warning(
        "O índice vetorial está vazio. Rode `python scripts/ingest.py` antes de fazer perguntas.",
        icon=":material/warning:",
    )
    st.stop()

for message in messages:
    avatar = APP_ICON if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources") or [])

question = None
is_retry = False

if conversation and conversation.get("last_error"):
    err = conversation["last_error"]
    with _assistant_message():
        st.error(err["message"], icon=err.get("icon", ":material/error:"))
        if st.button("Tentar novamente", icon=":material/refresh:"):
            question = err["question"]
            is_retry = True
            conversation["last_error"] = None

# As sugestões vivem num slot próprio para poderem sumir no mesmo run em que a
# pergunta é enviada. Antes elas continuavam na tela depois de uma pergunta
# digitada, e o clique seguinte caía num run que já não as renderizava — o
# botão simplesmente desaparecia sem responder nada.
suggestions_slot = st.empty()

if not messages and not question:
    with suggestions_slot.container():
        st.header("Sobre o que você quer saber?")
        st.caption(
            "As respostas saem do Relatório de Estabilidade Financeira, com página "
            "e seção citadas."
        )
        with st.container(horizontal=True, gap="small"):
            for i, (icon, sugestao) in enumerate(SUGESTOES):
                if st.button(sugestao, icon=icon, key=f"sug_{i}"):
                    question = sugestao

chat_input = st.chat_input("Pergunte algo sobre o relatório de estabilidade financeira...")
if chat_input:
    question = chat_input
    is_retry = False

if question:
    suggestions_slot.empty()

    if not is_retry:
        if conversation is None:
            conversation = _create_conversation(_make_title(question))
        conversation["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

    with _assistant_message():
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

        actual_provider = response.get("provider_used") or provider
        if actual_provider != provider:
            st.info(
                f"{current_provider_label(provider)} indisponível no momento — resposta gerada "
                f"via {current_provider_label(actual_provider)}.",
                icon=":material/swap_horiz:",
            )

        def stream_answer(text):
            for word in text.split(" "):
                yield word + " "

        st.write_stream(stream_answer(response["answer"]))
        st.caption(
            f":material/schedule: {elapsed:.1f}s • modelo: {current_provider_label(actual_provider)}"
        )

        _render_sources(response["sources"])

    conversation["messages"].append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        }
    )
