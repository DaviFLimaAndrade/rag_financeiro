from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from rag_financeiro import config

REQUEST_TIMEOUT_SECONDS = 15

_llm_cache: dict[tuple[str, float], object] = {}


class LLMTimeoutError(Exception):
    pass


_QUOTA_MARKERS = ("rate limit", "quota", "resource_exhausted", "resource exhausted", "429")
_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "token limit",
    "input is too long",
)


def classify_error(error: Exception) -> str:
    """Classifica um erro de provedor pra dar uma mensagem útil ao usuário.

    Em vez de importar o SDK do provedor só pra pegar o tipo certo de exceção (limite de cota,
    contexto grande demais), inspeciona o texto do erro por marcadores conhecidos.
    """
    text = str(error).lower()
    if any(marker in text for marker in _QUOTA_MARKERS):
        return "quota"
    if any(marker in text for marker in _CONTEXT_MARKERS):
        return "context"
    return "unknown"


def invoke_with_timeout(
    provider: str | None,
    messages,
    temperature: float = 0,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    purpose: str = "generate",
):
    def _call():
        llm = get_llm(provider=provider, temperature=temperature, purpose=purpose)
        return llm.invoke(messages)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        raise LLMTimeoutError(f"O provedor não respondeu em {timeout:.0f}s. Tente de novo.")
    finally:
        executor.shutdown(wait=False)


def get_llm(provider: str | None = None, temperature: float = 0, purpose: str = "generate"):
    provider = provider or config.LLM_PROVIDER
    cache_key = (provider, temperature, purpose)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY não configurada no .env")
        from langchain_groq import ChatGroq
        model_name = config.GROQ_REWRITE_MODEL if purpose == "rewrite" else config.GROQ_MODEL
        llm = ChatGroq(
            model=model_name,
            api_key=config.GROQ_API_KEY,
            temperature=temperature,
            request_timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    else:
        raise RuntimeError(f"LLM_PROVIDER desconhecido: {provider!r} (use 'groq')")

    _llm_cache[cache_key] = llm
    return llm


def current_provider_label(provider: str | None = None) -> str:
    provider = provider or config.LLM_PROVIDER
    if provider == "groq":
        return f"Groq ({config.GROQ_MODEL})"
    return provider


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content)
