from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from rag_financeiro import config

REQUEST_TIMEOUT_SECONDS = 15

_llm_cache: dict[tuple[str, float], object] = {}


class LLMTimeoutError(Exception):
    pass


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
        raise LLMTimeoutError(
            f"O provedor não respondeu em {timeout:.0f}s. Tente trocar de provedor (Groq/Gemini)."
        )
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
        )

    elif provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY não configurada no .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = config.GEMINI_REWRITE_MODEL if purpose == "rewrite" else config.GEMINI_MODEL
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=config.GEMINI_API_KEY,
            temperature=temperature,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    else:
        raise RuntimeError(f"LLM_PROVIDER desconhecido: {provider!r} (use 'groq' ou 'gemini')")

    _llm_cache[cache_key] = llm
    return llm


def current_provider_label(provider: str | None = None) -> str:
    provider = provider or config.LLM_PROVIDER
    if provider == "groq":
        return f"Groq ({config.GROQ_MODEL})"
    if provider == "gemini":
        return f"Gemini ({config.GEMINI_MODEL})"
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
