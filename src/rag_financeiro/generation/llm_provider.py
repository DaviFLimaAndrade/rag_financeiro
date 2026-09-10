from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from rag_financeiro import config

REQUEST_TIMEOUT_SECONDS = 15
# Modelos free-tier compartilhados (ex: openrouter/free) tendem a ser mais lentos e menos
# previsíveis que o Groq — nesse ponto a alternativa a esperar mais é falhar de vez, então vale
# dar mais tempo pro fallback antes de desistir.
FALLBACK_TIMEOUT_SECONDS = 30

_llm_cache: dict[tuple[str, float], object] = {}


class LLMTimeoutError(Exception):
    pass


_QUOTA_MARKERS = (
    "rate limit",
    "rate_limit",
    "quota",
    "resource_exhausted",
    "resource exhausted",
    "429",
    "error code: 413",
    "request too large",
    "tokens per minute",
)
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


def _invoke_once(provider, messages, temperature, timeout, purpose):
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


def invoke_with_timeout(
    provider: str | None,
    messages,
    temperature: float = 0,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    purpose: str = "generate",
):
    provider = provider or config.LLM_PROVIDER
    try:
        return _invoke_once(provider, messages, temperature, timeout, purpose), provider
    except LLMTimeoutError:
        raise
    except Exception as error:
        # Cota/rate limit do free tier: tenta uma vez no provider de fallback antes de desistir.
        fallback = config.FALLBACK_PROVIDER
        if not fallback or fallback == provider or classify_error(error) != "quota":
            raise
        fallback_timeout = max(timeout, FALLBACK_TIMEOUT_SECONDS)
        return _invoke_once(fallback, messages, temperature, fallback_timeout, purpose), fallback


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
            max_tokens=config.LLM_MAX_TOKENS,
        )

    elif provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY não configurada no .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=temperature,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_output_tokens=config.LLM_MAX_TOKENS,
        )

    elif provider == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY não configurada no .env")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
            max_tokens=config.LLM_MAX_TOKENS,
        )

    else:
        raise RuntimeError(
            f"provider desconhecido: {provider!r} (use 'groq', 'gemini' ou 'openrouter')"
        )

    _llm_cache[cache_key] = llm
    return llm


def current_provider_label(provider: str | None = None) -> str:
    provider = provider or config.LLM_PROVIDER
    if provider == "groq":
        return f"Groq ({config.GROQ_MODEL})"
    if provider == "gemini":
        return f"Gemini ({config.GEMINI_MODEL})"
    if provider == "openrouter":
        return f"OpenRouter ({config.OPENROUTER_MODEL})"
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
