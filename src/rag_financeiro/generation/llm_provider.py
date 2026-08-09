from rag_financeiro import config


def get_llm(provider: str | None = None, temperature: float = 0):
    provider = provider or config.LLM_PROVIDER

    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY não configurada no .env")
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=temperature)

    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY não configurada no .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL, google_api_key=config.GEMINI_API_KEY, temperature=temperature
        )

    raise RuntimeError(f"LLM_PROVIDER desconhecido: {provider!r} (use 'groq' ou 'gemini')")


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
