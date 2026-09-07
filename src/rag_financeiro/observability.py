"""Tracing via Arize AX (OpenTelemetry + OpenInference).

`setup_tracing()` deve ser chamado antes de qualquer import que puxe langchain_core/langgraph
(LangChainInstrumentor precisa instrumentar antes desses módulos serem importados pra capturar os
spans corretamente) — por isso é a primeira coisa em app/streamlit_app.py e scripts/evaluate.py.

`tracer` cobre os passos do pipeline que não passam por LangChain (retrieval híbrido, rerank,
cache lookup): sem setup_tracing() configurado, a API do OpenTelemetry já devolve um tracer no-op
por padrão, então esses `with tracer.start_as_current_span(...)` são sempre seguros de deixar no
código, configurado ou não.
"""

from opentelemetry import trace

from rag_financeiro import config

tracer = trace.get_tracer("rag_financeiro")

_instrumented = False


def setup_tracing() -> bool:
    global _instrumented
    if _instrumented:
        return True
    if not (config.ARIZE_SPACE_ID and config.ARIZE_API_KEY):
        return False

    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    tracer_provider = register(
        space_id=config.ARIZE_SPACE_ID,
        api_key=config.ARIZE_API_KEY,
        project_name=config.ARIZE_PROJECT_NAME,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _instrumented = True
    return True
