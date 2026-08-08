from google import genai

client = genai.Client()

def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """task_type: RETRIEVAL_DOCUMENT na indexação, RETRIEVAL_QUERY na busca."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"task_type": task_type},
    )
    return result.embeddings[0].values

def embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    return [embed_text(t, task_type) for t in texts]