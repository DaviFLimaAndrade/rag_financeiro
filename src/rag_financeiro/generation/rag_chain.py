from google import genai
from rag_financeiro.retrieval.retriever import retrieve

client = genai.Client()

def answer_question(question: str, k: int = 5) -> dict:
    chunks = retrieve(question, k=k)
    context = "\n\n".join(f"[p.{c['page_no']}] {c['text']}" for c in chunks)
    prompt = f"Responda com base apenas no contexto abaixo, citando a página.\n\nContexto:\n{context}\n\nPergunta: {question}"

    response = client.models.generate_content(model="gemini-3-flash", contents=prompt)
    return {"answer": response.text, "sources": chunks}