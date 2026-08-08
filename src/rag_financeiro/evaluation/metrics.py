def retrieval_recall(retrieved_chunks: list[dict], expected_pages: list[int]) -> float:
    retrieved_pages = {c["page_no"] for c in retrieved_chunks}
    hits = sum(1 for p in expected_pages if p in retrieved_pages)
    return hits / len(expected_pages) if expected_pages else 0.0

# depois dá pra adicionar faithfulness, answer_relevancy etc via RAGAS