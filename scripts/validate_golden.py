"""
validate_golden.py

Checagem offline (sem nenhuma chamada de LLM) do golden dataset:

1. o `expected_source` de cada caso existe de fato no vector store;
2. cada fato-chave em **negrito** do `ground_truth` aparece literalmente nos chunks
   daquela fonte.

É o teto do `key_fact_recall` medido em scripts/evaluate.py: se um fato não está
nem no documento inteiro, nenhum retrieval consegue trazê-lo e a nota baixa seria
culpa do dataset, não do RAG.
"""

import sys

from rag_financeiro.evaluation.golden_dataset import load_golden_dataset
from rag_financeiro.evaluation.metrics import BOLD_SPAN, _found, _normalize
from rag_financeiro.vector_store import chroma_store


def chunks_by_source() -> dict[str, str]:
    data = chroma_store.get_all()
    texts: dict[str, list[str]] = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        texts.setdefault(str(meta.get("source", "")), []).append(doc)
    return {src: _normalize(" ".join(docs)) for src, docs in texts.items()}


def main() -> int:
    corpus = chunks_by_source()
    cases = load_golden_dataset()
    problems = 0

    print(f"{len(cases)} casos | {len(corpus)} documentos no índice\n")

    for i, case in enumerate(cases, start=1):
        expected = case.get("expected_source")
        expected = [expected] if isinstance(expected, str) else (expected or [])

        matched = {
            src: text for src in expected
            for fname, text in corpus.items()
            if src.lower() in fname.lower()
        }
        missing_sources = [s for s in expected if not any(s.lower() in f.lower() for f in corpus)]
        if missing_sources:
            problems += 1
            print(f"[{i}] expected_source não existe no índice: {missing_sources}")
            continue

        context = " ".join(matched.values())
        facts = BOLD_SPAN.findall(case["ground_truth"])
        missing_facts = [f for f in facts if not _found(f, context)]
        if missing_facts:
            problems += 1
            print(f"[{i}] {case['question'][:70]}...")
            for fact in missing_facts:
                print(f"      fato não encontrado em {expected}: {fact!r}")

    print(f"\n{len(cases) - problems} casos ok, {problems} com problema.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
