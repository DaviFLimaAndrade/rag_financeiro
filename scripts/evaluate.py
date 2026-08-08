"""
evaluate.py

Avaliação do sistema RAG usando um golden dataset (golden_dataset.jsonl)
e a técnica de LLM-as-judge: um LLM compara a resposta gerada com o
ground_truth e atribui uma nota de correção/fidelidade, capturando
nuance semântica que keyword matching não consegue.

Métricas produzidas:
- Nota média do judge (1-5)
- Taxa de aprovação (score >= 4)
- Acurácia de retrieval (fonte esperada foi recuperada?)
- Latência média
"""

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from rag_llm import get_llm, current_provider_label, extract_text
from retriever import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path("golden_dataset.jsonl")

JUDGE_PROMPT = """Você é um avaliador rigoroso de sistemas de RAG (Retrieval-Augmented Generation).

Compare a RESPOSTA GERADA com a RESPOSTA ESPERADA (ground truth) para a pergunta abaixo.
Avalie a RESPOSTA GERADA numa escala de 1 a 5, considerando:
- Correção factual em relação ao ground truth
- Se cobre os pontos principais do ground truth (não precisa ser idêntica em texto)
- Se não inventa informação que não está no ground truth nem contradiz ele
- Se NÃO inclui informação extra, não solicitada, que não está no ground truth (isso deve reduzir a nota, mesmo que a informação extra seja verdadeira)

REGRAS DE CALIBRAÇÃO (aplique de forma consistente, sempre da mesma forma para o mesmo padrão):

A) Se a RESPOSTA GERADA afirma que não encontrou a informação (ex: "não encontrei", "não há dados"),
   e o ground truth contém um valor/fato específico real -> isso é uma FALHA DE RETRIEVAL.
   Nota-se SEMPRE 2, independentemente de quão "educada" ou "bem escrita" seja a recusa.
   Não dê nota 3 ou 4 para esse padrão só porque a resposta é honesta sobre não ter encontrado.

B) Se a RESPOSTA GERADA afirma que não encontrou a informação, e o ground truth TAMBÉM afirma
   que a informação não está disponível/fora de escopo (caso de guardrail/anti-alucinação)
   -> isso é um ACERTO. Nota 5.

C) Se a RESPOSTA GERADA contém o valor/fato correto, mas com texto adicional não solicitado
   -> nota 4 (não 5), pela falta de concisão.

D) Reserve nota 1 apenas para alucinação real (a resposta inventa ou contradiz um fato do
   ground truth) -- não para "não encontrei".

Escala:
5 = Correta e completa, equivalente em conteúdo ao ground truth (ou acerto de guardrail, regra B)
4 = Correta, mas incompleta, com detalhes a menos, ou com informação extra não solicitada (regra C)
3 = Parcialmente correta, com alguma imprecisão factual relevante (não é caso de "não encontrei")
2 = Falha de retrieval: não encontrou informação que existia no documento (regra A)
1 = Alucinação: inventa ou contradiz o ground truth (regra D)

Pergunta: {question}
Resposta esperada (ground truth): {ground_truth}
Resposta gerada pelo sistema: {generated_answer}

Responda APENAS com um JSON no formato:
{{"score": <número de 1 a 5>, "failure_type": "<'retrieval', 'generation', 'guardrail_ok' ou 'none'>", "justificativa": "<uma frase curta>"}}
"""


@dataclass
class EvalCase:
    question: str
    ground_truth: str
    expected_source: str | None


@dataclass
class EvalResult:
    question: str
    score: int
    failure_type: str
    justificativa: str
    source_ok: bool
    latency_seconds: float
    generated_answer: str


def load_dataset(path: Path) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {path.resolve()}")

    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cases.append(
                EvalCase(
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                    expected_source=item.get("expected_source"),
                )
            )
    return cases


def judge_answer(judge_llm, question: str, ground_truth: str, generated_answer: str) -> tuple[int, str, str]:
    prompt = JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, generated_answer=generated_answer
    )
    response = judge_llm.invoke(prompt)
    text = extract_text(response.content).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning(f"Judge não retornou JSON válido: {text!r}")
        return 1, "generation", "Falha ao interpretar avaliação do judge."

    try:
        parsed = json.loads(match.group())
        return (
            int(parsed["score"]),
            parsed.get("failure_type", "none"),
            parsed.get("justificativa", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning(f"Erro ao parsear resposta do judge: {text!r}")
        return 1, "generation", "Falha ao interpretar avaliação do judge."


def run_evaluation() -> list[EvalResult]:
    cases = load_dataset(DATASET_PATH)
    logger.info(f"{len(cases)} casos carregados de {DATASET_PATH}")
    logger.info(f"Provider/modelo em uso: {current_provider_label()}")

    judge_llm = get_llm(temperature=0)
    results = []

    for i, case in enumerate(cases, start=1):
        logger.info(f"[{i}/{len(cases)}] Avaliando: {case.question}")

        start = time.perf_counter()
        response = answer_question(case.question)
        elapsed = time.perf_counter() - start

        if case.expected_source is None:
            source_ok = True
        else:
            source_ok = any(
                case.expected_source.lower() in str(s.get("source", "")).lower()
                for s in response["sources"]
            )

        score, failure_type, justificativa = judge_answer(
            judge_llm, case.question, case.ground_truth, response["answer"]
        )

        results.append(
            EvalResult(
                question=case.question,
                score=score,
                failure_type=failure_type,
                justificativa=justificativa,
                source_ok=source_ok,
                latency_seconds=round(elapsed, 2),
                generated_answer=response["answer"][:200],
            )
        )

    return results


def print_report(results: list[EvalResult]):
    total = len(results)
    avg_score = sum(r.score for r in results) / total if total else 0
    approved = sum(1 for r in results if r.score >= 4)
    source_accuracy = sum(1 for r in results if r.source_ok) / total if total else 0
    avg_latency = sum(r.latency_seconds for r in results) / total if total else 0
    retrieval_failures = sum(1 for r in results if r.failure_type == "retrieval")

    print("\n" + "=" * 60)
    print("RELATÓRIO DE AVALIAÇÃO — RAG FINANCEIRO (LLM-as-judge)")
    print("=" * 60)
    for r in results:
        status = "✅" if r.score >= 4 else "⚠️" if r.score == 3 else "❌"
        print(f"\n[{status} nota {r.score}/5] ({r.failure_type}) {r.question}")
        print(f"  Justificativa do judge: {r.justificativa}")
        print(f"  Fonte correta: {r.source_ok} | Latência: {r.latency_seconds}s")
        print(f"  Resposta (preview): {r.generated_answer}...")

    print("\n" + "-" * 60)
    print(f"Nota média (judge): {avg_score:.2f}/5")
    print(f"Taxa de aprovação (nota >= 4): {approved}/{total} ({approved/total*100:.0f}%)")
    print(f"Acurácia de retrieval (fonte correta): {source_accuracy*100:.0f}%")
    print(f"Falhas de retrieval (dado existia mas não foi encontrado): {retrieval_failures}/{total}")
    print(f"Latência média: {avg_latency:.2f}s")
    print("=" * 60 + "\n")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": [asdict(r) for r in results],
                "summary": {
                    "provider": current_provider_label(),
                    "avg_score": round(avg_score, 2),
                    "approval_rate": round(approved / total, 2) if total else 0,
                    "source_accuracy": round(source_accuracy, 2),
                    "retrieval_failures": retrieval_failures,
                    "avg_latency_seconds": round(avg_latency, 2),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Resultados salvos em eval_results.json")


if __name__ == "__main__":
    results = run_evaluation()
    print_report(results)