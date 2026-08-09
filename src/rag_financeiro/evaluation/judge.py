import json
import logging
import re

from rag_financeiro.generation.llm_provider import extract_text

logger = logging.getLogger(__name__)

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
