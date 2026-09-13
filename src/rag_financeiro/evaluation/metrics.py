import re
import unicodedata

BOLD_SPAN = re.compile(r"\*\*(.+?)\*\*")
BARE_PERCENT = re.compile(r"^[\d.,]+%$")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def _found(fact: str, context: str) -> bool:
    norm = _normalize(fact)
    if norm in context:
        return True
    if BARE_PERCENT.match(norm):
        return f"= {norm[:-1]}" in context
    return False


def key_fact_recall(ground_truth: str, retrieved_chunks: list[dict]) -> float | None:
    """Fração dos fatos em **negrito** do ground_truth presentes nos chunks recuperados."""
    facts = BOLD_SPAN.findall(ground_truth)
    if not facts:
        return None

    context = _normalize(" ".join(c["text"] for c in retrieved_chunks))
    hits = sum(1 for fact in facts if _found(fact, context))
    return hits / len(facts)
