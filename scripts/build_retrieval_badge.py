import json
from pathlib import Path

RESULTS_PATH = Path("retrieval_results.json")
BADGE_PATH = Path("retrieval_badge.json")


def color_for_recall(recall: float) -> str:
    if recall >= 0.9:
        return "brightgreen"
    if recall >= 0.8:
        return "green"
    if recall >= 0.6:
        return "yellow"
    if recall >= 0.4:
        return "orange"
    return "red"


def main():
    summary = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["summary"]

    recall = summary.get("avg_key_fact_recall") or 0
    source = summary.get("source_accuracy") or 0
    cases = summary.get("cases", 0)

    badge = {
        "schemaVersion": 1,
        "label": "RAG retrieval",
        "message": f"{recall*100:.0f}% key-fact recall · {source*100:.0f}% fonte · {cases} perguntas",
        "color": color_for_recall(recall),
    }
    BADGE_PATH.write_text(json.dumps(badge, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Badge escrito em {BADGE_PATH}: {badge['message']}")


if __name__ == "__main__":
    main()
