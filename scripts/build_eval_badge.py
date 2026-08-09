import json
from pathlib import Path

RESULTS_PATH = Path("eval_results.json")
BADGE_PATH = Path("eval_badge.json")


def color_for_score(avg_score: float) -> str:
    if avg_score >= 4.5:
        return "brightgreen"
    if avg_score >= 4.0:
        return "green"
    if avg_score >= 3.0:
        return "yellow"
    if avg_score >= 2.0:
        return "orange"
    return "red"


def main():
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    summary = data["summary"]

    avg_score = summary["avg_score"]
    approval_rate = round(summary["approval_rate"] * 100)

    badge = {
        "schemaVersion": 1,
        "label": "RAG eval (LLM-as-judge)",
        "message": f"{avg_score:.1f}/5 · {approval_rate}% aprovação",
        "color": color_for_score(avg_score),
    }
    BADGE_PATH.write_text(json.dumps(badge, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Badge escrito em {BADGE_PATH}: {badge['message']}")


if __name__ == "__main__":
    main()
