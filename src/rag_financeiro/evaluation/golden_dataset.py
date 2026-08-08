import json

def load_golden_dataset(path: str = "data/golden_dataset_reduzido.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]