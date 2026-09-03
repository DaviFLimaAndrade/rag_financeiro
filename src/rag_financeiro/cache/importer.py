import json
import re

from rag_financeiro import config
from rag_financeiro.cache.builder import save_cache

DEFAULT_SOURCE = "relatorio_estabilidade_bcb.pdf"

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ROW_RE = re.compile(r"^\|(.+)\|$")


def _clean(text: str) -> str:
    return _BOLD_RE.sub(r"\1", text).strip()


def _is_separator_or_header(cells: list[str]) -> bool:
    first = cells[0].strip().lower()
    if first == "pergunta":
        return True
    return bool(first) and set(first) <= {"-", ":"}


def parse_markdown_table(markdown: str, source: str = DEFAULT_SOURCE) -> list[dict]:
    entries = []
    for line in markdown.splitlines():
        match = _ROW_RE.match(line.strip())
        if not match:
            continue
        cells = [c.strip() for c in match.group(1).split("|")]
        if len(cells) < 3 or _is_separator_or_header(cells):
            continue

        question, answer, page_no = cells[0], cells[1], cells[2]
        entries.append(
            {
                "question": _clean(question),
                "answer": _clean(answer),
                "page_no": page_no.strip(),
                "source": source,
            }
        )
    return entries


def import_from_file(path: str, source: str = DEFAULT_SOURCE, merge: bool = True) -> list[dict]:
    markdown = open(path, encoding="utf-8").read()
    new_entries = parse_markdown_table(markdown, source=source)

    existing = []
    if merge and config.CACHE_PATH.exists():
        existing = json.loads(config.CACHE_PATH.read_text(encoding="utf-8")).get("entries", [])

    seen_questions = {e["question"] for e in existing}
    merged = existing + [e for e in new_entries if e["question"] not in seen_questions]

    save_cache(merged, provider="notebooklm")
    return merged
